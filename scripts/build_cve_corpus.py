"""Build cve_corpus.parquet from raw NVD CVE JSON feeds.

This script reads one or more NVD JSON files (single-file, glob, or directory)
and writes a Polars parquet that the search engine loads at startup to populate
the ``content`` field in search results.

Output
------
    data/cve_embeddings_demo/cve_corpus.parquet
        Two columns: ``cve_id`` (str, uppercase), ``text`` (str).

Usage
-----
    # Single file
    uv run python scripts/build_cve_corpus.py --src data/raw/nvdcve-1.1-2021.json

    # All files in a directory (recursive)
    uv run python scripts/build_cve_corpus.py --src data/raw/

    # Glob
    uv run python scripts/build_cve_corpus.py --src 'data/raw/nvdcve-*.json'

    # Custom output path
    uv run python scripts/build_cve_corpus.py --src data/raw/ --out data/cve_embs/cve_corpus.parquet

NVD JSON formats supported
---------------------------
- NVD JSON 1.1 feed  (CVE_Items array, cve.CVE_data_meta.ID)
- NVD JSON 2.0 feed  (vulnerabilities array, cve.id)
- Any JSON whose top-level object or any nested list contains dicts
  with an 'id' key matching ^CVE-\\d{4}-\\d{4,} and a 'descriptions' list.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import re
import sys
from pathlib import Path

import polars as pl

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

_CVE_RE = re.compile(r"^CVE-\d{4}-\d{4,}$", re.IGNORECASE)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "data" / "cve_embeddings_demo" / "cve_corpus.parquet"


# ---------------------------------------------------------------------------
# NVD parsers
# ---------------------------------------------------------------------------

def _descriptions_to_text(descriptions: list[dict]) -> str:
    """Return the first English description, or any description if none is 'en'."""
    en = [d.get("value", "") for d in descriptions if d.get("lang", "").startswith("en")]
    if en:
        return en[0].strip()
    fallback = [d.get("value", "") for d in descriptions if d.get("value")]
    return fallback[0].strip() if fallback else ""


def _parse_nvd_11(data: dict) -> list[tuple[str, str]]:
    """Parse NVD JSON 1.1 feed (CVE_Items list)."""
    rows: list[tuple[str, str]] = []
    for item in data.get("CVE_Items", []):
        cve_block = item.get("cve", {})
        cve_id = (
            cve_block.get("CVE_data_meta", {}).get("ID")
            or cve_block.get("id", "")
        )
        if not cve_id or not _CVE_RE.match(cve_id):
            continue
        desc_data = (
            cve_block.get("description", {})
                     .get("description_data", [])
        )
        # 1.1 uses 'value' directly in description_data items
        text = _descriptions_to_text(desc_data)
        if text:
            rows.append((cve_id.upper(), text))
    return rows


def _parse_nvd_20(data: dict) -> list[tuple[str, str]]:
    """Parse NVD JSON 2.0 feed (vulnerabilities list)."""
    rows: list[tuple[str, str]] = []
    for vuln in data.get("vulnerabilities", []):
        cve_block = vuln.get("cve", {})
        cve_id = cve_block.get("id", "")
        if not cve_id or not _CVE_RE.match(cve_id):
            continue
        descriptions = cve_block.get("descriptions", [])
        text = _descriptions_to_text(descriptions)
        if text:
            rows.append((cve_id.upper(), text))
    return rows


def _parse_generic(data: object, depth: int = 0) -> list[tuple[str, str]]:
    """Fallback: walk any JSON structure looking for CVE-like dicts.

    Looks for dicts that have an 'id' matching CVE-YYYY-NNNN and either:
    - a 'descriptions' list  (NVD 2.0-style)
    - a 'description' str    (simplified format)
    """
    rows: list[tuple[str, str]] = []
    if depth > 6:
        return rows
    if isinstance(data, list):
        for item in data:
            rows.extend(_parse_generic(item, depth + 1))
    elif isinstance(data, dict):
        cve_id = str(data.get("id", "") or data.get("cve_id", ""))
        if _CVE_RE.match(cve_id):
            text = ""
            if "descriptions" in data:
                text = _descriptions_to_text(data["descriptions"])
            elif "description" in data:
                desc = data["description"]
                text = desc if isinstance(desc, str) else ""
            if text:
                rows.append((cve_id.upper(), text))
        else:
            for v in data.values():
                rows.extend(_parse_generic(v, depth + 1))
    return rows


def parse_nvd_file(path: Path) -> list[tuple[str, str]]:
    """Parse a single NVD JSON file, auto-detecting the format."""
    log.info("Parsing %s", path)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    if "CVE_Items" in data:
        rows = _parse_nvd_11(data)
        log.info("  NVD 1.1 → %d entries", len(rows))
        return rows

    if "vulnerabilities" in data:
        rows = _parse_nvd_20(data)
        log.info("  NVD 2.0 → %d entries", len(rows))
        return rows

    # Fallback generic walker
    rows = _parse_generic(data)
    log.info("  generic walker → %d entries", len(rows))
    return rows


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def collect_paths(src: str) -> list[Path]:
    """Expand src into a list of .json paths (file, glob, or directory)."""
    p = Path(src)
    if p.is_file():
        return [p]
    if p.is_dir():
        paths = sorted(p.rglob("*.json"))
        log.info("Found %d JSON files under %s", len(paths), p)
        return paths
    # Treat as glob
    paths = sorted(Path(g) for g in glob.glob(src, recursive=True))
    if not paths:
        log.error("No files matched: %s", src)
        sys.exit(1)
    return paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build(src: str, out: Path, overwrite: bool = False) -> None:
    paths = collect_paths(src)
    if not paths:
        log.error("No input files found for: %s", src)
        sys.exit(1)

    all_rows: list[tuple[str, str]] = []
    for path in paths:
        try:
            all_rows.extend(parse_nvd_file(path))
        except Exception as exc:  # noqa: BLE001
            log.warning("Skipping %s (%s)", path, exc)

    if not all_rows:
        log.error(
            "No CVE entries extracted from %d file(s). "
            "Check that the files are valid NVD JSON (1.1 or 2.0 format).",
            len(paths),
        )
        sys.exit(1)

    # Deduplicate — keep last occurrence (latest file wins)
    seen: dict[str, str] = {}
    for cve_id, text in all_rows:
        seen[cve_id] = text

    df = pl.DataFrame({"cve_id": list(seen.keys()), "text": list(seen.values())})
    df = df.sort("cve_id")

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not overwrite:
        log.warning(
            "%s already exists. Pass --overwrite to replace it.", out
        )
        sys.exit(1)

    df.write_parquet(out)
    log.info(
        "Wrote %d CVE entries to %s  (%.1f MB)",
        len(df),
        out,
        out.stat().st_size / 1_048_576,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build cve_corpus.parquet from NVD JSON feeds.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--src",
        required=True,
        help="Path to a JSON file, directory of JSON files, or glob pattern.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"Output parquet path (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing parquet if present.",
    )
    args = parser.parse_args()
    build(args.src, Path(args.out), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
