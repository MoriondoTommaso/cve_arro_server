# MODIFIED FILE
# Original source: Genefold/arro-server (https://github.com/Genefold/arro-server)
# Copyright 2026 GENEFOLD AI LTD — Apache License 2.0
# Modifications by Tommaso Moriondo for the CVE drift demo:
#   - Ported from Prompt-Kaban POC -> CVE corpus
#   - Fixed 5 bugs (see inline NOTE comments)
#   - Removed MMR re-ranking and salience weighting
#   - Pure ArrowSpace spectral search only
#   - Zarr fallback for embedding load when .npy files are absent
# Fix: settings.data_dir -> settings.prompt_data_dir (field was renamed
#      in settings.py; this caused 503 on every /api/prompts/* call)
# Fix: _load_dataset path & id normalisation so content is never empty
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from arrowspace import ArrowSpaceBuilder
import polars as pl

log = logging.getLogger(__name__)

# Keys that ArrowSpaceBuilder.build() accepts at graph-construction time.
# tau is a query-time parameter and must NEVER be passed to build().
_BUILD_KEYS = frozenset({"eps", "k", "topk", "p", "sigma"})
_DIM = 384
DEFAULT_TAU = 0.7

# Hardcoded fallback params used when tuner results are absent (fresh clone / CI).
# Values from the tuner run in search_engine_demo.ipynb cell 22.
_DEFAULT_BUILD_PARAMS: dict = {"eps": 1.2311, "k": 38, "topk": 19, "p": 2.0, "sigma": None}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_best_params(data_dir: Path) -> dict:
    """Load graph build params from the latest tuner run.

    Falls back to _DEFAULT_BUILD_PARAMS when the tuner output directory is
    missing (fresh clone, container, CI). tau is stripped because it is a
    query-time parameter and must not be passed to ArrowSpaceBuilder.build().
    """
    tuner_dir = data_dir.parent / "notebooks" / "results" / "arrowspace_tuner"
    try:
        candidates = sorted(tuner_dir.iterdir())
        if not candidates:
            raise FileNotFoundError("no tuner run directories found")
        latest = candidates[-1] / "best_params.json"
        raw = json.loads(latest.read_text())
        params = raw.get("params", raw)
        filtered = {k: v for k, v in params.items() if k in _BUILD_KEYS}
        if not filtered:
            raise ValueError("best_params.json contained no recognised build keys")
        filtered.setdefault("topk", _DEFAULT_BUILD_PARAMS["topk"])
        log.info("Loaded ArrowSpace build params from tuner: %s", filtered)
        return filtered
    except (FileNotFoundError, IndexError, KeyError, ValueError, json.JSONDecodeError) as exc:
        log.warning(
            "Tuner results unavailable (%s) — using hardcoded defaults: %s",
            exc,
            _DEFAULT_BUILD_PARAMS,
        )
        return dict(_DEFAULT_BUILD_PARAMS)


def _load_embeddings(data_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Load corpus embeddings and ids, preferring .npy then falling back to Zarr.

    Resolution order
    ----------------
    1. data_dir/cve_embeddings_demo/embs_99_to_25.npy  (+ ids_99_to_25.npy)
    2. data_dir/cve_zarr/cve_99_25.zarr
    """
    npy_embs = data_dir / "cve_embeddings_demo" / "embs_99_to_25.npy"
    npy_ids  = data_dir / "cve_embeddings_demo" / "ids_99_to_25.npy"

    if npy_embs.exists() and npy_ids.exists():
        embs = np.load(npy_embs).astype(np.float64)
        ids  = list(np.load(npy_ids, allow_pickle=True))
        log.info("Loaded embeddings from .npy: %s (shape=%s)", npy_embs, embs.shape)
        return embs, [str(x) for x in ids]

    # Zarr fallback
    repo_root = data_dir.parent if data_dir.name == "data" else data_dir
    zarr_path = repo_root / "cve_zarr" / "cve_99_25.zarr"
    if not zarr_path.exists():
        alt = data_dir / "cve_zarr" / "cve_99_25.zarr"
        if alt.exists():
            zarr_path = alt
        else:
            raise FileNotFoundError(
                "CveSearchEngine: no embeddings found. Looked for:\n"
                f"  - {npy_embs}\n"
                f"  - {zarr_path}\n"
                f"  - {alt}\n"
                "Set ARRO_SERVER_PROMPT_DATA_DIR to a directory containing cve_embeddings_demo/"
            )

    try:
        import zarr as _zarr
    except ImportError as exc:
        raise RuntimeError(
            "zarr is required for the .zarr fallback; pip install 'zarr>=3.0'"
        ) from exc

    arr  = _zarr.open_array(str(zarr_path), mode="r")
    embs = np.asarray(arr[:], dtype=np.float64)

    if npy_ids.exists():
        ids = [str(x) for x in np.load(npy_ids, allow_pickle=True)]
    else:
        n = embs.shape[0]
        ids = [f"cve_{i:05d}" for i in range(n)]
        log.warning("ids .npy not found — generated %d sequential ids", n)

    log.info("Loaded embeddings from Zarr fallback: %s (shape=%s)", zarr_path, embs.shape)
    return embs, ids


def _find_parquet(data_dir: Path) -> Path | None:
    """Locate cve_corpus.parquet trying several layout variants.

    Layout variants supported:
      data_dir/cve_embs/cve_corpus.parquet          (primary)
      data_dir/cve_embeddings_demo/cve_corpus.parquet
      data_dir/cve_corpus.parquet                   (flat)
    Returns None if not found anywhere.
    """
    candidates = [
        data_dir / "cve_embs" / "cve_corpus.parquet",
        data_dir / "cve_embeddings_demo" / "cve_corpus.parquet",
        data_dir / "cve_corpus.parquet",
    ]
    for p in candidates:
        if p.exists():
            log.info("Found CVE corpus parquet at %s", p)
            return p
    return None


def _load_dataset(data_dir: Path) -> dict[str, str]:
    """Load CVE corpus parquet and return a {cve_id: text} mapping.

    IDs are normalised to uppercase (CVE-YYYY-NNNNN) so they always match
    the ids produced by _load_embeddings regardless of case.

    Returns an empty dict if the file is absent (non-fatal: search still works
    but result dicts will have empty content fields).
    """
    dataset_path = _find_parquet(data_dir)
    if dataset_path is None:
        log.warning(
            "CVE corpus parquet not found in %s (tried cve_embs/, cve_embeddings_demo/, ./) "
            "— content fields will be empty until the parquet is present.",
            data_dir,
        )
        return {}
    try:
        lazy_df = pl.scan_parquet(dataset_path)
        df = lazy_df.select(["cve_id", "text"]).collect()
        # Normalise IDs to uppercase so .npy IDs (e.g. 'CVE-2021-1234') and
        # parquet IDs always hit the same key.
        return {
            str(k).upper(): str(v)
            for k, v in zip(df["cve_id"].to_list(), df["text"].to_list())
        }
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to load CVE corpus: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class CveSearchEngine:
    """Singleton that holds the ArrowSpace index and exposes CVE semantic search.

    Pure spectral search via ArrowSpace taumode — no re-ranking layer.
    The data directory is resolved from Settings.prompt_data_dir so the engine
    works correctly both in dev layout and inside containers.
    """

    _instance: "CveSearchEngine | None" = None

    def __init__(self, data_dir: Path) -> None:
        # 1. Embeddings + ids (.npy -> Zarr fallback)
        self.embs, self.ids = _load_embeddings(data_dir)

        # Normalise ids to uppercase so _meta lookup always hits
        self.ids = [str(i).upper() for i in self.ids]

        # 2. CVE text corpus — {cve_id: text} lookup, built once at startup
        self._meta: dict[str, str] = _load_dataset(data_dir)

        # 3. Build ArrowSpace graph-Laplacian index
        build_params = _load_best_params(data_dir)
        self.aspace, self.gl = ArrowSpaceBuilder().build(build_params, self.embs)

        # 4. Audit cache — populated lazily on first /api/cve/audit
        self._audit_cache: dict | None = None

        CveSearchEngine._instance = self

    @classmethod
    def get(cls) -> "CveSearchEngine":
        """Return (or lazily create) the singleton instance."""
        if cls._instance is None:
            from .settings import get_settings
            settings = get_settings()
            cls._instance = cls(Path(settings.prompt_data_dir))
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Clear the singleton. For tests and hot-reload scenarios."""
        cls._instance = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: np.ndarray,
        tau: float = DEFAULT_TAU,
        k: int = 10,
    ) -> list[dict]:
        """Return top-k CVE results using ArrowSpace spectral (taumode) search.

        Args:
            query : 384-d query embedding (float64).
            tau   : Spectral sharpness for taumode.
            k     : Number of results to return.

        Returns:
            List of result dicts with 'content' always populated from parquet.
        """
        if query.ndim != 1 or query.shape[0] != _DIM:
            raise ValueError(
                f"Query must be 1-D with dim={_DIM}, got shape {query.shape}"
            )
        query = query.astype(np.float64)

        hits: list[tuple[int, float]] = self.aspace.search(query, self.gl, tau)

        pool_size = len(hits)
        if pool_size == 0:
            return []
        if k > pool_size:
            log.warning(
                "Requested k=%d but search pool has only %d results; returning %d.",
                k, pool_size, pool_size,
            )
            k = pool_size

        results: list[dict] = []
        for rank, (corpus_i, score) in enumerate(hits[:k]):
            cve_id  = self.ids[corpus_i]                   # already uppercased
            content = self._meta.get(cve_id, "")           # parquet text
            results.append({
                "rank"    : rank + 1,
                "id"      : cve_id,
                "title"   : cve_id,                        # PromptSearchResult needs title
                "tau"     : tau,
                "score"   : float(score),
                "content" : content,
                "body"    : content,                       # keep body in sync
                "salience": 0.0,
                "upvotes" : 0,
                "views"   : 0,
            })
        return results


# ---------------------------------------------------------------------------
# Backwards-compat alias for any routes not yet renamed
# ---------------------------------------------------------------------------
PromptSearchEngine = CveSearchEngine
