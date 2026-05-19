# MODIFIED FILE
# Original source: Genefold/arro-server (https://github.com/Genefold/arro-server)
# Copyright 2026 GENEFOLD AI LTD — Apache License 2.0
# Modifications by Tommaso Moriondo for the CVE drift demo
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from arrowspace import ArrowSpaceBuilder
import polars as pl

log = logging.getLogger(__name__)

_BUILD_KEYS = frozenset({"eps", "k", "topk", "p", "sigma"})
_DIM = 384
DEFAULT_TAU = 0.7

_DEFAULT_BUILD_PARAMS: dict = {"eps": 1.2311, "k": 38, "topk": 19, "p": 2.0, "sigma": None}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_best_params(data_dir: Path) -> dict:
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
            exc, _DEFAULT_BUILD_PARAMS,
        )
        return dict(_DEFAULT_BUILD_PARAMS)


def _find_parquet(data_dir: Path) -> Path | None:
    """Locate cve_corpus.parquet across all known layout variants.

    Variants tried (in order):
      data_dir/cve_embs/cve_corpus.parquet           <- ARRO_SERVER_PROMPT_DATA_DIR=data/data
      data_dir/data/cve_embs/cve_corpus.parquet      <- ARRO_SERVER_PROMPT_DATA_DIR=data   (double-data)
      data_dir/../data/cve_embs/cve_corpus.parquet   <- ARRO_SERVER_PROMPT_DATA_DIR=repo root
      data_dir/cve_embeddings_demo/cve_corpus.parquet
      data_dir/cve_corpus.parquet                    <- flat
    """
    candidates = [
        data_dir / "cve_embs" / "cve_corpus.parquet",
        data_dir / "data" / "cve_embs" / "cve_corpus.parquet",
        data_dir.parent / "data" / "cve_embs" / "cve_corpus.parquet",
        data_dir / "cve_embeddings_demo" / "cve_corpus.parquet",
        data_dir / "cve_corpus.parquet",
    ]
    for p in candidates:
        if p.exists():
            log.info("Found CVE corpus parquet at %s", p)
            return p
    log.warning(
        "CVE corpus parquet not found. Tried:\n%s",
        "\n".join(f"  - {p}" for p in candidates),
    )
    return None


def _load_dataset(data_dir: Path) -> dict[str, str]:
    """Load {cve_id -> text} from parquet. IDs normalised to uppercase."""
    dataset_path = _find_parquet(data_dir)
    if dataset_path is None:
        return {}
    try:
        # Select only the two text columns — skip 'embedding' (large float array)
        lazy_df = pl.scan_parquet(dataset_path)
        schema_names = lazy_df.collect_schema().names()
        cols = [c for c in ("cve_id", "text") if c in schema_names]
        if len(cols) < 2:
            log.error(
                "Parquet at %s is missing required columns. Found: %s",
                dataset_path, schema_names,
            )
            return {}
        df = lazy_df.select(cols).collect()
        return {
            str(k).upper(): str(v)
            for k, v in zip(df["cve_id"].to_list(), df["text"].to_list())
        }
    except Exception as exc:  # noqa: BLE001
        log.error("Failed to load CVE corpus: %s", exc)
        return {}


def _load_embeddings_from_parquet(parquet_path: Path) -> tuple[np.ndarray, list[str]]:
    """Load embeddings and ids directly from the corpus parquet.

    Used as a last-resort fallback when .npy and .zarr files are absent.
    Requires columns: 'embedding' (list<float>) and 'cve_id' (str).
    """
    log.info("Loading embeddings from parquet fallback: %s", parquet_path)
    df = pl.read_parquet(parquet_path, columns=["cve_id", "embedding"])
    ids  = [str(x).upper() for x in df["cve_id"].to_list()]
    embs = np.array(df["embedding"].to_list(), dtype=np.float64)
    log.info("  parquet embeddings shape: %s", embs.shape)
    return embs, ids


def _load_embeddings(data_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Load corpus embeddings and ids.

    Resolution order
    ----------------
    1. data_dir/cve_embeddings_demo/embs_99_to_25.npy  (+ ids_99_to_25.npy)
    2. data_dir/cve_zarr/cve_99_25.zarr
    3. cve_corpus.parquet  (via _find_parquet — has 'embedding' column)
    """
    npy_embs = data_dir / "cve_embeddings_demo" / "embs_99_to_25.npy"
    npy_ids  = data_dir / "cve_embeddings_demo" / "ids_99_to_25.npy"

    if npy_embs.exists() and npy_ids.exists():
        embs = np.load(npy_embs).astype(np.float64)
        # allow_pickle=True required: id arrays are saved as object/str dtype
        ids  = list(np.load(npy_ids, allow_pickle=True))
        log.info("Loaded embeddings from .npy (shape=%s)", embs.shape)
        return embs, [str(x).upper() for x in ids]

    # Zarr fallback
    zarr_candidates = [
        data_dir / "cve_zarr" / "cve_99_25.zarr",
        data_dir.parent / "cve_zarr" / "cve_99_25.zarr",
    ]
    for zarr_path in zarr_candidates:
        if zarr_path.exists():
            try:
                import zarr as _zarr
            except ImportError as exc:
                raise RuntimeError("pip install 'zarr>=3.0'") from exc
            arr  = _zarr.open_array(str(zarr_path), mode="r")
            embs = np.asarray(arr[:], dtype=np.float64)
            if npy_ids.exists():
                ids = [str(x).upper() for x in np.load(npy_ids, allow_pickle=True)]
            else:
                ids = [f"CVE_{i:05d}" for i in range(embs.shape[0])]
            log.info("Loaded embeddings from Zarr (shape=%s)", embs.shape)
            return embs, ids

    # Parquet fallback — parquet already contains the embedding column
    parquet_path = _find_parquet(data_dir)
    if parquet_path is not None:
        try:
            return _load_embeddings_from_parquet(parquet_path)
        except Exception as exc:  # noqa: BLE001
            log.error("Parquet embedding fallback failed: %s", exc)

    raise FileNotFoundError(
        "CveSearchEngine: no embeddings found. Provide one of:\n"
        f"  - {npy_embs} (+ ids_99_to_25.npy)\n"
        "  - data/cve_zarr/cve_99_25.zarr\n"
        "  - data/data/cve_embs/cve_corpus.parquet  (must have 'embedding' column)\n"
        "Set ARRO_SERVER_PROMPT_DATA_DIR correctly."
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class CveSearchEngine:
    """Singleton — ArrowSpace spectral search over the CVE corpus."""

    _instance: "CveSearchEngine | None" = None

    def __init__(self, data_dir: Path) -> None:
        self.embs, self.ids = _load_embeddings(data_dir)
        self._meta: dict[str, str] = _load_dataset(data_dir)

        log.info(
            "CveSearchEngine: %d embeddings, %d meta entries loaded.",
            len(self.ids), len(self._meta),
        )
        if self._meta:
            overlap = sum(1 for i in self.ids if i in self._meta)
            log.info(
                "  ID overlap (embs ∩ meta): %d / %d (%.1f%%)",
                overlap, len(self.ids), 100 * overlap / len(self.ids),
            )

        build_params = _load_best_params(data_dir)
        self.aspace, self.gl = ArrowSpaceBuilder().build(build_params, self.embs)
        self._audit_cache: dict | None = None
        CveSearchEngine._instance = self

    @classmethod
    def get(cls) -> "CveSearchEngine":
        if cls._instance is None:
            from .settings import get_settings
            settings = get_settings()
            cls._instance = cls(Path(settings.prompt_data_dir))
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def search(
        self,
        query: np.ndarray,
        tau: float = DEFAULT_TAU,
        k: int = 10,
    ) -> list[dict]:
        if query.ndim != 1 or query.shape[0] != _DIM:
            raise ValueError(f"Query must be 1-D with dim={_DIM}, got shape {query.shape}")
        query = query.astype(np.float64)

        hits: list[tuple[int, float]] = self.aspace.search(query, self.gl, tau)
        pool_size = len(hits)
        if pool_size == 0:
            return []
        k = min(k, pool_size)

        results: list[dict] = []
        for rank, (corpus_i, score) in enumerate(hits[:k]):
            cve_id  = self.ids[corpus_i]
            content = self._meta.get(cve_id, "")
            results.append({
                "rank"    : rank + 1,
                "id"      : cve_id,
                "title"   : cve_id,
                "tau"     : tau,
                "score"   : float(score),
                "content" : content,
                "body"    : content,
                "salience": 0.0,
                "upvotes" : 0,
                "views"   : 0,
            })
        return results


PromptSearchEngine = CveSearchEngine
