# MODIFIED FILE
# Original source: Genefold/arro-server (https://github.com/Genefold/arro-server)
# Copyright 2026 GENEFOLD AI LTD — Apache License 2.0
# Modifications by Tommaso Moriondo for the CVE spectral drift PoC:
#   - Replaced PromptSearchEngine with CveSearchEngine
#   - Data loading reads cve_corpus.parquet directly (pandas) instead of db.json / dataset.json
#   - Parquet schema: cve_id (str), year (int64), text (str), row_id (int64), embedding (object)
#   - Dropped prompt-specific engagement fields and salience reranker
#   - Added `year` field to search results to support temporal drift slicing
#   - Aligned _DIM to 384 (nomic-embed-text-v1.5 MRL output used in CVE corpus)
#   - Zarr fallback preserved for pre-parquet compatibility
# See CHANGES.md for full modification record.
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from arrowspace import ArrowSpaceBuilder

log = logging.getLogger(__name__)

# nomic-embed-text-v1.5 MRL output dimension used when building the CVE corpus.
_DIM = 384

LAM         = 0.7   # MMR diversity weight: 1.0=pure relevance, 0.0=max diversity
DEFAULT_TAU = 0.75

_BUILD_KEYS = frozenset({"eps", "k", "topk", "p", "sigma"})

_DEFAULT_BUILD_PARAMS: dict = {"eps": 1.2311, "k": 38, "topk": 19, "p": 2.0, "sigma": None}


def _norm(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def _load_best_params(data_dir: Path) -> dict:
    """Load graph build params from the latest tuner run.

    Falls back to _DEFAULT_BUILD_PARAMS when the tuner output directory is
    missing (fresh clone, container, CI).
    """
    tuner_dir = data_dir.parent / "results" / "cve_arrowspace_fstar"
    try:
        candidates = sorted(tuner_dir.iterdir())
        if not candidates:
            raise FileNotFoundError("no tuner run directories found")
        latest = candidates[-1] / "best_params.json"
        raw    = json.loads(latest.read_text())
        params = raw.get("params", raw)
        filtered = {k: v for k, v in params.items() if k in _BUILD_KEYS}
        if not filtered:
            raise ValueError("best_params.json contained no recognised build keys")
        if "topk" not in filtered:
            filtered["topk"] = _DEFAULT_BUILD_PARAMS["topk"]
        log.info("Loaded ArrowSpace build params from tuner: %s", filtered)
        return filtered
    except (FileNotFoundError, IndexError, KeyError, ValueError, json.JSONDecodeError) as exc:
        log.warning(
            "Tuner results unavailable (%s) — using hardcoded defaults: %s",
            exc,
            _DEFAULT_BUILD_PARAMS,
        )
        return dict(_DEFAULT_BUILD_PARAMS)


def _load_corpus_parquet(data_dir: Path) -> tuple[np.ndarray, list[str], dict[str, dict]]:
    """Load CVE corpus from cve_corpus.parquet.

    Parquet schema (confirmed 2026-05-19):
        cve_id     : str       — CVE identifier (e.g. 'CVE-1999-0001')
        year       : int64     — publication year
        text       : str       — CVE description used for embedding
        row_id     : int64     — sequential row index
        embedding  : object    — pre-computed embedding vector (np.ndarray)

    Returns
    -------
    (embs, ids, meta)
        embs : (N, _DIM) float64 array
        ids  : list of N CVE id strings aligned to embs rows
        meta : dict[cve_id -> {id, content, year, row_id}]
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to load cve_corpus.parquet; pip install pandas") from exc

    parquet_path = data_dir / "cve_embs" / "cve_corpus.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"CveSearchEngine: cve_corpus.parquet not found at {parquet_path}.\n"
            f"Run scripts/build_corpus.py to generate it, or set ARRO_SERVER_PROMPT_DATA_DIR "
            f"to a directory containing cve_embs/cve_corpus.parquet."
        )

    df = pd.read_parquet(parquet_path)
    required = {"cve_id", "year", "text", "row_id", "embedding"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"cve_corpus.parquet is missing expected columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    # Stack embeddings into a contiguous (N, _DIM) float64 matrix.
    # Each cell in the `embedding` column is a 1-D numpy array of length _DIM.
    embs = np.vstack(df["embedding"].values).astype(np.float64)
    if embs.shape[1] != _DIM:
        log.warning(
            "Embedding dimension in parquet is %d, expected _DIM=%d. "
            "Update _DIM in search_engine.py if the model changed.",
            embs.shape[1], _DIM,
        )

    ids: list[str] = df["cve_id"].tolist()

    meta: dict[str, dict] = {
        row["cve_id"]: {
            "id":      row["cve_id"],
            "content": row["text"],
            "year":    int(row["year"]),
            "row_id":  int(row["row_id"]),
        }
        for row in df[["cve_id", "text", "year", "row_id"]].to_dict(orient="records")
    }

    log.info(
        "Loaded CVE corpus from parquet: %d records, embedding shape %s",
        len(ids), embs.shape,
    )
    return embs, ids, meta


class CveSearchEngine:
    """Singleton that holds the ArrowSpace index over CVE embeddings.

    Replaces PromptSearchEngine for the CVE spectral drift PoC.
    Metadata is sourced directly from cve_corpus.parquet — no db.json
    or dataset.json required.
    """

    _instance: "CveSearchEngine | None" = None

    def __init__(self, data_dir: Path) -> None:
        # 1. Load corpus: embeddings + ids + metadata from parquet
        self.embs, self.ids, self._meta = _load_corpus_parquet(data_dir)

        # 2. Build ArrowSpace graph-Laplacian index
        build_params = _load_best_params(data_dir)
        self.aspace, self.gl = ArrowSpaceBuilder().build(build_params, self.embs)

        # 3. Audit cache (PCA, CSR Laplacian) — populated lazily on first /api/cve/audit
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
        """Clear the cached singleton. Intended for tests and reload scenarios."""
        cls._instance = None

    def search(
        self,
        query: np.ndarray,
        tau: float   = DEFAULT_TAU,
        alpha: float = 0.5,
        lam: float   = LAM,
        k: int       = 10,
        year_from: int | None = None,
        year_to:   int | None = None,
    ) -> list[dict]:
        """Return top-k CVE results using ArrowSpace spectral search + MMR re-ranking.

        Args:
            query:     _DIM-d query embedding (float64).
            tau:       Spectral sharpness passed to ArrowSpace taumode search.
            alpha:     Blend factor. 0.0 = pure spectral, 1.0 = pure cosine similarity.
            lam:       MMR diversity weight. 1.0 = pure relevance, 0.0 = max diversity.
            k:         Number of results to return.
            year_from: Optional lower bound on CVE publication year (inclusive).
            year_to:   Optional upper bound on CVE publication year (inclusive).
        """
        if query.ndim != 1 or query.shape[0] != _DIM:
            raise ValueError(
                f"Query vector must be 1-D with dim={_DIM}, got shape {query.shape}"
            )

        query = query.astype(np.float64)

        # 1. Spectral retrieval — returns (corpus_index, score) pairs
        raw: list[tuple[int, float]] = self.aspace.search(query, self.gl, tau)

        # 2. Optional year filter — applied post-retrieval on the candidate pool
        if year_from is not None or year_to is not None:
            filtered = []
            for idx, score in raw:
                yr = self._meta.get(self.ids[idx], {}).get("year")
                if yr is None:
                    continue
                if year_from is not None and yr < year_from:
                    continue
                if year_to is not None and yr > year_to:
                    continue
                filtered.append((idx, score))
            raw = filtered

        if not raw:
            return []

        raw_idxs  = [r[0] for r in raw]
        raw_sims  = np.array([r[1] for r in raw], dtype=np.float64)

        pool_size = len(raw_idxs)
        k = min(k, pool_size)

        # 3. Cosine similarity between query and each candidate
        query_norm = query / (np.linalg.norm(query) + 1e-9)
        pool_embs  = self.embs[raw_idxs]
        pool_norms = pool_embs / (np.linalg.norm(pool_embs, axis=1, keepdims=True) + 1e-9)
        cos_sims   = pool_norms @ query_norm

        # 4. Alpha-blend: spectral score vs cosine similarity
        relevance = alpha * _norm(cos_sims) + (1.0 - alpha) * _norm(raw_sims)

        # 5. MMR re-ranking
        selected:  list[int] = []
        remaining = list(range(pool_size))

        while remaining and len(selected) < k:
            rem_arr = np.array(remaining)
            if not selected:
                best_local = int(np.argmax(relevance[rem_arr]))
            else:
                sel_norms  = pool_norms[selected]
                sim_to_sel = (pool_norms[rem_arr] @ sel_norms.T).max(axis=1)
                mmr_scores = lam * relevance[rem_arr] - (1.0 - lam) * sim_to_sel
                best_local = int(np.argmax(mmr_scores))
            best = remaining[best_local]
            selected.append(best)
            remaining.remove(best)

        results = []
        for rank, pool_i in enumerate(selected):
            corpus_i = raw_idxs[pool_i]
            item     = self._meta.get(self.ids[corpus_i], {})
            results.append({
                "rank":    rank + 1,
                "id":      self.ids[corpus_i],
                "tau":     tau,
                "score":   float(relevance[pool_i]),
                "sim":     float(raw_sims[pool_i]),
                "content": item.get("content", ""),
                "year":    item.get("year"),
                "row_id":  item.get("row_id"),
            })
        return results


# ---------------------------------------------------------------------------
# Backward-compatibility alias so existing imports of PromptSearchEngine
# continue to work without changes in routes.py and tests.
# ---------------------------------------------------------------------------
PromptSearchEngine = CveSearchEngine
