# MODIFIED FILE
# Original source: Genefold/arro-server (https://github.com/Genefold/arro-server)
# Copyright 2026 GENEFOLD AI LTD — Apache License 2.0
# Modifications by Tommaso Moriondo for the LEAF Prompt-Kaban POC:
#   - Added PromptSearchEngine class with MMR re-ranking and saliency weighting
#   - Added _load_best_params() tuner integration with hardcoded fallback defaults
#   - Added _norm() min-max normalisation helper
# See CHANGES.md for full modification record.
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from arrowspace import ArrowSpaceBuilder

log = logging.getLogger(__name__)

_DIM = 768

W_UP, W_LK, W_REP, W_VIEW = 0.35, 0.35, 0.20, 0.10
SAL_WEIGHT  = 0.30
LAM         = 0.7   # MMR diversity weight: 1.0=pure relevance, 0.0=max diversity
DEFAULT_TAU = 0.75

# Keys that ArrowSpaceBuilder.build() accepts at graph-construction time.
# tau/alpha/lam are query-time parameters and must never be passed to build().
_BUILD_KEYS = frozenset({"eps", "k", "topk", "p", "sigma"})

# Hardcoded fallback params used when tuner results are absent (fresh clone / container).
# Values come from the tuner run documented in the notebook (search_engine_demo.ipynb).
_DEFAULT_BUILD_PARAMS: dict = {"eps": 1.2311, "k": 38, "topk": 19, "p": 2.0}


def _norm(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def _load_best_params(data_dir: Path) -> dict:
    """Load graph build params from the latest tuner run.

    Falls back to _DEFAULT_BUILD_PARAMS when the tuner output directory is
    missing (fresh clone, container, CI).  tau is stripped because it is a
    query-time parameter and must not be passed to ArrowSpaceBuilder.build().
    """
    tuner_dir = data_dir.parent / "notebooks" / "results" / "arrowspace_tuner"
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
        log.info("Loaded ArrowSpace build params from tuner: %s", filtered)
        return filtered
    except (FileNotFoundError, IndexError, KeyError, ValueError, json.JSONDecodeError) as exc:
        log.warning(
            "Tuner results unavailable (%s) — using hardcoded defaults: %s",
            exc,
            _DEFAULT_BUILD_PARAMS,
        )
        return dict(_DEFAULT_BUILD_PARAMS)


class PromptSearchEngine:
    """Singleton that holds the ArrowSpace index and exposes semantic search.

    The data directory is resolved from Settings.prompt_data_dir so the
    engine works correctly both in development layout and inside containers.
    """

    _instance: "PromptSearchEngine | None" = None

    def __init__(self, data_dir: Path) -> None:
        embs_path    = data_dir / "nomic_embs" / f"embeddings_nomic_structured_{_DIM}d_raw.npy"
        ids_path     = data_dir / "nomic_embs" / f"embeddings_nomic_structured_{_DIM}d_ids.npy"
        dataset_path = data_dir / "dataset.json"

        for p in (embs_path, ids_path, dataset_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"PromptSearchEngine: required file missing: {p}\n"
                    f"Set ARRO_SERVER_PROMPT_DATA_DIR to the directory containing "
                    f"dataset.json and nomic_embs/."
                )

        # 1. embeddings + ids
        self.embs: np.ndarray = np.load(embs_path).astype(np.float64)
        self.ids:  list[str]  = list(np.load(ids_path, allow_pickle=True))

        # 2. dataset metadata — build lookup once, not on every search call
        with dataset_path.open() as f:
            self.dataset: list[dict] = json.load(f)
        self._meta: dict[str, dict] = {item["id"]: item for item in self.dataset}

        # 3. build ArrowSpace graph-Laplacian index.
        #    build() signature: build(graph_params: dict | None, items: np.ndarray)
        #    returns (ArrowSpace, GraphLaplacian)
        build_params = _load_best_params(data_dir)
        self.aspace, self.gl = ArrowSpaceBuilder().build(build_params, self.embs)

        PromptSearchEngine._instance = self

    @classmethod
    def get(cls) -> "PromptSearchEngine":
        if cls._instance is None:
            from .settings import get_settings
            settings = get_settings()
            cls._instance = cls(Path(settings.prompt_data_dir))
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Clear the cached singleton instance. Intended for tests and reload scenarios."""
        cls._instance = None

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: np.ndarray,
        tau: float   = DEFAULT_TAU,
        alpha: float = 0.5,
        lam: float   = LAM,
        k: int       = 10,
    ) -> list[dict]:
        """Return top-k results using ArrowSpace spectral search + MMR re-ranking.

        Args:
            query: 768-d query embedding (float64).
            tau:   Spectral sharpness passed to ArrowSpace taumode search.
            alpha: Blend factor. 0.0 = pure spectral, 1.0 = pure cosine similarity.
            lam:   MMR diversity weight. 1.0 = pure relevance, 0.0 = maximum diversity.
            k:     Number of results to return. Must be <= topk set at build time.
        """
        if query.ndim != 1 or query.shape[0] != _DIM:
            raise ValueError(f"Query vector must be 1-D with dim={_DIM}, got shape {query.shape}")

        query = query.astype(np.float64)

        # 1. Spectral retrieval.
        #    Signature: aspace.search(item: np.ndarray[f64, 1d], gl: GraphLaplacian, tau: float)
        #    Returns:   list[tuple[int, float]]  — (corpus_index, score) pairs.
        #    Pool size is fixed at build-time topk; k must be <= topk.
        raw: list[tuple[int, float]] = self.aspace.search(query, self.gl, tau)
        raw_idxs = [r[0] for r in raw]
        raw_sims  = np.array([r[1] for r in raw], dtype=np.float64)

        pool_size = len(raw_idxs)
        if k > pool_size:
            log.warning(
                "Requested k=%d but search pool only has %d results (topk at build time). "
                "Returning %d results.",
                k, pool_size, pool_size,
            )
            k = pool_size

        # 2. Cosine similarity between query and each candidate.
        query_norm = query / (np.linalg.norm(query) + 1e-9)
        pool_embs  = self.embs[raw_idxs]                                      # (pool, D)
        pool_norms = pool_embs / (np.linalg.norm(pool_embs, axis=1, keepdims=True) + 1e-9)
        cos_sims   = pool_norms @ query_norm                                   # (pool,)

        # 3. Alpha-blend: spectral score vs cosine similarity.
        spectral_norm = _norm(raw_sims)
        cos_norm      = _norm(cos_sims)
        relevance     = alpha * cos_norm + (1.0 - alpha) * spectral_norm

        # 4. Saliency: upvotes / likes / uses / views signal blend.
        sal_scores = np.array([
            (
                W_UP   * float(self._meta.get(self.ids[i], {}).get("upvotes", 0)) +
                W_LK   * float(self._meta.get(self.ids[i], {}).get("likes",   0)) +
                W_REP  * float(self._meta.get(self.ids[i], {}).get("uses",    0)) +
                W_VIEW * float(self._meta.get(self.ids[i], {}).get("views",   0))
            )
            for i in raw_idxs
        ], dtype=np.float64)

        # 5. Combined score: relevance blend + saliency.
        combined = (1.0 - SAL_WEIGHT) * relevance + SAL_WEIGHT * _norm(sal_scores)

        # 6. MMR re-ranking.
        #    Use L2-normalised embeddings for inter-candidate cosine similarity
        #    so the diversity penalty is directional, not magnitude-dependent.
        selected : list[int] = []
        remaining = list(range(pool_size))

        while remaining and len(selected) < k:
            rem_arr = np.array(remaining)
            if not selected:
                best_local = int(np.argmax(combined[rem_arr]))
            else:
                sel_norms  = pool_norms[selected]                              # (sel, D)
                sim_to_sel = (pool_norms[rem_arr] @ sel_norms.T).max(axis=1)  # (rem,)
                mmr_scores = lam * combined[rem_arr] - (1.0 - lam) * sim_to_sel
                best_local = int(np.argmax(mmr_scores))
            best = remaining[best_local]
            selected.append(best)
            remaining.remove(best)

        results = []
        for rank, pool_i in enumerate(selected):
            corpus_i = raw_idxs[pool_i]
            item     = self._meta.get(self.ids[corpus_i], {})
            results.append({
                "rank":      rank + 1,
                "id":        self.ids[corpus_i],
                "score":     float(combined[pool_i]),
                "sim":       float(raw_sims[pool_i]),
                "content":   item.get("content", item.get("text", "")),
                "_salience": float(sal_scores[pool_i]),
                "tags":      item.get("tags", []),
                "upvotes":   item.get("upvotes", 0),
                "likes":     item.get("likes", 0),
                "uses":      item.get("uses", 0),
                "views":     item.get("views", 0),
            })
        return results
    