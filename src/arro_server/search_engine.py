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
# tau is a *query-time* parameter and must never be passed to build().
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

        # 3. build ArrowSpace graph-Laplacian index
        build_params = _load_best_params(data_dir)
        builder      = ArrowSpaceBuilder()
        self.aspace, self.gl = builder.build(self.embs, **build_params)

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
        k: int       = 10,
        tau: float   = DEFAULT_TAU,
        alpha: float = 0.5,
        lam: float   = LAM,
    ) -> list[dict]:
        """Return top-k results using ArrowSpace spectral search + MMR re-ranking.

        Args:
            query: 768-d query embedding.
            k:     Number of results to return.
            tau:   Spectral sharpness passed to ArrowSpace taumode search.
            alpha: Blend factor. 0.0 = pure spectral, 1.0 = pure cosine similarity.
                   Intermediate values blend both scores after min-max normalisation.
            lam:   MMR diversity weight. 1.0 = pure relevance, 0.0 = maximum diversity.
        """
        if query.shape[0] != _DIM:
            raise ValueError(f"Query vector must have dim={_DIM}, got {query.shape[0]}")

        # 1. spectral retrieval — fetch a larger candidate pool for MMR
        pool_k   = min(k * 4, len(self.ids))
        q_dict   = {"vector": query.tolist(), "tau": tau}
        raw      = self.aspace.search(q_dict, k=pool_k)
        raw_idxs = [int(r["index"]) for r in raw]
        raw_sims = np.array([float(r.get("score", 0.0)) for r in raw])

        # 2. cosine similarity between query and each candidate
        #    (raw ArrowSpace scores may use a different metric internally)
        query_norm = query / (np.linalg.norm(query) + 1e-9)
        pool_embs  = self.embs[raw_idxs]
        pool_norms = pool_embs / (np.linalg.norm(pool_embs, axis=1, keepdims=True) + 1e-9)
        cos_sims   = pool_norms @ query_norm  # shape: (pool_k,)

        # 3. alpha-blend: spectral score vs cosine similarity
        spectral_norm = _norm(raw_sims)
        cos_norm      = _norm(cos_sims)
        relevance     = alpha * cos_norm + (1.0 - alpha) * spectral_norm

        # 4. saliency: upvotes / likes / uses / views signal blend
        #    Key names match PromptSearchResult and dataset.json fields.
        sal_scores = np.array([
            (
                W_UP   * float(self._meta.get(self.ids[i], {}).get("upvotes", 0)) +
                W_LK   * float(self._meta.get(self.ids[i], {}).get("likes",   0)) +
                W_REP  * float(self._meta.get(self.ids[i], {}).get("uses",    0)) +
                W_VIEW * float(self._meta.get(self.ids[i], {}).get("views",   0))
            )
            for i in raw_idxs
        ])

        # 5. combined score: relevance blend + saliency
        combined = (1.0 - SAL_WEIGHT) * relevance + SAL_WEIGHT * _norm(sal_scores)

        # 6. MMR re-ranking
        emb_pool   = self.embs[raw_idxs]
        selected   : list[int] = []
        remaining  = list(range(len(raw_idxs)))

        while remaining and len(selected) < k:
            rem_arr = np.array(remaining)
            if not selected:
                best_local = int(np.argmax(combined[rem_arr]))
            else:
                sel_embs   = emb_pool[selected]
                sim_to_sel = (emb_pool[rem_arr] @ sel_embs.T).max(axis=1)
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
                # Field name 'content' matches PromptSearchResult.content and
                # dataset.json; '_salience' matches the AliasChoices in the schema.
                "content":   item.get("content", item.get("text", "")),
                "_salience": float(sal_scores[pool_i]),
                "tags":      item.get("tags", []),
                "upvotes":   item.get("upvotes", 0),
                "likes":     item.get("likes", 0),
                "uses":      item.get("uses", 0),
                "views":     item.get("views", 0),
            })
        return results
