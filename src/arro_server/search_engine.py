# src/arro_server/search_engine.py
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from arrowspace import ArrowSpaceBuilder

_ROOT       = Path(__file__).parents[2]
_EMBS_DIR   = _ROOT / "data" / "nomic_embs"
_DATASET    = _ROOT / "data" / "dataset.json"
_DIM        = 768

W_UP, W_LK, W_REP, W_VIEW = 0.35, 0.35, 0.20, 0.10
SAL_WEIGHT  = 0.30
LAM         = 1.0
DEFAULT_TAU = 0.75

# Keys that ArrowSpaceBuilder.build() accepts at graph-construction time.
# tau is a *query-time* parameter and must never be passed to build().
_BUILD_KEYS = frozenset({"eps", "k", "topk", "p", "sigma"})


def _norm(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-9)


def _load_best_params() -> dict:
    """Load eps & k from the latest tuner run.

    tau is stripped here — it is a query-time parameter set by the
    caller at search time, not a corpus-topology constant.
    Only keys in _BUILD_KEYS are forwarded to ArrowSpaceBuilder.build().
    """
    tuner_dir = _ROOT / "notebooks" / "results" / "arrowspace_tuner"
    latest    = sorted(tuner_dir.iterdir())[-1] / "best_params.json"
    raw       = json.loads(latest.read_text())
    params    = raw.get("params", raw)          # unwrap {"params": {...}} if present
    return {k: v for k, v in params.items() if k in _BUILD_KEYS}


class PromptSearchEngine:
    _instance: "PromptSearchEngine | None" = None

    def __init__(self) -> None:
        # 1. embeddings + ids
        embs_path = _EMBS_DIR / f"embeddings_nomic_structured_{_DIM}d_raw.npy"
        ids_path  = _EMBS_DIR / f"embeddings_nomic_structured_{_DIM}d_ids.npy"
        self.embs: np.ndarray = np.load(embs_path).astype(np.float64)
        self.ids:  list[str]  = list(np.load(ids_path))
        assert self.embs.shape[0] == len(self.ids), (
            f"embeddings rows ({self.embs.shape[0]}) != id count ({len(self.ids)})"
        )

        # 2. dataset map  pk_NNNNN -> full JSON record
        dataset: list[dict] = json.loads(_DATASET.read_text())
        self.dataset_map: dict[str, dict] = {r["id"]: r for r in dataset}

        # 3. salience vector (parallel to embs rows)
        records    = [self.dataset_map[pk] for pk in self.ids]
        upvotes    = _norm(np.array([r.get("upvotes", 0)           for r in records], dtype=float))
        likes      = _norm(np.array([r.get("likes", 0)             for r in records], dtype=float))
        reputation = _norm(np.array([r.get("author_reputation", 0) for r in records], dtype=float))
        views      = _norm(np.log1p(np.array([r.get("views", 0)    for r in records], dtype=float)))
        sal_arr    = _norm(W_UP * upvotes + W_LK * likes + W_REP * reputation + W_VIEW * views)
        self.salience: dict[str, float] = {
            self.ids[i]: float(sal_arr[i]) for i in range(len(self.ids))
        }

        # 4. build graph index — eps & k only, tau excluded
        build_params = _load_best_params()
        self.aspace, self.gl = ArrowSpaceBuilder().build(build_params, self.embs)

    @classmethod
    def get(cls) -> "PromptSearchEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── public search ─────────────────────────────────────────────────────────
    def search(
        self,
        query_vec: np.ndarray,
        k: int        = 10,
        tau: float    = DEFAULT_TAU,
        alpha: float  = 0.6,          # kept for MMR salience blend; not forwarded to aspace
    ) -> list[dict]:
        """
        Parameters
        ----------
        query_vec : 768-d float64 nomic embedding (already L2-normalised by caller)
        k         : number of final results after MMR rerank
        tau       : spectral sharpness passed to aspace.search(vec, gl, tau)
        alpha     : cosine-vs-salience blend used inside _mmr(); not an aspace param
        """
        q = np.asarray(query_vec, dtype=np.float64).ravel()

        # aspace.search signature: (vec, gl, tau)  — three positional args, no kwargs
        raw_candidates: list[tuple[int, float]] = self.aspace.search(q, self.gl, tau)

        # keep the top k*3 raw hits for MMR diversity reranking
        candidates = raw_candidates[: k * 3]

        reranked = self._mmr(candidates, k, alpha)

        out = []
        for row_idx, score in reranked:
            pk     = self.ids[row_idx]
            record = dict(self.dataset_map[pk])
            record["_score"]    = round(score, 6)
            record["_salience"] = round(self.salience.get(pk, 0.0), 6)
            record["_tau"]      = tau
            out.append(record)
        return out

    # ── MMR reranker ──────────────────────────────────────────────────────────
    def _mmr(
        self,
        candidates: list[tuple[int, float]],
        k: int,
        alpha: float,
    ) -> list[tuple[int, float]]:
        """Maximal Marginal Relevance reranker with salience boosting.

        rel(i)  = (1 - SAL_WEIGHT) * cosine_score + SAL_WEIGHT * salience
        mmr(i)  = LAM * rel(i) - (1 - LAM) * max_sim_to_selected

        With LAM=1.0 (current default) MMR degenerates to pure relevance
        ordering, so the loop is effectively a salience-blended sort.
        Set LAM < 1.0 to activate diversity.
        """
        def rel(i: int) -> float:
            row_idx, cos_score = candidates[i]
            return (1 - SAL_WEIGHT) * cos_score + SAL_WEIGHT * self.salience.get(self.ids[row_idx], 0.0)

        selected: list[int] = []
        remaining: list[int] = list(range(len(candidates)))

        while len(selected) < k and remaining:
            if not selected:
                best = max(remaining, key=rel)
            else:
                sel_embs = np.array([self.embs[candidates[i][0]] for i in selected])

                def mmr_score(i: int) -> float:
                    e   = self.embs[candidates[i][0]]
                    nrm = np.linalg.norm(sel_embs, axis=1) * np.linalg.norm(e) + 1e-9
                    sim = float(np.max(sel_embs @ e / nrm))
                    return LAM * rel(i) - (1 - LAM) * sim

                best = max(remaining, key=mmr_score)

            selected.append(best)
            remaining.remove(best)

        return [candidates[i] for i in selected]
