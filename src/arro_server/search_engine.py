# MODIFIED FILE
# Original source: Genefold/arro-server (https://github.com/Genefold/arro-server)
# Copyright 2026 GENEFOLD AI LTD — Apache License 2.0
# Modifications by Tommaso Moriondo for the LEAF Prompt-Kaban POC:
#   - Added PromptSearchEngine class with MMR re-ranking and saliency weighting
#   - Added _load_best_params() tuner integration with hardcoded fallback defaults
#   - Added _norm() min-max normalisation helper
#   - Added Zarr fallback for embedding load when .npy files are absent
#   - Aligned salience formula with notebook (author_reputation + log1p(views),
#     per-field _norm before weighting)
# See CHANGES.md for full modification record.
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from arrowspace import ArrowSpaceBuilder

log = logging.getLogger(__name__)

_DIM = 768

# Notebook cell 39 weights — applied to *individually normalised* fields.
W_UP, W_LK, W_REP, W_VIEW = 0.35, 0.35, 0.20, 0.10
SAL_WEIGHT  = 0.30
LAM         = 0.7   # MMR diversity weight: 1.0=pure relevance, 0.0=max diversity
DEFAULT_TAU = 0.75

# Keys that ArrowSpaceBuilder.build() accepts at graph-construction time.
# tau/alpha/lam are query-time parameters and must never be passed to build().
_BUILD_KEYS = frozenset({"eps", "k", "topk", "p", "sigma"})

# Hardcoded fallback params used when tuner results are absent (fresh clone / container).
# Values come from the tuner run documented in the notebook (search_engine_demo.ipynb cell 22):
#   {"eps": 1.2311, "k": 38, "topk": 19, "p": 2.0, "sigma": None}
_DEFAULT_BUILD_PARAMS: dict = {"eps": 1.2311, "k": 38, "topk": 19, "p": 2.0, "sigma": None}


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
    Also fills in `topk` from the notebook default (19) if the tuner output
    lacks it — required to size the candidate pool consistently.
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
        # Tuner output sometimes omits topk; fall back to the notebook value (19)
        # so the candidate pool size matches the notebook's behaviour.
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


def _load_embeddings(data_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Load corpus embeddings and ids, preferring .npy then falling back to Zarr.

    Returns
    -------
    (embs, ids):
        embs: (N, _DIM) float64 array
        ids : list of N string ids (sequential `pk_NNNNN` if no ids source exists)

    Resolution order
    ----------------
    1. data_dir / nomic_embs / embeddings_nomic_structured_{D}d_raw.npy (+ _ids.npy)
    2. <repo_root> / embeddings_nomic_structured_{D}d_raw.zarr
       Ids are sourced from <repo_root>/notebooks/db.json if available,
       otherwise generated as `pk_{i:05d}` aligned to row order.
    """
    npy_embs = data_dir / "nomic_embs" / f"embeddings_nomic_structured_{_DIM}d_raw.npy"
    npy_ids  = data_dir / "nomic_embs" / f"embeddings_nomic_structured_{_DIM}d_ids.npy"
    if npy_embs.exists() and npy_ids.exists():
        embs = np.load(npy_embs).astype(np.float64)
        ids  = list(np.load(npy_ids, allow_pickle=True))
        log.info("Loaded embeddings from .npy: %s (shape=%s)", npy_embs, embs.shape)
        return embs, [str(x) for x in ids]

    # Fallback: load the Zarr that ships with the repo
    repo_root = data_dir.parent if data_dir.name == "data" else data_dir
    zarr_path = repo_root / f"embeddings_nomic_structured_{_DIM}d_raw.zarr"
    if not zarr_path.exists():
        # Search alternative locations the demo may use
        alt = data_dir / f"embeddings_nomic_structured_{_DIM}d_raw.zarr"
        if alt.exists():
            zarr_path = alt
        else:
            raise FileNotFoundError(
                f"PromptSearchEngine: no embeddings found. Looked for:\n"
                f"  - {npy_embs}\n"
                f"  - {zarr_path}\n"
                f"  - {alt}\n"
                f"Set ARRO_SERVER_PROMPT_DATA_DIR to a directory containing nomic_embs/ "
                f"or place embeddings_nomic_structured_{_DIM}d_raw.zarr at the repo root."
            )

    try:
        import zarr  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "zarr is required to load embeddings from .zarr fallback; "
            "pip install zarr>=3.0"
        ) from exc
    import zarr as _zarr
    arr = _zarr.open(str(zarr_path), mode="r")
    embs = np.asarray(arr[:], dtype=np.float64)
    log.info("Loaded embeddings from Zarr fallback: %s (shape=%s)", zarr_path, embs.shape)

    # Ids: prefer notebooks/db.json (id-aligned to row order), else synthesise sequential
    ids: list[str] = []
    db_path = repo_root / "notebooks" / "db.json"
    if db_path.exists():
        try:
            with db_path.open() as f:
                db = json.load(f)
            entries = db.get("_default", db)
            # tinydb-style db: keys are stringified ints "1", "2", ... in insertion order
            ordered = sorted(entries.items(), key=lambda kv: int(kv[0]))
            ids = [str(item.get("id")) for _, item in ordered if item.get("id")]
            if len(ids) != embs.shape[0]:
                log.warning(
                    "db.json id count (%d) != embedding rows (%d); generating sequential ids",
                    len(ids), embs.shape[0],
                )
                ids = []
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            log.warning("Could not parse db.json (%s); generating sequential ids", exc)
            ids = []
    if not ids:
        ids = [f"pk_{i:05d}" for i in range(embs.shape[0])]
    return embs, ids


def _load_dataset(data_dir: Path, ids: list[str]) -> list[dict]:
    """Load dataset.json (prompt metadata) with fallbacks.

    Resolution order
    ----------------
    1. data_dir / dataset.json
    2. <repo_root>/notebooks/db.json (tinydb dump with id + doc_string only —
       useful enough to display content even without engagement metadata)

    Raises FileNotFoundError with an actionable message if nothing is available.
    """
    dataset_path = data_dir / "dataset.json"
    if dataset_path.exists():
        with dataset_path.open() as f:
            return json.load(f)

    repo_root = data_dir.parent if data_dir.name == "data" else data_dir
    db_path = repo_root / "notebooks" / "db.json"
    if db_path.exists():
        try:
            with db_path.open() as f:
                db = json.load(f)
            entries = db.get("_default", db)
            # Build minimal records: id + content (from doc_string).
            # Engagement fields default to 0; salience will be uniform 0 for these
            # entries, which is correct (no signal beyond cosine/spectral relevance).
            records: list[dict] = []
            for _, item in entries.items():
                if "id" not in item:
                    continue
                records.append({
                    "id":       item["id"],
                    "content":  item.get("doc_string", ""),
                    "title":    item.get("title"),
                    "tags":     item.get("tags", []),
                    "upvotes":  item.get("upvotes", 0),
                    "likes":    item.get("likes", 0),
                    "uses":     item.get("uses", 0),
                    "views":    item.get("views", 0),
                    "author_reputation": item.get("author_reputation", 0),
                })
            log.info("Loaded dataset from db.json fallback: %d records", len(records))
            return records
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            log.warning("Could not parse db.json (%s)", exc)

    raise FileNotFoundError(
        f"PromptSearchEngine: dataset.json not found at {dataset_path} "
        f"and no fallback in {repo_root}/notebooks/db.json. "
        f"Provide dataset.json in {data_dir} or set ARRO_SERVER_PROMPT_DATA_DIR."
    )


class PromptSearchEngine:
    """Singleton that holds the ArrowSpace index and exposes semantic search.

    The data directory is resolved from Settings.prompt_data_dir so the
    engine works correctly both in development layout and inside containers.
    """

    _instance: "PromptSearchEngine | None" = None

    def __init__(self, data_dir: Path) -> None:
        # 1. embeddings + ids (with .npy → .zarr fallback)
        self.embs, self.ids = _load_embeddings(data_dir)

        # 2. dataset metadata — build lookup once, not on every search call
        self.dataset: list[dict] = _load_dataset(data_dir, self.ids)
        self._meta: dict[str, dict] = {item["id"]: item for item in self.dataset if "id" in item}

        # 3. build ArrowSpace graph-Laplacian index.
        #    build() signature: build(graph_params: dict | None, items: np.ndarray)
        #    returns (ArrowSpace, GraphLaplacian)
        build_params = _load_best_params(data_dir)
        self.aspace, self.gl = ArrowSpaceBuilder().build(build_params, self.embs)

        # 4. Cached audit artefacts (PCA, CSR Laplacian) — populated lazily on
        #    first /api/prompts/audit call; reused for subsequent requests so
        #    the 20k × 768 PCA fit does not run on every search.
        self._audit_cache: dict | None = None

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

    def _compute_salience(self, raw_idxs: list[int]) -> np.ndarray:
        """Compute salience for a pool of candidates using the notebook formula.

        Each metadata field is normalised individually across the candidate
        pool before being weighted, matching `_norm()` calls in cell 39 of
        search_engine_demo.ipynb.  Views are log1p-transformed before
        normalisation (long-tail distribution).
        """
        records = [self._meta.get(self.ids[i], {}) for i in raw_idxs]
        upvotes_raw    = np.array([float(r.get("upvotes", 0))             for r in records])
        likes_raw      = np.array([float(r.get("likes", 0))               for r in records])
        # Prefer notebook field `author_reputation`; fall back to `uses` so
        # datasets that only carry the older engagement schema still work.
        reputation_raw = np.array([
            float(r.get("author_reputation", r.get("uses", 0)))
            for r in records
        ])
        views_log_raw  = np.log1p(np.array([float(r.get("views", 0))      for r in records]))

        upvotes    = _norm(upvotes_raw)
        likes      = _norm(likes_raw)
        reputation = _norm(reputation_raw)
        views      = _norm(views_log_raw)
        return (W_UP * upvotes + W_LK * likes + W_REP * reputation + W_VIEW * views)

    def search(
        self,
        query: np.ndarray,
        tau: float      = DEFAULT_TAU,
        alpha: float    = 0.5,
        lam: float      = LAM,
        k: int          = 10,
        salience: float = SAL_WEIGHT,
    ) -> list[dict]:
        """Return top-k results using ArrowSpace spectral search + MMR re-ranking.

        Args:
            query:    768-d query embedding (float64).
            tau:      Spectral sharpness passed to ArrowSpace taumode search.
            alpha:    Blend factor. 0.0 = pure spectral, 1.0 = pure cosine similarity.
            lam:      MMR diversity weight. 1.0 = pure relevance, 0.0 = maximum diversity.
            k:        Number of results to return. Must be <= topk set at build time.
            salience: Salience weight in [0, 1]. 0 disables metadata salience,
                      1 gives it full influence in the combined score.
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

        # 4. Saliency: notebook formula — author_reputation + log1p(views), each
        #    field normalised individually before weighting.
        sal_scores = self._compute_salience(raw_idxs)

        # 5. Combined score: relevance blend + saliency.
        sal_w = float(np.clip(salience, 0.0, 1.0))
        combined = (1.0 - sal_w) * relevance + sal_w * _norm(sal_scores)

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
                'tau':     tau,
                "score":     float(combined[pool_i]),
                "sim":       float(raw_sims[pool_i]),
                "content":   item.get("content", item.get("text", "")),
                "salience": float(sal_scores[pool_i]),
                "tags":      item.get("tags", []),
                "upvotes":   item.get("upvotes", 0),
                "likes":     item.get("likes", 0),
                "uses":      item.get("uses", 0),
                "views":     item.get("views", 0),
            })
        return results
