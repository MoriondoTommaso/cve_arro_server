# MODIFIED FILE
# Original source: Genefold/arro-server (https://github.com/Genefold/arro-server)
# Copyright 2026 GENEFOLD AI LTD — Apache License 2.0
# Modifications by Tommaso Moriondo for the LEAF Prompt-Kaban POC:
#   - Added /api/prompts/* route group (health, warm, lambdas, graph_laplacian,
#     audit, search, nl_search) for LEAF Kaban semantic prompt search
#   - Updated /api/health to report prompt_engine_ready and embedder_ready
# Modifications for CVE spectral drift demo:
#   - Added /api/drift/* route group (health, lambdas, score, search)
#     backed by CveDriftEngine (two-period CVE spectral comparison)
# See CHANGES.md for full modification record.
"""API route handlers for arro-server.

All routes are mounted under the /api prefix.
Dataset IDs use '--' as the root/path separator (e.g. 'main--matrix').

Endpoint map
------------
GET  /api/health
GET  /api/datasets
GET  /api/datasets/{id}/metadata
GET  /api/datasets/{id}/data
GET  /api/datasets/{id}/slice
GET  /api/datasets/{id}/stats
GET  /api/datasets/{id}/manifold
GET  /api/datasets/{id}/search               -- keyword (sidecar)
POST /api/datasets/{id}/index                -- build index
GET  /api/datasets/{id}/lambdas              -- eigenvalue distribution
GET  /api/datasets/{id}/graph_laplacian      -- GL metadata
GET  /api/datasets/{id}/items                -- all items from index
GET  /api/datasets/{id}/items/{n}            -- single item
POST /api/datasets/{id}/search               -- spectral vector search
POST /api/datasets/{id}/search/energy        -- energy vector search
POST /api/datasets/{id}/search/hybrid        -- hybrid vector search
POST /api/datasets/{id}/search/linear        -- linear sorted search
POST /api/datasets/{id}/search/batch         -- batch vector search
GET  /api/datasets/{id}/spot/motives/eigen
GET  /api/datasets/{id}/spot/motives/energy
GET  /api/datasets/{id}/spot/subgraphs/centroids
GET  /api/datasets/{id}/spot/subgraphs/motives

GET  /api/prompts/health                     -- embedder + engine readiness
GET  /api/prompts/warm                       -- build aspace+gl, return index stats
GET  /api/prompts/lambdas                    -- eigenvalue distribution for prompt corpus
GET  /api/prompts/graph_laplacian            -- GL metadata for prompt corpus
GET  /api/prompts/audit                      -- full audit payload: degree stats, Fiedler, PCA 2D
POST /api/prompts/search                     -- LEAF kaban semantic search (pre-embedded vector)
POST /api/prompts/nl_search                  -- LEAF kaban NL search (server embeds query)

GET  /api/drift/health                       -- are both CVE period indices ready?
GET  /api/drift/lambdas                      -- eigenvalues for period_a and period_b
GET  /api/drift/score                        -- scalar Wasserstein drift score
POST /api/drift/search                       -- side-by-side spectral search across both periods
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from fastapi import APIRouter, Depends, HTTPException, Query

from .. import __version__
from ..arrowspace_adapter import DEFAULT_GRAPH_PARAMS, ArrowSpaceAdapter
from ..arrowspace_adapter import load as load_arrowspace
from ..errors import OptionalDependencyMissing
from ..search_engine import PromptSearchEngine
from ..settings import Settings, get_settings
from ..slicing import enforce_window_budget, parse_slice
from ..storage import StorageRegistry, get_registry
from ..storage.zarr_fs import zarr_available
from .schemas import (
    DriftSearchRequest,
    DriftSearchResponse,
    DriftPeriodResult,
    IndexBuildRequest,
    NLSearchRequest,
    PromptSearchRequest,
    PromptSearchResponse,
    PromptSearchResult,
    SearchBatchRequest,
    SearchEnergyRequest,
    SearchHybridRequest,
    SearchLinearRequest,
    SearchRequest,
)
from .serializers import array_to_payload

router = APIRouter(prefix="/api")


def _registry() -> StorageRegistry:
    return get_registry()


def _arrowspace() -> ArrowSpaceAdapter:
    return load_arrowspace()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@router.get("/health")
def health(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    engine_ready   = PromptSearchEngine._instance is not None
    embedder_ready = False
    try:
        from ..embedder import EmbedderService
        embedder_ready = EmbedderService._instance is not None
    except Exception:
        pass

    return {
        "status": "ok",
        "version": __version__,
        "zarr_available": zarr_available(),
        "arrowspace_backend": load_arrowspace().backend,
        "arrowspace_available": load_arrowspace().available,
        "data_roots": list(settings.resolved_roots.keys()),
        "prompt_engine_ready": engine_ready,
        "embedder_ready": embedder_ready,
    }


# ---------------------------------------------------------------------------
# Dataset discovery + raw Zarr access
# ---------------------------------------------------------------------------


@router.get("/datasets")
def list_datasets(reg: StorageRegistry = Depends(_registry)) -> dict[str, Any]:
    items = reg.list_datasets()
    return {
        "count": len(items),
        "datasets": [
            {
                "id": s.dataset_id,
                "root": s.root,
                "path": s.path,
                "kind": s.kind,
                "shape": list(s.shape),
                "dtype": s.dtype,
                "chunks": list(s.chunks) if s.chunks else None,
            }
            for s in items
        ],
    }


@router.get("/datasets/{dataset_id}/metadata")
def dataset_metadata(
    dataset_id: str,
    reg: StorageRegistry = Depends(_registry),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    attrs = h.metadata.get("attrs", {})
    return {
        "id":     h.summary.dataset_id,
        "root":   h.summary.root,
        "path":   h.summary.path,
        "kind":   h.summary.kind,
        "shape":  list(h.summary.shape),
        "dtype":  h.summary.dtype,
        "chunks": list(h.summary.chunks) if h.summary.chunks else None,
        "attrs":  attrs,
    }


@router.get("/datasets/{dataset_id}/data")
def dataset_data(
    dataset_id: str,
    offset: int = Query(0, ge=0),
    limit: int  = Query(-1),
    reg: StorageRegistry = Depends(_registry),
    settings: Settings   = Depends(get_settings),
) -> dict[str, Any]:
    h     = reg.open(dataset_id)
    shape = h.summary.shape
    if not shape:
        raise HTTPException(status_code=422, detail="Dataset is a group, not an array.")
    if limit < 0:
        limit = settings.default_window
    limit = min(limit, settings.max_window)
    rs = parse_slice(None, shape, offset=offset, limit=limit)
    try:
        enforce_window_budget(rs, settings.max_window * shape[1] if len(shape) > 1 else settings.max_window)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    arr = h.read_window(rs)
    payload = array_to_payload(arr)
    payload["offset"] = offset
    payload["limit"]  = arr.shape[0]
    return payload


@router.get("/datasets/{dataset_id}/slice")
def dataset_slice(
    dataset_id: str,
    spec: str          = Query(..., description="NumPy-style slice, e.g. '0:10,2:5'."),
    reg: StorageRegistry = Depends(_registry),
    settings: Settings   = Depends(get_settings),
) -> dict[str, Any]:
    h     = reg.open(dataset_id)
    shape = h.summary.shape
    try:
        rs = parse_slice(spec, shape)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        enforce_window_budget(rs, settings.max_window * (shape[1] if len(shape) > 1 else 1))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    arr = h.read_window(rs)
    return array_to_payload(arr)


@router.get("/datasets/{dataset_id}/stats")
def dataset_stats(
    dataset_id: str,
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    if h.fs_path is None:
        return {}
    try:
        return adapter.stats_data(h.summary.dataset_id)    
    except Exception:
        return {} 


@router.get("/datasets/{dataset_id}/manifold")
def dataset_manifold(
    dataset_id: str,
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    if h.fs_path is None:
        return {}
    try:
        return adapter.manifold_data(h.summary.dataset_id) 
    except Exception:
        return {} 


@router.get("/datasets/{dataset_id}/search")
def dataset_keyword_search(
    dataset_id: str,
    q: str               = Query(..., description="Keyword query."),
    limit: int           = Query(20, ge=1, le=200),
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    if h.fs_path is None:
        raise HTTPException(status_code=404, detail="No filesystem path for this dataset.")
    results = adapter.sidecar_search(h.fs_path, q, limit=limit)
    return {"query": q, "results": results}


@router.post("/datasets/{dataset_id}/index")
def build_index(
    dataset_id: str,
    body: IndexBuildRequest | None = None,
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
    settings: Settings   = Depends(get_settings),
) -> dict[str, Any]:
    """Build and persist an ArrowSpace graph-Laplacian index for a dataset."""
    h            = reg.open(dataset_id)
    graph_params = (body.graph_params if body else None) or DEFAULT_GRAPH_PARAMS

    raw_arr = getattr(h, "_arr", None)
    if raw_arr is None:
        raise HTTPException(
            status_code=422,
            detail="Dataset does not expose a raw array (not a Zarr array handle).",
        )

    arr_shape = getattr(raw_arr, "shape", None)
    if arr_shape is not None:
        total_elements = 1
        for dim in arr_shape:
            total_elements *= dim
        max_elements = settings.max_window * (arr_shape[1] if len(arr_shape) > 1 else 1)
        if total_elements > max_elements:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Array has {total_elements:,} elements which exceeds the "
                    f"max_window budget ({max_elements:,}). "
                    "Increase ARRO_SERVER_MAX_WINDOW or use a smaller dataset."
                ),
            )

    try:
        array = np.asarray(raw_arr[:], dtype=np.float64)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to read array: {exc}") from exc

    roots = list(settings.resolved_roots.values())
    if roots:
        index_store = Path(roots[0]).parent / "_arrowspace_index"
    elif h.fs_path is not None:
        index_store = h.fs_path.parent / "_arrowspace_index"
    else:
        index_store = Path(".") / "_arrowspace_index"
    index_store.mkdir(parents=True, exist_ok=True)

    try:
        result = adapter.build_index(
            dataset_id,
            array,
            index_store,
            graph_params=graph_params,
        )
    except OptionalDependencyMissing as exc:
        raise HTTPException(
            status_code=501,
            detail="Index building requires the arrowspace package: pip install arrowspace",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"build_index failed: {exc}") from exc

    return result


@router.get("/datasets/{dataset_id}/lambdas")
def dataset_lambdas(
    dataset_id: str,
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    if h.fs_path is None:
        raise HTTPException(status_code=404, detail="No filesystem path for this dataset.")
    try:
        return adapter.lambdas(h.summary.dataset_id)
    except (FileNotFoundError, NotImplementedError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/datasets/{dataset_id}/graph_laplacian")
def dataset_graph_laplacian(
    dataset_id: str,
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    if h.fs_path is None:
        raise HTTPException(status_code=404, detail="No filesystem path for this dataset.")
    try:
        return adapter.graph_laplacian_info(h.summary.dataset_id)
    except (FileNotFoundError, NotImplementedError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/datasets/{dataset_id}/items")
def dataset_items(
    dataset_id: str,
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    try:
        return adapter.get_all_items(h.summary.dataset_id)
    except (FileNotFoundError, NotImplementedError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/datasets/{dataset_id}/items/{item_index}")
def dataset_item(
    dataset_id: str,
    item_index: int,
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    try:
        return adapter.get_item(h.summary.dataset_id, item_index)
    except (FileNotFoundError, IndexError, NotImplementedError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/datasets/{dataset_id}/search")
def spectral_search(
    dataset_id: str,
    body: SearchRequest,
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    h      = reg.open(dataset_id)
    q_dict = {"vector": body.vector, "tau": body.tau}
    try:
        return adapter.search(h.summary.dataset_id, q_dict)
    except OptionalDependencyMissing:
        raise HTTPException(status_code=501, detail="spectral search requires arrowspace package") from None


@router.post("/datasets/{dataset_id}/search/energy")
def energy_search(
    dataset_id: str,
    body: SearchEnergyRequest,
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    h      = reg.open(dataset_id)
    q_dict = {"vector": body.vector, "k": body.k}
    try:
        return adapter.search_energy(h.summary.dataset_id, q_dict)
    except OptionalDependencyMissing:
        raise HTTPException(status_code=501, detail="energy search requires arrowspace package") from None


@router.post("/datasets/{dataset_id}/search/hybrid")
def hybrid_search(
    dataset_id: str,
    body: SearchHybridRequest,
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    h      = reg.open(dataset_id)
    q_dict = {"vector": body.vector, "alpha": body.alpha}
    try:
        return adapter.search_hybrid(h.summary.dataset_id, q_dict)
    except OptionalDependencyMissing:
        raise HTTPException(status_code=501, detail="hybrid search requires arrowspace package") from None


@router.post("/datasets/{dataset_id}/search/linear")
def linear_search(
    dataset_id: str,
    body: SearchLinearRequest,
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    h      = reg.open(dataset_id)
    q_dict = {"vector": body.vector, "k": body.k}
    try:
        return adapter.search_linear_sorted(h.summary.dataset_id, q_dict)
    except OptionalDependencyMissing:
        raise HTTPException(status_code=501, detail="linear search requires arrowspace package") from None


@router.post("/datasets/{dataset_id}/search/batch")
def batch_search(
    dataset_id: str,
    body: SearchBatchRequest,
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    h      = reg.open(dataset_id)
    q_dict = {"vectors": body.vectors, "tau": body.tau}
    try:
        return adapter.search_batch(h.summary.dataset_id, q_dict)
    except OptionalDependencyMissing:
        raise HTTPException(status_code=501, detail="batch search requires arrowspace package") from None


# ---------------------------------------------------------------------------
# Spot endpoints
# ---------------------------------------------------------------------------


@router.get("/datasets/{dataset_id}/spot/motives/eigen")
def spot_motives_eigen(
    dataset_id: str,
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    try:
        return adapter.spot_motives_eigen(h.summary.dataset_id)
    except (OptionalDependencyMissing, NotImplementedError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/datasets/{dataset_id}/spot/motives/energy")
def spot_motives_energy(
    dataset_id: str,
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    try:
        return adapter.spot_motives_energy(h.summary.dataset_id)
    except (OptionalDependencyMissing, NotImplementedError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/datasets/{dataset_id}/spot/subgraphs/centroids")
def spot_subgraph_centroids(
    dataset_id: str,
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    try:
        return adapter.spot_subg_centroids(h.summary.dataset_id)
    except (OptionalDependencyMissing, NotImplementedError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/datasets/{dataset_id}/spot/subgraphs/motives")
def spot_subgraph_motives(
    dataset_id: str,
    reg: StorageRegistry = Depends(_registry),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    try:
        return adapter.spot_subg_motives(h.summary.dataset_id)
    except (OptionalDependencyMissing, NotImplementedError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Prompt / LEAF Kaban endpoints
# ---------------------------------------------------------------------------


def _get_engine() -> PromptSearchEngine:
    """Return the PromptSearchEngine singleton, raising a JSON 503 on any failure.

    Catches all exception types so that errors from ArrowSpaceBuilder.build()
    (ValueError, RuntimeError, library-internal errors) are returned as a
    proper JSON body instead of the plain-text 'Internal Server Error' that
    FastAPI emits for unhandled exceptions.
    """
    try:
        return PromptSearchEngine.get()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Prompt search engine: required data file not found: {exc}. "
                "Set ARRO_SERVER_PROMPT_DATA_DIR to the directory that contains "
                "dataset.json and nomic_embs/ then restart the server."
            ),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Prompt search engine failed to initialise: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc


def _get_embedder():
    """Return the EmbedderService singleton, raising a JSON 503 on any failure."""
    try:
        from ..embedder import EmbedderService
        return EmbedderService.get()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Embedder service failed to initialise: "
                f"{type(exc).__name__}: {exc}. "
                "Ensure sentence-transformers is installed: "
                "pip install 'arro-server[nlp]'"
            ),
        ) from exc


@router.get("/prompts/health")
def prompts_health(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """Return readiness of the prompt engine and embedder.

    Also reports the resolved prompt_data_dir so operators can verify the
    path without needing server logs.
    """
    engine_ready   = PromptSearchEngine._instance is not None
    embedder_ready = False
    embedder_model = None
    try:
        from ..embedder import EmbedderService
        embedder_ready = EmbedderService._instance is not None
        if embedder_ready:
            embedder_model = EmbedderService._instance.model_name  # type: ignore[union-attr]
    except Exception:
        pass
    status = "ready" if (engine_ready and embedder_ready) else "warming"
    return {
        "status":              status,
        "prompt_engine_ready": engine_ready,
        "embedder_ready":      embedder_ready,
        "embedder_model":      embedder_model,
        # Expose the resolved path so operators can confirm it without logs
        "prompt_data_dir":     settings.prompt_data_dir,
    }


@router.get("/prompts/warm")
def prompts_warm() -> dict[str, Any]:
    """Initialise the search engine (builds ArrowSpace index on first call).

    Returns 200 with a JSON body in all outcomes:
    - 'warm'        → engine + embedder both ready
    - 'engine_only' → engine ready, embedder not installed / not yet loaded
    - raises 503    → engine failed to initialise (detail contains the error)
    """
    engine = _get_engine()  # raises JSON 503 on failure
    embedder_ready = False
    try:
        _get_embedder()
        embedder_ready = True
    except HTTPException:
        pass
    try:
        return {
            "status":         "warm" if embedder_ready else "engine_only",
            "nitems":         engine.aspace.nitems,
            "nfeatures":      engine.aspace.nfeatures,
            "nclusters":      engine.aspace.nclusters,
            "embedder_ready": embedder_ready,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Engine initialised but failed to read index stats: {type(exc).__name__}: {exc}",
        ) from exc


@router.get("/prompts/lambdas")
def prompts_lambdas() -> dict[str, Any]:
    engine = _get_engine()
    try:
        lambdas = engine.aspace.lambdas().tolist()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to compute lambdas: {exc}") from exc
    return {"lambdas": lambdas, "n": len(lambdas)}


@router.get("/prompts/graph_laplacian")
def prompts_graph_laplacian() -> dict[str, Any]:
    engine = _get_engine()
    nnodes = int(engine.gl.nnodes)
    try:
        gl_shape = list(engine.gl.shape)
    except TypeError:
        gl_shape = [nnodes, nnodes]
    return {
        "nitems":    engine.aspace.nitems,
        "nfeatures": engine.aspace.nfeatures,
        "nclusters": engine.aspace.nclusters,
        "gl_nodes":  nnodes,
        "gl_shape":  gl_shape,
    }


@router.get("/prompts/audit")
def prompts_audit() -> dict[str, Any]:
    engine = _get_engine()

    # Return cached payload if available — PCA on 20k × 768 is the expensive
    # part of this handler, and the result does not change unless the engine
    # is rebuilt. Cache lives on the engine singleton and is cleared on
    # PromptSearchEngine.reset().
    cached = getattr(engine, "_audit_cache", None)
    if cached is not None:
        return cached

    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail="scikit-learn is required for /prompts/audit."
        ) from exc


    n = engine.aspace.nfeatures if hasattr(engine, 'aspace') else 0
    degrees = np.zeros(n)
    fiedler_val, spectral_gap = 0.0, 0.0
    hub_count, isolated_count = 0, 0
    hub_fraction, tail_fraction = 0.0, 0.0
    n_edges, sparsity, degree_cv = 0, 1.0, 0.0

    
    try:
        data, indices, indptr, shape = engine.gl.to_csr()
        L = sp.csr_matrix(
            (np.array(data, dtype=np.float64),
             np.array(indices, dtype=np.int32),
             np.array(indptr, dtype=np.int32)),
            shape=tuple(shape),
        )
        
        n = L.shape[0]
        nnz = L.nnz
        
        if n > 0:
            n_edges = (nnz - n) // 2
            sparsity = 1.0 - (nnz / (n * n))
            
            # 3. Extract Degrees
            degrees = np.array(L.diagonal(), dtype=np.float64)
            avg_degree = float(degrees.mean())
            std_degree = float(degrees.std())
            degree_cv = std_degree / avg_degree if avg_degree > 0 else 0.0
            
            # 4. Degree Buckets (Hubs & Tails)
            p10, p90 = np.percentile(degrees, [10, 90])
            hub_count = int((degrees > p90).sum())
            isolated_count = int((degrees < p10).sum())
            hub_fraction = float(hub_count / n)
            tail_fraction = float(isolated_count / n)
            
            # 5. Fiedler & Spectral Gap (via Normalized Laplacian)
            try:
                safe_d = np.where(degrees > 1e-12, degrees, 1e-12)
                d_inv_sqrt = sp.diags(1.0 / np.sqrt(safe_d))
                L_norm = d_inv_sqrt @ L @ d_inv_sqrt
                
                # k=6 to safely grab lambda 2 and lambda 3
                eigs = spla.eigsh(
                    L_norm, k=6, which="SM", 
                    return_eigenvectors=False, tol=1e-5, maxiter=3000
                )
                eigs_sorted = sorted(np.real(eigs))
                fiedler_val = max(0.0, float(eigs_sorted[1])) if len(eigs_sorted) > 1 else 0.0
                spectral_gap = float(eigs_sorted[2] - eigs_sorted[1]) if len(eigs_sorted) > 2 else 0.0
            except Exception as e:
                print(f"[warn] Spectral computation failed: {e}")
                
    except Exception as e:
        print(f"[error] Laplacian reconstruction failed: {e}")

    # 6. Dimension Reduction for Visualization
    #
    # The graph Laplacian is built over the *feature* axis (n_nodes == nfeatures,
    # e.g. 768) — not over the corpus rows. The audit manifold colours each node
    # by its true L_ii degree, so the PCA used for layout must produce exactly
    # n_nodes points. We therefore PCA the columns of the embedding matrix
    # (each feature gets a vector of length nitems) when n_nodes matches
    # nfeatures, and otherwise fall back to PCA over rows.
    try:
        pca = PCA(n_components=2)
        if n > 0 and hasattr(engine, "embs") and engine.embs is not None:
            embs = np.asarray(engine.embs)
            if embs.shape[1] == n:
                node_vecs = embs.T          # (n_nodes, nitems) — one row per feature/node
            elif embs.shape[0] == n:
                node_vecs = embs            # (n_nodes, nfeatures) — one row per sample/node
            else:
                # Mismatched shapes: degrade gracefully — surface still renders below
                node_vecs = embs[:n] if embs.shape[0] >= n else embs
            coords = pca.fit_transform(node_vecs)
            explained_variance = pca.explained_variance_ratio_.tolist()
            coords_list = coords.tolist()
        else:
            coords = np.zeros((0, 2))
            coords_list = []
            explained_variance = []
    except Exception as exc:
        print(f"[warn] PCA over graph nodes failed: {exc}")
        coords = np.zeros((0, 2))
        coords_list = []
        explained_variance = []

    # 7. Pre-compute Graph Laplacian Manifold (server-side, mirrors the
    #    reference Python script). The frontend renders this payload directly
    #    via Plotly Surface + Scatter3d hubs — no IDW guessing in the browser.
    laplacian_manifold: dict[str, Any] | None = None
    pc1_pct: float | None = None
    pc2_pct: float | None = None
    if explained_variance:
        pc1_pct = float(explained_variance[0] * 100) if len(explained_variance) > 0 else None
        pc2_pct = float(explained_variance[1] * 100) if len(explained_variance) > 1 else None
    try:
        if n > 0 and coords.shape[0] == n and float(degrees.std()) > 0.0:
            x_pts = np.asarray(coords[:, 0], dtype=np.float64)
            y_pts = np.asarray(coords[:, 1], dtype=np.float64)
            deg_arr = np.asarray(degrees, dtype=np.float64)

            p05, p95 = np.percentile(deg_arr, [5, 95])
            z_pts = np.clip(deg_arr, p05, p95)

            grid_res = 100 if n > 5000 else 120
            x_min, x_max = float(x_pts.min()), float(x_pts.max())
            y_min, y_max = float(y_pts.min()), float(y_pts.max())
            # guard against degenerate spans
            if x_max - x_min < 1e-9:
                x_max = x_min + 1e-6
            if y_max - y_min < 1e-9:
                y_max = y_min + 1e-6
            xi = np.linspace(x_min, x_max, grid_res)
            yi = np.linspace(y_min, y_max, grid_res)
            Xi, Yi = np.meshgrid(xi, yi)

            pts_xy = np.column_stack([x_pts, y_pts])
            Zi = griddata(pts_xy, z_pts, (Xi, Yi), method="cubic")
            mask = np.isnan(Zi)
            if mask.any():
                Zi_nn = griddata(pts_xy, z_pts, (Xi, Yi), method="nearest")
                Zi = np.where(mask, Zi_nn, Zi)
            Zi = gaussian_filter(Zi, sigma=2.5)

            # Hubs: top 15% by degree, like the reference script.
            hub_thr = float(np.percentile(deg_arr, 85))
            hub_mask = deg_arr > hub_thr
            hub_idx = np.flatnonzero(hub_mask)
            hub_x = x_pts[hub_idx]
            hub_y = y_pts[hub_idx]
            hub_z = deg_arr[hub_idx]
            ids_list = engine.ids if hasattr(engine, "ids") else []
            hub_text = []
            for i in hub_idx.tolist():
                label = ids_list[i] if i < len(ids_list) else None
                if label is None:
                    label = f"node {i}"
                hub_text.append(f"{label} (Lᵢᵢ={float(deg_arr[i]):.3f})")

            x_label = f"PC1 ({pc1_pct:.1f}%)" if pc1_pct is not None else "PC1"
            y_label = f"PC2 ({pc2_pct:.1f}%)" if pc2_pct is not None else "PC2"
            subtitle = (
                f"Node connectivity as proxy for local manifold curvature "
                f"(PC1 {pc1_pct:.1f}%, PC2 {pc2_pct:.1f}%)"
                if pc1_pct is not None and pc2_pct is not None
                else "Node connectivity as proxy for local manifold curvature"
            )

            laplacian_manifold = {
                "n_nodes": int(n),
                "x_grid": xi.tolist(),
                "y_grid": yi.tolist(),
                "z_grid": Zi.tolist(),
                "x_label": x_label,
                "y_label": y_label,
                "subtitle": subtitle,
                "degree_p05": float(p05),
                "degree_p95": float(p95),
                "degree_min": float(deg_arr.min()),
                "degree_max": float(deg_arr.max()),
                "degree_std": float(deg_arr.std()),
                "hub_threshold": hub_thr,
                "hub_x": hub_x.tolist(),
                "hub_y": hub_y.tolist(),
                "hub_z": hub_z.tolist(),
                "hub_degree": hub_z.tolist(),
                "hub_text": hub_text,
                "title": f"Graph Laplacian Manifold ({int(n)} nodes)",
            }
            print(
                f"[audit] laplacian_manifold built: n_nodes={n} grid={grid_res} "
                f"deg min/max/std={float(deg_arr.min()):.4f}/{float(deg_arr.max()):.4f}/"
                f"{float(deg_arr.std()):.4f} hubs={int(hub_mask.sum())}"
            )
        else:
            print(
                f"[audit] laplacian_manifold skipped: n_nodes={n} "
                f"coords_rows={int(coords.shape[0]) if hasattr(coords,'shape') else 0} "
                f"deg_std={float(degrees.std()) if n>0 else 0.0:.6g}"
            )
    except Exception as exc:
        print(f"[warn] laplacian_manifold pre-compute failed: {exc}")
        laplacian_manifold = None

    # Build-time graph params used to construct ArrowSpace, exposed so the
    # frontend can display the manifold/build settings in sync with the engine.
    build_params: dict[str, Any] = {}
    try:
        gl_params = getattr(engine.gl, "graph_params", None)
        if isinstance(gl_params, dict) and gl_params:
            build_params = dict(gl_params)
    except Exception:
        build_params = {}
    if not build_params:
        from ..search_engine import _DEFAULT_BUILD_PARAMS
        build_params = dict(_DEFAULT_BUILD_PARAMS)
    # Ensure sigma key is always present (notebook explicitly sets sigma=None)
    build_params.setdefault("sigma", None)

    payload = {
        "graph_stats": {
            "n_nodes": n,
            "n_edges": n_edges,
            "sparsity": float(sparsity),
        },
        "degree_stats": {
            "min": float(degrees.min()) if n > 0 else 0.0,
            "max": float(degrees.max()) if n > 0 else 0.0,
            "mean": float(degrees.mean()) if n > 0 else 0.0,
            "std": float(degrees.std()) if n > 0 else 0.0,
            "cv": float(degree_cv),
            "hubs": {
                "count": hub_count,
                "fraction": hub_fraction
            },
            "tails": {
                "count": isolated_count,
                "fraction": tail_fraction
            }
        },
        # Per-node Laplacian diagonal (L_ii = degree of node i). Needed by the
        # audit frontend to colour/elevate the 3D manifold surface like the
        # reference Plotly script.
        "degrees": degrees.tolist() if n > 0 else [],
        "spectral_stats": {
            "fiedler_value": float(fiedler_val),
            "spectral_gap": float(spectral_gap)
        },
        "pca_2d": coords_list,
        "pca_explained_variance": explained_variance,
        "ids": engine.ids if hasattr(engine, 'ids') else [],
        "build_params": build_params,
        "laplacian_manifold": laplacian_manifold,
    }
    # Stash on the engine singleton so subsequent calls return instantly.
    try:
        engine._audit_cache = payload
    except AttributeError:
        pass
    return payload


@router.post("/prompts/search", response_model=PromptSearchResponse)
def prompts_search(body: PromptSearchRequest) -> PromptSearchResponse:
    """Run the LEAF semantic search using a pre-computed 768-d nomic embedding.

    Use this endpoint when the caller has already embedded the query (e.g. a
    long-lived client batch process).  For natural-language queries, prefer
    POST /api/prompts/nl_search, which embeds the text server-side.
    """
    engine    = _get_engine()
    query_vec = np.asarray(body.vector, dtype=np.float64)
    try:
        raw = engine.search(
            query_vec,
            k=body.k,
            tau=body.tau,
            alpha=body.alpha,
            lam=body.lam,
            salience=body.salience,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PromptSearchResponse(
        query=None,
        k=body.k,
        tau=body.tau,
        lam=body.lam,
        results=[PromptSearchResult(**r) for r in raw],
        result_count=len(raw),
    )


@router.post("/prompts/nl_search", response_model=PromptSearchResponse)
def prompts_nl_search(body: NLSearchRequest) -> PromptSearchResponse:
    embedder  = _get_embedder()
    engine    = _get_engine()
    query_vec = embedder.embed(body.query)
    try:
        raw = engine.search(
            query_vec,
            k=body.k,
            tau=body.tau,
            alpha=body.alpha,
            lam=body.lam,
            salience=body.salience,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PromptSearchResponse(
        query=body.query,
        k=body.k,
        tau=body.tau,
        lam=body.lam,
        results=[PromptSearchResult(**r) for r in raw],
        result_count=len(raw),
    )


# ---------------------------------------------------------------------------
# CVE spectral drift endpoints
# ---------------------------------------------------------------------------


def _get_drift_engine():
    """Return the CveDriftEngine singleton, raising JSON 503 on failure."""
    try:
        from ..drift_engine import CveDriftEngine
        return CveDriftEngine.get()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                f"CVE drift engine: embedding file not found: {exc}. "
                "Set ARRO_SERVER_CVE_PERIOD_A and ARRO_SERVER_CVE_PERIOD_B to valid paths "
                "and restart the server."
            ),
        ) from exc
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail=f"CVE drift engine requires pyarrowspace: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"CVE drift engine failed to initialise: {type(exc).__name__}: {exc}",
        ) from exc


@router.get("/drift/health")
def drift_health(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """Report readiness of the two CVE period indices.

    Returns the configured paths and whether the engine singleton is built.
    Does NOT trigger engine initialisation — call GET /api/drift/score or
    POST /api/drift/search to warm it up.
    """
    from ..drift_engine import CveDriftEngine
    engine_ready = CveDriftEngine._instance is not None
    return {
        "status": "ready" if engine_ready else "cold",
        "engine_ready": engine_ready,
        "cve_period_a": settings.cve_period_a,
        "cve_period_b": settings.cve_period_b,
    }


@router.get("/drift/lambdas")
def drift_lambdas() -> dict[str, Any]:
    """Return eigenvalue distributions for both CVE periods.

    Response shape::

        {
            "period_a": {"label": "period_a", "lambdas": [...], "n": 128},
            "period_b": {"label": "period_b", "lambdas": [...], "n": 128},
            "drift_score": 0.0034
        }
    """
    engine = _get_drift_engine()
    return {
        "period_a": {
            "label": engine.period_a.label,
            "lambdas": engine.period_a.lambdas,
            "n": len(engine.period_a.lambdas),
        },
        "period_b": {
            "label": engine.period_b.label,
            "lambdas": engine.period_b.lambdas,
            "n": len(engine.period_b.lambdas),
        },
        "drift_score": engine.drift_score,
    }


@router.get("/drift/score")
def drift_score() -> dict[str, Any]:
    """Return the scalar Wasserstein-1 drift score between the two CVE periods.

    This is the cheapest way to check whether spectral drift is significant
    after a data reload. The score is recomputed each time the engine is
    rebuilt (call CveDriftEngine.reset() + this endpoint to refresh).
    """
    engine = _get_drift_engine()
    return {
        "drift_score": engine.drift_score,
        "period_a": engine.period_a.label,
        "period_b": engine.period_b.label,
        "period_a_n_lambdas": len(engine.period_a.lambdas),
        "period_b_n_lambdas": len(engine.period_b.lambdas),
        "interpretation": (
            "low" if engine.drift_score < 0.01
            else "medium" if engine.drift_score < 0.05
            else "high"
        ),
    }


@router.post("/drift/search", response_model=DriftSearchResponse)
def drift_search(body: DriftSearchRequest) -> DriftSearchResponse:
    """Run the same query vector against both CVE period indices.

    Returns side-by-side spectral search results together with the current
    drift score, so callers can compare how the same concept is positioned
    differently in the 1999-2014 vs 2015-2025 CVE embedding spaces.
    """
    engine    = _get_drift_engine()
    query_vec = np.asarray(body.vector, dtype=np.float64)
    raw       = engine.search_both(query_vec, k=body.k, tau=body.tau)
    return DriftSearchResponse(
        drift_score=raw["drift_score"],
        period_a=DriftPeriodResult(**raw["period_a"]),
        period_b=DriftPeriodResult(**raw["period_b"]),
    )
