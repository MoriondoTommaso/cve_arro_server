from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Depends, Query

from .. import __version__
from ..arrowspace_adapter import ArrowSpaceAdapter
from ..arrowspace_adapter import DEFAULT_GRAPH_PARAMS
from ..arrowspace_adapter import load as load_arrowspace
from ..errors import DatasetNotFound, InvalidSlice
from ..settings import Settings, get_settings
from ..slicing import enforce_window_budget, parse_slice
from ..storage import StorageRegistry, get_registry
from ..storage.zarr_fs import zarr_available
from .serializers import array_to_payload

router = APIRouter(prefix="/api")


def _registry() -> StorageRegistry:
    return get_registry()


def _arrowspace() -> ArrowSpaceAdapter:
    return load_arrowspace()


def _resolve_dataset_path(settings: Settings, dataset_id: str) -> Path:
    if "/" in dataset_id:
        label, rel = dataset_id.split("/", 1)
    else:
        label, rel = dataset_id, "."
    roots = settings.resolved_roots()
    root = roots.get(label)
    if root is None:
        raise DatasetNotFound(dataset_id)
    return root if rel in (".", "") else root / rel


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/health")
def health(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    return {
        "status": "ok",
        "version": __version__,
        "zarr_available": zarr_available(),
        "arrowspace_backend": load_arrowspace().backend,
        "data_roots": list(settings.resolved_roots().keys()),
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


@router.get("/datasets/{dataset_id:path}/metadata")
def dataset_metadata(
    dataset_id: str,
    reg: StorageRegistry = Depends(_registry),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    return {
        "id": h.summary.dataset_id,
        "root": h.summary.root,
        "path": h.summary.path,
        "kind": h.summary.kind,
        "shape": list(h.summary.shape),
        "dtype": h.summary.dtype,
        "chunks": list(h.summary.chunks) if h.summary.chunks else None,
        "metadata": h.metadata,
    }


@router.get("/datasets/{dataset_id:path}/data")
def dataset_data(
    dataset_id: str,
    offset: int = Query(0, ge=0),
    limit: int | None = Query(None, ge=1),
    reg: StorageRegistry = Depends(_registry),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    """Row-oriented window over the leading axis. Suited to infinite scroll."""
    h = reg.open(dataset_id)
    if not h.summary.shape:
        raise InvalidSlice("dataset has no shape")
    eff_limit = limit or settings.default_window
    rs = parse_slice(None, h.summary.shape, offset=offset, limit=eff_limit)
    try:
        enforce_window_budget(rs, settings.max_window * max(1, _trailing_product(h.summary.shape)))
    except ValueError as e:
        raise InvalidSlice(str(e)) from e
    arr = h.read_window(rs)
    payload = array_to_payload(arr, preview_max_rows=eff_limit)
    total = h.summary.shape[0]
    next_offset = offset + payload["shape"][0] if payload["shape"] else offset
    return {
        "id": h.summary.dataset_id,
        "offset": offset,
        "limit": eff_limit,
        "total": total,
        "next_offset": next_offset if next_offset < total else None,
        "data": payload,
    }


@router.get("/datasets/{dataset_id:path}/slice")
def dataset_slice(
    dataset_id: str,
    spec: str = Query(..., alias="slice", description="Comma-separated per-axis slice spec"),
    reg: StorageRegistry = Depends(_registry),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    h = reg.open(dataset_id)
    try:
        rs = parse_slice(spec, h.summary.shape)
        enforce_window_budget(rs, settings.max_window * max(1, _trailing_product(h.summary.shape)))
    except ValueError as e:
        raise InvalidSlice(str(e)) from e
    arr = h.read_window(rs)
    return {
        "id": h.summary.dataset_id,
        "slice": spec,
        "out_shape": list(arr.shape),
        "data": array_to_payload(arr),
    }


# ---------------------------------------------------------------------------
# Zarr-level stats  (no arrowspace dependency)
# ---------------------------------------------------------------------------

@router.get("/datasets/{dataset_id:path}/stats")
def dataset_stats(
    dataset_id: str,
    reg: StorageRegistry = Depends(_registry),
) -> dict[str, Any]:
    """Basic Zarr array statistics: shape, dtype, chunks, size.

    For ArrowSpace graph-Laplacian analytics use the /index, /lambdas, and
    /search endpoints below.
    """
    h = reg.open(dataset_id)
    return {
        "id": dataset_id,
        "stats": h.stats(),
    }


# ---------------------------------------------------------------------------
# Sidecar manifold  (static JSON sidecar, no arrowspace package required)
# ---------------------------------------------------------------------------

@router.get("/datasets/{dataset_id:path}/manifold")
def dataset_manifold(
    dataset_id: str,
    settings: Settings = Depends(get_settings),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    """Read ``_arrowspace/manifold.json`` sidecar from the dataset directory.

    This endpoint serves static metadata written by upstream tooling and does
    not require the ``arrowspace`` package.
    """
    path = _resolve_dataset_path(settings, dataset_id)
    try:
        data = adapter.sidecar_manifold(path)
    except Exception as e:
        data = {"unavailable": str(e)}
    return {
        "id": dataset_id,
        "backend": adapter.backend,
        "manifold": data,
    }


# ---------------------------------------------------------------------------
# ArrowSpace index lifecycle
# ---------------------------------------------------------------------------

@router.post("/datasets/{dataset_id:path}/index")
def build_index(
    dataset_id: str,
    graph_params: dict[str, Any] | None = Body(
        default=None,
        example=DEFAULT_GRAPH_PARAMS,
        description="ArrowSpaceBuilder graph params.  Omit to use server defaults.",
    ),
    reg: StorageRegistry = Depends(_registry),
    settings: Settings = Depends(get_settings),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    """Build (or rebuild) the ArrowSpace graph-Laplacian index for a dataset.

    Reads the full Zarr array, calls ``ArrowSpaceBuilder().build()``, persists
    the graph Laplacian as Zarr v3 CSR arrays under ``ARROSPACE_INDEX_STORE``,
    and caches the result in memory for fast /lambdas and /search access.

    The source Zarr array must be 2-D (rows = items, columns = features).
    """
    h = reg.open(dataset_id)
    # Read the full array — arrowspace needs all items at build time.
    rs = parse_slice(None, h.summary.shape, offset=0, limit=h.summary.shape[0])
    arr = h.read_window(rs)

    index_store = Path(settings.index_store).expanduser().resolve()
    meta = adapter.build_index(
        dataset_id=dataset_id,
        array=arr,
        index_store=index_store,
        graph_params=graph_params,
    )
    return {
        "id": dataset_id,
        "built": True,
        "graph_params": graph_params or DEFAULT_GRAPH_PARAMS,
        **meta,
    }


# ---------------------------------------------------------------------------
# ArrowSpace query endpoints
# ---------------------------------------------------------------------------

@router.get("/datasets/{dataset_id:path}/lambdas")
def dataset_lambdas(
    dataset_id: str,
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    """Return the Laplacian eigenvalue distribution for a built index.

    Requires a prior call to ``POST /datasets/{id}/index``.
    Returns ``{nitems, lambdas, lambdas_sorted}``.
    """
    return {"id": dataset_id} | adapter.lambdas(dataset_id)


@router.post("/datasets/{dataset_id:path}/search")
def dataset_search(
    dataset_id: str,
    body: dict[str, Any] = Body(
        ...,
        example={"vector": [0.1, 0.2, 0.3], "tau": 1.0},
        description="Search body: 'vector' (list[float]) and optional 'tau' (float).",
    ),
    adapter: ArrowSpaceAdapter = Depends(_arrowspace),
) -> dict[str, Any]:
    """Vector search against the in-memory ArrowSpace index.

    Requires a prior call to ``POST /datasets/{id}/index``.
    Body: ``{\"vector\": [f64, ...], \"tau\": 1.0}``.
    Returns ``{backend, results: [{index, score}, ...]}`.
    """
    return {"id": dataset_id} | adapter.search(dataset_id, body)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trailing_product(shape: tuple[int, ...]) -> int:
    if len(shape) <= 1:
        return 1
    p = 1
    for d in shape[1:]:
        p *= int(d)
    return p
