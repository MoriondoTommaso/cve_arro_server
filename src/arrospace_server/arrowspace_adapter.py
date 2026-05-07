"""Adapter for arrowspace / ArrowSpace graph-Laplacian index.

The ``arrowspace`` package (pip install arrowspace,
repo: https://github.com/tuned-org-uk/pyarrowspace) may not be available in
every environment.  We never import it at module load — :func:`load` attempts
the import lazily and returns a stub adapter that falls back to the sidecar
JSON adapter.

Design
------
arrowspace is a **pure in-memory builder**.  Its only public entry-point is::

    from arrowspace import ArrowSpaceBuilder
    aspace, gl = ArrowSpaceBuilder().build(graph_params, np_array_float64)

The raw dataset stays as the Zarr v3 array (served by the existing Zarr
backend).  The *graph Laplacian* produced by arrowspace is persisted as a
second Zarr v3 array (CSR components) under::

    <ARROSPACE_INDEX_STORE>/<dataset_id_slug>/
        data.zarr    # gl.to_csr() → data array  (float32)
        indices.zarr # CSR column indices         (int64)
        indptr.zarr  # CSR row pointers           (int64)
        meta.json    # {nitems, nfeatures, nclusters, shape}

The ``ArrowSpace`` object itself is kept in-process memory; on the next server
start the Zarr arrays can be loaded back (Phase 2 work).  For now the index
lives in memory and is rebuilt on ``POST /datasets/{id}/index``.

Sidecar fallback
----------------
ArrowSpace metadata can also be advertised via sidecar files next to a
dataset::

    <dataset_root>/_arrowspace/manifold.json
    <dataset_root>/_arrowspace/stats.json
    <dataset_root>/_arrowspace/index.json

This allows a dataset to expose ArrowSpace metadata without the Python package.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .errors import MetadataUnavailable, OptionalDependencyMissing

log = logging.getLogger(__name__)

SIDECAR_DIR = "_arrowspace"

DEFAULT_GRAPH_PARAMS: dict[str, Any] = {
    "eps": 1.0,
    "k": 6,
    "topk": 3,
    "p": 2.0,
    "sigma": 1.0,
}


@dataclass
class ArrowSpaceAdapter:
    available: bool
    backend: str  # "arrowspace" | "sidecar" | "none"

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------

    def build_index(
        self,
        dataset_id: str,
        array: np.ndarray,
        index_store: Path,
        graph_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build (or rebuild) the graph-Laplacian index for *array*.

        Returns a summary dict ``{nitems, nfeatures, nclusters}``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Query methods (only what ArrowSpace actually exposes)
    # ------------------------------------------------------------------

    def lambdas(
        self,
        dataset_id: str,
    ) -> dict[str, Any]:
        """Return the Laplacian eigenvalue distribution.

        ``{nitems, lambdas: [float, ...], lambdas_sorted: [[float, int], ...]}``
        """
        raise NotImplementedError

    def search(
        self,
        dataset_id: str,
        query: dict[str, Any],
    ) -> dict[str, Any]:
        """Vector search.  *query* must contain ``vector`` (list[float]).

        Optionally ``tau`` (float, default 1.0).
        Returns ``{backend, results: [{index, score}, ...]}`.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Sidecar helpers (used by routes for /manifold and raw /stats)
    # ------------------------------------------------------------------

    def sidecar_manifold(self, dataset_path: Path) -> dict[str, Any]:
        """Read ``_arrowspace/manifold.json`` sidecar if present."""
        raise NotImplementedError

    def sidecar_stats(self, dataset_path: Path) -> dict[str, Any]:
        """Read ``_arrowspace/stats.json`` sidecar if present."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Sidecar-only adapter
# ---------------------------------------------------------------------------

class _SidecarAdapter(ArrowSpaceAdapter):
    def __init__(self) -> None:
        super().__init__(available=True, backend="sidecar")

    @staticmethod
    def _read(path: Path, name: str) -> dict[str, Any]:
        f = path / SIDECAR_DIR / name
        if not f.exists():
            raise MetadataUnavailable(f"{name} not present at {f}")
        try:
            return json.loads(f.read_text())
        except Exception as e:
            raise MetadataUnavailable(f"failed to parse {f}: {e}") from e

    def sidecar_manifold(self, dataset_path: Path) -> dict[str, Any]:
        return self._read(dataset_path, "manifold.json")

    def sidecar_stats(self, dataset_path: Path) -> dict[str, Any]:
        return self._read(dataset_path, "stats.json")

    def build_index(
        self,
        dataset_id: str,
        array: np.ndarray,
        index_store: Path,
        graph_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise OptionalDependencyMissing("arrowspace", "build_index")

    def lambdas(self, dataset_id: str) -> dict[str, Any]:
        raise OptionalDependencyMissing("arrowspace", "lambdas")

    def search(self, dataset_id: str, query: dict[str, Any]) -> dict[str, Any]:
        # Sidecar text-based fallback via index.json
        raise OptionalDependencyMissing(
            "arrowspace",
            "vector search (install arrowspace or provide sidecar index.json)",
        )


# ---------------------------------------------------------------------------
# No-op adapter
# ---------------------------------------------------------------------------

class _UnavailableAdapter(ArrowSpaceAdapter):
    def __init__(self) -> None:
        super().__init__(available=False, backend="none")

    def build_index(self, dataset_id, array, index_store, graph_params=None):
        raise OptionalDependencyMissing("arrowspace", "build_index")

    def lambdas(self, dataset_id):
        raise OptionalDependencyMissing("arrowspace", "lambdas")

    def search(self, dataset_id, query):
        raise OptionalDependencyMissing("arrowspace", "search")

    def sidecar_manifold(self, dataset_path):
        raise OptionalDependencyMissing("arrowspace", "manifold sidecar")

    def sidecar_stats(self, dataset_path):
        raise OptionalDependencyMissing("arrowspace", "stats sidecar")


# ---------------------------------------------------------------------------
# Live arrowspace adapter
# ---------------------------------------------------------------------------

@dataclass
class _IndexEntry:
    """In-memory cache slot for one built index."""
    aspace: Any
    gl: Any
    nitems: int
    nfeatures: int
    nclusters: int


class _ArrowSpaceAdapter(ArrowSpaceAdapter):  # pragma: no cover - optional dep
    """Live adapter backed by the ``arrowspace`` package.

    arrowspace is a **pure in-memory builder**; it has no path-based open API.
    This adapter:

    1. Accepts a ``numpy.ndarray`` (already read from Zarr by the route
       handler) and calls ``ArrowSpaceBuilder().build(graph_params, array)``.
    2. Persists the graph Laplacian CSR components as Zarr v3 arrays under
       ``<index_store>/<slug>/``.
    3. Caches ``(ArrowSpace, GraphLaplacian)`` in memory keyed by
       ``dataset_id`` to avoid rebuilds within a server lifetime.
    4. Exposes only the methods that ``ArrowSpace`` actually provides:
       ``search()``, ``lambdas()``, and ``lambdas_sorted()``.
    """

    def __init__(self, module: Any) -> None:
        super().__init__(available=True, backend="arrowspace")
        self._mod = module
        self._cache: dict[str, _IndexEntry] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _slug(dataset_id: str) -> str:
        """File-system-safe slug for a dataset_id."""
        return dataset_id.replace("/", "__").replace("\\", "__")

    def _persist_csr(
        self,
        index_store: Path,
        slug: str,
        gl: Any,
        meta: dict[str, Any],
    ) -> None:
        """Write CSR arrays + meta.json to ``index_store/<slug>/``."""
        try:
            import zarr  # type: ignore
        except ImportError:
            log.warning("zarr not installed; graph-Laplacian will not be persisted")
            return

        csr = gl.to_csr()  # returns (data, indices, indptr, shape)
        csr_data, csr_indices, csr_indptr, csr_shape = csr

        dest = index_store / slug
        dest.mkdir(parents=True, exist_ok=True)

        for arr_name, arr_val in (
            ("data", np.asarray(csr_data, dtype=np.float32)),
            ("indices", np.asarray(csr_indices, dtype=np.int64)),
            ("indptr", np.asarray(csr_indptr, dtype=np.int64)),
        ):
            zarr_path = dest / f"{arr_name}.zarr"
            z = zarr.open(
                str(zarr_path),
                mode="w",
                shape=arr_val.shape,
                dtype=arr_val.dtype,
                chunks=True,
                zarr_format=3,
            )
            z[:] = arr_val

        meta_dict = dict(meta)
        meta_dict["csr_shape"] = list(csr_shape)
        (dest / "meta.json").write_text(json.dumps(meta_dict))
        log.info("Persisted graph-Laplacian CSR to %s", dest)

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------

    def build_index(
        self,
        dataset_id: str,
        array: np.ndarray,
        index_store: Path,
        graph_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the ArrowSpace graph-Laplacian index for *array*.

        The input *array* must be a 2-D float64 ndarray (rows = items,
        columns = features).  It is read from the Zarr backend by the
        route handler before calling this method.
        """
        gp = graph_params or DEFAULT_GRAPH_PARAMS
        arr64 = np.asarray(array, dtype=np.float64)
        if arr64.ndim != 2:
            raise ValueError(
                f"arrowspace requires a 2-D array (items × features); got shape {arr64.shape}"
            )

        log.info("Building arrowspace index for %s (shape=%s, params=%s)", dataset_id, arr64.shape, gp)
        aspace, gl = self._mod.ArrowSpaceBuilder().build(gp, arr64)

        entry = _IndexEntry(
            aspace=aspace,
            gl=gl,
            nitems=int(aspace.nitems),
            nfeatures=int(aspace.nfeatures),
            nclusters=int(aspace.nclusters),
        )
        self._cache[dataset_id] = entry

        meta = {
            "nitems": entry.nitems,
            "nfeatures": entry.nfeatures,
            "nclusters": entry.nclusters,
        }
        self._persist_csr(index_store, self._slug(dataset_id), gl, meta)

        return meta

    # ------------------------------------------------------------------
    # Query methods — only real ArrowSpace API surface
    # ------------------------------------------------------------------

    def _get_entry(self, dataset_id: str) -> _IndexEntry:
        entry = self._cache.get(dataset_id)
        if entry is None:
            raise MetadataUnavailable(
                f"No index built for '{dataset_id}'. "
                "Call POST /datasets/{id}/index first."
            )
        return entry

    def lambdas(self, dataset_id: str) -> dict[str, Any]:
        """Return Laplacian eigenvalue distribution for a built index."""
        entry = self._get_entry(dataset_id)
        lam = list(entry.aspace.lambdas())
        lam_sorted = [[float(v), int(i)] for v, i in entry.aspace.lambdas_sorted()]
        return {
            "nitems": entry.nitems,
            "lambdas": [float(v) for v in lam],
            "lambdas_sorted": lam_sorted,
        }

    def search(self, dataset_id: str, query: dict[str, Any]) -> dict[str, Any]:
        """Vector search against the in-memory arrowspace index.

        *query* must contain ``vector`` (list of floats).  Optional ``tau``
        (float, default 1.0).
        """
        entry = self._get_entry(dataset_id)
        vec = query.get("vector")
        if vec is None:
            raise MetadataUnavailable(
                "arrowspace search requires 'vector' (list of float64 values)"
            )
        tau = float(query.get("tau", 1.0))
        q_arr = np.asarray(vec, dtype=np.float64)
        hits = entry.aspace.search(q_arr, entry.gl, tau)
        return {
            "backend": "arrowspace",
            "results": [{"index": int(i), "score": float(s)} for i, s in hits],
        }

    # ------------------------------------------------------------------
    # Sidecar helpers (delegate to the sidecar reader)
    # ------------------------------------------------------------------

    def sidecar_manifold(self, dataset_path: Path) -> dict[str, Any]:
        return _SidecarAdapter._read(dataset_path, "manifold.json")

    def sidecar_stats(self, dataset_path: Path) -> dict[str, Any]:
        return _SidecarAdapter._read(dataset_path, "stats.json")


# ---------------------------------------------------------------------------
# Module-level cache + loader
# ---------------------------------------------------------------------------

_cached: ArrowSpaceAdapter | None = None


def load() -> ArrowSpaceAdapter:
    """Pick the best available adapter.

    Order of preference:
        1. ``arrowspace`` package if importable  (pip install arrowspace)
        2. Sidecar JSON files under ``_arrowspace/``
    """
    global _cached
    if _cached is not None:
        return _cached
    try:
        import arrowspace  # type: ignore

        _cached = _ArrowSpaceAdapter(arrowspace)
        log.info("ArrowSpace adapter: arrowspace package")
        return _cached
    except Exception as e:
        log.info("arrowspace unavailable (%s); falling back to sidecar adapter", e)
    _cached = _SidecarAdapter()
    return _cached


def reset_adapter_cache() -> None:
    """Test / lifecycle helper."""
    global _cached
    _cached = None
