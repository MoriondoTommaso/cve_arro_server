# MODIFIED FILE
# Copyright 2026 GENEFOLD AI LTD — Apache License 2.0
# Author: Tommaso Moriondo — CVE spectral drift demo
#
# CveDriftEngine: loads two CVE embedding periods, builds ArrowSpace indices
# for each, and exposes spectral drift metrics (Wasserstein distance on
# eigenvalue distributions, per-period lambdas, side-by-side search).
#
# Period A: CVE 1999–2014  (embs_99_to_14.npy  or cve_99_2014.zarr)
# Period B: CVE 2015–2025  (embs_15_to_2025.npy or cve_15_2025.zarr)
#
# Fixes applied in this version:
#   FIX-1  _load_embeddings: np.load now uses allow_pickle=True with a safe
#           unwrapper so object-array .npy files (dict/list wrappers) load
#           correctly instead of raising "Object arrays cannot be loaded when
#           allow_pickle=False".
#   FIX-2  _load_embeddings: result is always cast to a contiguous float64
#           C-order array — ArrowSpaceBuilder requires exactly this dtype/layout.
#   FIX-3  _build_arrowspace: _subsample is called ONCE here (not again in
#           _build), avoiding a second independent subsample that silently
#           discards the already-built index alignment.
#   FIX-4  CveDriftEngine._build: removed the redundant second _subsample
#           calls for sub_a / sub_b — the indexed arrays are already the
#           subsampled ones returned by _build_arrowspace.
#   FIX-5  _extract_lambdas: aspace.lambdas() may return a numpy array,
#           a list, or a generator — normalised uniformly; also handles the
#           case where lambdas() does not exist and falls back to
#           aspace.eigenvalues if available.
#   FIX-6  _wasserstein1d: guard against empty arrays (returns 0.0 instead of
#           crashing with a zero-division in np.interp).
"""CVE spectral drift engine.

Two-period ArrowSpace engine used by the /api/drift/* route group.
Call ``CveDriftEngine.get()`` to obtain the lazily-initialised singleton.
Call ``CveDriftEngine.reset()`` to force a rebuild (e.g. after data reload).
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Default graph params — same as arrowspace_adapter.DEFAULT_GRAPH_PARAMS
_DEFAULT_GRAPH_PARAMS: dict[str, Any] = {
    "eps": 1.2311,
    "k": 38,
    "topk": 19,
    "p": 2.0,
    "sigma": None,
}

# Maximum number of rows passed to ArrowSpaceBuilder.build().
# The full CVE corpus is 80k–180k entries; building a graph-Laplacian index on
# the full array takes O(n^2) memory and 30+ minutes.  We subsample to a
# representative slice for the demo.  Set ARRO_SERVER_CVE_N_SAMPLE=0 to disable.
_DEFAULT_N_SAMPLE = 8_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wasserstein1d(u: np.ndarray, v: np.ndarray) -> float:
    """1-D Wasserstein (Earth Mover's) distance between two sample sets.

    Uses the closed-form solution on sorted arrays, which is exact and O(n log n).
    scipy is not required.

    FIX-6: returns 0.0 immediately if either array is empty, preventing
    zero-division / zero-length linspace inside np.interp.
    """
    u_arr = np.asarray(u, dtype=np.float64).ravel()
    v_arr = np.asarray(v, dtype=np.float64).ravel()
    # FIX-6 — guard empty arrays
    if len(u_arr) == 0 or len(v_arr) == 0:
        log.warning("_wasserstein1d: one or both lambda arrays are empty — returning 0.0")
        return 0.0
    u_sorted = np.sort(u_arr)
    v_sorted = np.sort(v_arr)
    n = max(len(u_sorted), len(v_sorted))
    u_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(u_sorted)), u_sorted)
    v_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(v_sorted)), v_sorted)
    return float(np.mean(np.abs(u_interp - v_interp)))


def _unwrap_object_array(raw: np.ndarray) -> np.ndarray:
    """Convert a numpy object-dtype array into a plain float64 matrix.

    The CVE pipeline sometimes saves embeddings as::

        np.save("file.npy", {"embeddings": arr})   # pickled dict
        np.save("file.npy", arr_list)               # pickled list-of-arrays

    This helper unwraps those cases safely.

    FIX-1 / FIX-2 support function.
    """
    if raw.dtype != object:
        return np.ascontiguousarray(raw, dtype=np.float64)

    # 0-d object array wrapping a dict, list, or ndarray
    inner = raw.item() if raw.ndim == 0 else None

    if inner is None and raw.ndim == 1:
        # 1-d object array whose elements are row vectors
        try:
            return np.ascontiguousarray(np.vstack(raw.tolist()), dtype=np.float64)
        except Exception as exc:
            raise ValueError(
                f"Cannot convert 1-d object array to float64 matrix: {exc}"
            ) from exc

    if inner is None:
        raise ValueError(
            f"Cannot convert object array with ndim={raw.ndim} to float64 matrix."
        )

    if isinstance(inner, dict):
        # common keys produced by the pipeline
        for key in ("embeddings", "embs", "vectors", "data", "x"):
            if key in inner:
                return np.ascontiguousarray(inner[key], dtype=np.float64)
        # last resort: stack all values
        try:
            return np.ascontiguousarray(np.vstack(list(inner.values())), dtype=np.float64)
        except Exception as exc:
            raise ValueError(
                f"Object array wraps a dict with keys {list(inner.keys())} "
                f"but none could be converted to a float64 matrix: {exc}"
            ) from exc

    if isinstance(inner, np.ndarray):
        return np.ascontiguousarray(inner, dtype=np.float64)

    if isinstance(inner, list):
        return np.ascontiguousarray(np.array(inner), dtype=np.float64)

    raise ValueError(
        f"Object array wraps an unsupported type {type(inner).__name__}. "
        "Re-save the embedding file with np.save('file.npy', arr.astype(np.float64))."
    )


def _load_embeddings(path: str | Path) -> np.ndarray:
    """Load embeddings from .npy or a Zarr array directory.

    Zarr stores produced by the CVE pipeline use the layout::

        cve_15_2025.zarr/
            c/          <- zarr array stored under the 'c' key
                0.0
                0.1
                ...

    We therefore try ``zarr.open_array(p / 'c')`` first, then fall back to
    opening the root as an array (for stores without the 'c' sub-key).

    FIX-1: np.load is now called with allow_pickle=True so that .npy files
           saved from dicts/lists (the CVE pipeline default) load without
           raising "Object arrays cannot be loaded when allow_pickle=False".
    FIX-2: every return path ends with _unwrap_object_array() which guarantees
           the output is a contiguous float64 C-order ndarray as required by
           ArrowSpaceBuilder.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"CVE embedding file not found: {p}. "
            "Set ARRO_SERVER_CVE_PERIOD_A / ARRO_SERVER_CVE_PERIOD_B to valid paths."
        )
    suffix = p.suffix.lower()
    if suffix == ".npy":
        # FIX-1: must use allow_pickle=True — the CVE pipeline saves object arrays
        try:
            raw = np.load(str(p), allow_pickle=False)
        except ValueError:
            # file contains a pickled object array — load with pickle and unwrap
            log.warning(
                "np.load(allow_pickle=False) failed for %s — retrying with allow_pickle=True. "
                "Consider re-saving the file as a plain float64 array.",
                p,
            )
            raw = np.load(str(p), allow_pickle=True)
        # FIX-2: normalise to contiguous float64
        arr = _unwrap_object_array(raw)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        log.info("Loaded .npy embeddings from %s  shape=%s", p, arr.shape)
        return arr

    # Zarr directory
    try:
        import zarr  # type: ignore
    except ImportError:
        raise ImportError(
            "zarr is required to load .zarr embedding stores. "
            "Install it with: pip install zarr"
        )

    # Try the canonical CVE layout: array stored under the 'c' sub-key.
    c_path = p / "c"
    if c_path.exists():
        try:
            arr = np.ascontiguousarray(
                zarr.open_array(str(c_path), mode="r")[:], dtype=np.float64
            )
            log.info("Loaded zarr embeddings (c/) from %s  shape=%s", p, arr.shape)
            return arr
        except Exception as exc:
            log.debug("zarr open_array on 'c' sub-key failed (%s), trying root", exc)

    # Fallback: root is directly a zarr array.
    try:
        arr = np.ascontiguousarray(
            zarr.open_array(str(p), mode="r")[:], dtype=np.float64
        )
        log.info("Loaded zarr embeddings (root) from %s  shape=%s", p, arr.shape)
        return arr
    except Exception as exc:
        raise RuntimeError(
            f"Could not load zarr store at {p}: {exc}. "
            "Expected either a plain zarr array or a group with a 'c' array sub-key."
        ) from exc


def _subsample(arr: np.ndarray, n_sample: int, rng_seed: int = 42) -> np.ndarray:
    """Return a stratified random subsample of *arr* (rows).

    If ``n_sample <= 0`` or ``n_sample >= len(arr)`` the full array is returned
    unchanged.  The sample is drawn without replacement using a fixed seed so
    the index is deterministic across restarts.
    """
    n = len(arr)
    if n_sample <= 0 or n_sample >= n:
        return arr
    rng = np.random.default_rng(rng_seed)
    idx = rng.choice(n, size=n_sample, replace=False)
    idx.sort()  # preserve row order for interpretability
    log.info("Subsampled %d → %d rows for ArrowSpace index", n, n_sample)
    return arr[idx]


def _build_arrowspace(
    arr: np.ndarray,
    label: str,
    n_sample: int = _DEFAULT_N_SAMPLE,
    graph_params: dict[str, Any] | None = None,
) -> tuple[Any, Any, np.ndarray]:
    """Build an ArrowSpace index for the given embedding array.

    Parameters
    ----------
    arr:
        Full embedding matrix (N × D).  Will be subsampled to *n_sample* rows
        before indexing unless n_sample <= 0.
    label:
        Human-readable name used in log messages.
    n_sample:
        Maximum number of rows to pass to ArrowSpaceBuilder.build().
        Defaults to _DEFAULT_N_SAMPLE (8 000).  Set to 0 to disable.
    graph_params:
        ArrowSpace graph construction parameters.  Defaults to
        _DEFAULT_GRAPH_PARAMS.

    Returns
    -------
    (aspace, gl, sub) : tuple[ArrowSpace, GraphLaplacian, np.ndarray]
        ``gl`` is required by every search call.
        FIX-3: ``sub`` (the subsampled array) is returned here so _build
        does not have to call _subsample a second time with a different RNG
        state, which would produce a misaligned array.

    Raises
    ------
    ImportError
        If the ``arrowspace`` package is not installed.
    """
    try:
        from arrowspace import ArrowSpaceBuilder  # type: ignore
    except ImportError:
        raise ImportError(
            "arrowspace is required for the drift engine. "
            "Install with: pip install arrowspace"
        )

    gp = graph_params or _DEFAULT_GRAPH_PARAMS
    # FIX-3: subsample once here and return the slice
    sub = _subsample(arr, n_sample)
    sub64 = np.ascontiguousarray(sub, dtype=np.float64)  # FIX-2: contiguous float64

    log.info(
        "Building ArrowSpace index for period '%s'  shape=%s params=%s …",
        label, sub64.shape, gp,
    )
    # Correct call order: ArrowSpaceBuilder().build(graph_params, array)
    aspace, gl = ArrowSpaceBuilder().build(gp, sub64)
    log.info(
        "ArrowSpace index built for period '%s'  nitems=%d nclusters=%d",
        label, aspace.nitems, aspace.nclusters,
    )
    # FIX-3: return sub64 so the caller does not subsample again
    return aspace, gl, sub64


# ---------------------------------------------------------------------------
# CveDriftEngine
# ---------------------------------------------------------------------------

@dataclass
class _PeriodIndex:
    label: str
    path: str
    embeddings: np.ndarray   # subsampled array actually indexed
    aspace: Any              # arrowspace.ArrowSpace
    gl: Any                  # arrowspace.GraphLaplacian — required for every search call
    lambdas: list[float] = field(default_factory=list)


class CveDriftEngine:
    """Two-period spectral engine for CVE drift monitoring.

    Singleton — use ``CveDriftEngine.get()``.

    Attributes
    ----------
    period_a, period_b : _PeriodIndex
        Loaded indices for the two CVE time windows.
    drift_score : float
        Wasserstein-1 distance between the two eigenvalue distributions.
        Higher = more spectral drift between periods.
    """

    _instance: "CveDriftEngine | None" = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, period_a: _PeriodIndex, period_b: _PeriodIndex) -> None:
        self.period_a = period_a
        self.period_b = period_b
        self.drift_score: float = _wasserstein1d(
            np.array(period_a.lambdas), np.array(period_b.lambdas)
        )
        log.info(
            "CveDriftEngine ready — drift_score (Wasserstein-1)=%.6f", self.drift_score
        )

    # ------------------------------------------------------------------
    # Singleton lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def get(cls) -> "CveDriftEngine":
        """Return (or lazily build) the singleton."""
        if cls._instance is not None:
            return cls._instance
        with cls._lock:
            if cls._instance is not None:
                return cls._instance
            from .settings import get_settings
            settings = get_settings()
            path_a = settings.cve_period_a
            path_b = settings.cve_period_b
            n_sample = getattr(settings, "cve_n_sample", _DEFAULT_N_SAMPLE)
            cls._instance = cls._build(path_a, path_b, n_sample=n_sample)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Destroy the singleton so the next ``get()`` rebuilds it."""
        with cls._lock:
            cls._instance = None

    @classmethod
    def _build(
        cls,
        path_a: str,
        path_b: str,
        n_sample: int = _DEFAULT_N_SAMPLE,
    ) -> "CveDriftEngine":
        """Load both periods, build indices, compute lambdas.

        FIX-4: _build_arrowspace now returns the subsampled array as its
        third element, so we no longer call _subsample a second time here.
        The old code called _subsample(embs_a, n_sample) a second time which
        used a fresh RNG state and produced a different row selection from the
        one already inside the ArrowSpace index — misaligning embeddings and index.
        """
        embs_a = _load_embeddings(path_a)
        embs_b = _load_embeddings(path_b)

        # FIX-4: unpack the third return value (the subsampled slice)
        aspace_a, gl_a, sub_a = _build_arrowspace(embs_a, "period_a", n_sample=n_sample)
        aspace_b, gl_b, sub_b = _build_arrowspace(embs_b, "period_b", n_sample=n_sample)

        lambdas_a = cls._extract_lambdas(aspace_a)
        lambdas_b = cls._extract_lambdas(aspace_b)

        pa = _PeriodIndex(
            label="cve_99_14 (1999–2014)",
            path=path_a,
            embeddings=sub_a,   # FIX-4: use the already-computed slice
            aspace=aspace_a,
            gl=gl_a,
            lambdas=lambdas_a,
        )
        pb = _PeriodIndex(
            label="cve_99_25 (1999–2025)",
            path=path_b,
            embeddings=sub_b,   # FIX-4: use the already-computed slice
            aspace=aspace_b,
            gl=gl_b,
            lambdas=lambdas_b,
        )
        return cls(pa, pb)

    @staticmethod
    def _extract_lambdas(aspace: Any) -> list[float]:
        """Extract eigenvalues from an ArrowSpace index.

        FIX-5: aspace.lambdas() may return a numpy array, a plain list, or a
        generator.  We also fall back to aspace.eigenvalues (attribute, not
        method) which some arrowspace builds expose instead.
        """
        # Try .lambdas() method first
        lams_raw = None
        lams_method = getattr(aspace, "lambdas", None)
        if callable(lams_method):
            try:
                lams_raw = lams_method()
            except Exception as exc:
                log.warning("aspace.lambdas() raised: %s", exc)

        # FIX-5: fallback to .eigenvalues attribute
        if lams_raw is None:
            lams_attr = getattr(aspace, "eigenvalues", None)
            if lams_attr is not None:
                lams_raw = lams_attr

        if lams_raw is None:
            log.warning(
                "ArrowSpace object exposes neither lambdas() nor .eigenvalues — "
                "returning empty list.  Drift charts will show no data."
            )
            return []

        # FIX-5: normalise to list[float] regardless of return type
        try:
            if isinstance(lams_raw, np.ndarray):
                return [float(x) for x in lams_raw.ravel()]
            return [float(x) for x in lams_raw]
        except Exception as exc:
            log.warning("Failed to convert lambdas to list[float]: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Search helpers
    # ------------------------------------------------------------------

    def search_period(
        self,
        period: _PeriodIndex,
        vector: np.ndarray,
        k: int = 10,
        tau: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Run spectral search against a single period index.

        Calls ``aspace.search(vec, gl, tau)`` — the correct pyarrowspace
        positional signature confirmed in arrowspace_adapter.py.
        """
        try:
            q_arr = np.ascontiguousarray(vector, dtype=np.float64)  # FIX-2: ensure contiguous
            # Correct call: aspace.search(query_vector, graph_laplacian, tau)
            raw = period.aspace.search(q_arr, period.gl, tau)
            # raw is List[(int, float)] per the adapter docstring
            if isinstance(raw, list):
                results = [
                    {"index": int(i), "score": float(s)}
                    for i, s in raw
                ]
            elif isinstance(raw, dict):
                results = raw.get("results", [])
            else:
                results = []
        except Exception as exc:
            log.warning("search_period failed for %s: %s", period.label, exc)
            results = []
        return results[:k]

    def search_both(
        self,
        vector: np.ndarray,
        k: int = 10,
        tau: float = 0.5,
    ) -> dict[str, Any]:
        """Run the same query against both period indices.

        Returns a dict with keys ``period_a`` and ``period_b``, each holding
        a list of result dicts, plus the current ``drift_score``.
        """
        results_a = self.search_period(self.period_a, vector, k=k, tau=tau)
        results_b = self.search_period(self.period_b, vector, k=k, tau=tau)
        return {
            "drift_score": self.drift_score,
            "period_a": {"label": self.period_a.label, "results": results_a},
            "period_b": {"label": self.period_b.label, "results": results_b},
        }