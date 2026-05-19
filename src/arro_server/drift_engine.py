# ADDED FILE
# Copyright 2026 GENEFOLD AI LTD — Apache License 2.0
# Author: Tommaso Moriondo — CVE spectral drift demo
#
# CveDriftEngine: loads two CVE embedding periods, builds ArrowSpace indices
# for each, and exposes spectral drift metrics (Wasserstein distance on
# eigenvalue distributions, per-period lambdas, side-by-side search).
#
# Period A: CVE 1999–2014  (embs_99_to_14.npy  or cve_99_2014.zarr)
# Period B: CVE 2015–2025  (embs_15_to_2025.npy or cve_15_2025.zarr)
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wasserstein1d(u: np.ndarray, v: np.ndarray) -> float:
    """1-D Wasserstein (Earth Mover's) distance between two sample sets.

    Uses the closed-form solution on sorted arrays, which is exact and O(n log n).
    scipy is not required.
    """
    u_sorted = np.sort(u.astype(np.float64))
    v_sorted = np.sort(v.astype(np.float64))
    n = max(len(u_sorted), len(v_sorted))
    u_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(u_sorted)), u_sorted)
    v_interp = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(v_sorted)), v_sorted)
    return float(np.mean(np.abs(u_interp - v_interp)))


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
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"CVE embedding file not found: {p}. "
            "Set ARRO_SERVER_CVE_PERIOD_A / ARRO_SERVER_CVE_PERIOD_B to valid paths."
        )
    suffix = p.suffix.lower()
    if suffix == ".npy":
        arr = np.load(str(p)).astype(np.float64)
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
            arr = np.asarray(zarr.open_array(str(c_path), mode="r")[:], dtype=np.float64)
            log.info("Loaded zarr embeddings (c/) from %s  shape=%s", p, arr.shape)
            return arr
        except Exception as exc:
            log.debug("zarr open_array on 'c' sub-key failed (%s), trying root", exc)

    # Fallback: root is directly a zarr array.
    try:
        arr = np.asarray(zarr.open_array(str(p), mode="r")[:], dtype=np.float64)
        log.info("Loaded zarr embeddings (root) from %s  shape=%s", p, arr.shape)
        return arr
    except Exception as exc:
        raise RuntimeError(
            f"Could not load zarr store at {p}: {exc}. "
            "Expected either a plain zarr array or a group with a 'c' array sub-key."
        ) from exc


def _build_arrowspace(arr: np.ndarray, label: str) -> Any:
    """Build an ArrowSpace index for the given embedding array.

    Returns the ArrowSpace object (pyarrowspace.ArrowSpace).
    Raises ImportError if pyarrowspace is not installed.
    """
    try:
        from pyarrowspace import ArrowSpaceBuilder  # type: ignore
    except ImportError:
        raise ImportError(
            "pyarrowspace is required for the drift engine. "
            "Install with: pip install pyarrowspace"
        )
    log.info("Building ArrowSpace index for period '%s'  shape=%s …", label, arr.shape)
    builder = ArrowSpaceBuilder()
    aspace = builder.build(arr)
    log.info("ArrowSpace index built for period '%s'  nitems=%d", label, aspace.nitems)
    return aspace


# ---------------------------------------------------------------------------
# CveDriftEngine
# ---------------------------------------------------------------------------

@dataclass
class _PeriodIndex:
    label: str
    path: str
    embeddings: np.ndarray
    aspace: Any  # pyarrowspace.ArrowSpace
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
            cls._instance = cls._build(path_a, path_b)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Destroy the singleton so the next ``get()`` rebuilds it."""
        with cls._lock:
            cls._instance = None

    @classmethod
    def _build(cls, path_a: str, path_b: str) -> "CveDriftEngine":
        """Load both periods, build indices, compute lambdas."""
        embs_a = _load_embeddings(path_a)
        embs_b = _load_embeddings(path_b)

        aspace_a = _build_arrowspace(embs_a, "period_a")
        aspace_b = _build_arrowspace(embs_b, "period_b")

        lambdas_a = cls._extract_lambdas(aspace_a)
        lambdas_b = cls._extract_lambdas(aspace_b)

        pa = _PeriodIndex(
            label="period_a",
            path=path_a,
            embeddings=embs_a,
            aspace=aspace_a,
            lambdas=lambdas_a,
        )
        pb = _PeriodIndex(
            label="period_b",
            path=path_b,
            embeddings=embs_b,
            aspace=aspace_b,
            lambdas=lambdas_b,
        )
        return cls(pa, pb)

    @staticmethod
    def _extract_lambdas(aspace: Any) -> list[float]:
        """Extract eigenvalues from an ArrowSpace index."""
        try:
            lams = aspace.lambdas()
            return [float(x) for x in lams]
        except Exception as exc:
            log.warning("Failed to extract lambdas: %s", exc)
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
        """Run spectral search against a single period index."""
        try:
            raw = period.aspace.search({"vector": vector.tolist(), "tau": tau})
            results = raw if isinstance(raw, list) else raw.get("results", [])
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
