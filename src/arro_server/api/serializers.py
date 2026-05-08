"""Numpy -> JSON-friendly conversion helpers."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _safe_scalar(v: Any) -> Any:
    """Coerce a scalar to a JSON-safe Python type.

    Handles numpy scalars, float NaN/Inf (replaced with None), and
    complex numbers.  Plain Python types pass through unchanged.
    """
    if isinstance(v, float):
        return None if not math.isfinite(v) else v
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if not math.isfinite(f) else f
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, complex):
        return {"re": v.real, "im": v.imag}
    return v


def array_to_payload(arr: np.ndarray, *, preview_max_rows: int | None = None) -> dict[str, Any]:
    """Convert an ndarray to a JSON-friendly preview payload.

    For 2-D arrays we emit a row-oriented ``rows`` field (truncated to
    ``preview_max_rows`` if provided). Otherwise we emit a flat ``values``
    list along with ``shape`` so the client can reshape.
    """
    payload: dict[str, Any] = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }
    if arr.dtype.kind in {"S", "U", "O"}:
        # Stringy or object arrays: convert via tolist().
        if arr.ndim == 2:
            rows = arr.tolist()
            if preview_max_rows is not None:
                rows = rows[:preview_max_rows]
            payload["rows"] = rows
        else:
            payload["values"] = arr.tolist()
        return payload

    if arr.dtype.kind in {"c"}:
        # Complex -> {real, imag} pairs.
        flat = arr.reshape(-1)
        payload["values"] = [{"re": float(x.real), "im": float(x.imag)} for x in flat]
        return payload

    if arr.ndim == 2:
        rows = arr.tolist()
        if preview_max_rows is not None:
            rows = rows[:preview_max_rows]
        payload["rows"] = rows
    elif arr.ndim == 1:
        payload["values"] = arr.tolist()
    else:
        payload["values"] = arr.reshape(-1).tolist()
    return payload
