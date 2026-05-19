#!/usr/bin/env python
"""Standalone CVE spectral drift script.

Computes and prints the Wasserstein-1 spectral drift score between two
CVE embedding periods WITHOUT starting the HTTP server.

Usage
-----
    uv run python scripts/compute_drift.py

    # custom paths
    uv run python scripts/compute_drift.py \
        --period-a data/cve_embeddings_demo/embs_99_to_14.npy \
        --period-b data/cve_embeddings_demo/embs_15_to_2025.npy \
        --plot

Requirements
------------
    pip install -e '.[notebook]'   # adds pyarrowspace, matplotlib, scipy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_A = Path(__file__).parents[1] / "data" / "cve_embeddings_demo" / "embs_99_to_14.npy"
DEFAULT_B = Path(__file__).parents[1] / "data" / "cve_embeddings_demo" / "embs_15_to_2025.npy"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CVE spectral drift (standalone)")
    p.add_argument("--period-a", default=str(DEFAULT_A), help="Path to Period A .npy or .zarr")
    p.add_argument("--period-b", default=str(DEFAULT_B), help="Path to Period B .npy or .zarr")
    p.add_argument("--n-eigs", type=int, default=200, help="Eigenvalues to compute (default: 200)")
    p.add_argument("--k-neighbours", type=int, default=10, help="k-NN graph neighbours (default: 10)")
    p.add_argument("--n-sample", type=int, default=5_000, help="Subsample size per period (default: 5000)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot", action="store_true", help="Save KDE overlay to data/results/")
    p.add_argument("--api", default="http://localhost:8000", help="Server base URL (for live API check)")
    p.add_argument("--no-api", action="store_true", help="Skip live server check")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_embeddings(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] File not found: {p}", file=sys.stderr)
        print("        See data/README.md for setup instructions.", file=sys.stderr)
        sys.exit(1)
    if p.suffix.lower() == ".npy":
        arr = np.load(str(p), allow_pickle=True)
        if arr.dtype == object:
            arr = np.vstack(arr.tolist())
        return arr.astype(np.float32)
    # Zarr fallback
    try:
        import zarr  # type: ignore
    except ImportError:
        print("[ERROR] zarr not installed. Run: pip install zarr", file=sys.stderr)
        sys.exit(1)
    c_path = p / "c"
    target = str(c_path) if c_path.exists() else str(p)
    return np.asarray(zarr.open_array(target, mode="r")[:], dtype=np.float32)


def subsample(arr: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(arr) <= n:
        return arr
    idx = rng.choice(len(arr), size=n, replace=False)
    return arr[idx]


def build_normalised_laplacian(X: np.ndarray, k: int):
    try:
        from sklearn.neighbors import kneighbors_graph  # type: ignore
        import scipy.sparse as sp
    except ImportError:
        print("[ERROR] scikit-learn and scipy required. Run: pip install scikit-learn scipy", file=sys.stderr)
        sys.exit(1)
    A = kneighbors_graph(X, n_neighbors=k, mode="connectivity", include_self=False, n_jobs=-1)
    A = (A + A.T).astype(np.float32)
    A.data[:] = 1.0
    d = np.asarray(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D = sp.diags(d_inv_sqrt)
    return (sp.eye(X.shape[0]) - D @ A @ D).tocsr()


def compute_spectrum(L, n_eigs: int) -> np.ndarray:
    import scipy.sparse.linalg as spla
    k = min(n_eigs, L.shape[0] - 2)
    eigs, _ = spla.eigsh(L, k=k, which="SM", tol=1e-4, maxiter=3000)
    return np.sort(np.real(eigs))


def wasserstein1d(u: np.ndarray, v: np.ndarray) -> float:
    n = max(len(u), len(v))
    u_i = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(u)), np.sort(u))
    v_i = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(v)), np.sort(v))
    return float(np.mean(np.abs(u_i - v_i)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print("\n=== CVE Spectral Drift — Standalone Script ===")
    print(f"Period A: {args.period_a}")
    print(f"Period B: {args.period_b}")
    print(f"k-neighbours: {args.k_neighbours}  |  n_eigs: {args.n_eigs}  |  sample: {args.n_sample}")
    print()

    # 1. Load
    print("[1/4] Loading embeddings ...")
    embs_a = load_embeddings(args.period_a)
    embs_b = load_embeddings(args.period_b)
    print(f"      Period A: {embs_a.shape}")
    print(f"      Period B: {embs_b.shape}")

    # 2. Subsample
    print(f"[2/4] Subsampling to {args.n_sample} per period ...")
    X_a = subsample(embs_a, args.n_sample, rng)
    X_b = subsample(embs_b, args.n_sample, rng)

    # 3. Build Laplacians & spectra
    print("[3/4] Building Laplacians and computing spectra ...")
    L_a = build_normalised_laplacian(X_a, args.k_neighbours)
    eigs_a = compute_spectrum(L_a, args.n_eigs)
    print(f"      Spectrum A: min={eigs_a.min():.4f}  max={eigs_a.max():.4f}")

    L_b = build_normalised_laplacian(X_b, args.k_neighbours)
    eigs_b = compute_spectrum(L_b, args.n_eigs)
    print(f"      Spectrum B: min={eigs_b.min():.4f}  max={eigs_b.max():.4f}")

    # 4. Drift metrics
    print("[4/4] Computing drift metrics ...")
    w1 = wasserstein1d(eigs_a, eigs_b)
    gap_a = float(eigs_a[eigs_a > 1e-6][0]) if np.any(eigs_a > 1e-6) else 0.0
    gap_b = float(eigs_b[eigs_b > 1e-6][0]) if np.any(eigs_b > 1e-6) else 0.0
    mean_shift = float(np.mean(eigs_b) - np.mean(eigs_a))
    interpretation = "low" if w1 < 0.01 else "medium" if w1 < 0.05 else "high"

    print()
    print("┌─────────────────────────────────────────────┐")
    print("│         Spectral Drift Metrics               │")
    print("├─────────────────────────────────────────────┤")
    print(f"│  Wasserstein-1 (W₁):  {w1:.6f}  [{interpretation:6s}]   │")
    print(f"│  Spectral gap A:      {gap_a:.6f}              │")
    print(f"│  Spectral gap B:      {gap_b:.6f}              │")
    print(f"│  Mean eigenvalue shift (B-A): {mean_shift:+.6f}       │")
    print("└─────────────────────────────────────────────┘")
    print()

    # Optional: plot
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            from scipy.stats import gaussian_kde

            results_dir = Path(__file__).parents[1] / "data" / "results"
            results_dir.mkdir(parents=True, exist_ok=True)

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            x_grid = np.linspace(0, 2, 500)
            kde_a = gaussian_kde(eigs_a, bw_method="scott")
            kde_b = gaussian_kde(eigs_b, bw_method="scott")

            ax = axes[0]
            ax.fill_between(x_grid, kde_a(x_grid), alpha=0.35, color="steelblue", label="Period A (1999-2014)")
            ax.fill_between(x_grid, kde_b(x_grid), alpha=0.35, color="tomato", label="Period B (2015-2025)")
            ax.plot(x_grid, kde_a(x_grid), color="steelblue", lw=1.5)
            ax.plot(x_grid, kde_b(x_grid), color="tomato", lw=1.5)
            ax.set_title("Spectral Density — KDE Overlay")
            ax.set_xlabel("Eigenvalue λ")
            ax.set_ylabel("Density")
            ax.legend()
            ax.annotate(f"W₁ = {w1:.4f}  [{interpretation}]", xy=(0.55, 0.88),
                        xycoords="axes fraction", fontsize=10)

            ax2 = axes[1]
            ax2.plot(eigs_a, color="steelblue", lw=1.5, label="Period A")
            ax2.plot(eigs_b, color="tomato", lw=1.5, label="Period B")
            ax2.set_title("Sorted Eigenvalues (rank plot)")
            ax2.set_xlabel("Rank")
            ax2.set_ylabel("Eigenvalue λ")
            ax2.legend()

            plt.suptitle("CVE Corpus — Graph Laplacian Spectral Drift", fontsize=13, fontweight="bold")
            plt.tight_layout()
            out = results_dir / "spectral_drift_overview.png"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[plot] Saved to {out}")
        except ImportError as exc:
            print(f"[warn] matplotlib/scipy not available, skipping plot: {exc}")

    # Optional: live API check
    if not args.no_api:
        try:
            import urllib.request, json as _json
            url = args.api.rstrip("/") + "/api/drift/score"
            with urllib.request.urlopen(url, timeout=3) as resp:
                data = _json.loads(resp.read())
            print(f"[API]  Live server W₁ = {data['drift_score']:.6f}  ({data['interpretation']})")
            print(f"       {url}")
        except Exception as exc:
            print(f"[API]  Server not reachable at {args.api} ({type(exc).__name__}) — run 'uv run src/arro_server' to start it.")

    print()


if __name__ == "__main__":
    main()
