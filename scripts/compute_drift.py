"""compute_drift.py

Standalone script to compute spectral drift between CVE period embeddings.
Usage:
    uv run python scripts/compute_drift.py
    uv run python scripts/compute_drift.py --n-sample 3000 --k 15

Outputs:
    data/results/spectral_drift_overview.png
    data/results/drift_metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.stats import gaussian_kde, wasserstein_distance
from sklearn.neighbors import kneighbors_graph

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "cve_embeddings_demo"
RESULTS_DIR = ROOT / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CVE spectral drift computation")
    p.add_argument("--n-sample", type=int, default=5_000, help="Samples per period")
    p.add_argument("--k", type=int, default=10, help="k-NN graph neighbours")
    p.add_argument("--n-eigs", type=int, default=200, help="Laplacian eigenvalues")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_embeddings(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=True)
    if arr.dtype == object:
        return np.vstack(arr.tolist()).astype(np.float32)
    return arr.astype(np.float32)


def build_normalised_laplacian(X: np.ndarray, k: int) -> sp.csr_matrix:
    A = kneighbors_graph(X, n_neighbors=k, mode="connectivity",
                         include_self=False, n_jobs=-1)
    A = (A + A.T).astype(np.float32)
    A.data[:] = 1.0
    d = np.asarray(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    return (sp.eye(X.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt).tocsr()


def compute_spectrum(L: sp.csr_matrix, n_eigs: int) -> np.ndarray:
    k = min(n_eigs, L.shape[0] - 2)
    eigs, _ = spla.eigsh(L, k=k, which="SM", tol=1e-4, maxiter=3000)
    return np.sort(np.real(eigs))


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print("Loading embeddings ...")
    embs_A = load_embeddings(DATA_DIR / "embs_99_to_14.npy")
    embs_B = load_embeddings(DATA_DIR / "embs_15_to_2025.npy")
    print(f"  Period A: {embs_A.shape}  Period B: {embs_B.shape}")

    idx_A = rng.choice(len(embs_A), size=min(args.n_sample, len(embs_A)), replace=False)
    idx_B = rng.choice(len(embs_B), size=min(args.n_sample, len(embs_B)), replace=False)
    X_A, X_B = embs_A[idx_A], embs_B[idx_B]

    print("Building Laplacians ...")
    L_A = build_normalised_laplacian(X_A, args.k)
    L_B = build_normalised_laplacian(X_B, args.k)

    print("Computing spectra ...")
    eigs_A = compute_spectrum(L_A, args.n_eigs)
    eigs_B = compute_spectrum(L_B, args.n_eigs)

    # --- metrics ---
    w1 = float(wasserstein_distance(eigs_A, eigs_B))
    gap_A = float(eigs_A[eigs_A > 1e-6][0]) if np.any(eigs_A > 1e-6) else 0.0
    gap_B = float(eigs_B[eigs_B > 1e-6][0]) if np.any(eigs_B > 1e-6) else 0.0
    mean_shift = float(np.mean(eigs_B) - np.mean(eigs_A))

    metrics = {
        "wasserstein_1": w1,
        "spectral_gap_A_1999_2014": gap_A,
        "spectral_gap_B_2015_2025": gap_B,
        "mean_eigenvalue_shift": mean_shift,
        "n_sample_A": len(X_A),
        "n_sample_B": len(X_B),
        "k_neighbours": args.k,
        "n_eigenvalues": args.n_eigs,
    }

    out_json = RESULTS_DIR / "drift_metrics.json"
    out_json.write_text(json.dumps(metrics, indent=2))
    print("\n=== Drift Metrics ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\nSaved metrics to {out_json}")

    # --- plot ---
    x = np.linspace(0, 2, 500)
    kde_A = gaussian_kde(eigs_A, bw_method="scott")
    kde_B = gaussian_kde(eigs_B, bw_method="scott")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.fill_between(x, kde_A(x), alpha=0.35, color="steelblue", label="Period A (1999-2014)")
    ax1.fill_between(x, kde_B(x), alpha=0.35, color="tomato",    label="Period B (2015-2025)")
    ax1.plot(x, kde_A(x), color="steelblue", lw=1.5)
    ax1.plot(x, kde_B(x), color="tomato",    lw=1.5)
    ax1.set_title("Spectral Density — KDE Overlay")
    ax1.set_xlabel("Eigenvalue λ")
    ax1.set_ylabel("Density")
    ax1.legend()
    ax1.annotate(f"W₁ = {w1:.4f}", xy=(0.62, 0.88), xycoords="axes fraction", fontsize=10)

    ax2.plot(eigs_A, color="steelblue", lw=1.5, label="Period A")
    ax2.plot(eigs_B, color="tomato",    lw=1.5, label="Period B")
    ax2.set_title("Sorted Eigenvalues (rank plot)")
    ax2.set_xlabel("Rank")
    ax2.set_ylabel("Eigenvalue λ")
    ax2.legend()

    plt.suptitle("CVE Corpus — Graph Laplacian Spectral Drift", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_png = RESULTS_DIR / "spectral_drift_overview.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot to {out_png}")


if __name__ == "__main__":
    main()
