# Data Directory

This directory is **gitignored** — large binary files are not committed to the repository.
All files below must be generated locally or downloaded from a shared storage location.

## Structure

```
data/
├── cve_embs/
│   └── cve_corpus.parquet          # Primary data file — see schema below
├── cve_zarr/
│   ├── cve_99_2014.zarr/           # Pre-split ArrowSpace Zarr store (period A)
│   └── cve_15_2025.zarr/           # Pre-split ArrowSpace Zarr store (period B)
├── cve_embeddings_demo/
│   ├── embs_99_to_14.npy           # Embedding matrix for period A (1999–2014)
│   └── embs_15_to_2025.npy         # Embedding matrix for period B (2015–2025)
└── results/
    └── cve_arrowspace_fstar/       # Optuna tuner output (timestamped run dirs)
```

## `cve_corpus.parquet` schema

| Column | Type | Description |
|--------|------|-------------|
| `cve_id` | `str` | CVE identifier, e.g. `CVE-1999-0001` |
| `year` | `int64` | Publication year extracted from CVE ID |
| `text` | `str` | Vulnerability description (used for embedding) |
| `row_id` | `int64` | Sequential row index aligned to embedding matrix |
| `embedding` | `object` | Pre-computed embedding vector (`np.ndarray`, shape `(384,)`) |

**Total rows:** 307 251 CVEs (1999–2025)
**Embedding model:** `nomic-embed-text-v1.5` MRL output, dimension 384

## Regenerating the corpus

```bash
# 1. Parse cvelistV5 JSON tree → parquet
uv run python scripts/build_corpus.py

# 2. Optionally re-run Optuna tuning
uv run python scripts/tune_arrowspace.py
```

## Spectral drift demo

The two `.npy` files in `cve_embeddings_demo/` are pre-split slices of the full
embedding matrix by time period and are used directly by the drift notebooks:

- `notebooks/cve_spectral_drift.ipynb` — Laplacian eigenvalue comparison
- `notebooks/cve_drift_monitoring.ipynb` — sliding-window drift monitoring
