# Data Directory

This directory is **gitignored** — large binary files are not committed to the repository.
All files must be generated locally or downloaded from a shared storage location.

## Structure

```
data/
├── cve_embs/
│   └── cve_corpus.parquet          # Primary data file — see schema below
├── cve_zarr/
│   ├── cve_99_2014.zarr/           # Pre-split ArrowSpace Zarr store (period A: 1999–2014)
│   └── cve_15_2025.zarr/           # Pre-split ArrowSpace Zarr store (period B: 2015–2025)
├── cve_embeddings_demo/
│   ├── embs_99_to_14.npy           # Embedding matrix for period A (1999–2014)
│   └── embs_15_to_2025.npy         # Embedding matrix for period B (2015–2025)
└── results/
    └── cve_arrowspace_fstar/       # Optuna tuner output (timestamped run dirs)
        └── <timestamp>/
            └── best_params.json    # Loaded by _load_best_params() at startup
```

## `cve_corpus.parquet` schema (confirmed 2026-05-19)

| Column | Type | Description |
|--------|------|-------------|
| `cve_id` | `str` | CVE identifier, e.g. `CVE-1999-0001` |
| `year` | `int64` | Publication year extracted from the CVE ID |
| `text` | `str` | Vulnerability description (used for embedding) |
| `row_id` | `int64` | Sequential row index aligned to the embedding matrix |
| `embedding` | `object` | Pre-computed embedding vector (`np.ndarray`, shape `(384,)`) |

**Total rows:** 307 251 CVEs (1999–2025)  
**Embedding model:** `all-MiniLM-L6-v2`, output dimension `384`

## Regenerating the corpus

```bash
# 1. Parse cvelistV5 JSON tree → Parquet + embeddings
uv run python scripts/build_corpus.py

# 2. Optionally re-run Optuna hyperparameter tuning
uv run python scripts/tune_arrowspace.py
```

## Period splits for spectral drift

The two `.npy` files in `cve_embeddings_demo/` are row-aligned slices of the
full corpus embedding matrix, partitioned by time period:

| File | Period | Rows |
|------|--------|------|
| `embs_99_to_14.npy` | 1999–2014 | ~80k |
| `embs_15_to_2025.npy` | 2015–2025 | ~227k |

These are consumed by `CveDriftEngine` (`src/arro_server/drift_engine.py`) which
builds a separate ArrowSpace index per period and exposes the Wasserstein drift
score via `/api/drift/score`.

## Quick verification

```python
import pandas as pd
df = pd.read_parquet('data/cve_embs/cve_corpus.parquet')
print(df.dtypes)
print(df.shape)          # (307251, 5)
print(df.head(2))
```
