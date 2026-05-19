# Data Directory

Large binary files (`.npy`, `.zarr`) are **not tracked in git**.
This directory documents the expected layout and how to obtain or regenerate each file.

---

## Expected layout

```
data/
├── cve_embeddings_demo/
│   ├── embs_99_to_14.npy        ← Period A: CVE 1999–2014  (float32, shape N×D)
│   └── embs_15_to_2025.npy      ← Period B: CVE 2015–2025  (float32, shape N×D)
├── cve_zarr/
│   ├── cve_99_2014.zarr/        ← Optional: Zarr v3 source for Period A
│   └── cve_15_2025.zarr/        ← Optional: Zarr v3 source for Period B
└── results/                     ← Auto-created by notebooks
    ├── spectral_drift_overview.png
    └── yearly_spectral_drift.csv
```

---

## Obtaining the embedding files

### Option A — from a colleague / shared drive

Copy the two `.npy` files into `data/cve_embeddings_demo/` and you are done.
The server will pick them up on next start (or first request to `/api/drift/*`).

### Option B — regenerate from NVD source

Run the data preparation notebook:

```bash
uv pip install -e '.[notebook]'
jupyter lab notebooks/cve.ipynb
```

The notebook downloads the NVD JSON feeds, parses CVE descriptions, embeds
them with `nomic-embed-text-v1.5`, and saves the two period `.npy` files
to this directory.

### Option C — from the Zarr stores

If you have the `.zarr` directories instead of `.npy` files, the
`CveDriftEngine` can load them directly.  Point the env vars at the Zarr paths:

```bash
export ARRO_SERVER_CVE_PERIOD_A=$(pwd)/data/cve_zarr/cve_99_2014.zarr
export ARRO_SERVER_CVE_PERIOD_B=$(pwd)/data/cve_zarr/cve_15_2025.zarr
```

The engine auto-detects the `c/` sub-key layout used by the CVE embedding
pipeline, so no manual conversion is needed.

---

## File sizes (approximate)

| File | Shape | Dtype | Size on disk |
|------|-------|-------|--------------|
| `embs_99_to_14.npy` | ~80 000 × 384 | float32 | ~115 MB |
| `embs_15_to_2025.npy` | ~180 000 × 384 | float32 | ~260 MB |

---

## gitignore rules

The following patterns are ignored (see `.gitignore`):

```
data/cve_embeddings_demo/*.npy
data/cve_zarr/
data/results/*.png
data/results/*.csv
```

Only this `README.md` is tracked.
