# cve-arro-server — CVE Spectral Drift Demo

> **Demo for Reply data scientists** — uses the CVE corpus to illustrate
> **spectral drift monitoring** with ArrowSpace graph-Laplacian indices.
> Two time periods (1999-2014 vs 2015-2025) are compared through their
> eigenvalue distributions; drift is quantified as a Wasserstein-1 distance
> and served as a live REST API.

---

## What is spectral drift?

Every embedding corpus has a **graph-Laplacian spectrum** — a distribution
of eigenvalues derived from the k-NN connectivity of the data manifold.
When the corpus changes over time (new topics, new vocabulary, shifted
semantics) the spectral distribution shifts too.

This server exposes that shift as a **live, queryable signal**:

```
CVE embeddings (Period A: 1999-2014)   →  ArrowSpace index  →  λ-distribution A ──┐
                                                                                    ├──▶  W₁ drift score
CVE embeddings (Period B: 2015-2025)   →  ArrowSpace index  →  λ-distribution B ──┘
```

A higher Wasserstein-1 (W₁) score means the two periods differ more in their
manifold geometry — i.e., the semantic neighbourhood structure of CVE
descriptions has drifted significantly.

---

## Quick start

```bash
# 1. clone & install
git clone https://github.com/MoriondoTommaso/cve_arro_server.git
cd cve_arro_server
uv venv && uv sync
uv pip install -e '.[full]'

# 2. place CVE embedding files (see data/README.md)
#    data/cve_embeddings_demo/embs_99_to_14.npy
#    data/cve_embeddings_demo/embs_15_to_2025.npy

# 3. run the standalone drift script (no server needed)
uv run python scripts/compute_drift.py

# 4. start the API server
uv run src/arro_server
# → http://localhost:8000/docs
```

---

## Drift endpoints

| Method | Route | Purpose |
|--------|-------|--------|
| `GET` | `/api/drift/health` | Are both period indices built and ready? |
| `GET` | `/api/drift/score` | Scalar W₁ drift score + interpretation label |
| `GET` | `/api/drift/lambdas` | Full eigenvalue arrays for both periods |
| `POST` | `/api/drift/search` | Side-by-side spectral search across both periods |

### Example — check drift score

```bash
curl http://localhost:8000/api/drift/score
```

```jsonc
{
  "drift_score": 0.0412,
  "period_a": "period_a",
  "period_b": "period_b",
  "period_a_n_lambdas": 200,
  "period_b_n_lambdas": 200,
  "interpretation": "medium"
}
```

### Example — side-by-side search

```bash
curl -X POST http://localhost:8000/api/drift/search \
  -H 'Content-Type: application/json' \
  -d '{"vector": [/* 384 floats */], "k": 5, "tau": 0.5}'
```

Returns matched CVEs from **both periods** for the same query vector, so you
can directly compare what a model would retrieve from the old vs. the new
corpus.

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| [`notebooks/cve_spectral_drift.ipynb`](notebooks/cve_spectral_drift.ipynb) | **Primary demo** — standalone spectral analysis, KDE overlay, drift metrics, live API calls |
| [`notebooks/cve_drift_monitoring.ipynb`](notebooks/cve_drift_monitoring.ipynb) | Yearly sweep — tracks W₁, spectral gap, and CVE volume across all years |
| [`notebooks/cve.ipynb`](notebooks/cve.ipynb) | Data preparation — raw NVD → parquet → embeddings |

---

## Configuration

All variables prefixed `ARRO_SERVER_`. Can be set in `.env`.

| Variable | Default | Notes |
|----------|---------|-------|
| `ARRO_SERVER_CVE_PERIOD_A` | `data/cve_embeddings_demo/embs_99_to_14.npy` | Period A embedding file (`.npy` or `.zarr`) |
| `ARRO_SERVER_CVE_PERIOD_B` | `data/cve_embeddings_demo/embs_15_to_2025.npy` | Period B embedding file |
| `ARRO_SERVER_DATA_ROOTS` | `[]` | Zarr roots for `/api/datasets/*` |
| `ARRO_SERVER_PROMPT_DATA_DIR` | `data/` | LEAF Kaban prompt corpus directory |
| `ARRO_SERVER_EMBEDDER_MODEL` | `nomic-ai/nomic-embed-text-v1.5` | HuggingFace model ID |
| `ARRO_SERVER_CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `ARRO_SERVER_MAX_WINDOW` | `10000` | Hard cap on per-request elements |

See `.env.example` for a ready-to-copy template.

---

## Project layout

```
cve_arro_server/
├── data/
│   ├── README.md                         ← data setup instructions
│   ├── cve_embeddings_demo/
│   │   ├── embs_99_to_14.npy             ← Period A embeddings (not tracked in git)
│   │   └── embs_15_to_2025.npy           ← Period B embeddings (not tracked in git)
│   ├── cve_zarr/
│   │   ├── cve_99_2014.zarr/             ← Optional Zarr source for Period A
│   │   └── cve_15_2025.zarr/             ← Optional Zarr source for Period B
│   └── results/                          ← Notebook outputs (plots, CSVs)
├── notebooks/
│   ├── cve_spectral_drift.ipynb          ← Primary drift demo
│   ├── cve_drift_monitoring.ipynb        ← Yearly monitoring sweep
│   └── cve.ipynb                         ← Data preparation
├── scripts/
│   └── compute_drift.py                  ← Standalone drift script (no server needed)
├── src/arro_server/
│   ├── app.py
│   ├── drift_engine.py                   ← CveDriftEngine singleton
│   ├── settings.py
│   └── api/
│       ├── routes.py                     ← /api/drift/* endpoints
│       └── schemas.py
├── Containerfile
├── compose.yaml
└── pyproject.toml
```

---

## Install options

```bash
uv pip install -e .              # server only
uv pip install -e '.[nlp]'       # + NL embedding (sentence-transformers + torch)
uv pip install -e '.[notebook]'  # + Jupyter / matplotlib / scikit-learn
uv pip install -e '.[full]'      # everything
```

> **uv users:** `uv pip install` is equivalent to `pip install` inside a uv venv.

---

## How the drift engine works

```
settings.cve_period_a / cve_period_b
        │
        ▼
_load_embeddings()          reads .npy or .zarr (auto-detects 'c/' sub-key)
        │  np.ndarray  (N × D)
        ▼
_build_arrowspace()         ArrowSpaceBuilder.build(arr)
        │  ArrowSpace object
        ▼
_extract_lambdas()          aspace.lambdas()  →  List[float]
        │
        ▼
_wasserstein1d(λ_A, λ_B)    closed-form 1-D EMD on sorted arrays
        │  float
        ▼
CveDriftEngine.drift_score
```

The singleton is built lazily on the first request (or eagerly at startup
via the FastAPI lifespan). Both period indices stay in memory for fast
side-by-side search.

---

## LEAF Kaban prompt search (original feature)

The repo also contains the LEAF Kaban semantic prompt search backend.
See the original [LEAF Prompt-Kaban POC](https://github.com/Genefold/arro-server)
for full documentation on `/api/prompts/*` endpoints.

---

## Tests

```bash
pytest -q
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
Based on [Genefold/arro-server](https://github.com/Genefold/arro-server), © 2026 GENEFOLD AI LTD.
Modifications by Tommaso Moriondo.
