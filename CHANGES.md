# CHANGES

This file documents all modifications made to the upstream
[`Genefold/arro-server`](https://github.com/Genefold/arro-server)
as required by the Apache License 2.0, Section 4(b).

Original work: Copyright 2026 GENEFOLD AI LTD, licensed under Apache-2.0.
This derivative work: LEAF Prompt-Kaban POC, developed by Tommaso Moriondo.

---

## v0.1.0 — 2026-05-12

### New files (not present in upstream)

| File | Description |
|---|---|
| `src/arro_server/search_engine.py` | `PromptSearchEngine` singleton — ArrowSpace-backed semantic search over the LEAF prompt corpus using MMR re-ranking and saliency weighting |
| `src/arro_server/embedder.py` | `EmbedderService` singleton — lazy-loaded `nomic-embed-text-v1.5` sentence-transformer for natural-language query embedding |
| `frontend/app.js` | Vanilla-JS single-page frontend: dataset explorer, manifold canvas, tensor viewer, hybrid search UI |
| `frontend/index.html` | HTML shell for the frontend |
| `frontend/styles.css` | CSS stylesheet for the frontend |
| `CHANGES.md` | This file (Apache-2.0 §4(b) modification record) |

### Modified files (relative to upstream `Genefold/arro-server`)

| File | Changes |
|---|---|
| `src/arro_server/settings.py` | Added `prompt_data_dir` and `embedder_model` settings fields for LEAF Kaban data path and HuggingFace model configuration |
| `src/arro_server/api/routes.py` | Added `/api/prompts/*` route group: `health`, `warm`, `lambdas`, `graph_laplacian`, `audit`, `search` (vector), `nl_search` (natural-language); updated `/api/health` to report `prompt_engine_ready` and `embedder_ready` |
| `src/arro_server/api/schemas.py` | Added `PromptSearchRequest`, `NLSearchRequest`, `PromptSearchResponse`, `PromptSearchResult` Pydantic models |
| `pyproject.toml` | Added optional `[nlp]` dependency group (`sentence-transformers`, `torch`); bumped version to `0.1.0` |
| `.gitignore` | Added entries for `**/.DS_Store`, `*.zarr/`, `_arrowspace_index/`, `arrowspace_index/` |
