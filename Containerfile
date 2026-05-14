# syntax=docker/dockerfile:1.6
#
# LEAF Prompt-Kaban POC container.
#
# Multi-stage build:
#   1. `builder` — installs Python + native build deps (gcc, cargo) and resolves
#      all wheels into a virtualenv. arrowspace requires a Rust toolchain;
#      pytrec-eval-terrier and a few transitives require a C compiler.
#   2. `runtime` — slim image that copies the prepared virtualenv. Smaller and
#      no toolchain leaks into the final image.
#
# Build args:
#   INSTALL_NLP=1  Install the [nlp] extra (sentence-transformers + CPU torch).
#                  Default: 1 (NL search works out of the box).
#   INSTALL_NOTEBOOK=0  Install [notebook] extras. Default: 0 (server only).
#
# The torch wheels are pulled from the CPU index so we don't drag in ~1.5GB
# of CUDA runtime that the POC doesn't use.

ARG PYTHON_VERSION=3.12

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — builder
# ─────────────────────────────────────────────────────────────────────────────
FROM docker.io/library/python:${PYTHON_VERSION}-slim AS builder

ARG INSTALL_NLP=1
ARG INSTALL_NOTEBOOK=0

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    CARGO_HOME=/usr/local/cargo \
    RUSTUP_HOME=/usr/local/rustup \
    PATH=/usr/local/cargo/bin:/opt/venv/bin:$PATH

# Build toolchain:
#   - build-essential / gcc: pytrec-eval-terrier C extension, scientific wheels
#     compiled from source on non-amd64 archs.
#   - cargo + rustc: arrowspace (maturin/pyo3) — pip's puccinialin fallback
#     fails when `cc` is absent, so we install the toolchain explicitly.
#   - curl, ca-certificates: rustup bootstrap.
#   - pkg-config, libssl-dev: a few transitive Rust crates link OpenSSL.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        curl \
        ca-certificates \
        pkg-config \
        libssl-dev \
        git \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain stable \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create the runtime virtualenv up-front so every pip install lands in it
# and we can copy a single tree into the runtime stage.
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip setuptools wheel

# Copy build inputs. hatchling needs README, LICENSE, and the frontend/ dir
# (declared as shared-data); src/ is the package itself.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY frontend ./frontend

# Install dependencies.
#   - Use the PyTorch CPU index for torch wheels so we don't pull 1.5GB of
#     CUDA libraries the POC will never load.
#   - Compose the extras list from build args.
RUN set -eux; \
    EXTRAS="zarr,arrow,dev"; \
    if [ "${INSTALL_NLP}" = "1" ]; then EXTRAS="${EXTRAS},nlp"; fi; \
    if [ "${INSTALL_NOTEBOOK}" = "1" ]; then EXTRAS="${EXTRAS},notebook"; fi; \
    /opt/venv/bin/pip install \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        ".[${EXTRAS}]"

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — runtime
# ─────────────────────────────────────────────────────────────────────────────
FROM docker.io/library/python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/venv/bin:$PATH \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1 is needed by sklearn/torch OpenMP backends at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Bring the populated virtualenv across.
COPY --from=builder /opt/venv /opt/venv

# Application files. We copy the bundled demo data too so the container has a
# working prompt-search corpus without requiring a separate volume mount.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY frontend ./frontend
COPY embeddings_nomic_structured_768d_raw.zarr ./embeddings_nomic_structured_768d_raw.zarr
COPY notebooks/db.json ./notebooks/db.json
COPY notebooks/results ./notebooks/results

# Empty data/ — PromptSearchEngine looks for nomic_embs/*.npy here first and
# transparently falls back to the bundled .zarr at the repo root.
RUN mkdir -p /app/data

# Drop to a non-root user.
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

ENV ARRO_SERVER_HOST=0.0.0.0 \
    ARRO_SERVER_PORT=8000 \
    ARRO_SERVER_SERVE_FRONTEND=true \
    ARRO_SERVER_PROMPT_DATA_DIR=/app/data \
    ARRO_SERVER_INDEX_STORE=/app/arrowspace_index \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0) if urllib.request.urlopen('http://localhost:8000/api/health', timeout=3).status==200 else sys.exit(1)" \
    || exit 1

CMD ["python", "-m", "arro_server"]
