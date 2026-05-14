# Podman-compatible (also a valid Dockerfile). Build with:
#   podman build -t arro-server -f Containerfile .
#   docker build -t arro-server -f Containerfile .

FROM docker.io/library/python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install build prerequisites first to maximise layer reuse.
COPY pyproject.toml README.md ./
COPY src ./src
COPY frontend ./frontend

# Install everything via the [full] extra (zarr, arrow, nlp, notebook, dev).
# Using editable install so src/ changes are reflected without rebuilding.
RUN pip install --upgrade pip \
    && pip install -e '.[full]'

# Run as a non-root user for security.
RUN adduser --disabled-password --gecos "" appuser
USER appuser

EXPOSE 8000
ENV ARRO_SERVER_HOST=0.0.0.0 \
    ARRO_SERVER_PORT=8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

CMD ["python", "-m", "arro_server"]
