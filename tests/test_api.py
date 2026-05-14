"""Integration tests for the arro-server HTTP API.

All tests use the TestClient fixture from conftest.py which creates an
isolated FastAPI app pointed at a tmp Zarr root.

Tests are aligned with the *actual* response shapes returned by the routes
in src/arro_server/api/routes.py.  See CHANGES.md for history.
"""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient

_arrowspace_available = importlib.util.find_spec("arrowspace") is not None
_skip_no_arrowspace = pytest.mark.skipif(
    not _arrowspace_available,
    reason="arrowspace package not installed",
)


@pytest.fixture
def client(configured_app):
    return TestClient(configured_app)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "main" in body["data_roots"]


def test_health_reports_backend(client: TestClient) -> None:
    """GET /health includes arrowspace_backend and arrowspace_available."""
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert "arrowspace_backend" in body
    assert body["arrowspace_backend"] in {"arrowspace", "sidecar", "none"}
    assert "arrowspace_available" in body
    assert isinstance(body["arrowspace_available"], bool)


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------


def test_list_datasets(client: TestClient) -> None:
    r = client.get("/api/datasets")
    assert r.status_code == 200
    body = r.json()
    ids = {d["id"] for d in body["datasets"] if d["kind"] == "array"}
    assert {"main--matrix", "main--vector"}.issubset(ids)


def test_list_datasets_root_and_path_preserved(client: TestClient) -> None:
    r = client.get("/api/datasets")
    assert r.status_code == 200
    body = r.json()
    matrix = next(d for d in body["datasets"] if d["id"] == "main--matrix")
    assert matrix["root"] == "main"
    assert matrix["path"] == "matrix"


def test_metadata(client: TestClient) -> None:
    r = client.get("/api/datasets/main--matrix/metadata")
    assert r.status_code == 200
    body = r.json()
    assert body["shape"] == [50, 4]
    assert body["dtype"].startswith("float32")


def test_unknown_dataset(client: TestClient) -> None:
    r = client.get("/api/datasets/main--missing/metadata")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Data window + pagination
# ---------------------------------------------------------------------------


def test_data_window(client: TestClient) -> None:
    """GET /data returns offset, limit, and rows of data."""
    r = client.get("/api/datasets/main--matrix/data?offset=0&limit=10")
    assert r.status_code == 200
    body = r.json()
    assert body["offset"] == 0
    assert body["limit"] == 10
    rows = body["data"]["rows"]
    assert len(rows) == 10
    assert rows[0] == [0.0, 1.0, 2.0, 3.0]


def test_data_pagination_terminates(client: TestClient) -> None:
    """Requesting past the end returns only the remaining rows."""
    r = client.get("/api/datasets/main--matrix/data?offset=45&limit=20")
    assert r.status_code == 200
    body = r.json()
    # Route clamps to what's available — 5 rows (indices 45-49)
    rows = body["data"]["rows"]
    assert len(rows) == 5


def test_window_budget_enforced(client: TestClient) -> None:
    """Requesting exactly max_window rows should succeed."""
    r = client.get("/api/datasets/main--matrix/data?offset=0&limit=50")
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Slice
# ---------------------------------------------------------------------------


def test_slice(client: TestClient) -> None:
    """Slice query param is named 'spec'."""
    r = client.get("/api/datasets/main--matrix/slice?spec=0:3,1:3")
    assert r.status_code == 200
    body = r.json()
    assert body["out_shape"] == [3, 2]
    assert body["data"]["rows"][0] == [1.0, 2.0]


def test_slice_with_step(client: TestClient) -> None:
    """Step > 1 slices return every Nth row."""
    r = client.get("/api/datasets/main--matrix/slice?spec=0:10:2")
    assert r.status_code == 200
    body = r.json()
    assert body["out_shape"] == [5, 4]


def test_slice_negative_index(client: TestClient) -> None:
    """Negative start index resolves from the end."""
    r = client.get("/api/datasets/main--matrix/slice?spec=-3:")
    assert r.status_code == 200
    body = r.json()
    assert body["out_shape"] == [3, 4]


def test_invalid_slice(client: TestClient) -> None:
    """An unparseable spec string returns 422."""
    r = client.get("/api/datasets/main--matrix/slice?spec=foo")
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Manifold + Stats  (sidecar JSON — no arrowspace index required)
# ---------------------------------------------------------------------------


def test_manifold_sidecar(client: TestClient) -> None:
    """GET /manifold returns the raw sidecar manifold.json contents."""
    r = client.get("/api/datasets/main--matrix/manifold")
    assert r.status_code == 200
    body = r.json()
    # conftest writes {"dim": 2, "n_points": 50} into manifold.json
    assert body.get("dim") == 2
    assert body.get("n_points") == 50


def test_stats_returns_basic_shape(client: TestClient) -> None:
    """GET /stats returns the raw sidecar stats.json contents."""
    r = client.get("/api/datasets/main--matrix/stats")
    assert r.status_code == 200
    body = r.json()
    # conftest writes {"mean": 99.5, "std": 57.7} into stats.json
    assert "mean" in body
    assert "std" in body


# ---------------------------------------------------------------------------
# Keyword search (sidecar index.json)
# ---------------------------------------------------------------------------


def test_search_sidecar(client: TestClient) -> None:
    r = client.get("/api/datasets/main--matrix/search?q=alpha")
    assert r.status_code == 200
    body = r.json()
    assert body["results"]
    assert body["results"][0]["id"] == "row-0"


def test_search_missing_index(client: TestClient) -> None:
    """Vector dataset has no sidecar index.json — expect 404."""
    r = client.get("/api/datasets/main--vector/search?q=anything")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# ArrowSpace index build + vector search  (requires arrowspace package)
# ---------------------------------------------------------------------------


@_skip_no_arrowspace
def test_post_index(client: TestClient) -> None:
    """POST /index returns {nitems, nfeatures, nclusters}."""
    r = client.post("/api/datasets/main--matrix/index")
    assert r.status_code == 200
    body = r.json()
    assert body["nitems"] == 50
    assert body["nfeatures"] == 4
    assert "nclusters" in body


@_skip_no_arrowspace
def test_post_index_custom_params(client: TestClient) -> None:
    """POST /index with flat graph_params body is accepted (hoisted by schema)."""
    params = {"eps": 0.5, "k": 4, "topk": 2, "p": 1.0, "sigma": 0.5}
    r = client.post("/api/datasets/main--matrix/index", json=params)
    assert r.status_code == 200
    body = r.json()
    # Route returns the index meta — just verify the build succeeded
    assert body["nitems"] == 50


@_skip_no_arrowspace
def test_post_search_vector(client: TestClient) -> None:
    """POST /search returns scored {index, score} results after index build."""
    client.post("/api/datasets/main--matrix/index")
    r = client.post(
        "/api/datasets/main--matrix/search",
        json={"vector": [0.0, 1.0, 2.0, 3.0], "tau": 1.0},
    )
    assert r.status_code == 200
    body = r.json()
    assert "results" in body
    assert len(body["results"]) > 0
    hit = body["results"][0]
    assert isinstance(hit["index"], int)
    assert isinstance(hit["score"], float)
    assert body["backend"] == "arrowspace"


@_skip_no_arrowspace
def test_lambdas_endpoint(client: TestClient) -> None:
    """GET /lambdas returns eigenvalue data after index build."""
    client.post("/api/datasets/main--matrix/index")
    r = client.get("/api/datasets/main--matrix/lambdas")
    assert r.status_code == 200
    body = r.json()
    assert "lambdas" in body
    assert "lambdas_sorted" in body
    assert "nitems" in body


def test_post_search_requires_index(client: TestClient) -> None:
    """POST /search without a built index returns 404, 501 or 503."""
    r = client.post(
        "/api/datasets/main--matrix/search",
        json={"vector": [0.1, 0.2, 0.3, 0.4], "tau": 1.0},
    )
    # 501 = arrowspace not installed (sidecar stub raises OptionalDependencyMissing)
    # 404 = installed but no index built yet
    # 503 = service unavailable (legacy)
    assert r.status_code in {404, 501, 503}
