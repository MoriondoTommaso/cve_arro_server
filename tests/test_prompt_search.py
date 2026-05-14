"""Tests for POST /api/prompts/nl_search, POST /api/prompts/search,
GET /api/prompts/health, and GET /api/prompts/warm.

All heavy dependencies (PromptSearchEngine, EmbedderService) are replaced
with lightweight stubs so tests run without GPU, model weights, or data files.
"""
from __future__ import annotations

import json
import os
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

FAKE_DIM   = 768
FAKE_QUERY = "write a DALL-E prompt for minimalist art"


def _fake_record(i: int) -> dict[str, Any]:
    """Produce a result dict with the same key names that search() emits.

    Uses canonical names (score, _salience) that match PromptSearchResult's
    AliasChoices so Pydantic v2 deserialises and re-serialises them correctly.
    """
    return {
        "id":        f"pk_{i:05d}",
        "title":     f"Prompt title {i}",
        "content":   f"Prompt body {i}",
        "tags":      ["art", "dalle"],
        "upvotes":   i * 3,
        "likes":     i * 2,
        "uses":      i,
        "views":     i * 100,
        # Canonical scoring keys emitted by search() and accepted by
        # PromptSearchResult via AliasChoices.
        "score":     round(0.9 - i * 0.05, 6),
        "_salience": round(0.8 - i * 0.05, 6),
    }


def _make_stub_engine(n: int = 5):
    """Return a MagicMock that quacks like PromptSearchEngine."""
    engine = MagicMock()
    engine.aspace.nitems    = 1000
    engine.aspace.nfeatures = FAKE_DIM
    engine.aspace.nclusters = 20
    engine.gl.nnodes        = 1000
    engine.ids              = [f"pk_{i:05d}" for i in range(n)]
    engine.embs             = np.random.rand(n, FAKE_DIM)
    engine.search.return_value = [_fake_record(i) for i in range(n)]
    return engine


def _make_stub_embedder():
    """Return a MagicMock that quacks like EmbedderService."""
    embedder = MagicMock()
    embedder.model_name = "nomic-ai/nomic-embed-text-v1.5"
    embedder.dim        = FAKE_DIM
    embedder.embed.return_value = np.random.rand(FAKE_DIM).astype(np.float64)
    return embedder


# ---------------------------------------------------------------------------
# App fixture with stubs injected
# ---------------------------------------------------------------------------

@pytest.fixture
def stub_app():
    """
    Creates the FastAPI app with PromptSearchEngine and EmbedderService
    replaced by stubs.  No data files or GPU required.
    """
    from arro_server import settings as settings_mod
    from arro_server import arrowspace_adapter
    from arro_server.app import create_app
    from arro_server import storage as storage_mod
    from arro_server.search_engine import PromptSearchEngine

    # env
    os.environ["ARRO_SERVER_DATA_ROOTS"]       = ""
    os.environ["ARRO_SERVER_SERVE_FRONTEND"]   = "false"
    os.environ["ARRO_SERVER_PROMPT_DATA_DIR"]  = "/nonexistent"

    settings_mod.reset_settings_cache()
    storage_mod.registry.reset_registry_cache()
    arrowspace_adapter.reset_adapter_cache()
    PromptSearchEngine.reset()

    stub_engine   = _make_stub_engine()
    stub_embedder = _make_stub_embedder()

    # Patch the singletons that the route helpers call
    with (
        patch("arro_server.api.routes.PromptSearchEngine") as MockEngine,
        patch("arro_server.embedder.EmbedderService")      as MockEmbedder,
    ):
        # _get_engine() calls PromptSearchEngine.get() -> return stub
        MockEngine.get.return_value             = stub_engine
        MockEngine._instance                    = stub_engine
        # _get_embedder() does `from ..embedder import EmbedderService; EmbedderService.get()`
        MockEmbedder.get.return_value           = stub_embedder
        MockEmbedder._instance                  = stub_embedder

        app = create_app()
        yield app

    # teardown
    PromptSearchEngine.reset()
    os.environ.pop("ARRO_SERVER_DATA_ROOTS",      None)
    os.environ.pop("ARRO_SERVER_SERVE_FRONTEND",   None)
    os.environ.pop("ARRO_SERVER_PROMPT_DATA_DIR",  None)
    settings_mod.reset_settings_cache()
    storage_mod.registry.reset_registry_cache()
    arrowspace_adapter.reset_adapter_cache()


@pytest.fixture
def client(stub_app):
    return TestClient(stub_app)


# ---------------------------------------------------------------------------
# GET /api/prompts/health
# ---------------------------------------------------------------------------

class TestPromptsHealth:
    def test_returns_200(self, client: TestClient):
        r = client.get("/api/prompts/health")
        assert r.status_code == 200

    def test_schema(self, client: TestClient):
        data = client.get("/api/prompts/health").json()
        assert "status" in data
        assert "prompt_engine_ready" in data
        assert "embedder_ready" in data


# ---------------------------------------------------------------------------
# POST /api/prompts/nl_search
# ---------------------------------------------------------------------------

class TestNLSearch:
    def test_valid_query_returns_200(self, client: TestClient):
        r = client.post("/api/prompts/nl_search", json={"query": FAKE_QUERY, "k": 5})
        assert r.status_code == 200

    def test_response_schema(self, client: TestClient):
        data = client.post(
            "/api/prompts/nl_search", json={"query": FAKE_QUERY, "k": 5}
        ).json()
        assert data["query"] == FAKE_QUERY
        assert data["k"] == 5
        assert isinstance(data["results"], list)
        assert "result_count" in data
        assert "tau" in data
        assert "lam" in data

    def test_result_fields(self, client: TestClient):
        """Check that canonical output keys are present and content/body are in sync."""
        results = client.post(
            "/api/prompts/nl_search", json={"query": FAKE_QUERY}
        ).json()["results"]
        assert len(results) > 0
        first = results[0]
        assert "id" in first
        # Canonical serialised names (Pydantic v2 uses field names, not aliases, on output)
        assert "score" in first
        assert "salience" in first
        # content and body must both be populated (sync validator)
        assert first.get("content") is not None
        assert first.get("body") is not None
        assert first["content"] == first["body"]

    def test_empty_query_422(self, client: TestClient):
        r = client.post("/api/prompts/nl_search", json={"query": ""})
        assert r.status_code == 422

    def test_missing_query_422(self, client: TestClient):
        r = client.post("/api/prompts/nl_search", json={"k": 5})
        assert r.status_code == 422

    def test_k_out_of_range_422(self, client: TestClient):
        r = client.post("/api/prompts/nl_search", json={"query": FAKE_QUERY, "k": 0})
        assert r.status_code == 422

    def test_tau_propagated(self, client: TestClient):
        data = client.post(
            "/api/prompts/nl_search",
            json={"query": FAKE_QUERY, "k": 3, "tau": 1.5},
        ).json()
        assert data["tau"] == 1.5

    def test_lam_propagated(self, client: TestClient):
        data = client.post(
            "/api/prompts/nl_search",
            json={"query": FAKE_QUERY, "lam": 0.5},
        ).json()
        assert data["lam"] == 0.5


# ---------------------------------------------------------------------------
# POST /api/prompts/search  (pre-embedded vector)
# ---------------------------------------------------------------------------

class TestPromptVectorSearch:
    def _vec(self) -> list[float]:
        return list(np.random.rand(FAKE_DIM).astype(float))

    def test_valid_vector_returns_200(self, client: TestClient):
        r = client.post("/api/prompts/search", json={"vector": self._vec(), "k": 5})
        assert r.status_code == 200

    def test_response_schema(self, client: TestClient):
        data = client.post(
            "/api/prompts/search", json={"vector": self._vec(), "k": 5}
        ).json()
        assert "results" in data
        assert "result_count" in data
        assert data["query"] is None  # no NL query for this endpoint

    def test_wrong_dim_422(self, client: TestClient):
        r = client.post("/api/prompts/search", json={"vector": [0.1, 0.2, 0.3], "k": 5})
        assert r.status_code == 422

    def test_missing_vector_422(self, client: TestClient):
        r = client.post("/api/prompts/search", json={"k": 5})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Salience plumbing — routes accept and propagate to PromptSearchEngine.search()
# ---------------------------------------------------------------------------

class TestSaliencePlumbing:
    """Verify salience is plumbed end-to-end from request to engine.search()."""

    def test_nl_search_passes_salience_to_engine(self, client: TestClient, stub_app):
        from arro_server.api import routes as routes_mod
        engine = routes_mod.PromptSearchEngine.get()
        engine.search.reset_mock()
        client.post(
            "/api/prompts/nl_search",
            json={"query": FAKE_QUERY, "k": 3, "salience": 0.85},
        )
        kwargs = engine.search.call_args.kwargs
        assert kwargs["salience"] == 0.85

    def test_vector_search_passes_salience_to_engine(self, client: TestClient, stub_app):
        from arro_server.api import routes as routes_mod
        engine = routes_mod.PromptSearchEngine.get()
        engine.search.reset_mock()
        client.post(
            "/api/prompts/search",
            json={"vector": list(np.random.rand(FAKE_DIM).astype(float)),
                  "k": 3, "salience": 0.42},
        )
        kwargs = engine.search.call_args.kwargs
        assert kwargs["salience"] == 0.42

    def test_salience_default_is_used_when_omitted(self, client: TestClient, stub_app):
        from arro_server.api import routes as routes_mod
        engine = routes_mod.PromptSearchEngine.get()
        engine.search.reset_mock()
        client.post("/api/prompts/nl_search", json={"query": FAKE_QUERY, "k": 3})
        kwargs = engine.search.call_args.kwargs
        # NLSearchRequest default is 0.3
        assert kwargs["salience"] == pytest.approx(0.3)

    def test_salience_out_of_range_422(self, client: TestClient):
        r = client.post(
            "/api/prompts/nl_search",
            json={"query": FAKE_QUERY, "salience": 1.5},
        )
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Salience computation — engine produces varied scores when metadata varies
# ---------------------------------------------------------------------------

class TestSalienceComputation:
    """Direct tests against PromptSearchEngine internals (no API).

    These tests do not load embeddings or build an ArrowSpace index; they
    construct an engine instance manually and exercise _compute_salience to
    confirm that varying engagement metadata produces varied salience scores
    (i.e. the salience slider has signal to act on).
    """

    def _make_engine_with(self, dataset: list[dict]):
        from arro_server.search_engine import PromptSearchEngine
        engine = PromptSearchEngine.__new__(PromptSearchEngine)
        engine.ids     = [item["id"] for item in dataset]
        engine.dataset = dataset
        engine._meta   = {item["id"]: item for item in dataset}
        return engine

    def test_varying_metadata_produces_varying_salience(self):
        dataset = [
            {"id": "a", "upvotes":   0, "likes":   0, "views":      0, "author_reputation":  0.0},
            {"id": "b", "upvotes":  50, "likes": 100, "views":   1000, "author_reputation": 10.0},
            {"id": "c", "upvotes": 500, "likes": 900, "views":  50000, "author_reputation": 99.0},
        ]
        engine = self._make_engine_with(dataset)
        scores = engine._compute_salience([0, 1, 2])
        assert scores[0] != scores[1]
        assert scores[1] != scores[2]
        # Highest engagement → highest salience.
        assert scores.argmax() == 2

    def test_uniform_metadata_produces_uniform_salience(self):
        dataset = [
            {"id": "a", "upvotes": 5, "likes": 5, "views": 100, "author_reputation": 1.0},
            {"id": "b", "upvotes": 5, "likes": 5, "views": 100, "author_reputation": 1.0},
        ]
        engine = self._make_engine_with(dataset)
        scores = engine._compute_salience([0, 1])
        # Both fields tie → _norm returns zeros → salience is zero everywhere.
        assert float(scores[0]) == float(scores[1])

    def test_synthesised_engagement_is_deterministic_and_varied(self):
        from arro_server.search_engine import _synthesise_engagement
        a = _synthesise_engagement("pk_00001")
        b = _synthesise_engagement("pk_00001")
        c = _synthesise_engagement("pk_00002")
        assert a == b              # deterministic
        assert a != c              # varies with id

    def test_db_json_fallback_yields_varying_salience(self, tmp_path):
        """End-to-end fallback path: when only id + doc_string is available,
        synthetic engagement metadata still drives a non-degenerate salience
        signal.  This is the path the Docker container uses.
        """
        from arro_server.search_engine import _load_dataset
        repo_root = tmp_path
        (repo_root / "notebooks").mkdir()
        # Minimal tinydb-style dump with only id + doc_string (the bundled shape).
        (repo_root / "notebooks" / "db.json").write_text(json.dumps({
            "_default": {
                "1": {"id": "pk_00001", "doc_string": "alpha"},
                "2": {"id": "pk_00002", "doc_string": "beta"},
                "3": {"id": "pk_00003", "doc_string": "gamma"},
            }
        }))
        data_dir = repo_root / "data"
        data_dir.mkdir()
        records = _load_dataset(data_dir, ["pk_00001", "pk_00002", "pk_00003"])
        assert len(records) == 3
        # Synthetic engagement should differ across ids.
        upvotes = {r["id"]: r["upvotes"] for r in records}
        assert len(set(upvotes.values())) > 1

        engine_dataset = records
        from arro_server.search_engine import PromptSearchEngine
        engine = PromptSearchEngine.__new__(PromptSearchEngine)
        engine.ids     = [r["id"] for r in engine_dataset]
        engine.dataset = engine_dataset
        engine._meta   = {r["id"]: r for r in engine_dataset}
        scores = engine._compute_salience([0, 1, 2])
        # Varied synthetic metadata → varied salience (not all equal).
        assert not np.allclose(scores, scores[0])
