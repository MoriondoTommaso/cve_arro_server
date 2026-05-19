"""Pydantic request/response schemas for ArrowSpace endpoints.

Using explicit Pydantic models instead of dict[str, Any] for all POST
bodies ensures FastAPI validates inputs and returns 422 automatically for
missing or wrongly-typed fields — before the route body ever runs.
"""

from __future__ import annotations

from typing import Any

from pydantic import AliasChoices, BaseModel, Field, model_validator

# Keys that belong to the ArrowSpaceBuilder graph_params dict.
_GRAPH_PARAM_KEYS = frozenset({"eps", "k", "topk", "p", "sigma"})


class SearchRequest(BaseModel):
    """Body for POST /datasets/{id}/search (spectral taumode search)."""

    vector: list[float] = Field(..., description="Query vector (float64 values).")
    tau: float = Field(1.0, description="Taumode tau parameter.")


class SearchEnergyRequest(BaseModel):
    """Body for POST /datasets/{id}/search/energy."""

    vector: list[float] = Field(..., description="Query vector (float64 values).")
    k: int = Field(10, ge=1, description="Number of results to return.")


class SearchHybridRequest(BaseModel):
    """Body for POST /datasets/{id}/search/hybrid."""

    vector: list[float] = Field(..., description="Query vector (float64 values).")
    alpha: float = Field(0.5, ge=0.0, le=1.0, description="Blend factor (0=spectral, 1=linear).")


class SearchLinearRequest(BaseModel):
    """Body for POST /datasets/{id}/search/linear."""

    vector: list[float] = Field(..., description="Query vector (float64 values).")
    k: int = Field(10, ge=1, description="Number of results to return.")


class SearchBatchRequest(BaseModel):
    """Body for POST /datasets/{id}/search/batch."""

    vectors: list[list[float]] = Field(..., description="Batch of query vectors.")
    tau: float = Field(1.0, description="Taumode tau parameter.")


class PromptSearchRequest(BaseModel):
    """Body for POST /api/prompts/search."""

    vector: list[float] = Field(..., description="768-dim nomic-embed-text-v1.5 query vector.")
    k: int              = Field(10, ge=1, le=100, description="Number of results to return.")
    tau: float          = Field(0.75, ge=0.0, le=5.0, description="Spectral sharpness (0=broad, 5=sharp).")
    alpha: float        = Field(0.6, ge=0.0, le=1.0, description="Spectral-vs-cosine blend.")
    lam: float          = Field(0.7, ge=0.0, le=1.0, description="MMR diversity weight.")
    salience: float     = Field(0.3, ge=0.0, le=1.0, description="Metadata salience influence.")

    @model_validator(mode="after")
    def _check_vector_dim(self) -> "PromptSearchRequest":
        if len(self.vector) != 768:
            raise ValueError(f"vector must have exactly 768 dimensions, got {len(self.vector)}")
        return self


class NLSearchRequest(BaseModel):
    """Body for POST /api/prompts/nl_search."""

    query: str   = Field(..., min_length=1, max_length=2048, description="Natural language search query.")
    k: int       = Field(10, ge=1, le=100, description="Number of results to return.")
    tau: float   = Field(0.75, ge=0.0, le=5.0, description="Spectral sharpness.")
    alpha: float = Field(0.6, ge=0.0, le=1.0, description="Spectral-vs-cosine blend.")
    lam: float   = Field(0.7, ge=0.0, le=1.0, description="MMR diversity weight.")
    salience: float = Field(0.3, ge=0.0, le=1.0, description="Metadata salience influence.")


class PromptSearchResult(BaseModel):
    """A single result returned by /prompts/search or /prompts/nl_search."""

    id: str
    title: str | None = None
    content: str | None = None
    body: str | None = None
    tags: list[str] = Field(default_factory=list)
    upvotes: int | None = None
    views: int | None = None
    author_reputation: float | None = None
    version: int | None = None
    fork_count: int | None = None
    likes: int | None = None
    downvotes: int | None = None
    uses: int | None = None
    created_at: str | None = None
    category: str | None = None
    subcategory: str | None = None
    has_placeholders: bool | None = None
    placeholders: list[str] = Field(default_factory=list)
    difficulty: str | None = None
    language: str | None = None
    target_model: str | None = None

    score: float = Field(
        default=0.0,
        validation_alias=AliasChoices("score", "_score"),
    )
    salience: float = Field(
        default=0.0,
        validation_alias=AliasChoices("salience", "_salience"),
    )
    tau: float = Field(
        default=0.0,
        validation_alias=AliasChoices("tau", "_tau"),
    )

    @model_validator(mode="after")
    def _sync_body_content(self) -> "PromptSearchResult":
        if self.content is not None and self.body is None:
            self.body = self.content
        elif self.body is not None and self.content is None:
            self.content = self.body
        return self

    model_config = {"extra": "allow", "populate_by_name": True}


class PromptSearchResponse(BaseModel):
    """Response envelope for /prompts/search and /prompts/nl_search."""

    query: str | None = Field(None, description="Original NL query (nl_search only).")
    k: int
    tau: float
    lam: float
    results: list[PromptSearchResult]
    result_count: int


class IndexBuildRequest(BaseModel):
    """Optional body for POST /datasets/{id}/index."""

    graph_params: dict[str, Any] = Field(
        default_factory=dict,
        description="ArrowSpace build parameters (eps, k, topk, p, sigma).",
    )

    @model_validator(mode="before")
    @classmethod
    def _hoist_flat_params(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "graph_params" not in data and data.keys() <= _GRAPH_PARAM_KEYS:
                return {"graph_params": data}
        return data


# ---------------------------------------------------------------------------
# CVE drift schemas
# ---------------------------------------------------------------------------

class DriftSearchRequest(BaseModel):
    """Body for POST /api/drift/search."""

    vector: list[float] = Field(
        ...,
        description="Query embedding vector. Dimensionality must match the CVE embeddings.",
    )
    k: int    = Field(10, ge=1, le=200, description="Results to return per period.")
    tau: float = Field(0.5, ge=0.0, le=5.0, description="Taumode spectral sharpness.")


class DriftPeriodResult(BaseModel):
    """Results for one time period in a drift search response.

    extra='allow' ensures that any additional keys returned by the underlying
    ArrowSpace search (e.g. score, tau, distance) are forwarded to the caller
    rather than being silently discarded by Pydantic v2.
    """

    label: str
    results: list[dict[str, Any]]

    model_config = {"extra": "allow", "populate_by_name": True}


class DriftSearchResponse(BaseModel):
    """Response for POST /api/drift/search."""

    drift_score: float = Field(
        ...,
        description="Wasserstein-1 distance between period_a and period_b eigenvalue distributions.",
    )
    period_a: DriftPeriodResult
    period_b: DriftPeriodResult
