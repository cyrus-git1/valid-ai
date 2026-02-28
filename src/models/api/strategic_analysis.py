"""Pydantic models for the /strategic-analysis router."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ── Shared / reusable ─────────────────────────────────────────────────────────


class AnalysisParams(BaseModel):
    """Shared tuning knobs reused across single, batch, and all requests."""

    top_k: int = Field(default=10, description="Number of KG nodes to retrieve.")
    hop_limit: int = Field(default=1, description="Graph expansion hops.")
    web_search_queries: List[str] = Field(
        default_factory=list,
        description=(
            "Additional search queries for Serper. If empty, one is auto-generated "
            "from the focus_query and client profile."
        ),
    )
    llm_model: str = Field(default="gpt-4o-mini", description="LLM to use for analysis.")


class ActionPoint(BaseModel):
    """A single actionable recommendation."""

    title: str
    description: str
    priority: str = Field(description="high | medium | low")
    evidence: List[str] = Field(
        default_factory=list,
        description="Sources that support this recommendation (chunk refs, web links, etc.).",
    )


class StrategicAnalysisResult(BaseModel):
    """Core analysis output — shared by single, batch, and all responses."""

    tenant_id: UUID
    client_id: UUID
    focus_query: str

    # Core outputs
    executive_summary: str
    convergent_themes: List[str] = Field(
        default_factory=list,
        description="Key themes that emerged across all data sources.",
    )
    action_points: List[ActionPoint] = Field(default_factory=list)
    future_recommendations: List[str] = Field(default_factory=list)

    # Metadata about scope / data used
    analysis_depth: str = Field(
        description=(
            "foundational | developing | comprehensive | deep — "
            "scales with the volume of transcript data available."
        ),
    )
    transcript_count: int = Field(
        default=0,
        description="Number of video transcript documents contributing to this analysis.",
    )
    sources_used: Dict[str, Any] = Field(
        default_factory=dict,
        description="Counts of chunks, KG nodes, web results, etc. consumed.",
    )
    generated_at: Optional[datetime] = None


# ── Single ────────────────────────────────────────────────────────────────────


class StrategicAnalysisRequest(BaseModel):
    """Request body for POST /strategic-analysis/generate (single)."""

    tenant_id: UUID
    client_id: UUID
    focus_query: str = Field(
        description=(
            "The business question or area of focus for the analysis. "
            "e.g. 'How can we improve customer retention?'"
        ),
    )
    client_profile: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional company labels / client profile overrides.",
    )
    top_k: int = Field(default=10, description="Number of KG nodes to retrieve.")
    hop_limit: int = Field(default=1, description="Graph expansion hops.")
    web_search_queries: List[str] = Field(
        default_factory=list,
        description=(
            "Additional search queries for Serper. If empty, one is auto-generated "
            "from the focus_query and client profile."
        ),
    )
    llm_model: str = Field(default="gpt-4o-mini", description="LLM to use for analysis.")


class StrategicAnalysisResponse(BaseModel):
    """Response from the single generate endpoint."""

    tenant_id: UUID
    client_id: UUID
    focus_query: str

    executive_summary: str
    convergent_themes: List[str] = Field(default_factory=list)
    action_points: List[ActionPoint] = Field(default_factory=list)
    future_recommendations: List[str] = Field(default_factory=list)

    analysis_depth: str
    transcript_count: int = 0
    sources_used: Dict[str, Any] = Field(default_factory=dict)
    generated_at: Optional[datetime] = None


# ── Batch (multiple focus queries, same tenant+client) ────────────────────────


class BatchAnalysisRequest(BaseModel):
    """Request body for POST /strategic-analysis/generate/batch.

    Runs multiple focus queries against the same tenant+client in sequence.
    Shared context (KG, transcripts, summary, profile) is gathered once
    and reused across all queries to avoid redundant data fetching.
    """

    tenant_id: UUID
    client_id: UUID
    focus_queries: List[str] = Field(
        min_length=1,
        description="List of business questions to analyse (1-10).",
    )
    client_profile: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional company labels / client profile overrides.",
    )
    top_k: int = Field(default=10, description="Number of KG nodes to retrieve per query.")
    hop_limit: int = Field(default=1, description="Graph expansion hops.")
    web_search_queries: List[str] = Field(
        default_factory=list,
        description="Shared web search queries applied to every focus query.",
    )
    llm_model: str = Field(default="gpt-4o-mini", description="LLM to use for analysis.")


class BatchAnalysisResponse(BaseModel):
    """Response from the batch generate endpoint."""

    tenant_id: UUID
    client_id: UUID
    total: int = Field(description="Number of focus queries submitted.")
    completed: int = Field(description="Number that succeeded.")
    failed: int = Field(default=0, description="Number that failed.")
    results: List[StrategicAnalysisResult] = Field(default_factory=list)
    errors: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Per-query errors: [{focus_query, error}].",
    )


# ── All (every client_id under a tenant) ──────────────────────────────────────


class AllAnalysisRequest(BaseModel):
    """Request body for POST /strategic-analysis/generate/all.

    Runs the same focus query across every client_id that has ingested
    data under the given tenant_id. Useful for cross-client benchmarking
    or org-wide strategic reviews.
    """

    tenant_id: UUID
    focus_query: str = Field(
        description="The business question to analyse across all clients.",
    )
    client_profile: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional shared company labels applied to every client.",
    )
    top_k: int = Field(default=10, description="Number of KG nodes to retrieve per client.")
    hop_limit: int = Field(default=1, description="Graph expansion hops.")
    web_search_queries: List[str] = Field(
        default_factory=list,
        description="Shared web search queries applied to every client analysis.",
    )
    llm_model: str = Field(default="gpt-4o-mini", description="LLM to use for analysis.")


class AllAnalysisResponse(BaseModel):
    """Response from the all-clients generate endpoint."""

    tenant_id: UUID
    focus_query: str
    total_clients: int = Field(description="Number of client_ids discovered.")
    completed: int = Field(description="Number that succeeded.")
    failed: int = Field(default=0)
    results: List[StrategicAnalysisResult] = Field(default_factory=list)
    errors: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Per-client errors: [{client_id, error}].",
    )
