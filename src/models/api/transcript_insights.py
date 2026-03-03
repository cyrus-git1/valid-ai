"""Pydantic models for the /transcript-insights router."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ── Shared sub-models ────────────────────────────────────────────────────────


class ActionableInsight(BaseModel):
    """A single actionable insight extracted from a transcript."""
    title: str = Field(description="Short label for the insight.")
    description: str = Field(
        description="What the participant said or suggested and why it matters."
    )
    category: str = Field(
        description="Category: feature_request | pain_point | improvement | workflow | ux | other."
    )
    source_quote: Optional[str] = Field(
        default=None,
        description="Verbatim quote from the transcript that supports this insight.",
    )
    priority: str = Field(
        default="medium",
        description="Suggested priority: high | medium | low.",
    )


# ── Request / Response ───────────────────────────────────────────────────────


class TranscriptInsightsRequest(BaseModel):
    tenant_id: UUID
    survey_id: UUID
    llm_model: str = Field(default="gpt-4o-mini", description="LLM to use.")
    chunk_limit: int = Field(
        default=60, ge=1, le=200,
        description="Max transcript chunks to analyse.",
    )


class TranscriptInsightsResponse(BaseModel):
    tenant_id: UUID
    survey_id: UUID
    summary: str = Field(description="Full summary of the transcript.")
    actionable_insights: List[ActionableInsight] = Field(default_factory=list)
    transcript_count: int = 0
    chunks_analysed: int = 0
    status: str = "complete"
    error: Optional[str] = None
    generated_at: Optional[datetime] = None
