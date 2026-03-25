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


# ── Response ─────────────────────────────────────────────────────────────────


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


# ── Summary sub-models ──────────────────────────────────────────────────────


class ActionItem(BaseModel):
    action: str = Field(description="Description of what needs to be done.")
    owner: Optional[str] = Field(
        default=None, description="Person responsible, or null if unspecified."
    )
    deadline: Optional[str] = Field(
        default=None, description="Deadline or timeframe, or null if unspecified."
    )
    timestamp: Optional[str] = Field(
        default=None, description="WebVTT timestamp (HH:MM:SS) where this was discussed."
    )


class Decision(BaseModel):
    decision: str = Field(description="What was decided.")
    timestamp: Optional[str] = Field(
        default=None, description="WebVTT timestamp (HH:MM:SS) where it was decided."
    )


class TopicGroup(BaseModel):
    topic: str = Field(description="Subject header for this discussion segment.")
    timestamp_start: Optional[str] = Field(
        default=None, description="WebVTT timestamp (HH:MM:SS) when topic started."
    )
    timestamp_end: Optional[str] = Field(
        default=None, description="WebVTT timestamp (HH:MM:SS) when topic ended."
    )
    summary: str = Field(
        description="Brief summary of what was discussed under this topic."
    )


class TranscriptSummaryResponse(BaseModel):
    tenant_id: str
    summary_type: str = Field(description="The summary type used (e.g. 'general').")
    summary: str = Field(description="Concise summary of the entire meeting.")
    action_items: List[ActionItem] = Field(default_factory=list)
    decisions: List[Decision] = Field(default_factory=list)
    topic_groups: List[TopicGroup] = Field(default_factory=list)
    status: str = "complete"
    error: Optional[str] = None
    generated_at: Optional[datetime] = None
