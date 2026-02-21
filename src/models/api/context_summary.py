"""Pydantic models for the /context-summary router."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ── Request models ────────────────────────────────────────────────────────────


class ContextSummaryGenerateRequest(BaseModel):
    """Request body for POST /context-summary/generate.

    Triggers LLM-based context analysis and stores the resulting summary.
    """

    tenant_id: UUID
    client_id: UUID
    client_profile: Optional[Dict[str, Any]] = None
    force_regenerate: bool = Field(
        default=False,
        description="If True, regenerates even if a recent summary exists.",
    )


class ContextSummaryGetRequest(BaseModel):
    """Request body for POST /context-summary/get (retrieve existing)."""

    tenant_id: UUID
    client_id: UUID


# ── Response models ───────────────────────────────────────────────────────────


class ContextSummaryResponse(BaseModel):
    """Standard response containing one context summary."""

    id: UUID
    tenant_id: UUID
    client_id: UUID
    summary: str
    topics: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_stats: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ContextSummaryGenerateResponse(BaseModel):
    """Response from the generate endpoint."""

    summary: ContextSummaryResponse
    status: str = "complete"
    regenerated: bool = False
    error: Optional[str] = None


class ContextSummaryDeleteResponse(BaseModel):
    """Response from the delete endpoint."""

    deleted: bool
    tenant_id: UUID
    client_id: UUID
