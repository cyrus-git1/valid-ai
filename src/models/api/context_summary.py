"""Pydantic models for the /context-summary router."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from src.models.base import StatusResponse, TenantScoped, TenantScopedRequest


# ── Request models ────────────────────────────────────────────────────────────


class ContextSummaryGenerateRequest(TenantScopedRequest):
    """Request body for POST /context-summary/generate.

    Triggers LLM-based context analysis and stores the resulting summary.
    """

    force_regenerate: bool = Field(
        default=False,
        description="If True, regenerates even if a recent summary exists.",
    )


class ContextSummaryGetRequest(TenantScoped):
    """Request body for POST /context-summary/get (retrieve existing)."""


# ── Response models ───────────────────────────────────────────────────────────


class ContextSummaryResponse(TenantScoped):
    """Standard response containing one context summary."""

    id: UUID
    summary: str
    topics: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_stats: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ContextSummaryGenerateResponse(StatusResponse):
    """Response from the generate endpoint."""

    summary: ContextSummaryResponse
    regenerated: bool = False


class ContextSummaryDeleteResponse(TenantScoped):
    """Response from the delete endpoint."""

    deleted: bool
