"""Domain models for context summaries."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from src.models.base import TenantScoped
from src.models.domain._time import TimestampMixin

JsonDict = Dict[str, Any]


class ContextSummaryUpsert(TenantScoped):
    """
    Upsert intent for a context summary.

    Natural key: (tenant_id, client_id)
    One active summary per tenant+client pair.
    """

    summary: str
    topics: List[str] = Field(default_factory=list)
    metadata: JsonDict = Field(default_factory=dict)
    source_stats: JsonDict = Field(default_factory=dict)


class ContextSummaryRow(ContextSummaryUpsert, TimestampMixin):
    """Full database row returned after insert/select."""

    id: UUID
