"""Domain models for context summaries."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from src.models.domain._time import utcnow

JsonDict = Dict[str, Any]


class ContextSummaryUpsert(BaseModel):
    """
    Upsert intent for a context summary.

    Natural key: (tenant_id, client_id)
    One active summary per tenant+client pair.
    """

    tenant_id: UUID
    client_id: UUID

    summary: str
    topics: List[str] = Field(default_factory=list)
    metadata: JsonDict = Field(default_factory=dict)
    source_stats: JsonDict = Field(default_factory=dict)


class ContextSummaryRow(ContextSummaryUpsert):
    """Full database row returned after insert/select."""

    id: UUID
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
