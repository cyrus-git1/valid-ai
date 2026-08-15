"""
src/models/api/canvas_events.py
-------------------------------
Read envelopes for the canvas change log (status/confidence transitions on
CanvasBlock and Hypothesis nodes).
"""
from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from src.models.base import TenantOwned


class CanvasEventsRequest(TenantOwned):
    study_id: Optional[UUID] = Field(default=None, description="Study scope; omit for the tenant-wide feed.")
    limit: int = Field(default=100, ge=1, le=500)


class CanvasEvent(BaseModel):
    entity_type: str                       # canvas_block | hypothesis
    entity_key: Optional[str] = None       # block_key or hypothesis external_id
    field: str                             # status | confidence
    from_value: Optional[str] = None       # null on create
    to_value: Optional[str] = None
    study_id: Optional[str] = None
    at: Optional[str] = None


class CanvasEventsResponse(BaseModel):
    events: List[CanvasEvent] = Field(default_factory=list)
