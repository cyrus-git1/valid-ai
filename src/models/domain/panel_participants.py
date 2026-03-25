"""Domain models for panel participant data."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from ._time import utcnow, TimestampMixin

JsonDict = Dict[str, Any]


class PanelParticipantCreate(BaseModel):
    """Single participant upsert intent — one 'document' per participant."""
    tenant_id: UUID
    client_id: UUID
    vendor_name: str
    participant_data: JsonDict
    metadata: JsonDict = Field(default_factory=dict)


class PanelParticipantRow(PanelParticipantCreate, TimestampMixin):
    """Full row with server-assigned id and timestamps."""
    id: UUID


class PanelBatchUpsert(BaseModel):
    """Batch input: a vendor sends a list of participants."""
    tenant_id: UUID
    client_id: UUID
    vendor_name: str
    participants: List[JsonDict]
    metadata: JsonDict = Field(default_factory=dict)
    embed_model: str = "text-embedding-3-small"
    embed_batch_size: int = 64
