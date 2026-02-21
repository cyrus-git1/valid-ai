from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from ._time import utcnow

JsonDict = Dict[str, Any]


class ChunkUpsert(BaseModel):
    # Idempotent upsert intent model.
    # Natural key: (tenant_id, document_id, chunk_index)
    tenant_id: UUID
    document_id: UUID
    chunk_index: int

    page_start: Optional[int] = None
    page_end: Optional[int] = None

    content: str
    content_tokens: Optional[int] = None
    metadata: JsonDict = Field(default_factory=dict)

    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=utcnow)


class ChunkRow(ChunkUpsert):
    id: UUID
