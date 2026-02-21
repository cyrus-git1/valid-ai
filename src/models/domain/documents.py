from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel, Field

JsonDict = Dict[str, Any]


class DocumentCreate(BaseModel):
    tenant_id: UUID
    client_id: Optional[UUID] = None

    source_type: str
    source_uri: Optional[str] = None
    title: Optional[str] = None
    metadata: JsonDict = Field(default_factory=dict)


class DocumentRow(DocumentCreate):
    id: UUID
    created_at: datetime
    updated_at: datetime
