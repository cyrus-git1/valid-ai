from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from src.models.base import TenantOwned
from src.models.domain._time import TimestampMixin

JsonDict = Dict[str, Any]


class DocumentCreate(TenantOwned):
    source_type: str
    source_uri: Optional[str] = None
    title: Optional[str] = None
    metadata: JsonDict = Field(default_factory=dict)


class DocumentRow(DocumentCreate, TimestampMixin):
    id: UUID
