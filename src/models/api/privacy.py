"""Pydantic models for the /privacy router."""
from __future__ import annotations

from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class RevealRequest(BaseModel):
    tenant_id: UUID
    alias: str = Field(min_length=1)
    reason: str = Field(min_length=4, description="Required for SOC 2 audit trail.")
    actor_id: Optional[str] = None


class RevealResponse(BaseModel):
    alias: str
    original: str
    pii_type: str
    first_seen_at: Optional[str] = None
    seen_count: int = 0


class EraseRequest(BaseModel):
    tenant_id: UUID
    alias: Optional[str] = None
    actor_id: Optional[str] = Field(
        default=None,
        description="If set, erases all aliases originally tied to this actor (subject erasure).",
    )
    reason: str = Field(min_length=4)


class EraseResponse(BaseModel):
    aliases_erased: int
    chunks_redacted: int
