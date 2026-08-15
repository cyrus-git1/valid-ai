"""
src/models/api/canvas_impact.py
-------------------------------
Impact-ledger links: a shipped change tied to the hypothesis/block it targeted.
"""
from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from src.models.base import TenantOwned


class ImpactLinkUpsertRequest(TenantOwned):
    change_text: str = Field(min_length=1, description="The shipped change.")
    study_id: Optional[UUID] = None
    shipped_at: Optional[str] = Field(default=None, description="ISO date the change shipped.")
    hypothesis_external_id: Optional[str] = Field(default=None, description="Hypothesis this change targeted.")
    block_key: Optional[str] = Field(default=None, description="Or the canvas block it targeted.")
    id: Optional[UUID] = Field(default=None, description="Update an existing link when set.")


class ImpactLinkUpsertResponse(BaseModel):
    id: str


class ImpactLinksRequest(TenantOwned):
    study_id: Optional[UUID] = None


class ImpactLink(BaseModel):
    id: str
    change_text: str
    shipped_at: Optional[str] = None
    hypothesis_external_id: Optional[str] = None
    block_key: Optional[str] = None
    study_id: Optional[str] = None
    created_at: Optional[str] = None


class ImpactLinksResponse(BaseModel):
    links: List[ImpactLink] = Field(default_factory=list)
