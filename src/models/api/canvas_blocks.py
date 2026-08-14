"""
src/models/api/canvas_blocks.py
-------------------------------
Request/response envelopes for the grounding canvas (a near-clone of
app_entities). A canvas block is a business-frame node stored as a kg_node,
keyed by scope + block_key, recalled by scope (not nearest).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from src.models.base import TenantOwned


# ── upsert ──────────────────────────────────────────────────────────────────


class CanvasBlockUpsertRequest(TenantOwned):
    study_id: Optional[UUID] = Field(
        default=None, description="Study overlay; omit for the org-level canvas."
    )
    block_key: str = Field(min_length=1, description="One of the twelve canvas block keys.")
    statement: Optional[str] = Field(default=None, description="The current best one-liner (embedded).")
    stated: Optional[str] = Field(default=None, description="What the docs say (assumption layer).")
    evidenced: Optional[str] = Field(default=None, description="What the spine shows (agent-written).")
    source: str = Field(default="agent", description="seed | agent | human")
    status: Optional[str] = Field(default=None, description="assumption | validated | invalidated | refined")
    confidence: Optional[str] = Field(default=None, description="high | medium | low")
    pinned: Optional[bool] = Field(default=None, description="Human-authored; agent won't overwrite the statement.")
    divergence: Optional[bool] = Field(default=None, description="Evidence contradicts a pinned statement.")
    evidence_refs: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Stance-tagged SourceRefs [{type,id,origin,stance,weight,study_id}].",
    )


class CanvasBlockUpsertResponse(BaseModel):
    node_id: str
    block_key: str
    scope: str
    created: bool
    pinned: bool = False
    divergence: bool = False


# ── by-scope (fetch the canvas) ─────────────────────────────────────────────


class CanvasByScopeRequest(TenantOwned):
    study_id: Optional[UUID] = Field(
        default=None, description="Study overlay; omit for the org-level canvas."
    )


class CanvasBlockRow(BaseModel):
    node_id: str
    block_key: str
    statement: Optional[str] = None
    stated: Optional[str] = None
    evidenced: Optional[str] = None
    source: Optional[str] = None
    status: Optional[str] = None
    confidence: Optional[str] = None
    pinned: bool = False
    divergence: bool = False
    study_id: Optional[str] = None
    evidence_refs: List[Dict[str, Any]] = Field(default_factory=list)


class CanvasByScopeResponse(BaseModel):
    blocks: List[CanvasBlockRow] = Field(default_factory=list)
