"""
src/models/api/hypotheses.py
----------------------------
Request/response envelopes for the hypothesis repo (a near-clone of
canvas_blocks). A hypothesis is a testable claim stored as a kg_node, keyed by
scope + external_id, embedded on its text so it can be linked to themes/evidence.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from src.models.base import TenantOwned


# ── upsert ──────────────────────────────────────────────────────────────────


class HypothesisUpsertRequest(TenantOwned):
    study_id: Optional[UUID] = Field(default=None, description="Study scope; omit for org-level.")
    external_id: str = Field(min_length=1, description="Stable hypothesis id (idempotency key).")
    text: Optional[str] = Field(default=None, description="The claim (embedded).")
    block_key: Optional[str] = Field(default=None, description="Canvas block this bears on.")
    status: Optional[str] = Field(default=None, description="untested | supported | refuted | mixed")
    confidence: Optional[str] = Field(default=None, description="high | medium | low")
    reasoning: Optional[str] = Field(default=None, description="Why we expect this.")
    theme_ids: Optional[List[str]] = Field(default=None, description="Linked concept (theme) ids.")
    evidence_refs: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Stance-tagged SourceRefs [{type,id,origin,stance,weight,study_id}]."
    )


class HypothesisUpsertResponse(BaseModel):
    node_id: str
    external_id: str
    scope: str
    created: bool


# ── by-scope ────────────────────────────────────────────────────────────────


class HypothesesByScopeRequest(TenantOwned):
    study_id: Optional[UUID] = Field(default=None, description="Study scope; omit for org-level.")


class HypothesisRow(BaseModel):
    node_id: str
    external_id: str
    text: Optional[str] = None
    block_key: Optional[str] = None
    status: Optional[str] = None
    confidence: Optional[str] = None
    reasoning: Optional[str] = None
    theme_ids: List[str] = Field(default_factory=list)
    evidence_refs: List[Dict[str, Any]] = Field(default_factory=list)
    study_id: Optional[str] = None


class HypothesesByScopeResponse(BaseModel):
    hypotheses: List[HypothesisRow] = Field(default_factory=list)
