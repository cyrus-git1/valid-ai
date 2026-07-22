"""
src/models/api/concepts.py
--------------------------
Request/response envelopes for the per-org concept spine (first slice).

Concept identity is PER-ORG: canonical_id is minted server-side (a random UUID),
never a label-only hash, and is unique within the tenant. Resolution logic lives
in the agent layer; the data plane provides nearest / create / merge primitives.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.models.base import TenantOwned


# ── nearest ─────────────────────────────────────────────────────────────────


class ConceptNearestRequest(TenantOwned):
    """Embedding (+ optional lexical hints) → candidate concepts, tenant-scoped."""

    query_text: Optional[str] = Field(
        default=None, description="Text to embed for similarity (provide this or `embedding`)"
    )
    embedding: Optional[List[float]] = Field(
        default=None, description="Pre-computed 1536-d embedding (overrides query_text)"
    )
    hints: List[str] = Field(default_factory=list, description="Lexical hints; boost label/alias matches")
    top_k: int = Field(default=10, ge=1, le=100)


class ConceptCandidate(BaseModel):
    id: str
    canonical_id: Optional[str] = None
    canonical_label: Optional[str] = None
    alias_set: Optional[Dict[str, Any]] = None
    merge_confidence: Optional[float] = None
    similarity: Optional[float] = None
    score: Optional[float] = None


class ConceptNearestResponse(BaseModel):
    candidates: List[ConceptCandidate] = Field(default_factory=list)


# ── create ──────────────────────────────────────────────────────────────────


class ConceptCreateRequest(TenantOwned):
    canonical_label: str = Field(min_length=1)
    aliases: List[str] = Field(default_factory=list)
    merge_confidence: float = 1.0
    embedding_text: Optional[str] = Field(
        default=None, description="Representative text to embed (defaults to canonical_label)"
    )
    embedding: Optional[List[float]] = Field(
        default=None, description="Pre-computed centroid/representative embedding"
    )
    canonical_id: Optional[str] = Field(
        default=None,
        description="Supply to make create idempotent (resolve-first re-ingest). Omit to mint a new per-org id.",
    )


class ConceptCreateResponse(BaseModel):
    concept_id: str
    canonical_id: str
    node_key: str
    created: bool
    # True when the supplied canonical_id resolved through a merged tombstone to a
    # surviving concept. concept_id/canonical_id are the SURVIVOR's; nothing was
    # written to it. Log these — each one is a resolve-first miss worth tuning on.
    redirected: bool = False


# ── merge ───────────────────────────────────────────────────────────────────


class ConceptMergeRequest(TenantOwned):
    surviving_concept_id: str = Field(description="kg_nodes.id of the canonical survivor")
    source_concept_id: str = Field(description="kg_nodes.id of the concept folded in")


class ConceptMergeResponse(BaseModel):
    merged: bool
    rewired_count: int = 0
    surviving_member_count: int = 0


# ── by-study ────────────────────────────────────────────────────────────────


class ConceptByStudyItem(BaseModel):
    concept_id: str
    label: Optional[str] = None


class ConceptsByStudyResponse(BaseModel):
    concepts: List[ConceptByStudyItem] = Field(default_factory=list)
