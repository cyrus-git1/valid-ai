"""
/entities router
----------------

The enriched-entity rollout in three tiers:

  Tier 1 — LLM extraction cache (sha256 keyed):
    GET  /entities/cache/{transcript_sha256}
    POST /entities/cache

  Tier 2 — Entity nodes + MENTIONS edges in the KG:
    POST /kg/entities/upsert
    POST /kg/entities/{canonical_id}/merge

  Tier 3 — Rollups (depends on Tier 2):
    GET  /kg/entities/{canonical_id}
    GET  /kg/entities/search

All tenant-scoped via AuthMiddleware. Entity rows live in kg_nodes with
type='Entity'; MENTIONS edges live in kg_edges with rel_type='mentions'.

HTTP-only: auth/tenant checks and input guards live here; the Supabase queries,
RPCs, and embedding orchestration live in EntityService.
"""
from __future__ import annotations

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request

from src.logging_config import get_logger
from src.models.api.entities import (
    CacheUpsertRequest,
    CacheUpsertResponse,
    CachedEntitiesResponse,
    EntityMergeRequest,
    EntityMergeResponse,
    EntityRollupResponse,
    EntitySearchResponse,
    EntityUpsertRequest,
    EntityUpsertResponse,
)
from src.services.entity_service import EntityService
from src.db.supabase_client import get_supabase

logger = get_logger(__name__)
router = APIRouter(prefix="/entities", tags=["entities"])


# ── Helpers ─────────────────────────────────────────────────────────────────


def _require_tenant(request: Request) -> str:
    """Match the existing tenant-confusion pattern from other routers."""
    tenant_id = getattr(request.state, "tenant_id", None)
    if not tenant_id:
        raise HTTPException(status_code=401, detail="authenticated tenant required")
    return str(tenant_id)


def _check_tenant_match(request: Request, body_tenant: UUID) -> str:
    auth_tenant = _require_tenant(request)
    if str(body_tenant) != auth_tenant:
        raise HTTPException(status_code=403, detail="Tenant mismatch")
    return auth_tenant


# ── Endpoints ───────────────────────────────────────────────────────────────


@router.get("/cache/{transcript_sha256}", response_model=CachedEntitiesResponse)
def get_cached_entities(
    transcript_sha256: str,
    request: Request,
    prompt_version: Optional[str] = None,
) -> CachedEntitiesResponse:
    """Fetch cached LLM extraction output. 404 on miss or expired."""
    if len(transcript_sha256) != 64:
        raise HTTPException(status_code=400, detail="transcript_sha256 must be 64 hex chars")
    tenant_id = _require_tenant(request)
    return EntityService(get_supabase()).get_cached(tenant_id, transcript_sha256, prompt_version)


@router.post("/cache", response_model=CacheUpsertResponse, status_code=201)
def upsert_cached_entities(
    body: CacheUpsertRequest,
    request: Request,
) -> CacheUpsertResponse:
    """Upsert cached LLM extraction output. Idempotent on transcript_sha256."""
    if len(body.transcript_sha256) != 64:
        raise HTTPException(status_code=400, detail="transcript_sha256 must be 64 hex chars")
    tenant_id = _check_tenant_match(request, body.tenant_id)
    return EntityService(get_supabase()).upsert_cache(tenant_id, body)


# ════════════════════════════════════════════════════════════════════════════
# TIER 2 — Entity KG nodes + MENTIONS edges
# ════════════════════════════════════════════════════════════════════════════


# Separate router so the URL prefix is /kg/entities/... (per the brief), still
# included alongside the cache endpoints when both are mounted.
kg_router = APIRouter(prefix="/kg/entities", tags=["entities-kg"])


@kg_router.post("/upsert", response_model=EntityUpsertResponse)
def upsert_entity_mentions(
    body: EntityUpsertRequest,
    request: Request,
) -> EntityUpsertResponse:
    """
    Upsert a batch of entities + their MENTIONS edges for one session/document.

    Idempotent: re-posting the same payload does not double-count or duplicate
    edges. Embeddings are computed once per *new* entity; existing entities
    keep their embedding (the brief's cost optimisation).
    """
    tenant_id = _check_tenant_match(request, body.tenant_id)
    return EntityService(get_supabase()).upsert_mentions(tenant_id, body)


@kg_router.post("/{canonical_id}/merge", response_model=EntityMergeResponse)
def merge_entity(
    canonical_id: str,
    body: EntityMergeRequest,
    request: Request,
    label: Optional[str] = Query(None, description="Label of the source entity (disambiguates multi-label)"),
) -> EntityMergeResponse:
    """
    Merge the (canonical_id, label) entity into (surviving_canonical_id,
    surviving_label). All MENTIONS edges rewire to the survivor. The source
    is flagged status='merged' with merged_into pointing at the survivor.
    Idempotent.
    """
    tenant_id = _check_tenant_match(request, body.tenant_id)
    return EntityService(get_supabase()).merge(tenant_id, canonical_id, label, body)


# ════════════════════════════════════════════════════════════════════════════
# TIER 3 — Rollups (read endpoints)
# ════════════════════════════════════════════════════════════════════════════


@kg_router.get("/search", response_model=EntitySearchResponse)
def search_entities(
    request: Request,
    tenant_id: UUID = Query(...),
    client_id: Optional[UUID] = Query(None),
    q: Optional[str] = Query(None, description="Free-text semantic search"),
    label: Optional[str] = Query(None),
    valence: Optional[str] = Query(None, description="positive|negative|neutral|mixed"),
    min_mentions: int = Query(1, ge=0),
    top_k: int = Query(20, ge=1, le=200),
) -> EntitySearchResponse:
    """List or semantic-search tenant entities. Tenant-scoped, never crosses tenants."""
    auth_tenant = _check_tenant_match(request, tenant_id)
    return EntityService(get_supabase()).search(
        auth_tenant, client_id, q, label, valence, min_mentions, top_k,
    )


@kg_router.get("/{canonical_id}", response_model=EntityRollupResponse)
def get_entity_rollup(
    canonical_id: str,
    request: Request,
    tenant_id: UUID = Query(...),
    client_id: Optional[UUID] = Query(None),
    label: Optional[str] = Query(None),
) -> EntityRollupResponse:
    """
    Get full entity profile + all MENTIONS. If `label` is omitted and the
    canonical_id matches multiple entities (different labels), an error is
    raised — pass `label` to disambiguate.
    """
    auth_tenant = _check_tenant_match(request, tenant_id)
    return EntityService(get_supabase()).rollup(auth_tenant, canonical_id, client_id, label)
