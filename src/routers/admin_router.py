"""
/admin router
-------------
Operational endpoints — health and stats.

GET  /admin/health — Liveness check (Supabase + OpenAI reachable)
GET  /admin/stats  — Document/chunk/node/edge counts for a client

Reindex and rebuild-kg have moved to the agent service.
"""
from __future__ import annotations

import logging
import os
from uuid import UUID

import secrets
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.supabase.supabase_client import get_supabase
from src.services.memory_state_service import MemoryStateService
from src.services.api_key_service import hash_key, key_prefix
from src.models.api.admin import (
    HealthResponse,
    MaintenanceClientResult,
    MaintenanceRunRequest,
    MaintenanceRunResponse,
    StatsResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])


def _discover_client_ids(sb, tenant_id: UUID) -> list[str]:
    client_ids: set[str] = set()
    for table in ("documents", "kg_nodes", "memory_state"):
        try:
            res = (
                sb.table(table)
                .select("client_id")
                .eq("tenant_id", str(tenant_id))
                .execute()
            )
            for row in (res.data or []):
                client_id = row.get("client_id")
                if client_id:
                    client_ids.add(client_id)
        except Exception as e:
            logger.warning("Failed to discover client_ids from %s for tenant=%s: %s", table, tenant_id, e)
    return sorted(client_ids)


# ── API key management ────────────────────────────────────────────────────────


class ApiKeyCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    scopes: List[str] = Field(default_factory=list)
    expires_at: Optional[str] = None


class ApiKeyCreateResponse(BaseModel):
    id: str
    raw_key: str = Field(description="Shown once — store it now.")
    key_prefix: str
    name: str
    scopes: List[str]


class ApiKeyListItem(BaseModel):
    id: str
    key_prefix: str
    name: str
    scopes: List[str]
    status: str
    last_used_at: Optional[str] = None
    expires_at: Optional[str] = None
    created_at: str
    revoked_at: Optional[str] = None


def _require_admin(request: Request) -> str:
    scopes = getattr(request.state, "scopes", []) or []
    if "admin" not in scopes:
        raise HTTPException(status_code=403, detail="admin scope required")
    return getattr(request.state, "tenant_id")


@router.post("/api-keys", response_model=ApiKeyCreateResponse)
def create_api_key(body: ApiKeyCreateRequest, request: Request) -> ApiKeyCreateResponse:
    """Issue a new API key for the caller's tenant. Raw key returned ONCE."""
    tenant_id = _require_admin(request)
    sb = get_supabase()

    # Generate 32 bytes of entropy → 43-char urlsafe token
    raw = "dp_" + secrets.token_urlsafe(32)
    res = sb.table("api_keys").insert({
        "tenant_id": str(tenant_id),
        "key_hash": hash_key(raw),
        "key_prefix": key_prefix(raw),
        "name": body.name,
        "scopes": body.scopes,
        "status": "active",
        "expires_at": body.expires_at,
    }).execute()
    row = (res.data or [{}])[0]

    logger.info(
        "admin.api_key.create tenant=%s key_id=%s prefix=%s scopes=%s",
        tenant_id, row.get("id"), key_prefix(raw), body.scopes,
    )
    return ApiKeyCreateResponse(
        id=row.get("id", ""),
        raw_key=raw,
        key_prefix=key_prefix(raw),
        name=body.name,
        scopes=body.scopes,
    )


@router.get("/api-keys", response_model=List[ApiKeyListItem])
def list_api_keys(request: Request) -> List[ApiKeyListItem]:
    """List API keys for the caller's tenant (prefixes only, never raw)."""
    tenant_id = _require_admin(request)
    sb = get_supabase()
    res = (
        sb.table("api_keys")
        .select("id, key_prefix, name, scopes, status, last_used_at, expires_at, created_at, revoked_at")
        .eq("tenant_id", str(tenant_id))
        .order("created_at", desc=True)
        .execute()
    )
    return [ApiKeyListItem(**row) for row in (res.data or [])]


@router.delete("/api-keys/{key_id}")
def revoke_api_key(key_id: str, request: Request) -> dict:
    """Revoke an API key."""
    tenant_id = _require_admin(request)
    sb = get_supabase()
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    res = (
        sb.table("api_keys")
        .update({"status": "revoked", "revoked_at": now})
        .eq("id", key_id)
        .eq("tenant_id", str(tenant_id))
        .execute()
    )
    if not res.data:
        raise HTTPException(status_code=404, detail="API key not found")
    logger.info("admin.api_key.revoke tenant=%s key_id=%s", tenant_id, key_id)
    return {"revoked": True, "id": key_id}


# ── Tenant plan management (subscription tiers) ─────────────────────────────


class TenantPlanResponse(BaseModel):
    tenant_id: str
    plan: str
    notes: Optional[str] = None
    daily_embedding_tokens_limit: int
    daily_embedding_tokens_used: int = 0
    max_body_bytes: int
    max_chunks_per_ingest: int


class TenantPlanSetRequest(BaseModel):
    plan: str = Field(description="free | pro | enterprise")
    notes: Optional[str] = None


def _build_plan_response(tenant_id: str, plan: str, notes: Optional[str] = None) -> TenantPlanResponse:
    from src.config.plan_limits import get_limit, VALID_PLANS
    from src.services.tenant_plan_service import EmbeddingQuotaService, TenantPlanService
    if plan not in VALID_PLANS:
        plan = "free"
    sb = get_supabase()
    plan_svc = TenantPlanService(sb)
    used = EmbeddingQuotaService(plan_svc).current_usage(tenant_id)
    return TenantPlanResponse(
        tenant_id=tenant_id,
        plan=plan,
        notes=notes,
        daily_embedding_tokens_limit=get_limit(plan, "daily_embedding_tokens"),
        daily_embedding_tokens_used=used,
        max_body_bytes=get_limit(plan, "max_body_bytes"),
        max_chunks_per_ingest=get_limit(plan, "max_chunks_per_ingest"),
    )


@router.get("/plan", response_model=TenantPlanResponse)
def get_my_plan(request: Request) -> TenantPlanResponse:
    """Return the authenticated tenant's plan + current usage."""
    tenant_id = getattr(request.state, "tenant_id", None)
    if not tenant_id:
        raise HTTPException(status_code=401, detail="authenticated tenant required")
    from src.services.tenant_plan_service import TenantPlanService
    sb = get_supabase()
    plan_svc = TenantPlanService(sb)
    plan = plan_svc.get_plan(str(tenant_id))
    notes = None
    try:
        res = sb.table("tenant_plans").select("notes").eq("tenant_id", str(tenant_id)).limit(1).execute()
        if res.data:
            notes = res.data[0].get("notes")
    except Exception:
        pass
    return _build_plan_response(str(tenant_id), plan, notes)


@router.put("/plan", response_model=TenantPlanResponse)
def set_my_plan(body: TenantPlanSetRequest, request: Request) -> TenantPlanResponse:
    """Update the authenticated tenant's plan. Admin scope required."""
    tenant_id = _require_admin(request)
    from src.config.plan_limits import VALID_PLANS
    from src.services.tenant_plan_service import TenantPlanService
    if body.plan not in VALID_PLANS:
        raise HTTPException(status_code=400, detail=f"plan must be one of {VALID_PLANS}")
    sb = get_supabase()
    TenantPlanService(sb).set_plan(str(tenant_id), body.plan, body.notes)
    logger.info("admin.plan.set tenant=%s plan=%s", tenant_id, body.plan)
    return _build_plan_response(str(tenant_id), body.plan, body.notes)


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness + dependency check."""
    sb_ok = False
    detail = None

    try:
        sb = get_supabase()
        sb.table("documents").select("id").limit(1).execute()
        sb_ok = True
    except Exception as e:
        detail = f"Supabase unreachable: {e}"
        logger.error(detail)

    openai_ok = bool(os.environ.get("OPENAI_API_KEY"))
    if not openai_ok:
        detail = (detail or "") + " OPENAI_API_KEY missing."

    overall = "ok" if (sb_ok and openai_ok) else "degraded"

    return HealthResponse(
        status=overall,
        supabase=sb_ok,
        openai=openai_ok,
        detail=detail,
    )


@router.get("/stats", response_model=StatsResponse)
def stats(
    tenant_id: UUID = Query(...),
    client_id: UUID = Query(...),
) -> StatsResponse:
    """Document, chunk, KG node, and KG edge counts for a tenant+client."""
    sb = get_supabase()
    memory_state = MemoryStateService(sb)

    def _count(table: str, filters: dict) -> int:
        q = sb.table(table).select("id", count="exact")
        for col, val in filters.items():
            q = q.eq(col, val)
        return q.execute().count or 0

    doc_count = _count("documents", {"tenant_id": str(tenant_id), "client_id": str(client_id)})
    doc_ids_res = (
        sb.table("documents")
        .select("id")
        .eq("tenant_id", str(tenant_id))
        .eq("client_id", str(client_id))
        .execute()
    )
    doc_ids = [row["id"] for row in (doc_ids_res.data or [])]
    if doc_ids:
        chunk_count = (
            sb.table("chunks")
            .select("id", count="exact")
            .eq("tenant_id", str(tenant_id))
            .in_("document_id", doc_ids)
            .execute()
            .count
            or 0
        )
    else:
        chunk_count = 0

    try:
        emb_res = sb.rpc(
            "fetch_chunks_with_embeddings",
            {
                "p_tenant_id": str(tenant_id),
                "p_client_id": str(client_id),
                "p_document_id": None,
                "p_limit": 1,
                "p_offset": 0,
            },
        ).execute()
        chunks_with_embeddings = chunk_count
    except Exception:
        chunks_with_embeddings = -1

    node_count = _count("kg_nodes", {"tenant_id": str(tenant_id), "client_id": str(client_id)})
    edge_count = _count("kg_edges", {"tenant_id": str(tenant_id), "client_id": str(client_id)})
    client_state = memory_state.get_state(tenant_id=tenant_id, client_id=client_id)
    tenant_state = memory_state.get_state(tenant_id=tenant_id, client_id=None)

    return StatsResponse(
        tenant_id=str(tenant_id),
        client_id=str(client_id),
        document_count=doc_count,
        chunk_count=chunk_count,
        chunks_with_embeddings=chunks_with_embeddings,
        kg_node_count=node_count,
        kg_edge_count=edge_count,
        client_memory_version=int(client_state.get("memory_version") or 0),
        tenant_memory_version=int(tenant_state.get("memory_version") or 0),
        client_last_ingested_at=client_state.get("last_ingested_at"),
        client_last_changed_at=client_state.get("last_changed_at"),
        client_last_summary_at=client_state.get("last_summary_at"),
    )


# ── Governance backfill: mirror seeded taxonomy + re-embed null-embedding rows ──
#
# Two operator-triggered sweeps that close the governance loop after an embedder
# outage (see migration 56). Both are system operations — the embedding cost is
# NOT charged to the tenant's daily quota (tenant_id=None on the embed call), so a
# large backfill can't exhaust a tenant's budget or fail halfway on a quota gate.
#
# Ordering matters (agent-layer contract):
#   1. /admin/mirror-taxonomy → governed Concept nodes exist WITH embeddings
#   2. /admin/reembed         → outage-era null-embedding rows become resolvable
#   3. (app) /spine/backfill  → observations re-resolve to the now-embedded
#                               governed nodes + candidates graduate
#
# NOTE: the data plane has no local codebook table (`transcript_tags` lives in the
# app/agent layer), so mirror-taxonomy takes the seed tags in the BODY rather than
# self-sourcing — the caller supplies {tag_id, label, description}. This is the
# batch form of the already-shipped per-tag POST /concepts/mirror-tag.


def _rpc_scalar(data):
    """Normalize a scalar/jsonb RPC result (PostgREST may list-wrap it)."""
    if isinstance(data, list):
        data = data[0] if data else None
    return data


class TaxonomyTag(BaseModel):
    tag_id: str = Field(description="transcript_tags uuid — becomes the concept node id + external_ref")
    label: str = Field(min_length=1)
    description: Optional[str] = None


class MirrorTaxonomyRequest(BaseModel):
    tenant_id: str
    client_id: Optional[str] = None
    tags: List[TaxonomyTag] = Field(default_factory=list, description="Seeded codebook tags to mirror")


class MirrorTaxonomyResultItem(BaseModel):
    tag_id: str
    concept_id: str = ""
    external_ref: str = ""
    embedded: bool = False
    error: Optional[str] = None


class MirrorTaxonomyResponse(BaseModel):
    tenant_id: str
    mirrored: int
    embedded: int
    failed: int
    results: List[MirrorTaxonomyResultItem]


@router.post("/mirror-taxonomy", response_model=MirrorTaxonomyResponse)
def mirror_taxonomy(body: MirrorTaxonomyRequest, request: Request) -> MirrorTaxonomyResponse:
    """Batch-mirror seeded taxonomy tags into the graph as governed Concept nodes
    (node id = tag id, external_ref = tag id), each embedded on label+description
    so observations can resolve to them via /concepts/nearest. Idempotent per tag.
    """
    from src.routers.ingest_router import _EMBED_MODEL, _embed_in_batches

    try:
        tenant_id = str(UUID(body.tenant_id))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid tenant_id: {e}")
    client_id: Optional[str] = None
    if body.client_id:
        try:
            client_id = str(UUID(body.client_id))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid client_id: {e}")

    if not body.tags:
        return MirrorTaxonomyResponse(tenant_id=tenant_id, mirrored=0, embedded=0, failed=0, results=[])

    sb = get_supabase()
    # One embed batch for the whole seed. System op → not billed to the tenant quota.
    texts = [
        (f"{t.label}. {t.description}".strip() if t.description else t.label)
        for t in body.tags
    ]
    try:
        embeddings = _embed_in_batches(texts, tenant_id=None)
    except Exception as ex:
        logger.exception("mirror_taxonomy embedding failed tenant=%s", tenant_id)
        raise HTTPException(status_code=502, detail=f"Embedding provider error: {ex}")

    results: List[MirrorTaxonomyResultItem] = []
    mirrored = embedded = failed = 0
    for tag, emb in zip(body.tags, embeddings):
        try:
            tag_uuid = str(UUID(tag.tag_id))
        except ValueError:
            failed += 1
            results.append(MirrorTaxonomyResultItem(tag_id=tag.tag_id, error="invalid tag_id (not a uuid)"))
            continue
        try:
            res = sb.rpc(
                "mirror_tag_concept",
                {
                    "p_tenant_id": tenant_id,
                    "p_client_id": client_id,
                    "p_tag_id": tag_uuid,
                    "p_label": tag.label,
                    "p_description": tag.description,
                    "p_embedding": emb,
                    "p_embedding_model": _EMBED_MODEL,
                },
            ).execute()
            ret = _rpc_scalar(res.data) or {}
            mirrored += 1
            embedded += 1
            results.append(MirrorTaxonomyResultItem(
                tag_id=tag.tag_id,
                concept_id=str(ret.get("concept_id", "")),
                external_ref=str(ret.get("external_ref", tag_uuid)),
                embedded=True,
            ))
        except Exception as ex:
            failed += 1
            logger.exception("mirror_tag_concept failed tenant=%s tag=%s", tenant_id, tag.tag_id)
            results.append(MirrorTaxonomyResultItem(tag_id=tag.tag_id, error=str(ex)))

    logger.info("admin.mirror_taxonomy tenant=%s mirrored=%d failed=%d", tenant_id, mirrored, failed)
    return MirrorTaxonomyResponse(
        tenant_id=tenant_id, mirrored=mirrored, embedded=embedded, failed=failed, results=results,
    )


class ReembedRequest(BaseModel):
    tenant_id: str
    types: Optional[List[str]] = Field(
        default=None, description="Node types to re-embed, e.g. ['Observation','Concept']. None = all embeddable."
    )
    limit: int = Field(default=500, ge=1, le=2000, description="Max nodes to re-embed this sweep (chunk large backfills).")


class ReembedResponse(BaseModel):
    tenant_id: str
    scanned: int          # null-embedding nodes picked this sweep
    reembedded: int       # rows actually filled
    remaining: int        # null-embedding nodes still left after this sweep
    embedding_model: str


@router.post("/reembed", response_model=ReembedResponse)
def reembed(body: ReembedRequest, request: Request) -> ReembedResponse:
    """Re-embed nodes that have a null embedding (e.g. written during an embedder
    outage), making them resolvable via nearest/search again. Idempotent and
    chunkable via `limit`; re-run until `remaining` is 0."""
    from src.routers.ingest_router import _EMBED_MODEL, _embed_in_batches

    try:
        tenant_id = str(UUID(body.tenant_id))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid tenant_id: {e}")
    types = body.types or None

    sb = get_supabase()
    cands = (sb.rpc(
        "reembed_candidates",
        {"p_tenant_id": tenant_id, "p_types": types, "p_limit": body.limit},
    ).execute().data) or []
    scanned = len(cands)

    reembedded = 0
    if scanned:
        ids = [c["id"] for c in cands]
        texts = [(c.get("embed_text") or "") for c in cands]
        try:
            embeddings = _embed_in_batches(texts, tenant_id=None)  # system op — not billed to tenant
        except Exception as ex:
            logger.exception("reembed embedding failed tenant=%s", tenant_id)
            raise HTTPException(status_code=502, detail=f"Embedding provider error: {ex}")
        applied = sb.rpc(
            "reembed_apply_batch",
            {
                "p_tenant_id": tenant_id,
                "p_ids": ids,
                "p_embeddings": embeddings,
                "p_embedding_model": _EMBED_MODEL,
            },
        ).execute().data
        reembedded = int(_rpc_scalar(applied) or 0)

    q = sb.table("kg_nodes").select("id", count="exact").eq("tenant_id", tenant_id).is_("embedding", "null")
    if types:
        q = q.in_("type", types)
    remaining = q.execute().count or 0

    logger.info("admin.reembed tenant=%s scanned=%d reembedded=%d remaining=%d", tenant_id, scanned, reembedded, remaining)
    return ReembedResponse(
        tenant_id=tenant_id, scanned=scanned, reembedded=reembedded, remaining=remaining, embedding_model=_EMBED_MODEL,
    )


# ── Dedup / purge sweeps (formalize the manual cleanups; see migration 58) ──────


class RetireOrphansRequest(BaseModel):
    tenant_id: str


class RetireOrphansResponse(BaseModel):
    retired: int
    retired_labels: List[str] = Field(default_factory=list)


@router.post("/retire-orphans", response_model=RetireOrphansResponse)
def retire_orphans(body: RetireOrphansRequest, request: Request) -> RetireOrphansResponse:
    """Delete CANDIDATE concept nodes with 0 live linked observations. Governed
    nodes (external_ref set) are preserved — a governed theme with no observations
    is a codebook mirror awaiting resolution, not an orphan. Idempotent. Run after
    a re-emit re-points duplicate candidates' observations onto the governed node."""
    try:
        tenant_id = str(UUID(body.tenant_id))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid tenant_id: {e}")
    sb = get_supabase()
    try:
        res = sb.rpc("retire_orphan_concepts", {"p_tenant_id": tenant_id}).execute()
    except Exception as ex:
        logger.exception("retire_orphan_concepts failed tenant=%s", tenant_id)
        raise HTTPException(status_code=500, detail=str(ex))
    ret = _rpc_scalar(res.data) or {}
    labels = [l for l in (ret.get("retired_labels") or []) if l is not None]
    logger.info("admin.retire_orphans tenant=%s retired=%s", tenant_id, ret.get("retired", 0))
    return RetireOrphansResponse(retired=int(ret.get("retired", 0)), retired_labels=labels)


class PurgeStudyRequest(BaseModel):
    tenant_id: str
    study_id: str


class PurgeStudyResponse(BaseModel):
    observations_deleted: int
    concepts_deleted: int


@router.post("/purge-study", response_model=PurgeStudyResponse)
def purge_study(body: PurgeStudyRequest, request: Request) -> PurgeStudyResponse:
    """Delete all observations for a study, then retire the candidate concepts those
    observations left orphaned (governed preserved). Idempotent."""
    try:
        tenant_id = str(UUID(body.tenant_id))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid tenant_id: {e}")
    try:
        study_id = str(UUID(body.study_id))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid study_id: {e}")
    sb = get_supabase()
    try:
        res = sb.rpc("purge_study", {"p_tenant_id": tenant_id, "p_study_id": study_id}).execute()
    except Exception as ex:
        logger.exception("purge_study failed tenant=%s study=%s", tenant_id, study_id)
        raise HTTPException(status_code=500, detail=str(ex))
    ret = _rpc_scalar(res.data) or {}
    logger.info("admin.purge_study tenant=%s study=%s obs=%s concepts=%s",
                tenant_id, study_id, ret.get("observations_deleted", 0), ret.get("concepts_deleted", 0))
    return PurgeStudyResponse(
        observations_deleted=int(ret.get("observations_deleted", 0)),
        concepts_deleted=int(ret.get("concepts_deleted", 0)),
    )


@router.post("/maintenance/run", response_model=MaintenanceRunResponse)
def run_maintenance(req: MaintenanceRunRequest) -> MaintenanceRunResponse:
    """Run KG pruning and orphan cleanup for one client or every client in a tenant."""
    sb = get_supabase()
    memory_state = MemoryStateService(sb)

    try:
        tenant_id = UUID(req.tenant_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid tenant_id: {e}")

    client_ids = [req.client_id] if req.client_id else _discover_client_ids(sb, tenant_id)
    if not client_ids:
        tenant_memory_version = memory_state.bump(
            tenant_id=tenant_id,
            client_id=None,
            change_type="maintenance",
            metadata={"clients_processed": 0, "cleanup_orphans": req.cleanup_orphans},
        ) or 0
        return MaintenanceRunResponse(
            tenant_id=req.tenant_id,
            client_ids_processed=[],
            total_clients=0,
            results=[],
            totals={
                "edges_archived": 0,
                "nodes_archived": 0,
                "edge_evidence_deleted": 0,
                "node_evidence_deleted": 0,
                "orphaned_nodes_deleted": 0,
            },
            tenant_memory_version=tenant_memory_version,
            metadata={"message": "No clients found for tenant."},
        )

    results: list[MaintenanceClientResult] = []
    totals = {
        "edges_archived": 0,
        "nodes_archived": 0,
        "edge_evidence_deleted": 0,
        "node_evidence_deleted": 0,
        "orphaned_nodes_deleted": 0,
    }

    for client_id in client_ids:
        try:
            prune_res = sb.rpc(
                "prune_kg",
                {
                    "p_tenant_id": str(tenant_id),
                    "p_client_id": client_id,
                    "p_edge_stale_days": req.edge_stale_days,
                    "p_node_stale_days": req.node_stale_days,
                    "p_min_degree": req.min_degree,
                    "p_keep_edge_evidence": req.keep_edge_evidence,
                    "p_keep_node_evidence": req.keep_node_evidence,
                },
            ).execute()
            prune_data = prune_res.data or {}

            orphaned_nodes_deleted = 0
            if req.cleanup_orphans:
                cleanup_res = sb.rpc(
                    "cleanup_orphaned_kg_nodes",
                    {"p_tenant_id": str(tenant_id), "p_client_id": client_id},
                ).execute()
                cleanup_data = cleanup_res.data or {}
                orphaned_nodes_deleted = int(cleanup_data.get("nodes_deleted") or 0)

            client_memory_version = memory_state.bump(
                tenant_id=tenant_id,
                client_id=UUID(client_id),
                change_type="maintenance",
                metadata={
                    "cleanup_orphans": req.cleanup_orphans,
                    "edges_archived": int(prune_data.get("edges_archived") or 0),
                    "nodes_archived": int(prune_data.get("nodes_archived") or 0),
                    "orphaned_nodes_deleted": orphaned_nodes_deleted,
                },
            ) or 0

            result = MaintenanceClientResult(
                client_id=client_id,
                edges_archived=int(prune_data.get("edges_archived") or 0),
                nodes_archived=int(prune_data.get("nodes_archived") or 0),
                edge_evidence_deleted=int(prune_data.get("edge_evidence_deleted") or 0),
                node_evidence_deleted=int(prune_data.get("node_evidence_deleted") or 0),
                orphaned_nodes_deleted=orphaned_nodes_deleted,
                client_memory_version=client_memory_version,
            )
            totals["edges_archived"] += result.edges_archived
            totals["nodes_archived"] += result.nodes_archived
            totals["edge_evidence_deleted"] += result.edge_evidence_deleted
            totals["node_evidence_deleted"] += result.node_evidence_deleted
            totals["orphaned_nodes_deleted"] += result.orphaned_nodes_deleted
            logger.info(
                "admin.maintenance tenant=%s client=%s edges_archived=%d nodes_archived=%d edge_evidence_deleted=%d node_evidence_deleted=%d orphaned_nodes_deleted=%d client_memory_version=%d",
                tenant_id,
                client_id,
                result.edges_archived,
                result.nodes_archived,
                result.edge_evidence_deleted,
                result.node_evidence_deleted,
                result.orphaned_nodes_deleted,
                result.client_memory_version,
            )
        except Exception as e:
            logger.exception("Maintenance failed tenant=%s client=%s", tenant_id, client_id)
            result = MaintenanceClientResult(client_id=client_id, error=str(e))
        results.append(result)

    tenant_memory_version = memory_state.bump(
        tenant_id=tenant_id,
        client_id=None,
        change_type="maintenance",
        metadata={
            "clients_processed": len(client_ids),
            "cleanup_orphans": req.cleanup_orphans,
            **totals,
        },
    ) or 0

    return MaintenanceRunResponse(
        tenant_id=req.tenant_id,
        client_ids_processed=client_ids,
        total_clients=len(client_ids),
        results=results,
        totals=totals,
        tenant_memory_version=tenant_memory_version,
        metadata={
            "cleanup_orphans": req.cleanup_orphans,
            "edge_stale_days": req.edge_stale_days,
            "node_stale_days": req.node_stale_days,
            "min_degree": req.min_degree,
            "keep_edge_evidence": req.keep_edge_evidence,
            "keep_node_evidence": req.keep_node_evidence,
        },
    )
