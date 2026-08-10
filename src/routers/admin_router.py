"""
/admin router
-------------
Operational endpoints — health, stats, API keys, tenant plans, governance
backfill, dedup/purge sweeps, KG maintenance.

HTTP-only: the `_require_admin` scope guard (also imported by genomes_router)
and the dependency-ping `/admin/health` live here; all DB/business logic lives
in AdminService.
"""
from __future__ import annotations

import logging
import os
from typing import List
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request

from src.db.supabase_client import get_supabase
from src.services.admin_service import AdminService
from src.models.api.admin import (
    ApiKeyCreateRequest,
    ApiKeyCreateResponse,
    ApiKeyListItem,
    HealthResponse,
    MaintenanceRunRequest,
    MaintenanceRunResponse,
    MirrorTaxonomyRequest,
    MirrorTaxonomyResponse,
    PurgeStudyRequest,
    PurgeStudyResponse,
    ReembedRequest,
    ReembedResponse,
    RetireOrphansRequest,
    RetireOrphansResponse,
    StatsResponse,
    TenantPlanResponse,
    TenantPlanSetRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])


def _require_admin(request: Request) -> str:
    scopes = getattr(request.state, "scopes", []) or []
    if "admin" not in scopes:
        raise HTTPException(status_code=403, detail="admin scope required")
    return getattr(request.state, "tenant_id")


# ── API key management ────────────────────────────────────────────────────────


@router.post("/api-keys", response_model=ApiKeyCreateResponse)
def create_api_key(body: ApiKeyCreateRequest, request: Request) -> ApiKeyCreateResponse:
    """Issue a new API key for the caller's tenant. Raw key returned ONCE."""
    tenant_id = _require_admin(request)
    return AdminService(get_supabase()).create_api_key(tenant_id, body)


@router.get("/api-keys", response_model=List[ApiKeyListItem])
def list_api_keys(request: Request) -> List[ApiKeyListItem]:
    """List API keys for the caller's tenant (prefixes only, never raw)."""
    tenant_id = _require_admin(request)
    return AdminService(get_supabase()).list_api_keys(tenant_id)


@router.delete("/api-keys/{key_id}")
def revoke_api_key(key_id: str, request: Request) -> dict:
    """Revoke an API key."""
    tenant_id = _require_admin(request)
    return AdminService(get_supabase()).revoke_api_key(tenant_id, key_id)


# ── Tenant plan management (subscription tiers) ─────────────────────────────


@router.get("/plan", response_model=TenantPlanResponse)
def get_my_plan(request: Request) -> TenantPlanResponse:
    """Return the authenticated tenant's plan + current usage."""
    tenant_id = getattr(request.state, "tenant_id", None)
    if not tenant_id:
        raise HTTPException(status_code=401, detail="authenticated tenant required")
    return AdminService(get_supabase()).get_my_plan(str(tenant_id))


@router.put("/plan", response_model=TenantPlanResponse)
def set_my_plan(body: TenantPlanSetRequest, request: Request) -> TenantPlanResponse:
    """Update the authenticated tenant's plan. Admin scope required."""
    tenant_id = _require_admin(request)
    return AdminService(get_supabase()).set_my_plan(tenant_id, body)


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

    # Reranker is OPTIONAL — its absence degrades retrieval quality but is not an
    # outage, so it is surfaced here (not folded into `status`).
    try:
        from src.services.reranker_service import get_reranker
        reranker_ok = get_reranker().is_available()
    except Exception:
        reranker_ok = False
    if not reranker_ok:
        detail = (detail or "") + " Reranker disabled (no COHERE_API_KEY)."

    overall = "ok" if (sb_ok and openai_ok) else "degraded"

    return HealthResponse(
        status=overall,
        supabase=sb_ok,
        openai=openai_ok,
        reranker=reranker_ok,
        detail=detail,
    )


@router.get("/stats", response_model=StatsResponse)
def stats(
    tenant_id: UUID = Query(...),
    client_id: UUID = Query(...),
) -> StatsResponse:
    """Document, chunk, KG node, and KG edge counts for a tenant+client."""
    return AdminService(get_supabase()).stats(tenant_id, client_id)


# ── Governance backfill: mirror seeded taxonomy + re-embed null-embedding rows ──


@router.post("/mirror-taxonomy", response_model=MirrorTaxonomyResponse)
def mirror_taxonomy(body: MirrorTaxonomyRequest, request: Request) -> MirrorTaxonomyResponse:
    """Batch-mirror seeded taxonomy tags into the graph as governed Concept nodes,
    each embedded on label+description. Idempotent per tag."""
    return AdminService(get_supabase()).mirror_taxonomy(body)


@router.post("/reembed", response_model=ReembedResponse)
def reembed(body: ReembedRequest, request: Request) -> ReembedResponse:
    """Re-embed nodes that have a null embedding. Idempotent and chunkable via
    `limit`; re-run until `remaining` is 0."""
    return AdminService(get_supabase()).reembed(body)


# ── Dedup / purge sweeps (formalize the manual cleanups; see migration 58) ──────


@router.post("/retire-orphans", response_model=RetireOrphansResponse)
def retire_orphans(body: RetireOrphansRequest, request: Request) -> RetireOrphansResponse:
    """Delete CANDIDATE concept nodes with 0 live linked observations. Governed
    nodes (external_ref set) are preserved. Idempotent."""
    return AdminService(get_supabase()).retire_orphans(body)


@router.post("/purge-study", response_model=PurgeStudyResponse)
def purge_study(body: PurgeStudyRequest, request: Request) -> PurgeStudyResponse:
    """Delete all observations for a study, then retire the candidate concepts those
    observations left orphaned (governed preserved). Idempotent."""
    return AdminService(get_supabase()).purge_study(body)


@router.post("/maintenance/run", response_model=MaintenanceRunResponse)
def run_maintenance(req: MaintenanceRunRequest) -> MaintenanceRunResponse:
    """Run KG pruning and orphan cleanup for one client or every client in a tenant."""
    return AdminService(get_supabase()).run_maintenance(req)
