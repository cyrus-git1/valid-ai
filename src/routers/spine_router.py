"""
/observations + /concepts + /app-entities router — the agent-layer memory spine
-------------------------------------------------------------------------------

  Observations:
    POST /observations/upsert      (idempotent by observation_id)
    GET  /observations/by-concept  ({concept_id, study_ids[], modality, segment})
    POST /observations/by-ids
    POST /observations/rollup

  Concepts (per-org entity spine):
    POST /concepts/nearest, /create, /merge, /mirror-tag, /link-tag, /graduate,
         /relations/compute
    GET  /concepts/by-study, /relations

  App entities:
    POST /app-entities/upsert, /nearest

Everything is tenant-scoped via AuthMiddleware + an explicit body/auth match.
Observation/Concept rows live in kg_nodes; links live in kg_edges. See migration 43.

HTTP-only: tenant resolution (_check_tenant_match) and query params live here;
embedding orchestration, Supabase RPCs, and response shaping live in SpineService.
"""
from __future__ import annotations

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request

from src.logging_config import get_logger
from src.models.api.concepts import (
    ConceptCreateRequest,
    ConceptCreateResponse,
    ConceptMergeRequest,
    ConceptMergeResponse,
    ConceptNearestRequest,
    ConceptNearestResponse,
    ConceptRelation,
    ConceptRelationsComputeRequest,
    ConceptRelationsComputeResponse,
    ConceptsByStudyResponse,
    GraduateRequest,
    GraduateResponse,
    LinkTagRequest,
    LinkTagResponse,
    MirrorTagRequest,
    MirrorTagResponse,
)
from src.models.api.app_entities import (
    AppEntityNearestRequest,
    AppEntityNearestResponse,
    AppEntityUpsertRequest,
    AppEntityUpsertResponse,
)
from src.models.api.canvas_blocks import (
    CanvasBlockUpsertRequest,
    CanvasBlockUpsertResponse,
    CanvasByScopeRequest,
    CanvasByScopeResponse,
)
from src.models.api.hypotheses import (
    HypothesesByScopeRequest,
    HypothesesByScopeResponse,
    HypothesisUpsertRequest,
    HypothesisUpsertResponse,
)
from src.models.api.canvas_events import (
    CanvasEventsRequest,
    CanvasEventsResponse,
)
from src.models.api.observations import (
    ObservationsByConceptResponse,
    ObservationsByIdsRequest,
    ObservationsRollupRequest,
    ObservationsRollupResponse,
    ObservationUpsertRequest,
    ObservationUpsertResponse,
)
from src.services.spine_service import SpineService
from src.db.supabase_client import get_supabase

logger = get_logger(__name__)

observations_router = APIRouter(prefix="/observations", tags=["observations"])
concepts_router = APIRouter(prefix="/concepts", tags=["concepts"])
app_entities_router = APIRouter(prefix="/app-entities", tags=["app-entities"])
canvas_router = APIRouter(prefix="/canvas", tags=["canvas"])
hypotheses_router = APIRouter(prefix="/hypotheses", tags=["hypotheses"])


# ── Tenant helpers (same pattern as entities_router) ────────────────────────


def _require_tenant(request: Request) -> Optional[str]:
    """The authenticated tenant from the auth context, or None when auth is off."""
    tenant_id = getattr(request.state, "tenant_id", None)
    return str(tenant_id) if tenant_id else None


def _check_tenant_match(request: Request, body_tenant: UUID) -> str:
    """Resolve the tenant for a spine request.

    - Auth ON: the key is authoritative — the body tenant_id must match it, else 403.
    - Auth OFF: fall back to the body tenant_id, exactly like the other body-tenant
      endpoints. Tightens automatically the moment auth is enabled.
    """
    auth_tenant = _require_tenant(request)
    if auth_tenant:
        if body_tenant is not None and str(body_tenant) != auth_tenant:
            raise HTTPException(status_code=403, detail="Tenant mismatch")
        return auth_tenant
    if body_tenant is None:
        raise HTTPException(status_code=400, detail="tenant_id required")
    return str(body_tenant)


# ════════════════════════════════════════════════════════════════════════════
# Observations
# ════════════════════════════════════════════════════════════════════════════


@observations_router.post("/upsert", response_model=ObservationUpsertResponse)
def upsert_observation(body: ObservationUpsertRequest, request: Request) -> ObservationUpsertResponse:
    """Upsert one observation. Idempotent on observation_id."""
    tenant_id = _check_tenant_match(request, body.tenant_id)
    return SpineService(get_supabase()).upsert_observation(tenant_id, body)


@observations_router.get("/by-concept", response_model=ObservationsByConceptResponse)
def observations_by_concept(
    request: Request,
    tenant_id: UUID = Query(...),
    concept_id: UUID = Query(...),
    study_ids: Optional[List[UUID]] = Query(None),
    modality: Optional[str] = Query(None),
    persona: Optional[str] = Query(None),
    variant_key: Optional[str] = Query(None),
) -> ObservationsByConceptResponse:
    """Scoped fetch backing the heatmap / drill-down. Tenant-scoped on every hop."""
    auth_tenant = _check_tenant_match(request, tenant_id)
    return SpineService(get_supabase()).observations_by_concept(
        auth_tenant, concept_id, study_ids, modality, persona, variant_key,
    )


@observations_router.post("/by-ids")
def observations_by_ids(body: ObservationsByIdsRequest, request: Request) -> dict:
    """Resolve observations by observation_id → map keyed by observation_id."""
    tenant_id = _check_tenant_match(request, body.tenant_id)
    return SpineService(get_supabase()).observations_by_ids(tenant_id, body)


@observations_router.post("/rollup", response_model=ObservationsRollupResponse)
def observations_rollup(body: ObservationsRollupRequest, request: Request) -> ObservationsRollupResponse:
    """The scope cube + top-N evidence per concept, in one call."""
    tenant_id = _check_tenant_match(request, body.tenant_id)
    return SpineService(get_supabase()).observations_rollup(tenant_id, body)


# ════════════════════════════════════════════════════════════════════════════
# Concepts
# ════════════════════════════════════════════════════════════════════════════


@concepts_router.post("/nearest", response_model=ConceptNearestResponse)
def concepts_nearest(body: ConceptNearestRequest, request: Request) -> ConceptNearestResponse:
    """Top-k candidate concepts in the tenant by cosine similarity + lexical hints."""
    tenant_id = _check_tenant_match(request, body.tenant_id)
    return SpineService(get_supabase()).concepts_nearest(tenant_id, body)


@concepts_router.post("/create", response_model=ConceptCreateResponse, status_code=201)
def concepts_create(body: ConceptCreateRequest, request: Request) -> ConceptCreateResponse:
    """Create (or idempotently re-affirm) a per-org concept. Mints canonical_id."""
    tenant_id = _check_tenant_match(request, body.tenant_id)
    return SpineService(get_supabase()).concepts_create(tenant_id, body)


@concepts_router.post("/merge", response_model=ConceptMergeResponse)
def concepts_merge(body: ConceptMergeRequest, request: Request) -> ConceptMergeResponse:
    """Fold source concept into survivor — union alias_set, rewire edges. Idempotent."""
    tenant_id = _check_tenant_match(request, body.tenant_id)
    return SpineService(get_supabase()).concepts_merge(tenant_id, body)


@concepts_router.get("/by-study", response_model=ConceptsByStudyResponse)
def concepts_by_study(
    request: Request,
    tenant_id: UUID = Query(...),
    study_ids: Optional[List[UUID]] = Query(None),
    client_id: Optional[UUID] = Query(None),
) -> ConceptsByStudyResponse:
    """Distinct concepts that have >=1 in-scope observation. Backs the synthesis board."""
    auth_tenant = _check_tenant_match(request, tenant_id)
    return SpineService(get_supabase()).concepts_by_study(auth_tenant, study_ids, client_id)


@concepts_router.post("/mirror-tag", response_model=MirrorTagResponse)
def mirror_tag(body: MirrorTagRequest, request: Request) -> MirrorTagResponse:
    """Mirror a transcript_tag into the graph as a governed Concept node. Idempotent."""
    tenant_id = _check_tenant_match(request, body.tenant_id)
    return SpineService(get_supabase()).mirror_tag(tenant_id, body)


@concepts_router.post("/link-tag", response_model=LinkTagResponse)
def link_tag(body: LinkTagRequest, request: Request) -> LinkTagResponse:
    """Stamp external_ref on an existing candidate concept (graduation in place)."""
    tenant_id = _check_tenant_match(request, body.tenant_id)
    return SpineService(get_supabase()).link_tag(tenant_id, body)


@concepts_router.post("/graduate", response_model=GraduateResponse)
def graduate(body: GraduateRequest, request: Request) -> GraduateResponse:
    """Graduate a candidate to governed in ONE server-side call."""
    tenant_id = _check_tenant_match(request, body.tenant_id)
    return SpineService(get_supabase()).graduate(tenant_id, body)


@concepts_router.post("/relations/compute", response_model=ConceptRelationsComputeResponse)
def compute_concept_relations(body: ConceptRelationsComputeRequest, request: Request) -> ConceptRelationsComputeResponse:
    """Batch pass: write signed supports/contradicts edges between co-occurring concepts."""
    tenant_id = _check_tenant_match(request, body.tenant_id)
    return SpineService(get_supabase()).compute_concept_relations(tenant_id, body)


@concepts_router.get("/relations", response_model=List[ConceptRelation])
def concept_relations(
    request: Request,
    tenant_id: UUID = Query(...),
    study_ids: Optional[List[UUID]] = Query(None),
    rel_types: Optional[List[str]] = Query(None),
) -> List[ConceptRelation]:
    """Read signed concept↔concept relations (cross-concept Tensions), confidence-weighted."""
    auth_tenant = _check_tenant_match(request, tenant_id)
    return SpineService(get_supabase()).concept_relations(auth_tenant, study_ids, rel_types)


# ════════════════════════════════════════════════════════════════════════════
# App entities — make app objects (insights, quotes, sessions) recallable by id
# ════════════════════════════════════════════════════════════════════════════


@app_entities_router.post("/upsert", response_model=AppEntityUpsertResponse)
def upsert_app_entity(body: AppEntityUpsertRequest, request: Request) -> AppEntityUpsertResponse:
    """Upsert one app object as an AppEntity kg_node, keyed by external_id. Idempotent."""
    tenant_id = _check_tenant_match(request, body.tenant_id)
    return SpineService(get_supabase()).upsert_app_entity(tenant_id, body)


@app_entities_router.post("/nearest", response_model=AppEntityNearestResponse)
def nearest_app_entities(body: AppEntityNearestRequest, request: Request) -> AppEntityNearestResponse:
    """Semantic recall of app objects by id, scoped to study_ids (+ optional kinds)."""
    tenant_id = _check_tenant_match(request, body.tenant_id)
    return SpineService(get_supabase()).nearest_app_entities(tenant_id, body)


# ════════════════════════════════════════════════════════════════════════════
# Canvas — persist the grounding-canvas blocks as CanvasBlock kg_nodes
# ════════════════════════════════════════════════════════════════════════════


@canvas_router.post("/upsert", response_model=CanvasBlockUpsertResponse)
def upsert_canvas_block(body: CanvasBlockUpsertRequest, request: Request) -> CanvasBlockUpsertResponse:
    """Upsert one canvas block (org or study scope), keyed by scope+block_key.
    A pinned human statement is never overwritten by an agent upsert."""
    tenant_id = _check_tenant_match(request, body.tenant_id)
    return SpineService(get_supabase()).upsert_canvas_block(tenant_id, body)


@canvas_router.post("/by-scope", response_model=CanvasByScopeResponse)
def canvas_by_scope(body: CanvasByScopeRequest, request: Request) -> CanvasByScopeResponse:
    """Fetch all canvas blocks for a scope (study overlay, or org-level when study_id omitted)."""
    tenant_id = _check_tenant_match(request, body.tenant_id)
    return SpineService(get_supabase()).canvas_by_scope(tenant_id, body)


@canvas_router.post("/events", response_model=CanvasEventsResponse)
def canvas_events(body: CanvasEventsRequest, request: Request) -> CanvasEventsResponse:
    """The canvas change log — status/confidence transitions for a scope."""
    tenant_id = _check_tenant_match(request, body.tenant_id)
    return SpineService(get_supabase()).canvas_events_by_scope(tenant_id, body)


# ════════════════════════════════════════════════════════════════════════════
# Hypotheses — store testable claims as Hypothesis kg_nodes (linker sets links)
# ════════════════════════════════════════════════════════════════════════════


@hypotheses_router.post("/upsert", response_model=HypothesisUpsertResponse)
def upsert_hypothesis(body: HypothesisUpsertRequest, request: Request) -> HypothesisUpsertResponse:
    """Upsert one hypothesis (org or study scope), keyed by external_id. Editing
    the text re-embeds it; the linker (agent-side) then re-derives theme_ids +
    evidence_refs."""
    tenant_id = _check_tenant_match(request, body.tenant_id)
    return SpineService(get_supabase()).upsert_hypothesis(tenant_id, body)


@hypotheses_router.post("/by-scope", response_model=HypothesesByScopeResponse)
def hypotheses_by_scope(body: HypothesesByScopeRequest, request: Request) -> HypothesesByScopeResponse:
    """Fetch all hypotheses for a scope (study, or org-level when study_id omitted)."""
    tenant_id = _check_tenant_match(request, body.tenant_id)
    return SpineService(get_supabase()).hypotheses_by_scope(tenant_id, body)
