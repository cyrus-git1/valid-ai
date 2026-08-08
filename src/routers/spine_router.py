"""
/observations + /concepts router — the agent-layer memory spine (first slice)
-----------------------------------------------------------------------------

  Observations:
    POST /observations/upsert      (idempotent by observation_id)
    GET  /observations/by-concept  ({concept_id, study_ids[], modality, segment})

  Concepts (per-org entity spine):
    POST /concepts/nearest         (embedding + hints → candidates, tenant-scoped)
    POST /concepts/create
    POST /concepts/merge

Everything is tenant-scoped via AuthMiddleware + an explicit body/auth match.
Observation/Concept rows live in kg_nodes (type 'Observation' / 'Concept'); the
observation→concept link is a kg_edges row (rel_type 'about_concept') and the
observation→evidence link is a kg_node_evidence row. See migration 43.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request

from src.logging_config import get_logger
from src.models.api.concepts import (
    ConceptByStudyItem,
    ConceptCandidate,
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
    AppEntityMatch,
    AppEntityNearestRequest,
    AppEntityNearestResponse,
    AppEntityUpsertRequest,
    AppEntityUpsertResponse,
)
from src.models.api.observations import (
    ObservationRecord,
    ObservationsByConceptResponse,
    ObservationsByIdsRequest,
    ObservationsRollupRequest,
    ObservationsRollupResponse,
    ObservationUpsertRequest,
    ObservationUpsertResponse,
)
from src.routers.ingest_router import _EMBED_MODEL, _embed_in_batches
from src.supabase.supabase_client import get_supabase

logger = get_logger(__name__)

observations_router = APIRouter(prefix="/observations", tags=["observations"])
concepts_router = APIRouter(prefix="/concepts", tags=["concepts"])
app_entities_router = APIRouter(prefix="/app-entities", tags=["app-entities"])


# ── Tenant helpers (same pattern as entities_router) ────────────────────────


def _require_tenant(request: Request) -> Optional[str]:
    """The authenticated tenant from the auth context, or None when auth is off."""
    tenant_id = getattr(request.state, "tenant_id", None)
    return str(tenant_id) if tenant_id else None


def _check_tenant_match(request: Request, body_tenant: UUID) -> str:
    """Resolve the tenant for a spine request.

    - Auth ON (AuthMiddleware set request.state.tenant_id): the key is
      authoritative — the body tenant_id must match it, else 403.
    - Auth OFF (no auth context): fall back to the body tenant_id, exactly like
      the other body-tenant endpoints (/search/graph, /ingest/*). This keeps the
      spine endpoints callable when AUTH_ENABLED=false and tightens automatically
      the moment auth is enabled.
    """
    auth_tenant = _require_tenant(request)
    if auth_tenant:
        if body_tenant is not None and str(body_tenant) != auth_tenant:
            raise HTTPException(status_code=403, detail="Tenant mismatch")
        return auth_tenant
    if body_tenant is None:
        raise HTTPException(status_code=400, detail="tenant_id required")
    return str(body_tenant)


def _rpc_scalar(res_data: Any) -> Dict[str, Any]:
    """Unwrap a jsonb-returning RPC (dict, or single-element list)."""
    ret = res_data or {}
    if isinstance(ret, list):
        ret = ret[0] if ret else {}
    return ret if isinstance(ret, dict) else {}


# ════════════════════════════════════════════════════════════════════════════
# Observations
# ════════════════════════════════════════════════════════════════════════════


@observations_router.post("/upsert", response_model=ObservationUpsertResponse)
def upsert_observation(
    body: ObservationUpsertRequest,
    request: Request,
) -> ObservationUpsertResponse:
    """
    Upsert one observation. Idempotent on observation_id: re-sending the same id
    updates in place, never duplicates. nl_text is embedded into the shared
    vector space; value{number,unit} is stored verbatim; source provenance and
    (when supplied) the evidence + concept links are persisted atomically.
    """
    tenant_id = _check_tenant_match(request, body.tenant_id)
    client_id = str(body.client_id) if body.client_id else None
    sb = get_supabase()

    # Embed nl_text once (re-embed on every upsert is acceptable for the slice;
    # the RPC stores whatever we pass).
    try:
        embedding = _embed_in_batches([body.nl_text], tenant_id=tenant_id)[0]
    except Exception as ex:
        logger.warning("observation embedding failed (storing null embedding): %s", ex)
        embedding = None

    # Assemble the verbatim properties payload. value is opaque.
    properties: Dict[str, Any] = {"value": body.value}
    for field in ("modality", "signal_type", "direction", "occurred_at", "confidence"):
        val = getattr(body, field)
        if val is not None:
            properties[field] = val
    if body.prevalence is not None:
        properties["prevalence"] = body.prevalence.model_dump(exclude_none=True)
    if body.reliability is not None:
        properties["reliability"] = body.reliability.model_dump(exclude_none=True)
    if body.segment is not None:
        properties["segment"] = body.segment.model_dump(exclude_none=True)
    if body.source is not None:
        properties["source"] = body.source.model_dump(exclude_none=True)

    try:
        res = sb.rpc(
            "upsert_observation",
            {
                "p_tenant_id":         tenant_id,
                "p_client_id":         client_id,
                "p_observation_id":    body.observation_id,
                "p_nl_text":           body.nl_text,
                "p_properties":        properties,
                "p_embedding":         embedding,
                "p_embedding_model":   _EMBED_MODEL,
                "p_study_id":          str(body.study_id) if body.study_id else None,
                "p_evidence_chunk_id": str(body.evidence_chunk_id) if body.evidence_chunk_id else None,
                "p_concept_id":        str(body.concept_id) if body.concept_id else None,
            },
        ).execute()
    except Exception as ex:
        logger.exception("observation upsert failed for %s", body.observation_id)
        raise HTTPException(status_code=500, detail=str(ex))

    ret = _rpc_scalar(res.data)
    return ObservationUpsertResponse(
        observation_id=ret.get("observation_id", body.observation_id),
        node_id=str(ret.get("node_id", "")),
        evidence_linked=bool(ret.get("evidence_linked", False)),
        concept_linked=bool(ret.get("concept_linked", False)),
    )


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
    """
    Scoped fetch backing the heatmap / drill-down. Returns every observation
    linked to the concept (within the study filter / segment), each with its
    verbatim value, prevalence, reliability, occurred_at, source provenance, and
    the chunk_ids it cites as evidence. Tenant-scoped on every hop.
    """
    auth_tenant = _check_tenant_match(request, tenant_id)
    sb = get_supabase()

    try:
        res = sb.rpc(
            "observations_by_concept",
            {
                "p_tenant_id":   auth_tenant,
                "p_concept_id":  str(concept_id),
                "p_study_ids":   [str(s) for s in study_ids] if study_ids else None,
                "p_modality":    modality,
                "p_persona":     persona,
                "p_variant_key": variant_key,
            },
        ).execute()
    except Exception as ex:
        logger.exception("observations_by_concept failed for concept %s", concept_id)
        raise HTTPException(status_code=500, detail=str(ex))

    rows = res.data or []
    if isinstance(rows, dict):
        rows = [rows]

    observations: List[ObservationRecord] = []
    for row in rows:
        observations.append(ObservationRecord(
            node_id=str(row.get("node_id", "")),
            observation_id=row.get("observation_id"),
            nl_text=row.get("nl_text"),
            value=row.get("value"),
            modality=row.get("modality"),
            signal_type=row.get("signal_type"),
            direction=row.get("direction"),
            prevalence=row.get("prevalence"),
            confidence=row.get("confidence"),
            reliability=row.get("reliability"),
            segment=row.get("segment"),
            occurred_at=row.get("occurred_at"),
            source=row.get("source"),
            study_id=str(row["study_id"]) if row.get("study_id") else None,
            evidence_chunk_ids=[str(c) for c in (row.get("evidence_chunk_ids") or [])],
        ))

    return ObservationsByConceptResponse(
        concept_id=str(concept_id),
        observations=observations,
    )


def _row_to_observation(row: Dict[str, Any]) -> ObservationRecord:
    return ObservationRecord(
        node_id=str(row.get("node_id", "")),
        observation_id=row.get("observation_id"),
        nl_text=row.get("nl_text"),
        value=row.get("value"),
        modality=row.get("modality"),
        signal_type=row.get("signal_type"),
        direction=row.get("direction"),
        prevalence=row.get("prevalence"),
        confidence=row.get("confidence"),
        reliability=row.get("reliability"),
        segment=row.get("segment"),
        occurred_at=row.get("occurred_at"),
        source=row.get("source"),
        study_id=str(row["study_id"]) if row.get("study_id") else None,
        evidence_chunk_ids=[str(c) for c in (row.get("evidence_chunk_ids") or [])],
        evidence=row.get("evidence"),
    )


@observations_router.post("/by-ids")
def observations_by_ids(
    body: ObservationsByIdsRequest,
    request: Request,
) -> Dict[str, ObservationRecord]:
    """Resolve observations by observation_id → map keyed by observation_id.

    Backs the agent's /spine/hydrate (stateless). Tenant + study scoped.
    """
    tenant_id = _check_tenant_match(request, body.tenant_id)
    sb = get_supabase()
    try:
        res = sb.rpc(
            "observations_by_ids",
            {
                "p_tenant_id":       tenant_id,
                "p_observation_ids": body.ids or [],
                "p_study_ids":       [str(s) for s in body.study_ids] if body.study_ids else None,
            },
        ).execute()
    except Exception as ex:
        logger.exception("observations_by_ids failed")
        raise HTTPException(status_code=500, detail=str(ex))

    rows = res.data or []
    if isinstance(rows, dict):
        rows = [rows]
    out: Dict[str, ObservationRecord] = {}
    for row in rows:
        rec = _row_to_observation(row)
        if rec.observation_id:
            out[rec.observation_id] = rec
    return out


@observations_router.post("/rollup", response_model=ObservationsRollupResponse)
def observations_rollup(
    body: ObservationsRollupRequest,
    request: Request,
) -> ObservationsRollupResponse:
    """The scope cube + top-N evidence per concept, in one call.

    Returns RAW direction counts + n_sum (agent applies the sign/threshold/
    divergence policy). Subsumes /concepts/by-study and kills the per-concept loop.
    """
    tenant_id = _check_tenant_match(request, body.tenant_id)
    sb = get_supabase()
    rng = body.range
    try:
        res = sb.rpc(
            "observations_rollup",
            {
                "p_tenant_id":     tenant_id,
                "p_study_ids":     [str(s) for s in body.study_ids] if body.study_ids else None,
                "p_range_from":    rng.from_ if rng else None,
                "p_range_to":      rng.to if rng else None,
                "p_top_evidence":  body.top_evidence,
            },
        ).execute()
    except Exception as ex:
        logger.exception("observations_rollup failed")
        raise HTTPException(status_code=500, detail=str(ex))

    ret = _rpc_scalar(res.data)
    return ObservationsRollupResponse(
        cube=ret.get("cube") or [],
        evidence=ret.get("evidence") or {},
    )


# ════════════════════════════════════════════════════════════════════════════
# Concepts
# ════════════════════════════════════════════════════════════════════════════


def _resolve_embedding(
    *, tenant_id: str, supplied: Optional[List[float]], text: Optional[str]
) -> Optional[List[float]]:
    if supplied:
        return supplied
    if text:
        try:
            return _embed_in_batches([text], tenant_id=tenant_id)[0]
        except HTTPException:
            raise
        except Exception as ex:
            # The caller DID supply text — a failure here is the embedding
            # PROVIDER being unavailable (OpenAI outage / exhausted quota), NOT a
            # malformed request. Surface it honestly as 502 with the real cause.
            # Previously this returned None, so the caller raised a misleading
            # 400 "provide query_text or embedding" — which disguised a billing
            # outage as a client contract error and buried the real reason.
            logger.error("concept embedding failed (provider): %s", ex, exc_info=True)
            raise HTTPException(
                status_code=502, detail=f"embedding provider unavailable: {ex}"
            )
    return None


@concepts_router.post("/nearest", response_model=ConceptNearestResponse)
def concepts_nearest(
    body: ConceptNearestRequest,
    request: Request,
) -> ConceptNearestResponse:
    """Top-k candidate concepts in the tenant by cosine similarity + lexical hints."""
    tenant_id = _check_tenant_match(request, body.tenant_id)
    sb = get_supabase()

    embedding = _resolve_embedding(tenant_id=tenant_id, supplied=body.embedding, text=body.query_text)
    if embedding is None:
        raise HTTPException(status_code=400, detail="provide query_text or embedding")

    try:
        res = sb.rpc(
            "nearest_concepts",
            {
                "p_tenant_id":       tenant_id,
                "p_embedding":       embedding,
                "p_hints":           body.hints or None,
                "p_top_k":           body.top_k,
                "p_embedding_model": _EMBED_MODEL,
            },
        ).execute()
    except Exception as ex:
        logger.exception("nearest_concepts failed")
        raise HTTPException(status_code=500, detail=str(ex))

    rows = res.data or []
    if isinstance(rows, dict):
        rows = [rows]

    candidates = [
        ConceptCandidate(
            id=str(row.get("id", "")),
            canonical_id=row.get("canonical_id"),
            canonical_label=row.get("canonical_label"),
            alias_set=row.get("alias_set"),
            merge_confidence=row.get("merge_confidence"),
            external_ref=row.get("external_ref"),
            similarity=row.get("similarity"),
            score=row.get("final_score"),
        )
        for row in rows
    ]
    return ConceptNearestResponse(candidates=candidates)


@concepts_router.post("/create", response_model=ConceptCreateResponse, status_code=201)
def concepts_create(
    body: ConceptCreateRequest,
    request: Request,
) -> ConceptCreateResponse:
    """Create (or idempotently re-affirm) a per-org concept. Mints canonical_id."""
    tenant_id = _check_tenant_match(request, body.tenant_id)
    client_id = str(body.client_id) if body.client_id else None
    sb = get_supabase()

    embedding = _resolve_embedding(
        tenant_id=tenant_id,
        supplied=body.embedding,
        text=body.embedding_text or body.canonical_label,
    )

    try:
        res = sb.rpc(
            "create_concept",
            {
                "p_tenant_id":        tenant_id,
                "p_client_id":        client_id,
                "p_canonical_label":  body.canonical_label,
                "p_aliases":          body.aliases or [],
                "p_embedding":        embedding,
                "p_embedding_model":  _EMBED_MODEL,
                "p_merge_confidence": body.merge_confidence,
                "p_canonical_id":     body.canonical_id,
            },
        ).execute()
    except Exception as ex:
        logger.exception("create_concept failed")
        raise HTTPException(status_code=500, detail=str(ex))

    ret = _rpc_scalar(res.data)
    return ConceptCreateResponse(
        concept_id=str(ret.get("concept_id", "")),
        canonical_id=str(ret.get("canonical_id", "")),
        node_key=str(ret.get("node_key", "")),
        created=bool(ret.get("created", False)),
        redirected=bool(ret.get("redirected", False)),
    )


@concepts_router.post("/merge", response_model=ConceptMergeResponse)
def concepts_merge(
    body: ConceptMergeRequest,
    request: Request,
) -> ConceptMergeResponse:
    """Fold source concept into survivor — union alias_set, rewire edges. Idempotent."""
    tenant_id = _check_tenant_match(request, body.tenant_id)
    sb = get_supabase()

    try:
        res = sb.rpc(
            "merge_concepts",
            {
                "p_tenant_id":            tenant_id,
                "p_surviving_concept_id": body.surviving_concept_id,
                "p_source_concept_id":    body.source_concept_id,
            },
        ).execute()
    except Exception as ex:
        logger.exception("merge_concepts failed")
        raise HTTPException(status_code=500, detail=str(ex))

    ret = _rpc_scalar(res.data)
    return ConceptMergeResponse(
        merged=bool(ret.get("merged", False)),
        rewired_count=int(ret.get("rewired_count", 0)),
        surviving_member_count=int(ret.get("surviving_member_count", 0)),
    )


@concepts_router.get("/by-study", response_model=ConceptsByStudyResponse)
def concepts_by_study(
    request: Request,
    tenant_id: UUID = Query(...),
    study_ids: Optional[List[UUID]] = Query(None),
    client_id: Optional[UUID] = Query(None),
) -> ConceptsByStudyResponse:
    """Distinct concepts that have >=1 in-scope observation. Backs the synthesis board."""
    auth_tenant = _check_tenant_match(request, tenant_id)
    sb = get_supabase()
    try:
        res = sb.rpc(
            "concepts_by_study",
            {
                "p_tenant_id": auth_tenant,
                "p_study_ids": [str(s) for s in study_ids] if study_ids else None,
                "p_client_id": str(client_id) if client_id else None,
            },
        ).execute()
    except Exception as ex:
        logger.exception("concepts_by_study failed")
        raise HTTPException(status_code=500, detail=str(ex))

    rows = res.data or []
    if isinstance(rows, dict):
        rows = [rows]
    concepts = [
        ConceptByStudyItem(
            concept_id=str(row.get("concept_id", "")),
            label=row.get("label"),
            external_ref=row.get("external_ref"),
        )
        for row in rows
    ]
    return ConceptsByStudyResponse(concepts=concepts)


@concepts_router.post("/mirror-tag", response_model=MirrorTagResponse)
def mirror_tag(
    body: MirrorTagRequest,
    request: Request,
) -> MirrorTagResponse:
    """Mirror a transcript_tag into the graph as a governed Concept node (id = tag_id,
    external_ref = tag_id). Idempotent; refresh label/description/embedding on edits."""
    tenant_id = _check_tenant_match(request, body.tenant_id)
    client_id = str(body.client_id) if body.client_id else None
    sb = get_supabase()
    text = body.embedding_text or " ".join(filter(None, [body.label, body.description]))
    embedding = _resolve_embedding(tenant_id=tenant_id, supplied=body.embedding, text=text)
    try:
        res = sb.rpc(
            "mirror_tag_concept",
            {
                "p_tenant_id":       tenant_id,
                "p_client_id":       client_id,
                "p_tag_id":          str(body.tag_id),
                "p_label":           body.label,
                "p_description":     body.description,
                "p_embedding":       embedding,
                "p_embedding_model": _EMBED_MODEL,
            },
        ).execute()
    except Exception as ex:
        logger.exception("mirror_tag_concept failed")
        raise HTTPException(status_code=500, detail=str(ex))
    ret = _rpc_scalar(res.data)
    return MirrorTagResponse(
        concept_id=str(ret.get("concept_id", "")),
        external_ref=str(ret.get("external_ref", "")),
        node_key=str(ret.get("node_key", "")),
    )


@concepts_router.post("/link-tag", response_model=LinkTagResponse)
def link_tag(
    body: LinkTagRequest,
    request: Request,
) -> LinkTagResponse:
    """Stamp external_ref on an existing candidate concept (graduation in place —
    preserves its accumulated observations)."""
    tenant_id = _check_tenant_match(request, body.tenant_id)
    sb = get_supabase()
    try:
        res = sb.rpc(
            "link_concept_tag",
            {
                "p_tenant_id":  tenant_id,
                "p_concept_id": str(body.concept_id),
                "p_tag_id":     str(body.tag_id),
            },
        ).execute()
    except Exception as ex:
        logger.exception("link_concept_tag failed")
        raise HTTPException(status_code=500, detail=str(ex))
    ret = _rpc_scalar(res.data)
    return LinkTagResponse(
        linked=bool(ret.get("linked", False)),
        concept_id=str(ret.get("concept_id", body.concept_id)),
        external_ref=str(ret.get("external_ref", "")),
    )


@concepts_router.post("/graduate", response_model=GraduateResponse)
def graduate(
    body: GraduateRequest,
    request: Request,
) -> GraduateResponse:
    """Graduate a candidate to governed in ONE server-side call: create-or-get the
    transcript_tags codebook row (idempotent on tenant+label), then stamp
    external_ref on the existing candidate node in place (preserves its accumulated
    observations). This is the whole graduation write — the agent no longer touches
    Supabase directly. No embedding (the candidate node already carries one)."""
    tenant_id = _check_tenant_match(request, body.tenant_id)
    client_id = str(body.client_id) if body.client_id else None
    sb = get_supabase()
    try:
        res = sb.rpc(
            "graduate_concept",
            {
                "p_tenant_id":    tenant_id,
                "p_concept_id":   str(body.concept_id),
                "p_label":        body.label,
                "p_description":  body.description,
                "p_client_id":    client_id,
                "p_cluster_id":   body.cluster_id,
                "p_evidence_ids": body.evidence_ids or [],
            },
        ).execute()
    except Exception as ex:
        logger.exception("graduate_concept failed")
        raise HTTPException(status_code=500, detail=str(ex))
    ret = _rpc_scalar(res.data) or {}
    return GraduateResponse(
        tag_id=str(ret.get("tag_id", "")),
        concept_id=str(ret.get("concept_id", body.concept_id)),
        external_ref=str(ret.get("external_ref", "")),
        created=bool(ret.get("created", False)),
    )


@concepts_router.post("/relations/compute", response_model=ConceptRelationsComputeResponse)
def compute_concept_relations(
    body: ConceptRelationsComputeRequest,
    request: Request,
) -> ConceptRelationsComputeResponse:
    """Batch pass: write signed supports/contradicts edges between co-occurring concepts."""
    tenant_id = _check_tenant_match(request, body.tenant_id)
    sb = get_supabase()
    try:
        res = sb.rpc(
            "compute_concept_relations",
            {
                "p_tenant_id":   tenant_id,
                "p_study_ids":   [str(s) for s in body.study_ids] if body.study_ids else None,
                "p_min_cooccur": body.min_cooccur,
            },
        ).execute()
    except Exception as ex:
        logger.exception("compute_concept_relations failed")
        raise HTTPException(status_code=500, detail=str(ex))
    ret = _rpc_scalar(res.data)
    return ConceptRelationsComputeResponse(relations_written=int(ret.get("relations_written", 0)))


@concepts_router.get("/relations", response_model=List[ConceptRelation])
def concept_relations(
    request: Request,
    tenant_id: UUID = Query(...),
    study_ids: Optional[List[UUID]] = Query(None),
    rel_types: Optional[List[str]] = Query(None),
) -> List[ConceptRelation]:
    """Read signed concept↔concept relations (cross-concept Tensions), confidence-weighted."""
    auth_tenant = _check_tenant_match(request, tenant_id)
    sb = get_supabase()
    try:
        res = sb.rpc(
            "concept_relations",
            {
                "p_tenant_id": auth_tenant,
                "p_study_ids": [str(s) for s in study_ids] if study_ids else None,
                "p_rel_types": rel_types or ["supports", "contradicts"],
            },
        ).execute()
    except Exception as ex:
        logger.exception("concept_relations failed")
        raise HTTPException(status_code=500, detail=str(ex))
    rows = res.data or []
    if isinstance(rows, dict):
        rows = [rows]
    return [
        ConceptRelation(
            src_concept_id=str(row.get("src_concept_id", "")),
            dst_concept_id=str(row.get("dst_concept_id", "")),
            rel_type=str(row.get("rel_type", "")),
            weight=row.get("weight"),
        )
        for row in rows
    ]


# ════════════════════════════════════════════════════════════════════════════
# App entities — make app objects (insights, quotes, sessions) recallable by id
# ════════════════════════════════════════════════════════════════════════════


@app_entities_router.post("/upsert", response_model=AppEntityUpsertResponse)
def upsert_app_entity(
    body: AppEntityUpsertRequest,
    request: Request,
) -> AppEntityUpsertResponse:
    """
    Upsert one app object as an AppEntity kg_node, keyed by external_id. Idempotent
    on (tenant, node_key, client). Re-embeds `text` only when it changed.
    status='inactive' soft-deletes (excluded from recall).
    """
    tenant_id = _check_tenant_match(request, body.tenant_id)
    client_id = str(body.client_id) if body.client_id else None
    sb = get_supabase()

    node_key = f"app:{body.kind}:{body.external_id}"

    # Re-embed only when the stored text changed (name holds the embedded text).
    unchanged = False
    try:
        q = (
            sb.table("kg_nodes")
            .select("name")
            .eq("tenant_id", tenant_id)
            .eq("node_key", node_key)
            .eq("type", "AppEntity")
        )
        q = q.eq("client_id", client_id) if client_id else q.is_("client_id", "null")
        rows = (q.limit(1).execute().data) or []
        unchanged = bool(rows) and rows[0].get("name") == body.text
    except Exception as ex:
        logger.debug("app-entity text-change check failed (will re-embed): %s", ex)

    embedding = None
    if not unchanged:
        try:
            embedding = _embed_in_batches([body.text], tenant_id=tenant_id)[0]
        except Exception as ex:
            logger.warning("app-entity embedding failed (storing null embedding): %s", ex)
            embedding = None

    try:
        res = sb.rpc(
            "upsert_app_entity",
            {
                "p_tenant_id":       tenant_id,
                "p_client_id":       client_id,
                "p_study_id":        str(body.study_id),
                "p_kind":            body.kind,
                "p_external_id":     body.external_id,
                "p_text":            body.text,
                "p_embedding":       embedding,
                "p_embedding_model": _EMBED_MODEL,
                "p_status":          body.status,
            },
        ).execute()
    except Exception as ex:
        logger.exception("upsert_app_entity failed for %s:%s", body.kind, body.external_id)
        raise HTTPException(status_code=500, detail=str(ex))

    ret = _rpc_scalar(res.data)
    return AppEntityUpsertResponse(
        node_id=str(ret.get("node_id", "")),
        external_id=ret.get("external_id", body.external_id),
        kind=ret.get("kind", body.kind),
        created=bool(ret.get("created", False)),
    )


@app_entities_router.post("/nearest", response_model=AppEntityNearestResponse)
def nearest_app_entities(
    body: AppEntityNearestRequest,
    request: Request,
) -> AppEntityNearestResponse:
    """Semantic recall of app objects by id, scoped to study_ids (+ optional kinds)."""
    tenant_id = _check_tenant_match(request, body.tenant_id)
    sb = get_supabase()

    embedding = _resolve_embedding(tenant_id=tenant_id, supplied=body.embedding, text=body.query_text)
    if embedding is None:
        raise HTTPException(status_code=400, detail="provide query_text or embedding")

    try:
        res = sb.rpc(
            "nearest_app_entities",
            {
                "p_tenant_id":       tenant_id,
                "p_embedding":       embedding,
                "p_study_ids":       [str(s) for s in body.study_ids] if body.study_ids else None,
                "p_kinds":           body.kinds or None,
                "p_top_k":           body.top_k,
                "p_embedding_model": _EMBED_MODEL,
            },
        ).execute()
    except Exception as ex:
        logger.exception("nearest_app_entities failed")
        raise HTTPException(status_code=500, detail=str(ex))

    rows = res.data or []
    if isinstance(rows, dict):
        rows = [rows]

    matches = [
        AppEntityMatch(
            external_id=str(row.get("external_id", "")),
            kind=str(row.get("kind", "")),
            study_id=str(row["study_id"]) if row.get("study_id") else None,
            similarity=float(row.get("similarity") or 0.0),
        )
        for row in rows
    ]
    return AppEntityNearestResponse(matches=matches)
