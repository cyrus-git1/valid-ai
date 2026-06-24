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
    ConceptCandidate,
    ConceptCreateRequest,
    ConceptCreateResponse,
    ConceptMergeRequest,
    ConceptMergeResponse,
    ConceptNearestRequest,
    ConceptNearestResponse,
)
from src.models.api.observations import (
    ObservationRecord,
    ObservationsByConceptResponse,
    ObservationUpsertRequest,
    ObservationUpsertResponse,
)
from src.routers.ingest_router import _EMBED_MODEL, _embed_in_batches
from src.supabase.supabase_client import get_supabase

logger = get_logger(__name__)

observations_router = APIRouter(prefix="/observations", tags=["observations"])
concepts_router = APIRouter(prefix="/concepts", tags=["concepts"])


# ── Tenant helpers (same pattern as entities_router) ────────────────────────


def _require_tenant(request: Request) -> str:
    tenant_id = getattr(request.state, "tenant_id", None)
    if not tenant_id:
        raise HTTPException(status_code=401, detail="authenticated tenant required")
    return str(tenant_id)


def _check_tenant_match(request: Request, body_tenant: UUID) -> str:
    auth_tenant = _require_tenant(request)
    if str(body_tenant) != auth_tenant:
        raise HTTPException(status_code=403, detail="Tenant mismatch")
    return auth_tenant


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
        except Exception as ex:
            logger.warning("concept embedding failed: %s", ex)
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
