"""
/data router
------------
Internal data endpoints for the agent service.

This router exposes only generic document metadata helpers and durable
context-summary reads/writes. Transcript- and survey-specific artifacts
should be serialized by the agent and ingested through the generic
processed ingest endpoints instead.

HTTP-only: query params and PII-reveal resolution (`caller_can_reveal`) live
here; Supabase queries, RPCs, redaction, memory-state, and audit live in
DataService.
"""
from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Query, Request

from src.models.api.data import (
    ContextSummaryReadResponse,
    DocumentTitlesRequest,
    DocumentTitlesResponse,
    KgEntitiesResponse,
    PreviewUrlResponse,
    SummaryListResponse,
    SurveyOutputsResponse,
    _BulkDeleteRequest,
    _DocumentPatchRequest,
)
from src.services.data_service import DataService, _PREVIEW_URL_TTL
from src.services.redaction import caller_can_reveal
from src.db.supabase_client import get_supabase

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/data", tags=["data"])


@router.get("/documents")
def list_documents(
    request: Request,
    tenant_id: UUID = Query(...),
    client_id: UUID = Query(...),
    include_summaries: bool = Query(
        False,
        description="If true, include ContextSummary/DocumentSummary/TopicSummary "
                    "rows in the listing. Defaults to false — agents listing source "
                    "documents almost never want summary rows mixed in.",
    ),
):
    """
    Return source documents for a tenant+client with all associated chunks.

    Summary documents (ContextSummary, DocumentSummary, TopicSummary) are
    EXCLUDED by default — use the dedicated /data/summaries endpoints for
    those. Pass include_summaries=true to opt in.

    Chunk content is PII-redacted (via chunks.pii_annotations) unless the
    caller's API key has the `pii:reveal` scope.
    """
    can_reveal = caller_can_reveal(request)
    key_id = getattr(getattr(request, "state", None), "key_id", None)
    return DataService(get_supabase()).list_documents(
        tenant_id, client_id, include_summaries, can_reveal, key_id,
    )


@router.patch("/documents/{document_id}")
def patch_document(
    document_id: str,
    body: _DocumentPatchRequest,
    request: Request,
    tenant_id: UUID = Query(...),
    client_id: UUID = Query(...),
):
    """
    Update document flags (status, is_pinned, is_canonical).
    Bumps memory state on change.
    """
    return DataService(get_supabase()).patch_document(document_id, body, tenant_id, client_id, request)


@router.post("/documents/delete")
def delete_documents(
    body: _BulkDeleteRequest,
    request: Request,
    tenant_id: UUID = Query(...),
    client_id: UUID = Query(...),
):
    """
    Delete documents by tenant_id + list of document_ids.
    Bumps memory state and cleans up orphaned KG nodes.
    Raw data endpoint for the agent service.
    """
    return DataService(get_supabase()).delete_documents(body, tenant_id, client_id, request)


@router.get("/documents/{document_id}/preview-url", response_model=PreviewUrlResponse)
def get_document_preview_url(
    document_id: str,
    request: Request,
    tenant_id: UUID = Query(...),
    client_id: UUID = Query(...),
    expires_in: int = Query(_PREVIEW_URL_TTL, ge=30, le=3600),
) -> PreviewUrlResponse:
    """
    Mint a short-lived signed URL for the original document file in storage, so
    the frontend can fetch + render (docx/pptx/xlsx/pdf) directly — bytes never
    pass through this service.

    Tenant-scoped: only signs for a document that exists under the given
    (tenant_id, client_id).
    """
    return DataService(get_supabase()).get_document_preview_url(document_id, tenant_id, client_id, expires_in)


@router.get("/survey-outputs", response_model=SurveyOutputsResponse)
def get_survey_outputs(
    request: Request,
    tenant_id: UUID = Query(...),
    client_id: Optional[UUID] = Query(None),
    study_id: Optional[UUID] = Query(None),
    output_type: Optional[str] = Query(None, description="'survey' | 'recommendation' | 'follow_up'"),
) -> SurveyOutputsResponse:
    """
    Read generated survey outputs (their `questions` array) for question alignment.

    BEST-EFFORT TRANSIENT CACHE — NOT the durable source of record. Rows expire
    after 7 days, so studies older than that return []. The durable survey
    definitions live in the app survey table; backfill / re-govern MUST pass
    `survey_outputs` INLINE and must never rely on this endpoint.
    """
    return DataService(get_supabase()).get_survey_outputs(tenant_id, client_id, study_id, output_type)


@router.get("/kg-entities", response_model=KgEntitiesResponse)
def get_kg_entities(
    request: Request,
    tenant_id: UUID = Query(...),
    client_id: UUID = Query(...),
    limit: int = Query(30, ge=1, le=200),
) -> KgEntitiesResponse:
    """Top active KG entities (name/type/description) for a tenant+client — powers
    the agent's per-tenant business glossary. Best-effort read; the agent degrades
    to no glossary on any failure."""
    return DataService(get_supabase()).get_kg_entities(tenant_id, client_id, limit)


@router.post("/document-titles", response_model=DocumentTitlesResponse)
def get_document_titles(req: DocumentTitlesRequest) -> DocumentTitlesResponse:
    """Resolve document IDs to titles."""
    return DataService(get_supabase()).get_document_titles(req)


@router.get("/context/summary/get", response_model=ContextSummaryReadResponse)
def get_context_summary(
    request: Request,
    tenant_id: UUID = Query(...),
    client_id: UUID = Query(...),
) -> ContextSummaryReadResponse:
    """Fetch the canonical ContextSummary document + its chunk content.

    Summary content is PII-redacted unless caller has `pii:reveal` scope.
    """
    return DataService(get_supabase()).get_context_summary(
        tenant_id, client_id, caller_can_reveal(request),
    )


@router.get("/summaries", response_model=SummaryListResponse)
def list_summaries(
    tenant_id: UUID = Query(...),
    client_id: UUID = Query(...),
    source_type: Optional[str] = Query(None, description="ContextSummary | DocumentSummary | TopicSummary"),
) -> SummaryListResponse:
    """List canonical summaries for a (tenant, client), optionally filtered by type."""
    return DataService(get_supabase()).list_summaries(tenant_id, client_id, source_type)


@router.get("/summaries/document/{document_id}", response_model=ContextSummaryReadResponse)
def get_document_summary(
    document_id: str,
    request: Request,
    tenant_id: UUID = Query(...),
    client_id: UUID = Query(...),
) -> ContextSummaryReadResponse:
    """Fetch the canonical DocumentSummary for a given source document_id."""
    return DataService(get_supabase()).get_document_summary(
        tenant_id, client_id, document_id, caller_can_reveal(request),
    )


@router.get("/summaries/topic", response_model=ContextSummaryReadResponse)
def get_topic_summary(
    request: Request,
    tenant_id: UUID = Query(...),
    client_id: UUID = Query(...),
    topic: str = Query(..., min_length=1),
) -> ContextSummaryReadResponse:
    """Fetch the canonical TopicSummary for a given topic string."""
    return DataService(get_supabase()).get_topic_summary(
        tenant_id, client_id, topic, caller_can_reveal(request),
    )
