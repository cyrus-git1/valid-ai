"""
/data router
-------------
Data endpoints for the agent service to consume.

Provides read access to transcript chunks, document titles, survey outputs,
and context summaries, plus write access for persisting survey outputs.

These endpoints exist so the agent service (valid-agents) can access
data without direct Supabase access.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.supabase.supabase_client import get_supabase
from src.services.base_service import BaseAnalysisService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/data", tags=["data"])


# ── Transcript chunks ────────────────────────────────────────────────────────


class TranscriptChunksResponse(BaseModel):
    chunks: List[Dict[str, Any]]
    count: int


@router.get("/transcript-chunks", response_model=TranscriptChunksResponse)
def get_transcript_chunks(
    tenant_id: UUID = Query(...),
    client_id: UUID = Query(...),
    limit: int = Query(default=50, ge=1, le=200),
) -> TranscriptChunksResponse:
    """Fetch transcript chunks for a tenant+client."""
    sb = get_supabase()
    svc = BaseAnalysisService(sb)
    chunks = svc._get_transcript_chunks(tenant_id, client_id, limit=limit)
    return TranscriptChunksResponse(chunks=chunks, count=len(chunks))


# ── Transcript count ─────────────────────────────────────────────────────────


class TranscriptCountResponse(BaseModel):
    count: int


@router.get("/transcript-count", response_model=TranscriptCountResponse)
def get_transcript_count(
    tenant_id: UUID = Query(...),
    client_id: UUID = Query(...),
) -> TranscriptCountResponse:
    """Count transcript documents for a tenant+client."""
    sb = get_supabase()
    svc = BaseAnalysisService(sb)
    count = svc._count_transcripts(tenant_id, client_id)
    return TranscriptCountResponse(count=count)


# ── Document titles ──────────────────────────────────────────────────────────


class DocumentTitlesRequest(BaseModel):
    document_ids: List[str]


class DocumentTitlesResponse(BaseModel):
    titles: Dict[str, str]


@router.post("/document-titles", response_model=DocumentTitlesResponse)
def get_document_titles(req: DocumentTitlesRequest) -> DocumentTitlesResponse:
    """Resolve document IDs to titles."""
    if not req.document_ids:
        return DocumentTitlesResponse(titles={})

    sb = get_supabase()
    try:
        res = (
            sb.table("documents")
            .select("id, title, source_uri, source_type")
            .in_("id", req.document_ids)
            .execute()
        )
        titles = {}
        for row in (res.data or []):
            if row.get("title"):
                titles[row["id"]] = row["title"]
            elif row.get("source_type") == "web" and row.get("source_uri"):
                titles[row["id"]] = row["source_uri"]
        return DocumentTitlesResponse(titles=titles)
    except Exception as e:
        logger.exception("Failed to resolve document titles")
        raise HTTPException(status_code=500, detail=str(e))


# ── Survey outputs ───────────────────────────────────────────────────────────


class SurveyOutputsResponse(BaseModel):
    outputs: List[Dict[str, Any]]


@router.get("/survey-outputs", response_model=SurveyOutputsResponse)
def get_survey_outputs(
    tenant_id: UUID = Query(...),
    client_id: UUID = Query(...),
    output_type: Optional[str] = Query(default=None),
    limit: int = Query(default=5, ge=1, le=50),
) -> SurveyOutputsResponse:
    """Fetch prior survey outputs for a tenant+client."""
    sb = get_supabase()
    try:
        q = (
            sb.table("survey_outputs")
            .select("*")
            .eq("tenant_id", str(tenant_id))
            .eq("client_id", str(client_id))
            .order("created_at", desc=True)
            .limit(limit)
        )
        if output_type:
            q = q.eq("output_type", output_type)
        res = q.execute()
        return SurveyOutputsResponse(outputs=res.data or [])
    except Exception as e:
        logger.exception("Failed to fetch survey outputs")
        raise HTTPException(status_code=500, detail=str(e))


class SaveSurveyOutputRequest(BaseModel):
    tenant_id: UUID
    client_id: UUID
    output_type: str
    request: str
    questions: List[Dict[str, Any]]
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SaveSurveyOutputResponse(BaseModel):
    status: str = "ok"


@router.post("/survey-outputs", response_model=SaveSurveyOutputResponse)
def save_survey_output(req: SaveSurveyOutputRequest) -> SaveSurveyOutputResponse:
    """Persist a survey output row."""
    sb = get_supabase()
    try:
        sb.rpc("cleanup_expired_survey_outputs", {}).execute()
    except Exception:
        logger.debug("Expired survey output cleanup skipped")

    try:
        sb.table("survey_outputs").insert({
            "tenant_id": str(req.tenant_id),
            "client_id": str(req.client_id),
            "output_type": req.output_type,
            "request": req.request,
            "questions": req.questions,
            "reasoning": req.reasoning,
            "metadata": req.metadata,
        }).execute()
    except Exception as e:
        logger.exception("Failed to save survey output")
        raise HTTPException(status_code=500, detail=str(e))

    return SaveSurveyOutputResponse()


# ── Context summary ────────────────────────────────────────────────────────


class ContextSummaryReadResponse(BaseModel):
    id: Optional[str] = None
    tenant_id: str
    client_id: str
    summary: str
    topics: List[Any] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_stats: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


@router.get("/context/summary/get", response_model=ContextSummaryReadResponse)
def get_context_summary(
    tenant_id: UUID = Query(...),
    client_id: UUID = Query(...),
) -> ContextSummaryReadResponse:
    """Fetch the stored context summary for a tenant+client."""
    sb = get_supabase()
    try:
        res = (
            sb.table("context_summaries")
            .select("*")
            .eq("tenant_id", str(tenant_id))
            .eq("client_id", str(client_id))
            .limit(1)
            .execute()
        )
        rows = res.data or []
        if not rows:
            raise HTTPException(
                status_code=404,
                detail=f"No context summary found for tenant={tenant_id}, client={client_id}.",
            )
        row = rows[0]
        return ContextSummaryReadResponse(
            id=row.get("id"),
            tenant_id=row["tenant_id"],
            client_id=row["client_id"],
            summary=row.get("summary", ""),
            topics=row.get("topics") or [],
            metadata=row.get("metadata") or {},
            source_stats=row.get("source_stats") or {},
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to fetch context summary")
        raise HTTPException(status_code=500, detail=str(e))
