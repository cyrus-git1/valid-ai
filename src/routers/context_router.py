"""
/context router
---------------
Unified context management: build, status, summary generation & retrieval.

POST   /context/build                          — Full pipeline: ingest all sources + build KG
GET    /context/status/{job_id}                — Poll context build job status
POST   /context/summary/generate               — Generate (or regenerate) a context summary
POST   /context/summary/get                    — Retrieve an existing summary
DELETE /context/summary/{tenant_id}/{client_id} — Delete a summary
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, HTTPException

from src.models.api.context import (
    ContextBuildRequest,
    ContextBuildResponse,
    ContextBuildStatusResponse,
)
from src.models.api.context_summary import (
    ContextSummaryGenerateRequest,
    ContextSummaryGenerateResponse,
    ContextSummaryGetRequest,
    ContextSummaryResponse,
    ContextSummaryDeleteResponse,
)
from src.workflows.context_build_workflow import build_context_graph
from src.supabase.supabase_client import get_supabase
from src.services.context_summary_service import ContextSummaryService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/context", tags=["context"])

# In-memory job store — replace with Redis or a DB table in production
_jobs: Dict[str, Dict[str, Any]] = {}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _ensure_parsed(value: Any, fallback: Any = None) -> Any:
    """Supabase may return JSONB columns as JSON strings — parse if needed."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return fallback
    return value if value is not None else fallback


def _row_to_response(row: Dict[str, Any]) -> ContextSummaryResponse:
    """Convert a raw Supabase row dict into a ContextSummaryResponse."""
    return ContextSummaryResponse(
        id=row["id"],
        tenant_id=row["tenant_id"],
        client_id=row["client_id"],
        summary=row["summary"],
        topics=_ensure_parsed(row.get("topics"), []),
        metadata=_ensure_parsed(row.get("metadata"), {}),
        source_stats=_ensure_parsed(row.get("source_stats"), {}),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


# ── Context build ────────────────────────────────────────────────────────────

def _run_context_build(job_id: str, req: ContextBuildRequest) -> None:
    """Background task: run the full context build LangGraph."""
    _jobs[job_id] = {"status": "running"}
    try:
        app = build_context_graph()
        result = app.invoke({
            "tenant_id": str(req.tenant_id),
            "client_id": str(req.client_id),
            "docs": req.context.docs,
            "weblinks": req.context.weblinks,
            "transcripts": req.context.transcripts,
            "client_profile": req.client_profile.model_dump(),
        })

        ingest_results = result.get("ingest_results", [])
        kg_result = result.get("kg_build_result", {})

        doc_count = sum(1 for r in ingest_results if r.get("source_type") not in ("web", "vtt"))
        web_count = sum(1 for r in ingest_results if r.get("source_type") == "web")
        vtt_count = sum(1 for r in ingest_results if r.get("source_type") == "vtt")
        total_chunks = sum(r.get("chunks_upserted", 0) for r in ingest_results)

        _jobs[job_id] = {
            "status": result.get("status", "complete"),
            "documents_ingested": doc_count,
            "weblinks_ingested": web_count,
            "transcripts_ingested": vtt_count,
            "total_chunks": total_chunks,
            "kg_nodes_upserted": kg_result.get("nodes_upserted", 0),
            "kg_edges_upserted": kg_result.get("edges_upserted", 0),
            "warnings": result.get("warnings", []),
        }
        logger.info("Context build job %s complete", job_id)

    except Exception as e:
        logger.exception("Context build job %s failed", job_id)
        _jobs[job_id] = {"status": "failed", "detail": str(e)}


@router.post("/build", response_model=ContextBuildResponse, status_code=202)
def build_context(
    req: ContextBuildRequest,
    background_tasks: BackgroundTasks,
) -> ContextBuildResponse:
    """
    Build a client's full knowledge base from documents, websites, and transcripts.

    Accepts a unified JSON payload with:
      - docs: list of file paths to PDFs/DOCX files
      - weblinks: list of URLs to scrape
      - transcripts: list of file paths to WebVTT (.vtt) Daily.js transcripts
      - client_profile: industry, headcount, demographic info

    Returns 202 immediately with a job_id. The full pipeline
    (ingest → embed → KG build) runs in the background.
    Poll GET /context/status/{job_id} to check completion.
    """
    job_id = str(uuid.uuid4())
    background_tasks.add_task(_run_context_build, job_id, req)
    return ContextBuildResponse(job_id=job_id, status="accepted")


@router.get("/status/{job_id}", response_model=ContextBuildStatusResponse)
def context_status(job_id: str) -> ContextBuildStatusResponse:
    """Poll the status of a context build job."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return ContextBuildStatusResponse(
        job_id=job_id,
        status=job.get("status", "unknown"),
        detail=job,
    )


# ── Context summary ─────────────────────────────────────────────────────────

@router.post("/summary/generate", response_model=ContextSummaryGenerateResponse)
def generate_context_summary(
    req: ContextSummaryGenerateRequest,
) -> ContextSummaryGenerateResponse:
    """
    Generate an LLM-powered context summary for a tenant+client.

    Uses the tenant's knowledge graph to retrieve relevant excerpts,
    then prompts an LLM to produce a summary and topic tags.
    The result is stored in the context_summaries table (upsert).

    If a summary already exists and force_regenerate is False, returns
    the existing summary without calling the LLM.
    """
    svc = ContextSummaryService(get_supabase())
    try:
        result = svc.generate_summary(
            tenant_id=req.tenant_id,
            client_id=req.client_id,
            client_profile=req.client_profile,
            force_regenerate=req.force_regenerate,
        )
    except Exception as e:
        logger.exception("Context summary generation failed")
        raise HTTPException(
            status_code=500,
            detail=f"Context summary generation failed: {e}",
        )

    row = result["summary_row"]
    return ContextSummaryGenerateResponse(
        summary=_row_to_response(row),
        status=result.get("status", "complete"),
        regenerated=result.get("regenerated", False),
    )


@router.post("/summary/get", response_model=ContextSummaryResponse)
def get_context_summary(
    req: ContextSummaryGetRequest,
) -> ContextSummaryResponse:
    """
    Retrieve the stored context summary for a tenant+client.

    Returns 404 if no summary has been generated yet.
    """
    svc = ContextSummaryService(get_supabase())
    row = svc.get_summary(tenant_id=req.tenant_id, client_id=req.client_id)

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"No context summary found for tenant={req.tenant_id}, client={req.client_id}. "
            "Call POST /context/summary/generate first.",
        )

    return _row_to_response(row)


@router.delete(
    "/summary/{tenant_id}/{client_id}",
    response_model=ContextSummaryDeleteResponse,
)
def delete_context_summary(
    tenant_id: UUID,
    client_id: UUID,
) -> ContextSummaryDeleteResponse:
    """
    Delete the context summary for a tenant+client.

    Returns whether a row was actually deleted.
    """
    svc = ContextSummaryService(get_supabase())
    deleted = svc.delete_summary(tenant_id=tenant_id, client_id=client_id)
    return ContextSummaryDeleteResponse(
        deleted=deleted,
        tenant_id=tenant_id,
        client_id=client_id,
    )
