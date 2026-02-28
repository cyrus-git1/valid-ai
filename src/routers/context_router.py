"""
/context router
---------------
Unified context build endpoint and agent interaction.

POST /context/build              — Full pipeline: ingest all sources + build KG
GET  /context/status/{job_id}    — Poll context build job status
POST /context/query              — Send a query through the routing agent
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict

from fastapi import APIRouter, BackgroundTasks, HTTPException

from src.models.api.context import (
    ContextBuildRequest,
    ContextBuildResponse,
    ContextBuildStatusResponse,
)
from src.workflows.context_build_workflow import build_context_graph

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/context", tags=["context"])

# In-memory job store — replace with Redis or a DB table in production
_jobs: Dict[str, Dict[str, Any]] = {}


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
