"""
/ingest router
--------------
Handles getting content into the system.

POST /ingest/file    — Upload PDF or DOCX (multipart), runs full pipeline in background
POST /ingest/web     — Kick off a website crawl by URL, runs in background
GET  /ingest/status/{job_id} — Poll job status

Both POST endpoints return immediately with a job_id. The actual ingest
(chunking + embedding + storage) runs in a FastAPI BackgroundTask so the
HTTP response is not held open during the full crawl/embed cycle.

For production with many concurrent ingests, swap BackgroundTasks for
an arq or Celery worker — the IngestService itself has no HTTP coupling.
"""
from __future__ import annotations

import uuid
import logging
from typing import Dict, Any

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from supabase import Client

from src.supabase.supabase_client import get_supabase
from src.models.api.ingest import (
    IngestFileResponse,
    IngestWebRequest,
    IngestWebResponse,
    IngestStatusResponse,
)
from src.services.ingest_service import IngestService, IngestInput

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingest"])

# In-memory job store — replace with Redis or a DB table in production
_jobs: Dict[str, Dict[str, Any]] = {}


def _run_file_ingest(
    job_id: str,
    sb: Client,
    file_bytes: bytes,
    file_name: str,
    tenant_id: uuid.UUID,
    client_id: uuid.UUID,
    title: str | None,
    prune_after_ingest: bool,
) -> None:
    """Background task: full PDF/DOCX ingest pipeline."""
    _jobs[job_id] = {"status": "running"}
    try:
        svc = IngestService(sb)
        result = svc.ingest(IngestInput(
            tenant_id=tenant_id,
            client_id=client_id,
            file_bytes=file_bytes,
            file_name=file_name,
            title=title,
            prune_after_ingest=prune_after_ingest,
        ))
        _jobs[job_id] = {
            "status": "complete",
            "document_id": str(result.document_id),
            "source_type": result.source_type,
            "source_uri": result.source_uri,
            "chunks_upserted": result.chunks_upserted,
            "warnings": result.warnings,
            "prune_result": result.prune_result,
        }
        logger.info("Job %s complete — %d chunks", job_id, result.chunks_upserted)
    except Exception as e:
        logger.exception("Job %s failed", job_id)
        _jobs[job_id] = {"status": "failed", "detail": str(e)}


def _run_web_ingest(
    job_id: str,
    sb: Client,
    url: str,
    tenant_id: uuid.UUID,
    client_id: uuid.UUID,
    title: str | None,
    prune_after_ingest: bool,
) -> None:
    """Background task: full web scrape + ingest pipeline."""
    _jobs[job_id] = {"status": "running"}
    try:
        svc = IngestService(sb)
        result = svc.ingest(IngestInput(
            tenant_id=tenant_id,
            client_id=client_id,
            web_url=url,
            title=title,
            prune_after_ingest=prune_after_ingest,
        ))
        _jobs[job_id] = {
            "status": "complete",
            "document_id": str(result.document_id),
            "source_type": result.source_type,
            "source_uri": result.source_uri,
            "chunks_upserted": result.chunks_upserted,
            "warnings": result.warnings,
            "prune_result": result.prune_result,
        }
        logger.info("Job %s complete — %d chunks", job_id, result.chunks_upserted)
    except Exception as e:
        logger.exception("Job %s failed", job_id)
        _jobs[job_id] = {"status": "failed", "detail": str(e)}


@router.post("/file", response_model=IngestFileResponse, status_code=202)
async def ingest_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF or DOCX file to ingest"),
    tenant_id: uuid.UUID = Form(...),
    client_id: uuid.UUID = Form(...),
    title: str | None = Form(default=None),
    prune_after_ingest: bool = Form(default=False),
) -> IngestFileResponse:
    """
    Upload a PDF or DOCX and run the full ingest pipeline.

    Returns 202 immediately with a job_id. Poll GET /ingest/status/{job_id}
    to check completion.

    The file is:
      1. Uploaded to Supabase "pdf" bucket
      2. Tokenized with spaCy + tiktoken
      3. Embedded with OpenAI text-embedding-3-small
      4. Stored in the chunks table
    """
    if file.content_type not in (
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Send a PDF or DOCX.",
        )

    file_bytes = await file.read()
    file_name = file.filename or f"upload_{uuid.uuid4().hex}.pdf"
    job_id = str(uuid.uuid4())
    sb = get_supabase()

    background_tasks.add_task(
        _run_file_ingest,
        job_id, sb, file_bytes, file_name,
        tenant_id, client_id, title, prune_after_ingest,
    )

    return IngestFileResponse(
        job_id=job_id,
        document_id="pending",
        source_type="pdf",
        source_uri="pending",
        chunks_upserted=0,
        warnings=[],
    )


@router.post("/web", response_model=IngestWebResponse, status_code=202)
def ingest_web(
    req: IngestWebRequest,
    background_tasks: BackgroundTasks,
) -> IngestWebResponse:
    """
    Scrape a website and run the full ingest pipeline.

    Returns 202 immediately with a job_id. Poll GET /ingest/status/{job_id}
    to check completion.

    The crawler:
      1. Follows all internal links from the root URL (respects robots.txt)
      2. Extracts clean text with trafilatura
      3. Tokenizes pages with spaCy + tiktoken
      4. Embeds with OpenAI text-embedding-3-small
      5. Stores in the chunks table
    """
    job_id = str(uuid.uuid4())
    sb = get_supabase()

    background_tasks.add_task(
        _run_web_ingest,
        job_id, sb, req.url,
        req.tenant_id, req.client_id,
        req.title, req.prune_after_ingest,
    )

    return IngestWebResponse(
        job_id=job_id,
        document_id="pending",
        source_type="web",
        source_uri=req.url,
        chunks_upserted=0,
        warnings=[],
    )


@router.get("/status/{job_id}", response_model=IngestStatusResponse)
def ingest_status(job_id: str) -> IngestStatusResponse:
    """
    Poll the status of a background ingest job.

    Returns:
      - status: "running" | "complete" | "failed"
      - detail: error message if failed
    """
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    return IngestStatusResponse(
        job_id=job_id,
        status=job["status"],
        detail=job.get("detail"),
    )
