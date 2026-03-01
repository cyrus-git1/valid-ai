"""
/ingest router
--------------
Handles getting content into the system.

POST /ingest/file    — Upload PDF, DOCX, VTT, or XLSX (multipart), runs full pipeline in background
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
from typing import Any, Dict, List

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from supabase import Client

from src.supabase.supabase_client import get_supabase
from src.models.api.ingest import (
    IngestFileResponse,
    IngestWebRequest,
    IngestWebResponse,
    IngestStatusResponse,
    BatchWebRequest,
    BatchIngestResponse,
    BatchIngestStatusResponse,
    BatchItemStatus,
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
    file: UploadFile = File(..., description="PDF, DOCX, VTT, or XLSX file to ingest"),
    tenant_id: uuid.UUID = Form(...),
    client_id: uuid.UUID = Form(...),
    title: str | None = Form(default=None),
    prune_after_ingest: bool = Form(default=False),
) -> IngestFileResponse:
    """
    Upload a PDF, DOCX, VTT, or XLSX file and run the full ingest pipeline.

    Returns 202 immediately with a job_id. Poll GET /ingest/status/{job_id}
    to check completion.

    The file is:
      1. Uploaded to Supabase storage bucket
      2. Tokenized (spaCy + tiktoken for docs, WebVTT parser for .vtt,
         pandas for .xlsx/.xls)
      3. Embedded with OpenAI text-embedding-3-small
      4. Stored in the chunks table
    """
    _ALLOWED_CONTENT_TYPES = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
        "text/vtt",
        "text/plain",  # some clients send .vtt as text/plain
    }
    if file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Send a PDF, DOCX, VTT, or XLSX.",
        )

    file_bytes = await file.read()
    file_name = file.filename or f"upload_{uuid.uuid4().hex}.bin"
    job_id = str(uuid.uuid4())
    sb = get_supabase()

    # Detect source type from file extension
    ext = (file_name.rsplit(".", 1)[-1] if "." in file_name else "").lower()
    _EXT_TO_TYPE = {"pdf": "pdf", "docx": "docx", "vtt": "vtt", "xlsx": "xlsx", "xls": "xlsx"}
    source_type = _EXT_TO_TYPE.get(ext, ext or "file")

    background_tasks.add_task(
        _run_file_ingest,
        job_id, sb, file_bytes, file_name,
        tenant_id, client_id, title, prune_after_ingest,
    )

    return IngestFileResponse(
        job_id=job_id,
        document_id="pending",
        source_type=source_type,
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


# ── Batch ingest ─────────────────────────────────────────────────────────────

# In-memory batch store — replace with Redis or a DB table in production
_batches: Dict[str, Dict[str, Any]] = {}


def _run_batch_file_ingest(
    batch_id: str,
    sb: Client,
    files_data: List[Dict[str, Any]],
    tenant_id: uuid.UUID,
    client_id: uuid.UUID,
    prune_after_ingest: bool,
) -> None:
    """Background task: ingest multiple files sequentially within a batch."""
    for i, item in enumerate(files_data):
        _batches[batch_id]["items"][i]["status"] = "running"
        try:
            svc = IngestService(sb)
            result = svc.ingest(IngestInput(
                tenant_id=tenant_id,
                client_id=client_id,
                file_bytes=item["file_bytes"],
                file_name=item["file_name"],
                title=item.get("title"),
                prune_after_ingest=prune_after_ingest and (i == len(files_data) - 1),
            ))
            _batches[batch_id]["items"][i].update({
                "status": "complete",
                "document_id": str(result.document_id),
                "chunks_upserted": result.chunks_upserted,
                "warnings": result.warnings,
            })
            logger.info("Batch %s item %d complete — %d chunks", batch_id, i, result.chunks_upserted)
        except Exception as e:
            logger.exception("Batch %s item %d failed", batch_id, i)
            _batches[batch_id]["items"][i].update({
                "status": "failed",
                "detail": str(e),
            })

    _finalise_batch(batch_id)


def _run_batch_web_ingest(
    batch_id: str,
    sb: Client,
    items: List[Dict[str, Any]],
    tenant_id: uuid.UUID,
    client_id: uuid.UUID,
    prune_after_ingest: bool,
) -> None:
    """Background task: ingest multiple web URLs sequentially within a batch."""
    for i, item in enumerate(items):
        _batches[batch_id]["items"][i]["status"] = "running"
        try:
            svc = IngestService(sb)
            result = svc.ingest(IngestInput(
                tenant_id=tenant_id,
                client_id=client_id,
                web_url=item["url"],
                title=item.get("title"),
                metadata=item.get("metadata", {}),
                prune_after_ingest=prune_after_ingest and (i == len(items) - 1),
            ))
            _batches[batch_id]["items"][i].update({
                "status": "complete",
                "document_id": str(result.document_id),
                "chunks_upserted": result.chunks_upserted,
                "warnings": result.warnings,
            })
            logger.info("Batch %s item %d complete — %d chunks", batch_id, i, result.chunks_upserted)
        except Exception as e:
            logger.exception("Batch %s item %d failed", batch_id, i)
            _batches[batch_id]["items"][i].update({
                "status": "failed",
                "detail": str(e),
            })

    _finalise_batch(batch_id)


def _finalise_batch(batch_id: str) -> None:
    """Set overall batch status based on item outcomes."""
    items = _batches[batch_id]["items"]
    failed = sum(1 for it in items if it["status"] == "failed")
    completed = sum(1 for it in items if it["status"] == "complete")

    if failed == len(items):
        _batches[batch_id]["status"] = "failed"
    elif failed > 0:
        _batches[batch_id]["status"] = "partial_failure"
    else:
        _batches[batch_id]["status"] = "complete"

    _batches[batch_id]["completed"] = completed
    _batches[batch_id]["failed"] = failed
    _batches[batch_id]["running"] = 0
    logger.info("Batch %s finalised — %d/%d complete, %d failed",
                batch_id, completed, len(items), failed)


@router.post("/batch/files", response_model=BatchIngestResponse, status_code=202)
async def batch_ingest_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Multiple PDF, DOCX, VTT, or XLSX files"),
    tenant_id: uuid.UUID = Form(...),
    client_id: uuid.UUID = Form(...),
    prune_after_ingest: bool = Form(default=False),
) -> BatchIngestResponse:
    """
    Upload multiple files and ingest them all in one batch.

    Returns 202 immediately with a batch_id. Poll GET /ingest/batch/status/{batch_id}
    to check progress of each item.
    """
    _ALLOWED_CONTENT_TYPES = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
        "text/vtt",
        "text/plain",
    }

    # Validate and read all files upfront
    files_data: List[Dict[str, Any]] = []
    for f in files:
        if f.content_type not in _ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type '{f.content_type}' for file '{f.filename}'. "
                       f"Send PDF, DOCX, VTT, or XLSX.",
            )
        file_bytes = await f.read()
        files_data.append({
            "file_bytes": file_bytes,
            "file_name": f.filename or f"upload_{uuid.uuid4().hex}.bin",
        })

    batch_id = str(uuid.uuid4())
    sb = get_supabase()

    items = [
        {
            "index": i,
            "source": fd["file_name"],
            "status": "pending",
            "document_id": None,
            "chunks_upserted": 0,
            "warnings": [],
            "detail": None,
        }
        for i, fd in enumerate(files_data)
    ]
    _batches[batch_id] = {
        "status": "running",
        "total": len(files_data),
        "completed": 0,
        "failed": 0,
        "running": len(files_data),
        "items": items,
    }

    background_tasks.add_task(
        _run_batch_file_ingest,
        batch_id, sb, files_data,
        tenant_id, client_id, prune_after_ingest,
    )

    return BatchIngestResponse(
        batch_id=batch_id,
        total=len(files_data),
        status="running",
        items=[BatchItemStatus(**it) for it in items],
    )


@router.post("/batch/web", response_model=BatchIngestResponse, status_code=202)
def batch_ingest_web(
    req: BatchWebRequest,
    background_tasks: BackgroundTasks,
) -> BatchIngestResponse:
    """
    Scrape multiple websites and ingest them all in one batch.

    Returns 202 immediately with a batch_id. Poll GET /ingest/batch/status/{batch_id}
    to check progress of each item.
    """
    batch_id = str(uuid.uuid4())
    sb = get_supabase()

    items_raw = [
        {
            "url": item.url,
            "title": item.title,
            "metadata": item.metadata,
        }
        for item in req.items
    ]

    items = [
        {
            "index": i,
            "source": it["url"],
            "status": "pending",
            "document_id": None,
            "chunks_upserted": 0,
            "warnings": [],
            "detail": None,
        }
        for i, it in enumerate(items_raw)
    ]
    _batches[batch_id] = {
        "status": "running",
        "total": len(items_raw),
        "completed": 0,
        "failed": 0,
        "running": len(items_raw),
        "items": items,
    }

    background_tasks.add_task(
        _run_batch_web_ingest,
        batch_id, sb, items_raw,
        req.tenant_id, req.client_id, req.prune_after_ingest,
    )

    return BatchIngestResponse(
        batch_id=batch_id,
        total=len(items_raw),
        status="running",
        items=[BatchItemStatus(**it) for it in items],
    )


@router.get("/batch/status/{batch_id}", response_model=BatchIngestStatusResponse)
def batch_ingest_status(batch_id: str) -> BatchIngestStatusResponse:
    """
    Poll the status of a batch ingest job.

    Returns per-item status and overall batch progress.
    """
    batch = _batches.get(batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail=f"Batch '{batch_id}' not found.")

    return BatchIngestStatusResponse(
        batch_id=batch_id,
        total=batch["total"],
        completed=batch["completed"],
        failed=batch["failed"],
        running=batch["running"],
        status=batch["status"],
        items=[BatchItemStatus(**it) for it in batch["items"]],
    )
