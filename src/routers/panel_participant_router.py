"""
/panel router
--------------
Panel participant ingest and filtering.

POST /panel/ingest                 — Ingest participant data (background task)
GET  /panel/ingest/status/{job_id} — Poll ingest job status
POST /panel/filter                 — Filter participants against business context
"""
from __future__ import annotations

import uuid
import logging
from typing import Dict

from fastapi import APIRouter, BackgroundTasks, HTTPException

from src.supabase.supabase_client import get_supabase
from src.models.api.panel_participants import (
    PanelFilterRequest,
    PanelFilterResponse,
    PanelIngestRequest,
    PanelIngestResponse,
)
from src.services.panel_participant_service import PanelParticipantService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/panel", tags=["panel"])

# In-memory job store — replace with Redis or a DB table in production
_jobs: Dict[str, Dict] = {}


# ── Background ingest task ───────────────────────────────────────────────────

def _run_panel_ingest(job_id: str, req: PanelIngestRequest) -> None:
    """Run the full panel ingest pipeline in background."""
    try:
        sb = get_supabase()
        svc = PanelParticipantService(sb)
        result = svc.ingest_participants(
            tenant_id=req.tenant_id,
            client_id=req.client_id,
            vendor_name=req.vendor_name,
            participants=req.participants,
            metadata=req.metadata,
            embed_model=req.embed_model,
            embed_batch_size=req.embed_batch_size,
            build_kg=req.build_kg,
        )
        _jobs[job_id] = {
            "status": "complete",
            "vendor_name": req.vendor_name,
            "total_participants": result["total_participants"],
            "completed": result["completed"],
            "failed": result["failed"],
            "results": result["results"],
            "warnings": result["warnings"],
        }
    except Exception as e:
        logger.exception("Panel ingest job %s failed", job_id)
        _jobs[job_id] = {
            "status": "failed",
            "vendor_name": req.vendor_name,
            "total_participants": len(req.participants),
            "completed": 0,
            "failed": len(req.participants),
            "results": [],
            "warnings": [str(e)],
        }


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/ingest", response_model=PanelIngestResponse)
async def ingest_panel_participants(
    req: PanelIngestRequest,
    background_tasks: BackgroundTasks,
) -> PanelIngestResponse:
    """
    Ingest a batch of panel participants from a vendor.

    Kicks off a background task that:
      1. Serializes each participant's nested JSON into natural-language text
      2. Chunks the text using SpaCy + tiktoken
      3. Embeds all chunks in a single batch
      4. Stores documents and chunks in Supabase
      5. Optionally builds KG nodes + edges and updates context summary
    """
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "status": "running",
        "vendor_name": req.vendor_name,
        "total_participants": len(req.participants),
        "completed": 0,
        "failed": 0,
        "results": [],
        "warnings": [],
    }
    background_tasks.add_task(_run_panel_ingest, job_id, req)

    return PanelIngestResponse(
        job_id=job_id,
        vendor_name=req.vendor_name,
        total_participants=len(req.participants),
        status="running",
    )


@router.get("/ingest/status/{job_id}", response_model=PanelIngestResponse)
async def get_panel_ingest_status(job_id: str) -> PanelIngestResponse:
    """Poll the status of a panel ingest job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return PanelIngestResponse(
        job_id=job_id,
        vendor_name=job["vendor_name"],
        total_participants=job["total_participants"],
        status=job["status"],
        completed=job.get("completed", 0),
        failed=job.get("failed", 0),
        results=job.get("results", []),
        warnings=job.get("warnings", []),
    )


@router.post("/filter", response_model=PanelFilterResponse)
async def filter_panel_participants(
    req: PanelFilterRequest,
) -> PanelFilterResponse:
    """
    Filter previously ingested panel participants against business context.

    Supports four modes:
      - 'label'     — rule-based term matching against business context (no LLM)
      - 'llm'       — LLM generates criteria from business context, then scores participants
      - 'embedding' — cosine similarity of participant embeddings vs business context centroid
      - 'full'      — label filter first, then embedding similarity on candidates
    """
    sb = get_supabase()
    svc = PanelParticipantService(sb)

    try:
        return svc.filter_participants(
            tenant_id=req.tenant_id,
            client_id=req.client_id,
            filter_mode=req.filter_mode,
            top_k=req.top_k,
            similarity_threshold=req.similarity_threshold,
            llm_model=req.llm_model,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Panel filtering failed")
        raise HTTPException(status_code=500, detail=f"Filtering failed: {e}")
