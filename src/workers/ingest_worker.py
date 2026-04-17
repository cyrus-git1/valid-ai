"""
arq worker for async ingest jobs.

Run with:
    arq src.workers.ingest_worker.WorkerSettings

Each task:
  1. Marks the ingest_jobs row as 'running'
  2. Invokes the synchronous ingest pipeline
  3. Updates the row with 'complete' + result, or 'failed' + error
"""
from __future__ import annotations

import os
import traceback
from datetime import datetime, timezone
from typing import Any, Dict

import dotenv
from arq.connections import RedisSettings

dotenv.load_dotenv()

# Configure logging so worker output matches the API
from src.logging_config import configure_logging, get_logger

configure_logging(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = get_logger(__name__)

from src.services.ingest_service import run_processed_ingest, run_processed_web_ingest
from src.supabase.supabase_client import get_supabase


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mark_running(job_id: str) -> None:
    try:
        get_supabase().table("ingest_jobs").update({
            "status": "running",
            "started_at": _now_iso(),
        }).eq("id", job_id).execute()
    except Exception as e:
        logger.warning("ingest_job_mark_running_failed", job_id=job_id, error=str(e))


def _mark_complete(job_id: str, result: Dict[str, Any]) -> None:
    try:
        get_supabase().table("ingest_jobs").update({
            "status": "complete",
            "completed_at": _now_iso(),
            "document_id": result.get("document_id"),
            "result": result,
        }).eq("id", job_id).execute()
    except Exception as e:
        logger.warning("ingest_job_mark_complete_failed", job_id=job_id, error=str(e))


def _mark_failed(job_id: str, error: str) -> None:
    try:
        get_supabase().table("ingest_jobs").update({
            "status": "failed",
            "completed_at": _now_iso(),
            "error": error[:2000],
        }).eq("id", job_id).execute()
    except Exception as e:
        logger.warning("ingest_job_mark_failed_failed", job_id=job_id, error=str(e))


async def run_processed_ingest_task(ctx, job_id: str, payload: Dict[str, Any]) -> None:
    logger.info("ingest_task_started", job_id=job_id, job_type="processed")
    _mark_running(job_id)
    try:
        result = run_processed_ingest(payload)
        _mark_complete(job_id, result)
        logger.info("ingest_task_complete", job_id=job_id)
    except Exception as e:
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        logger.error("ingest_task_failed", job_id=job_id, error=err)
        _mark_failed(job_id, err)
        raise


async def run_processed_web_ingest_task(ctx, job_id: str, payload: Dict[str, Any]) -> None:
    logger.info("ingest_task_started", job_id=job_id, job_type="processed-web")
    _mark_running(job_id)
    try:
        result = run_processed_web_ingest(payload)
        _mark_complete(job_id, result)
        logger.info("ingest_task_complete", job_id=job_id)
    except Exception as e:
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        logger.error("ingest_task_failed", job_id=job_id, error=err)
        _mark_failed(job_id, err)
        raise


class WorkerSettings:
    functions = [run_processed_ingest_task, run_processed_web_ingest_task]
    redis_settings = RedisSettings.from_dsn(
        os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    )
    # Long jobs are allowed — a large PDF can legitimately take a while
    job_timeout = 60 * 30  # 30 minutes
    max_jobs = int(os.environ.get("ARQ_MAX_JOBS", "4"))
    keep_result = 3600  # keep arq job result in Redis for 1h
