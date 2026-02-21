"""
/context-summary router
-----------------------
Manage LLM-generated context summaries per tenant+client.

POST /context-summary/generate  — Generate (or regenerate) a context summary
POST /context-summary/get       — Retrieve an existing summary
DELETE /context-summary/{tenant_id}/{client_id}  — Delete a summary
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict
from uuid import UUID

from fastapi import APIRouter, HTTPException

from src.models.api.context_summary import (
    ContextSummaryGenerateRequest,
    ContextSummaryGenerateResponse,
    ContextSummaryGetRequest,
    ContextSummaryResponse,
    ContextSummaryDeleteResponse,
)
from src.supabase.supabase_client import get_supabase
from src.services.context_summary_service import ContextSummaryService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/context-summary", tags=["context-summary"])


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


# ── POST /context-summary/generate ────────────────────────────────────────────

@router.post("/generate", response_model=ContextSummaryGenerateResponse)
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


# ── POST /context-summary/get ─────────────────────────────────────────────────

@router.post("/get", response_model=ContextSummaryResponse)
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
            "Call POST /context-summary/generate first.",
        )

    return _row_to_response(row)


# ── DELETE /context-summary/{tenant_id}/{client_id} ───────────────────────────

@router.delete(
    "/{tenant_id}/{client_id}",
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
