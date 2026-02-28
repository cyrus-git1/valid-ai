"""
/strategic-analysis router
---------------------------
Convergent problem-solving analysis that combines all tenant data sources
(vectorized chunks, knowledge graph, context summaries, company labels,
and Serper web search) to produce actionable strategic insights.

POST /strategic-analysis/generate       — Single focus query for one tenant+client
POST /strategic-analysis/generate/batch — Multiple focus queries, same tenant+client
POST /strategic-analysis/generate/all   — One focus query across all clients for a tenant
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from src.models.api.strategic_analysis import (
    ActionPoint,
    AllAnalysisRequest,
    AllAnalysisResponse,
    BatchAnalysisRequest,
    BatchAnalysisResponse,
    StrategicAnalysisRequest,
    StrategicAnalysisResponse,
    StrategicAnalysisResult,
)
from src.services.strategic_analysis_service import StrategicAnalysisService
from src.supabase.supabase_client import get_supabase

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/strategic-analysis", tags=["strategic-analysis"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dict_to_result(raw: dict) -> StrategicAnalysisResult:
    """Convert a raw service dict into a StrategicAnalysisResult model."""
    action_points = [
        ActionPoint(**ap) if isinstance(ap, dict) else ap
        for ap in raw.get("action_points", [])
    ]
    return StrategicAnalysisResult(
        tenant_id=raw["tenant_id"],
        client_id=raw["client_id"],
        focus_query=raw["focus_query"],
        executive_summary=raw["executive_summary"],
        convergent_themes=raw.get("convergent_themes", []),
        action_points=action_points,
        future_recommendations=raw.get("future_recommendations", []),
        analysis_depth=raw["analysis_depth"],
        transcript_count=raw.get("transcript_count", 0),
        sources_used=raw.get("sources_used", {}),
        generated_at=raw.get("generated_at"),
    )


# ── POST /strategic-analysis/generate (single) ───────────────────────────────

@router.post("/generate", response_model=StrategicAnalysisResponse)
def generate_strategic_analysis(
    req: StrategicAnalysisRequest,
) -> StrategicAnalysisResponse:
    """
    Generate a convergent strategic analysis for a single focus query.

    Pulls together KG chunks, context summary, client profile, Serper web
    search, and transcript data. Analysis depth scales with transcript count:
      0 → foundational | 1-3 → developing | 4-9 → comprehensive | 10+ → deep
    """
    svc = StrategicAnalysisService(get_supabase())

    try:
        result = svc.generate_analysis(
            tenant_id=req.tenant_id,
            client_id=req.client_id,
            focus_query=req.focus_query,
            client_profile=req.client_profile,
            top_k=req.top_k,
            hop_limit=req.hop_limit,
            web_search_queries=req.web_search_queries,
            llm_model=req.llm_model,
        )
    except Exception as e:
        logger.exception("Strategic analysis generation failed")
        raise HTTPException(status_code=500, detail=f"Strategic analysis failed: {e}")

    action_points = [
        ActionPoint(**ap) if isinstance(ap, dict) else ap
        for ap in result.get("action_points", [])
    ]

    return StrategicAnalysisResponse(
        tenant_id=result["tenant_id"],
        client_id=result["client_id"],
        focus_query=result["focus_query"],
        executive_summary=result["executive_summary"],
        convergent_themes=result.get("convergent_themes", []),
        action_points=action_points,
        future_recommendations=result.get("future_recommendations", []),
        analysis_depth=result["analysis_depth"],
        transcript_count=result.get("transcript_count", 0),
        sources_used=result.get("sources_used", {}),
        generated_at=result.get("generated_at"),
    )


# ── POST /strategic-analysis/generate/batch ───────────────────────────────────

@router.post("/generate/batch", response_model=BatchAnalysisResponse)
def generate_batch_analysis(
    req: BatchAnalysisRequest,
) -> BatchAnalysisResponse:
    """
    Run multiple focus queries against the same tenant+client.

    Shared context (transcripts, context summary, profile) is gathered once
    and reused across all queries to avoid redundant data fetching. Each
    focus query still gets its own KG retrieval and web search.

    Capped at 10 focus queries per request.
    """
    svc = StrategicAnalysisService(get_supabase())

    try:
        raw = svc.generate_batch(
            tenant_id=req.tenant_id,
            client_id=req.client_id,
            focus_queries=req.focus_queries,
            client_profile=req.client_profile,
            top_k=req.top_k,
            hop_limit=req.hop_limit,
            web_search_queries=req.web_search_queries,
            llm_model=req.llm_model,
        )
    except Exception as e:
        logger.exception("Batch strategic analysis failed")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {e}")

    results = [_dict_to_result(r) for r in raw.get("results", [])]

    return BatchAnalysisResponse(
        tenant_id=raw["tenant_id"],
        client_id=raw["client_id"],
        total=raw["total"],
        completed=raw["completed"],
        failed=raw.get("failed", 0),
        results=results,
        errors=raw.get("errors", []),
    )


# ── POST /strategic-analysis/generate/all ─────────────────────────────────────

@router.post("/generate/all", response_model=AllAnalysisResponse)
def generate_all_analysis(
    req: AllAnalysisRequest,
) -> AllAnalysisResponse:
    """
    Run the same focus query across every client_id that has ingested
    data under the given tenant_id.

    Discovers all client_ids with documents, then runs a full convergent
    analysis for each. Useful for cross-client benchmarking or org-wide
    strategic reviews.
    """
    svc = StrategicAnalysisService(get_supabase())

    try:
        raw = svc.generate_all(
            tenant_id=req.tenant_id,
            focus_query=req.focus_query,
            client_profile=req.client_profile,
            top_k=req.top_k,
            hop_limit=req.hop_limit,
            web_search_queries=req.web_search_queries,
            llm_model=req.llm_model,
        )
    except Exception as e:
        logger.exception("All-clients strategic analysis failed")
        raise HTTPException(status_code=500, detail=f"All-clients analysis failed: {e}")

    results = [_dict_to_result(r) for r in raw.get("results", [])]

    return AllAnalysisResponse(
        tenant_id=raw["tenant_id"],
        focus_query=raw["focus_query"],
        total_clients=raw["total_clients"],
        completed=raw["completed"],
        failed=raw.get("failed", 0),
        results=results,
        errors=raw.get("errors", []),
    )
