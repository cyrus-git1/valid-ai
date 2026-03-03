"""
/sentiment-analysis router
----------------------------
Sentiment analysis of WebVTT transcript content.

POST /sentiment-analysis/generate       — Single analysis for one tenant+client
POST /sentiment-analysis/generate/batch — Multiple focus queries, same tenant+client
POST /sentiment-analysis/generate/all   — One query across all clients for a tenant
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from src.models.api.sentiment import (
    AllSentimentRequest,
    AllSentimentResponse,
    BatchSentimentRequest,
    BatchSentimentResponse,
    NotableQuote,
    SentimentAnalysisRequest,
    SentimentAnalysisResponse,
    SentimentAnalysisResult,
    SentimentScore,
    ThemeSentiment,
)
from src.services.sentiment_analysis_service import SentimentAnalysisService
from src.supabase.supabase_client import get_supabase

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sentiment-analysis", tags=["sentiment-analysis"])


# ── Helpers ──────────────────────────────────────────────────────────────────


def _dict_to_result(raw: dict) -> SentimentAnalysisResult:
    """Convert a raw service dict into a SentimentAnalysisResult model."""
    return SentimentAnalysisResult(
        tenant_id=raw["tenant_id"],
        client_id=raw["client_id"],
        overall_sentiment=SentimentScore(**raw.get("overall_sentiment", {})),
        dominant_sentiment=raw.get("dominant_sentiment", "neutral"),
        themes=[
            ThemeSentiment(**t) if isinstance(t, dict) else t
            for t in raw.get("themes", [])
        ],
        notable_quotes=[
            NotableQuote(**q) if isinstance(q, dict) else q
            for q in raw.get("notable_quotes", [])
        ],
        summary=raw.get("summary", ""),
        transcript_count=raw.get("transcript_count", 0),
        chunks_analysed=raw.get("chunks_analysed", 0),
        generated_at=raw.get("generated_at"),
    )


# ── POST /sentiment-analysis/generate (single) ──────────────────────────────


@router.post("/generate", response_model=SentimentAnalysisResponse)
def generate_sentiment_analysis(
    req: SentimentAnalysisRequest,
) -> SentimentAnalysisResponse:
    """Generate sentiment analysis for transcript content of a single tenant+client.

    Analyses VTT transcript chunks and produces overall sentiment scores,
    themed breakdowns, notable quotes, and a summary paragraph.
    Optionally narrow to a focus area.
    """
    svc = SentimentAnalysisService(get_supabase())

    try:
        result = svc.generate_analysis(
            tenant_id=req.tenant_id,
            client_id=req.client_id,
            focus_query=req.focus_query,
            client_profile=req.client_profile,
            llm_model=req.llm_model,
            chunk_limit=req.chunk_limit,
        )
    except Exception as e:
        logger.exception("Sentiment analysis generation failed")
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {e}")

    return SentimentAnalysisResponse(
        tenant_id=result["tenant_id"],
        client_id=result["client_id"],
        overall_sentiment=SentimentScore(**result.get("overall_sentiment", {})),
        dominant_sentiment=result.get("dominant_sentiment", "neutral"),
        themes=[
            ThemeSentiment(**t) if isinstance(t, dict) else t
            for t in result.get("themes", [])
        ],
        notable_quotes=[
            NotableQuote(**q) if isinstance(q, dict) else q
            for q in result.get("notable_quotes", [])
        ],
        summary=result.get("summary", ""),
        transcript_count=result.get("transcript_count", 0),
        chunks_analysed=result.get("chunks_analysed", 0),
        generated_at=result.get("generated_at"),
    )


# ── POST /sentiment-analysis/generate/batch ─────────────────────────────────


@router.post("/generate/batch", response_model=BatchSentimentResponse)
def generate_batch_sentiment(
    req: BatchSentimentRequest,
) -> BatchSentimentResponse:
    """Run sentiment analysis for multiple focus areas against the same
    tenant+client. Shared context is gathered once and reused.
    Capped at 10 focus queries per request.
    """
    svc = SentimentAnalysisService(get_supabase())

    try:
        raw = svc.generate_batch(
            tenant_id=req.tenant_id,
            client_id=req.client_id,
            focus_queries=req.focus_queries,
            client_profile=req.client_profile,
            llm_model=req.llm_model,
            chunk_limit=req.chunk_limit,
        )
    except Exception as e:
        logger.exception("Batch sentiment analysis failed")
        raise HTTPException(status_code=500, detail=f"Batch sentiment failed: {e}")

    results = [_dict_to_result(r) for r in raw.get("results", [])]

    return BatchSentimentResponse(
        tenant_id=raw["tenant_id"],
        client_id=raw["client_id"],
        total=raw["total"],
        completed=raw["completed"],
        failed=raw.get("failed", 0),
        results=results,
        errors=raw.get("errors", []),
    )


# ── POST /sentiment-analysis/generate/all ────────────────────────────────────


@router.post("/generate/all", response_model=AllSentimentResponse)
def generate_all_sentiment(
    req: AllSentimentRequest,
) -> AllSentimentResponse:
    """Run sentiment analysis across every client_id that has data under
    the given tenant_id. Useful for cross-client sentiment benchmarking.
    """
    svc = SentimentAnalysisService(get_supabase())

    try:
        raw = svc.generate_all(
            tenant_id=req.tenant_id,
            focus_query=req.focus_query,
            client_profile=req.client_profile,
            llm_model=req.llm_model,
            chunk_limit=req.chunk_limit,
        )
    except Exception as e:
        logger.exception("All-clients sentiment analysis failed")
        raise HTTPException(status_code=500, detail=f"All-clients sentiment failed: {e}")

    results = [_dict_to_result(r) for r in raw.get("results", [])]

    return AllSentimentResponse(
        tenant_id=raw["tenant_id"],
        focus_query=raw.get("focus_query"),
        total_clients=raw["total_clients"],
        completed=raw["completed"],
        failed=raw.get("failed", 0),
        results=results,
        errors=raw.get("errors", []),
    )
