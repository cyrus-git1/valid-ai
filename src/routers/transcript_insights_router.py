"""
/transcript-insights router
------------------------------
Summarise WebVTT transcripts and extract actionable product/service
improvement insights.

POST /transcript-insights/generate — Summarise + extract insights for a survey
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from src.models.api.transcript_insights import (
    ActionableInsight,
    TranscriptInsightsRequest,
    TranscriptInsightsResponse,
)
from src.services.transcript_insights_service import TranscriptInsightsService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/transcript-insights", tags=["transcript-insights"])


@router.post("/generate", response_model=TranscriptInsightsResponse)
def generate_transcript_insights(
    req: TranscriptInsightsRequest,
) -> TranscriptInsightsResponse:
    """Summarise raw WebVTT content and extract actionable insights.

    Accepts a WebVTT transcript string and returns a structured summary with
    actionable insights that could improve the client's product or service.
    """
    svc = TranscriptInsightsService(supabase=None)

    try:
        result = svc.generate_from_vtt(
            tenant_id=req.tenant_id,
            survey_id=req.survey_id,
            vtt_content=req.vtt_content,
            llm_model=req.llm_model,
        )
    except Exception as e:
        logger.exception("Transcript insights generation failed")
        raise HTTPException(
            status_code=500,
            detail=f"Transcript insights generation failed: {e}",
        )

    insights = [
        ActionableInsight(**i) if isinstance(i, dict) else i
        for i in result.get("actionable_insights", [])
    ]

    return TranscriptInsightsResponse(
        tenant_id=result["tenant_id"],
        survey_id=result["survey_id"],
        summary=result.get("summary", ""),
        actionable_insights=insights,
        transcript_count=result.get("transcript_count", 0),
        chunks_analysed=result.get("chunks_analysed", 0),
        status=result.get("status", "complete"),
        error=result.get("error"),
        generated_at=result.get("generated_at"),
    )
