"""
/transcript-insights router
------------------------------
Summarise WebVTT transcripts and extract actionable product/service
improvement insights.

POST /transcript-insights/generate — Summarise + extract insights for a survey
"""
from __future__ import annotations

import logging

from uuid import UUID

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.models.api.transcript_insights import (
    ActionableInsight,
    TranscriptInsightsResponse,
)
from src.services.transcript_insights_service import TranscriptInsightsService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/transcript-insights", tags=["transcript-insights"])


@router.post("/generate", response_model=TranscriptInsightsResponse)
async def generate_transcript_insights(
    file: UploadFile = File(..., description="WebVTT (.vtt) transcript file"),
    tenant_id: UUID = Form(...),
    survey_id: UUID = Form(...),
    llm_model: str = Form("gpt-4o-mini"),
) -> TranscriptInsightsResponse:
    """Summarise an uploaded WebVTT file and extract actionable insights.

    Accepts a .vtt file upload and returns a structured summary with
    actionable insights that could improve the client's product or service.
    """
    vtt_content = (await file.read()).decode("utf-8")

    svc = TranscriptInsightsService(supabase=None)

    try:
        result = svc.generate_from_vtt(
            tenant_id=tenant_id,
            survey_id=survey_id,
            vtt_content=vtt_content,
            llm_model=llm_model,
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
