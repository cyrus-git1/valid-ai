"""
/confidence-interval router
-----------------------------
Compute confidence intervals for quantitative survey question types.

POST /confidence-interval/compute — Compute CIs for one or more questions
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from src.models.api.confidence_interval import (
    ConfidenceIntervalRequest,
    ConfidenceIntervalResponse,
)
from src.services.confidence_interval_service import ConfidenceIntervalService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/confidence-interval", tags=["confidence-interval"])


@router.post("/compute", response_model=ConfidenceIntervalResponse)
def compute_confidence_intervals(
    req: ConfidenceIntervalRequest,
) -> ConfidenceIntervalResponse:
    """Compute confidence intervals for quantitative survey questions.

    Supports: rating, nps, yes_no, multiple_choice, checkbox, ranking.
    Non-quantitative question types are silently skipped.

    Uses Wilson score intervals for proportions (yes_no, multiple_choice,
    checkbox) and t-distribution intervals for means (rating, nps, ranking).
    """
    svc = ConfidenceIntervalService()

    try:
        # Convert Pydantic models to dicts for the service
        question_dicts = [q.model_dump() for q in req.questions]
        results = svc.compute_all(question_dicts, req.confidence_level)
    except Exception as e:
        logger.exception("Confidence interval computation failed")
        raise HTTPException(
            status_code=500,
            detail=f"Confidence interval computation failed: {e}",
        )

    return ConfidenceIntervalResponse(
        tenant_id=req.tenant_id,
        survey_id=req.survey_id,
        confidence_level=req.confidence_level,
        results=results,
    )
