"""
/survey router
--------------
Dedicated survey generation endpoint. Skips intent classification entirely —
if this endpoint is called, a survey MUST be generated.

POST /survey/generate  — Generate a survey from context + tenant profile
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.workflows.survey_workflow import build_survey_graph

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/survey", tags=["survey"])


# ── Request / Response ────────────────────────────────────────────────────────

class SurveyGenerateRequest(BaseModel):
    tenant_id: UUID
    client_id: UUID
    request: str = Field(..., description="What kind of survey to generate")
    client_profile: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Tenant profile: industry, headcount, revenue, persona, demographic, etc.",
    )
    question_types: List[str] = Field(
        default=["multiple_choice"],
        description="Question types to generate",
    )


class SurveyQuestionItem(BaseModel):
    id: str
    type: str
    label: str
    options: List[str] = Field(default_factory=list)
    required: bool = False


class SurveyGenerateResponse(BaseModel):
    questions: List[SurveyQuestionItem]
    context_used: int = 0
    status: str = "complete"
    error: Optional[str] = None


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/generate", response_model=SurveyGenerateResponse)
def generate_survey(req: SurveyGenerateRequest) -> SurveyGenerateResponse:
    """
    Generate a survey. No intent classification — this endpoint IS the intent.

    Runs the full survey LangGraph workflow:
      retrieve context → analyze (LLM) → generate questions → validate output

    Accepts JSON, returns JSON.
    """
    try:
        graph = build_survey_graph()
        result = graph.invoke({
            "request": req.request,
            "tenant_id": str(req.tenant_id),
            "client_id": str(req.client_id),
            "client_profile": req.client_profile or {},
            "question_types": req.question_types,
        })
    except Exception as e:
        logger.exception("Survey generation failed")
        raise HTTPException(status_code=500, detail=f"Survey generation failed: {e}")

    # Parse the JSON string back to structured response
    status = result.get("status", "unknown")
    error = result.get("error")
    survey_json = result.get("survey", "[]")

    try:
        questions_raw = json.loads(survey_json)
    except json.JSONDecodeError:
        questions_raw = []
        status = "failed"
        error = "Could not parse survey output"

    questions = [
        SurveyQuestionItem(
            id=q.get("id", ""),
            type=q.get("type", "multiple_choice"),
            label=q.get("label", ""),
            options=q.get("options", []),
            required=q.get("required", False),
        )
        for q in questions_raw
    ]

    return SurveyGenerateResponse(
        questions=questions,
        context_used=result.get("context_used", 0),
        status=status,
        error=error,
    )
