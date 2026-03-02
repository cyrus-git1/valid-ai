"""
/survey router
--------------
Survey generation, question recommendations, follow-up generation,
and persisted survey output storage.

POST /survey/generate            — Generate a full survey
POST /survey/generate-question   — Recommend new questions for an in-progress survey
POST /survey/generate-follow-up  — Generate follow-up questions from a completed survey
GET  /survey/outputs             — List stored survey outputs for a tenant + client
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from supabase import create_client

from src.models.api.survey import (
    CardSortItem,
    GenerateFollowUpRequest,
    GenerateFollowUpResponse,
    GenerateQuestionRequest,
    GenerateQuestionResponse,
    SurveyGenerateRequest,
    SurveyGenerateResponse,
    SurveyOutputListResponse,
    SurveyOutputRow,
    SurveyQuestionItem,
)
from src.workflows.survey_workflow import (
    build_survey_graph,
    generate_follow_up_survey,
    generate_question,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/survey", tags=["survey"])


# ── Helpers ──────────────────────────────────────────────────────────────────


def _get_supabase():
    return create_client(
        os.environ["SUPABASE_URL"],
        os.environ["SUPABASE_SERVICE_KEY"],
    )


def _save_survey_output(
    tenant_id: UUID,
    client_id: UUID,
    output_type: str,
    request: str,
    questions: List[Dict[str, Any]],
    reasoning: str | None = None,
    metadata: Dict[str, Any] | None = None,
) -> None:
    """Persist a generated survey output to the survey_outputs table."""
    sb = _get_supabase()
    try:
        # Clean expired rows opportunistically
        sb.rpc("cleanup_expired_survey_outputs", {}).execute()
    except Exception:
        logger.debug("Expired survey output cleanup skipped (RPC may not exist yet)")

    sb.table("survey_outputs").insert({
        "tenant_id": str(tenant_id),
        "client_id": str(client_id),
        "output_type": output_type,
        "request": request,
        "questions": questions,
        "reasoning": reasoning,
        "metadata": metadata or {},
    }).execute()


def _parse_questions(questions_raw: List[Dict[str, Any]]) -> List[SurveyQuestionItem]:
    """Convert raw question dicts to SurveyQuestionItem models."""
    items = []
    for q in questions_raw:
        items.append(SurveyQuestionItem(
            id=q.get("id", ""),
            type=q.get("type", "multiple_choice"),
            label=q.get("label", ""),
            required=q.get("required", False),
            options=q.get("options"),
            min=q.get("min"),
            max=q.get("max"),
            lowLabel=q.get("lowLabel"),
            highLabel=q.get("highLabel"),
            items=q.get("items"),
            categories=q.get("categories"),
        ))
    return items


# ── POST /survey/generate ───────────────────────────────────────────────────


@router.post("/generate", response_model=SurveyGenerateResponse)
def survey_generate(req: SurveyGenerateRequest) -> SurveyGenerateResponse:
    """Generate a full survey via the LangGraph workflow.

    Runs: retrieve context → analyze (LLM) → generate questions → validate.
    Persists the output to survey_outputs automatically.
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

    status = result.get("status", "unknown")
    error = result.get("error")
    survey_json = result.get("survey", "[]")

    try:
        questions_raw = json.loads(survey_json)
    except json.JSONDecodeError:
        questions_raw = []
        status = "failed"
        error = "Could not parse survey output"

    questions = _parse_questions(questions_raw)

    # Persist output
    try:
        _save_survey_output(
            tenant_id=req.tenant_id,
            client_id=req.client_id,
            output_type="survey",
            request=req.request,
            questions=questions_raw,
        )
    except Exception:
        logger.warning("Failed to persist survey output", exc_info=True)

    return SurveyGenerateResponse(
        questions=questions,
        context_used=result.get("context_used", 0),
        status=status,
        error=error,
    )


# ── POST /survey/generate-question ──────────────────────────────────────────


@router.post("/generate-question", response_model=GenerateQuestionResponse)
def survey_generate_question(req: GenerateQuestionRequest) -> GenerateQuestionResponse:
    """Recommend new questions based on the questions already in a survey.

    Acts as a recommendation engine — analyses existing questions and suggests
    complementary ones that fill coverage gaps and strengthen the survey.
    """
    existing_dicts = [q.model_dump(exclude_none=True) for q in req.existing_questions]

    try:
        result = generate_question(
            request=req.request,
            existing_questions=existing_dicts,
            tenant_id=str(req.tenant_id),
            client_id=str(req.client_id),
            client_profile=req.client_profile,
            question_types=req.question_types,
            count=req.count,
        )
    except Exception as e:
        logger.exception("Question recommendation failed")
        raise HTTPException(status_code=500, detail=f"Question recommendation failed: {e}")

    recs = _parse_questions(result.get("recommendations", []))

    # Persist output
    try:
        _save_survey_output(
            tenant_id=req.tenant_id,
            client_id=req.client_id,
            output_type="recommendation",
            request=req.request,
            questions=result.get("recommendations", []),
            reasoning=result.get("reasoning"),
        )
    except Exception:
        logger.warning("Failed to persist recommendation output", exc_info=True)

    return GenerateQuestionResponse(
        recommendations=recs,
        reasoning=result.get("reasoning", ""),
        status=result.get("status", "complete"),
        error=result.get("error"),
    )


# ── POST /survey/generate-follow-up ─────────────────────────────────────────


@router.post("/generate-follow-up", response_model=GenerateFollowUpResponse)
def survey_generate_follow_up(req: GenerateFollowUpRequest) -> GenerateFollowUpResponse:
    """Generate follow-up survey questions from a completed survey.

    Takes completed questions (with optional response summaries) and produces
    deeper-dive questions that probe the findings.
    """
    completed_dicts = [q.model_dump(exclude_none=True) for q in req.completed_questions]

    try:
        result = generate_follow_up_survey(
            original_request=req.original_request,
            completed_questions=completed_dicts,
            tenant_id=str(req.tenant_id),
            client_id=str(req.client_id),
            client_profile=req.client_profile,
            question_types=req.question_types,
            count=req.count,
        )
    except Exception as e:
        logger.exception("Follow-up survey generation failed")
        raise HTTPException(status_code=500, detail=f"Follow-up generation failed: {e}")

    questions = _parse_questions(result.get("questions", []))

    # Persist output
    try:
        _save_survey_output(
            tenant_id=req.tenant_id,
            client_id=req.client_id,
            output_type="follow_up",
            request=req.original_request,
            questions=result.get("questions", []),
            reasoning=result.get("reasoning"),
        )
    except Exception:
        logger.warning("Failed to persist follow-up output", exc_info=True)

    return GenerateFollowUpResponse(
        questions=questions,
        reasoning=result.get("reasoning", ""),
        status=result.get("status", "complete"),
        error=result.get("error"),
    )


# ── GET /survey/outputs ─────────────────────────────────────────────────────


@router.get("/outputs", response_model=SurveyOutputListResponse)
def list_survey_outputs(
    tenant_id: UUID = Query(..., description="Tenant ID"),
    client_id: UUID = Query(..., description="Client ID"),
    output_type: Optional[str] = Query(
        default=None, description="Filter by type: survey, recommendation, follow_up"
    ),
) -> SurveyOutputListResponse:
    """List persisted survey outputs for a tenant + client.

    Returns all non-expired outputs ordered by creation time (newest first).
    Expired rows (older than 7 days) are cleaned up opportunistically.
    """
    sb = _get_supabase()

    # Opportunistic cleanup of expired rows
    try:
        sb.rpc("cleanup_expired_survey_outputs", {}).execute()
    except Exception:
        logger.debug("Expired survey output cleanup skipped")

    query = (
        sb.table("survey_outputs")
        .select("*")
        .eq("tenant_id", str(tenant_id))
        .eq("client_id", str(client_id))
        .order("created_at", desc=True)
    )

    if output_type:
        query = query.eq("output_type", output_type)

    resp = query.execute()
    rows = resp.data or []

    outputs = [SurveyOutputRow(**row) for row in rows]
    return SurveyOutputListResponse(outputs=outputs, count=len(outputs))
