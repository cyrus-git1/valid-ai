"""Pydantic models for the /survey router."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from src.models.base import StatusResponse, TenantScoped, TenantScopedRequest
from src.prompts.survey_prompts import ALL_QUESTION_TYPES


# ── Shared question models ───────────────────────────────────────────────────


class CardSortItem(BaseModel):
    id: str
    label: str


class SurveyQuestionItem(BaseModel):
    id: str
    type: str
    label: str
    required: bool = False
    # multiple_choice / checkbox
    options: Optional[List[str]] = None
    # rating
    min: Optional[int] = None
    max: Optional[int] = None
    lowLabel: Optional[str] = None
    highLabel: Optional[str] = None
    # ranking (list of strings) / card_sort (list of CardSortItem)
    items: Optional[List[Any]] = None
    # card_sort
    categories: Optional[List[CardSortItem]] = None


# ── Generate survey ──────────────────────────────────────────────────────────


class SurveyGenerateRequest(TenantScopedRequest):
    request: str = Field(..., description="What kind of survey to generate")
    question_types: List[str] = Field(
        default=ALL_QUESTION_TYPES,
        description="Question types to generate",
    )


class SurveyGenerateResponse(StatusResponse):
    questions: List[SurveyQuestionItem]
    context_used: int = 0

    model_config = {"json_schema_serialization_defaults_required": True}

    def model_dump(self, **kwargs):
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)


# ── Generate question recommendation ─────────────────────────────────────────


class GenerateQuestionRequest(TenantScopedRequest):
    request: str = Field(..., description="Original survey description / goal")
    existing_questions: List[SurveyQuestionItem] = Field(
        ..., description="Questions already created in this survey"
    )
    question_types: List[str] = Field(
        default=ALL_QUESTION_TYPES,
        description="Allowed question types for recommendations",
    )
    count: int = Field(default=3, ge=1, le=10, description="Number of recommendations")


class GenerateQuestionResponse(StatusResponse):
    recommendations: List[SurveyQuestionItem]
    reasoning: str = Field(
        default="", description="Explanation of why these questions are recommended"
    )


# ── Generate follow-up survey ────────────────────────────────────────────────


class CompletedSurveyQuestion(BaseModel):
    """A question from a completed survey, optionally including aggregated response data."""
    id: str
    type: str
    label: str
    options: Optional[List[str]] = None
    items: Optional[List[Any]] = None
    categories: Optional[List[CardSortItem]] = None
    response_summary: Optional[str] = Field(
        default=None,
        description="Aggregated/summary of responses for this question (e.g. '60% said Yes')",
    )


class GenerateFollowUpRequest(TenantScopedRequest):
    original_request: str = Field(..., description="Original survey goal / description")
    completed_questions: List[CompletedSurveyQuestion] = Field(
        ..., description="Questions from the completed survey with optional response summaries"
    )
    question_types: List[str] = Field(
        default=ALL_QUESTION_TYPES,
        description="Allowed question types for follow-up",
    )
    count: int = Field(default=5, ge=1, le=15, description="Number of follow-up questions")


class GenerateFollowUpResponse(StatusResponse):
    questions: List[SurveyQuestionItem]
    reasoning: str = Field(
        default="", description="Explanation of how follow-up questions build on the original survey"
    )


# ── Survey output storage ────────────────────────────────────────────────────


class SurveyOutputRow(TenantScoped):
    """A persisted survey output record."""
    id: UUID
    output_type: str = Field(
        description="Type of generation: 'survey', 'recommendation', or 'follow_up'"
    )
    request: str
    questions: List[Dict[str, Any]]
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    expires_at: datetime


class SurveyOutputListResponse(BaseModel):
    outputs: List[SurveyOutputRow]
    count: int
