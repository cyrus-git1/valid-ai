"""Pydantic models for the /confidence-interval router.

Only quantitative survey question types are supported:
  rating, nps, yes_no, multiple_choice, checkbox, ranking
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

# Quantitative question types that produce data suitable for CI calculation.
# Sourced from the canonical list in src/prompts/survey_prompts.QUESTION_TYPE_PROMPTS.
QUANTITATIVE_QUESTION_TYPES = [
    "multiple_choice",
    "checkbox",
    "rating",
    "yes_no",
    "nps",
    "ranking",
]

MIN_SAMPLE_SIZE = 3  # Hard floor — CI is undefined for n<2, unreliable for n<3


# ── Input models ─────────────────────────────────────────────────────────────


class QuestionResponses(BaseModel):
    """Response data for a single survey question."""

    question_id: str = Field(description="ID of the survey question.")
    question_type: str = Field(
        description=(
            "Question type — must be one of: "
            "multiple_choice, checkbox, rating, yes_no, nps, ranking."
        ),
    )
    label: str = Field(default="", description="Question text (for context in output).")

    # For rating / nps: list of numeric values e.g. [4, 5, 3, 2, 5]
    # For yes_no: list of booleans or strings e.g. [true, false, true]
    # For multiple_choice: list of selected option strings e.g. ["Option A", "Option B", "Option A"]
    # For checkbox: list of lists e.g. [["A","B"], ["B","C"], ["A"]]
    # For ranking: list of ordered lists e.g. [["A","B","C"], ["B","A","C"]]
    responses: List[Any] = Field(
        ..., description="Raw response values from respondents."
    )

    # Optional: provide the full set of options/items for proportion calculations
    options: Optional[List[str]] = Field(
        default=None,
        description="All possible options (multiple_choice, checkbox) or items (ranking).",
    )


class ConfidenceIntervalRequest(BaseModel):
    tenant_id: UUID
    survey_id: UUID
    questions: List[QuestionResponses] = Field(
        ..., min_length=1, description="Response data per question."
    )
    confidence_level: float = Field(
        default=0.95, ge=0.50, le=0.99,
        description="Confidence level (e.g. 0.95 for 95% CI).",
    )


# ── Output models ────────────────────────────────────────────────────────────


class MeanCI(BaseModel):
    """Confidence interval for a numeric mean (rating, nps)."""
    mean: float
    ci_lower: float
    ci_upper: float
    std_dev: float
    n: int


class ProportionCI(BaseModel):
    """Confidence interval for a single proportion (one option)."""
    option: str
    count: int
    proportion: float
    ci_lower: float
    ci_upper: float
    n: int


class RankCI(BaseModel):
    """Confidence interval for the mean rank of one item."""
    item: str
    mean_rank: float
    ci_lower: float
    ci_upper: float
    n: int


class QuestionCI(BaseModel):
    """Confidence interval result for a single question."""
    question_id: str
    question_type: str
    label: str = ""
    n: int = Field(description="Number of responses used.")

    # Populated depending on type
    mean_ci: Optional[MeanCI] = Field(
        default=None, description="For rating / nps questions."
    )
    proportion_cis: Optional[List[ProportionCI]] = Field(
        default=None, description="For multiple_choice / checkbox / yes_no questions."
    )
    rank_cis: Optional[List[RankCI]] = Field(
        default=None, description="For ranking questions."
    )

    warning: Optional[str] = Field(
        default=None, description="Sample-size or data-quality warning."
    )


class ConfidenceIntervalResponse(BaseModel):
    tenant_id: UUID
    survey_id: UUID
    confidence_level: float
    results: List[QuestionCI]
    status: str = "complete"
    error: Optional[str] = None
