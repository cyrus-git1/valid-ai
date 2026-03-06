"""Pydantic models for the /sentiment-analysis router."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from src.models.base import TenantScoped, TenantScopedRequest


# ── Shared sub-models ────────────────────────────────────────────────────────


class SentimentScore(BaseModel):
    """Overall sentiment distribution across transcript content."""
    positive: float = Field(description="Fraction of content with positive sentiment (0.0-1.0).")
    negative: float = Field(description="Fraction of content with negative sentiment (0.0-1.0).")
    neutral: float = Field(description="Fraction of content with neutral sentiment (0.0-1.0).")


class ThemeSentiment(BaseModel):
    """A key theme extracted from transcripts with per-theme sentiment."""
    theme: str = Field(description="Short label for the theme (e.g. 'Product Quality').")
    sentiment: str = Field(description="Dominant sentiment: positive | negative | neutral | mixed.")
    confidence: float = Field(description="Confidence in the sentiment assignment (0.0-1.0).")
    description: str = Field(
        description="1-2 sentence explanation of the theme and why it has this sentiment."
    )


class NotableQuote(BaseModel):
    """A direct quote from transcript content that illustrates a sentiment."""
    quote: str = Field(description="The verbatim excerpt from the transcript.")
    sentiment: str = Field(description="positive | negative | neutral.")
    theme: Optional[str] = Field(
        default=None, description="Which theme this quote relates to, if any."
    )


class SentimentAnalysisResult(TenantScoped):
    """Core sentiment analysis output — shared by single, batch, and all responses."""

    overall_sentiment: SentimentScore
    dominant_sentiment: str = Field(
        description="The single dominant sentiment: positive | negative | neutral."
    )
    themes: List[ThemeSentiment] = Field(default_factory=list)
    notable_quotes: List[NotableQuote] = Field(default_factory=list)
    summary: str = Field(description="A paragraph summarizing the overall sentiment landscape.")

    transcript_count: int = Field(
        default=0, description="Number of VTT documents analysed."
    )
    chunks_analysed: int = Field(
        default=0, description="Number of transcript chunks fed to the LLM."
    )
    generated_at: Optional[datetime] = None


# ── Single ────────────────────────────────────────────────────────────────────


class SentimentAnalysisResponse(BaseModel):
    """Response from the single generate endpoint."""
    tenant_id: UUID
    survey_id: UUID
    status: str = "complete"
    error: Optional[str] = None

    overall_sentiment: SentimentScore
    dominant_sentiment: str
    themes: List[ThemeSentiment] = Field(default_factory=list)
    notable_quotes: List[NotableQuote] = Field(default_factory=list)
    summary: str

    transcript_count: int = 0
    chunks_analysed: int = 0
    generated_at: Optional[datetime] = None


# ── Batch (multiple focus queries, same tenant+client) ────────────────────────


class BatchSentimentRequest(TenantScopedRequest):
    """Request body for POST /sentiment-analysis/generate/batch."""
    focus_queries: List[str] = Field(
        min_length=1, description="List of focus areas to analyse sentiment for (1-10)."
    )
    llm_model: str = Field(default="gpt-4o-mini")
    chunk_limit: int = Field(default=50)


class BatchSentimentResponse(TenantScoped):
    """Response from the batch generate endpoint."""
    total: int
    completed: int
    failed: int = 0
    results: List[SentimentAnalysisResult] = Field(default_factory=list)
    errors: List[Dict[str, str]] = Field(default_factory=list)


# ── All (same query across all clients) ───────────────────────────────────────


class AllSentimentRequest(BaseModel):
    """Request body for POST /sentiment-analysis/generate/all."""
    tenant_id: UUID
    focus_query: Optional[str] = Field(
        default=None, description="Optional focus area applied to every client."
    )
    client_profile: Optional[Dict[str, Any]] = Field(default=None)
    llm_model: str = Field(default="gpt-4o-mini")
    chunk_limit: int = Field(default=50)


class AllSentimentResponse(BaseModel):
    """Response from the all-clients generate endpoint."""
    tenant_id: UUID
    focus_query: Optional[str] = None
    total_clients: int
    completed: int
    failed: int = 0
    results: List[SentimentAnalysisResult] = Field(default_factory=list)
    errors: List[Dict[str, str]] = Field(default_factory=list)
