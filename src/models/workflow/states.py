"""LangGraph workflow state definitions.

Centralises all TypedDict state classes so that workflow files stay focused
on node / edge logic rather than type declarations.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.documents import Document


# ── RAG workflow state ───────────────────────────────────────────────────────


class RAGState(TypedDict, total=False):
    question: str
    tenant_id: str
    client_id: str
    client_profile: Dict[str, Any]
    documents: List[Document]
    context: str
    answer: str
    confidence: float
    top_similarity: float
    attempt: int
    model: str


# ── Context build workflow state ─────────────────────────────────────────────


class ContextBuildState(TypedDict, total=False):
    tenant_id: str
    client_id: str
    docs: List[str]
    weblinks: List[str]
    transcripts: List[str]
    client_profile: Dict[str, Any]
    ingest_results: List[Dict[str, Any]]
    documents: List[Document]
    status: str
    error: Optional[str]
    warnings: List[str]


# ── Survey workflow state ────────────────────────────────────────────────────


class SurveyState(TypedDict, total=False):
    request: str
    tenant_id: str
    client_id: str
    client_profile: Dict[str, Any]
    question_types: List[str]
    title: str                      # optional user-supplied title to guide generation
    description: str                # optional user-supplied description to guide generation

    # Populated by nodes
    documents: List[Document]
    context: str
    tenant_profile: str             # formatted tenant profile string
    context_analysis: str           # LLM-generated insights from context + profile
    profile_section: str
    prior_questions: str            # formatted prior questions from survey_outputs
    title_description_section: str  # formatted title/description for the prompt
    raw_output: str
    survey: str                     # final JSON string output
    generated_title: str            # LLM-generated title
    generated_description: str      # LLM-generated description
    context_used: int
    confidence: float               # top similarity score from retrieval
    attempt: int
    error: Optional[str]
    status: str
