"""
src/workflows/survey_workflow.py
----------------------------------
LangGraph survey generation workflow: request → retrieve → grade → build → analyze → generate → validate.

Implements confidence-gated context retrieval with automatic retry, then generates
a survey matching the flat-array output schema.

Usage
-----
    from src.workflows.survey_workflow import build_survey_graph

    app = build_survey_graph()
    result = app.invoke({
        "request": "Create a customer satisfaction survey",
        "tenant_id": "...",
        "client_id": "...",
        "client_profile": {...},
        "question_types": ["multiple_choice"],
    })
    print(result["survey"])  # JSON array string
"""
from __future__ import annotations

import json
import logging
import os
import re
import uuid
from typing import Any, Dict, List, Optional, TypedDict
from uuid import UUID

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from src.prompts.survey_prompts import (
    CONTEXT_ANALYSIS_PROMPT,
    SURVEY_GENERATION_PROMPT,
    get_question_type_instructions,
)
from src.services.search_service import SearchService

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.60


# ── State ────────────────────────────────────────────────────────────────────

class SurveyState(TypedDict, total=False):
    request: str
    tenant_id: str
    client_id: str
    client_profile: Dict[str, Any]
    question_types: List[str]

    # Populated by nodes
    documents: List[Document]
    context: str
    tenant_profile: str         # formatted tenant profile string
    context_analysis: str       # LLM-generated insights from context + profile
    profile_section: str
    raw_output: str
    survey: str              # final JSON string output
    context_used: int
    confidence: float        # top similarity score from retrieval
    attempt: int
    error: Optional[str]
    status: str


# ── Nodes ────────────────────────────────────────────────────────────────────

def retrieve_context(state: SurveyState) -> SurveyState:
    """Retrieve context from KG via graph-expanded search."""
    attempt = state.get("attempt", 0) + 1
    top_k = 10 if attempt == 1 else 15
    hop_limit = 1 if attempt == 1 else 2

    svc = SearchService(
        tenant_id=UUID(state["tenant_id"]),
        client_id=UUID(state["client_id"]),
    )

    docs = svc.graph_search(state["request"], top_k=top_k, hop_limit=hop_limit)

    top_sim = 0.0
    if docs:
        top_sim = docs[0].metadata.get("similarity_score", 0.0)

    return {
        **state,
        "documents": docs,
        "confidence": top_sim,
        "context_used": len(docs),
        "attempt": attempt,
        "status": "retrieving",
    }


def grade_context(state: SurveyState) -> SurveyState:
    """Grade retrieval quality for routing."""
    return state


def build_prompt(state: SurveyState) -> SurveyState:
    """Build context string and tenant profile for the analysis step."""
    docs = state.get("documents", [])
    context = ""
    if docs:
        context = "\n\n---\n\n".join(
            f"[Source {i + 1}]\n{doc.page_content}"
            for i, doc in enumerate(docs)
            if doc.page_content.strip()
        )

    context_section = ""
    if context:
        context_section = f"\n\n{context}"

    # Build full tenant profile from client_profile
    client_profile = state.get("client_profile", {})
    profile_parts = []
    if client_profile.get("industry"):
        profile_parts.append(f"Industry: {client_profile['industry']}")
    if client_profile.get("headcount"):
        profile_parts.append(f"Headcount: {client_profile['headcount']} employees")
    if client_profile.get("revenue"):
        profile_parts.append(f"Revenue: {client_profile['revenue']}")
    if client_profile.get("company_name"):
        profile_parts.append(f"Company: {client_profile['company_name']}")
    if client_profile.get("persona"):
        profile_parts.append(f"Target persona: {client_profile['persona']}")
    demo = client_profile.get("demographic", {})
    if demo.get("age_range"):
        profile_parts.append(f"Respondent age range: {demo['age_range']}")
    if demo.get("income_bracket"):
        profile_parts.append(f"Income bracket: {demo['income_bracket']}")
    if demo.get("occupation"):
        profile_parts.append(f"Respondent occupation: {demo['occupation']}")
    if demo.get("location"):
        profile_parts.append(f"Location: {demo['location']}")
    if demo.get("language") and demo["language"] != "en":
        profile_parts.append(f"Survey language: {demo['language']}")

    tenant_profile = "\n".join(profile_parts) if profile_parts else "No profile provided."
    profile_section = (
        f"\n\nOrganization profile:\n{tenant_profile}" if profile_parts else ""
    )

    return {
        **state,
        "context": context_section,
        "tenant_profile": tenant_profile,
        "profile_section": profile_section,
        "status": "analyzing",
    }


def analyze_context(state: SurveyState) -> SurveyState:
    """Use LLM to extract survey-relevant insights from KG context + tenant profile."""
    context = state.get("context", "")
    tenant_profile = state.get("tenant_profile", "No profile provided.")

    # If there's nothing to analyze, skip with a minimal analysis
    if not context.strip() and tenant_profile == "No profile provided.":
        return {
            **state,
            "context_analysis": "No context or profile available. Generate general-purpose survey questions.",
            "status": "generating",
        }

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    chain = CONTEXT_ANALYSIS_PROMPT | llm | StrOutputParser()

    try:
        analysis = chain.invoke({
            "tenant_profile": tenant_profile,
            "request": state["request"],
            "context": context if context.strip() else "No knowledge base context available.",
        })
    except Exception as e:
        logger.exception("Context analysis failed")
        analysis = f"Analysis unavailable: {e}. Proceed with general survey design."

    logger.info("Context analysis completed (%d chars) for request: %r", len(analysis), state["request"][:80])

    return {
        **state,
        "context_analysis": analysis,
        "status": "generating",
    }


def generate_survey(state: SurveyState) -> SurveyState:
    """Generate survey questions via LLM."""
    question_types = state.get("question_types", ["multiple_choice"])
    question_type_instructions = get_question_type_instructions(question_types)

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    chain = SURVEY_GENERATION_PROMPT | llm | StrOutputParser()

    try:
        raw_output = chain.invoke({
            "request": state["request"],
            "context_analysis": state.get("context_analysis", ""),
            "context_section": state.get("context", ""),
            "profile_section": state.get("profile_section", ""),
            "question_type_instructions": question_type_instructions,
        })
    except Exception as e:
        logger.exception("Survey generation failed")
        return {**state, "error": str(e), "status": "failed"}

    return {**state, "raw_output": raw_output}


def validate_output(state: SurveyState) -> SurveyState:
    """Parse and validate the LLM output into the required flat-array schema."""
    raw = state.get("raw_output", "")

    # Try direct JSON parse
    survey_data = None
    try:
        survey_data = json.loads(raw)
    except json.JSONDecodeError:
        # Try extracting from markdown code block
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if json_match:
            try:
                survey_data = json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

    if survey_data is None:
        return {**state, "error": "Could not parse JSON from LLM output", "status": "parse_error"}

    # Unwrap if LLM returned {"questions": [...]} instead of flat array
    if isinstance(survey_data, dict) and "questions" in survey_data:
        survey_data = survey_data["questions"]

    if not isinstance(survey_data, list):
        return {**state, "error": "Survey output is not a JSON array", "status": "parse_error"}

    # Normalize each question to the required schema
    normalized = []
    for q in survey_data:
        normalized.append({
            "id": q.get("id") if _is_valid_uuid(q.get("id")) else str(uuid.uuid4()),
            "type": q.get("type", "multiple_choice"),
            "label": q.get("label") or q.get("text", ""),
            "options": q.get("options", []),
            "required": bool(q.get("required", False)),
        })

    return {
        **state,
        "survey": json.dumps(normalized, indent=2),
        "status": "complete",
    }


def fallback_output(state: SurveyState) -> SurveyState:
    """Handle unparseable LLM output."""
    logger.error("Survey output parse failed: %s", state.get("error"))
    return {
        **state,
        "survey": json.dumps([]),
        "status": "failed",
    }


# ── Helpers ──────────────────────────────────────────────────────────────────

def _is_valid_uuid(val: Any) -> bool:
    if not isinstance(val, str):
        return False
    try:
        uuid.UUID(val, version=4)
        return True
    except ValueError:
        return False


# ── Routing ──────────────────────────────────────────────────────────────────

def route_on_context_confidence(state: SurveyState) -> str:
    """Route based on retrieval confidence. Proceeds to generation after one retry."""
    confidence = state.get("confidence", 0.0)
    attempt = state.get("attempt", 1)

    if confidence < CONFIDENCE_THRESHOLD and attempt < 2:
        return "retrieve_context"  # retry with broader search
    return "build_prompt"          # proceed regardless after retry


def route_on_validation(state: SurveyState) -> str:
    """Route based on output validation result."""
    if state.get("status") == "parse_error":
        return "fallback_output"
    return END


# ── Graph ────────────────────────────────────────────────────────────────────

def build_survey_graph():
    """Build and compile the survey generation LangGraph."""
    graph = StateGraph(SurveyState)

    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("grade_context", grade_context)
    graph.add_node("build_prompt", build_prompt)
    graph.add_node("analyze_context", analyze_context)
    graph.add_node("generate_survey", generate_survey)
    graph.add_node("validate_output", validate_output)
    graph.add_node("fallback_output", fallback_output)

    graph.set_entry_point("retrieve_context")

    graph.add_edge("retrieve_context", "grade_context")
    graph.add_conditional_edges("grade_context", route_on_context_confidence)
    graph.add_edge("build_prompt", "analyze_context")
    graph.add_edge("analyze_context", "generate_survey")
    graph.add_edge("generate_survey", "validate_output")
    graph.add_conditional_edges("validate_output", route_on_validation)
    graph.add_edge("fallback_output", END)

    return graph.compile()
