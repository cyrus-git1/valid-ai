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
import re
import uuid
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph

from src.config.llm import get_llm
from src.models.workflow import SurveyState
from src.prompts.survey_prompts import (
    ALL_QUESTION_TYPES,
    CONTEXT_ANALYSIS_PROMPT,
    FOLLOW_UP_SURVEY_PROMPT,
    QUESTION_RECOMMENDATION_PROMPT,
    SURVEY_DESCRIPTION_PROMPT,
    SURVEY_GENERATION_PROMPT,
    SURVEY_TITLE_PROMPT,
    get_question_type_instructions,
)
from src.services.search_service import SearchService
from src.supabase.supabase_client import get_supabase

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.60


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

    # Fetch prior questions from survey_outputs for this tenant+client
    prior_questions_section = ""
    try:
        sb = get_supabase()
        prior_res = (
            sb.table("survey_outputs")
            .select("questions")
            .eq("tenant_id", state["tenant_id"])
            .eq("client_id", state["client_id"])
            .eq("output_type", "survey")
            .order("created_at", desc=True)
            .limit(5)
            .execute()
        )
        if prior_res.data:
            # Flatten all questions, deduplicate by label, cap at 30
            seen_labels: set[str] = set()
            prior_list: list[dict] = []
            for row in prior_res.data:
                questions = row.get("questions") or []
                if isinstance(questions, str):
                    try:
                        questions = json.loads(questions)
                    except json.JSONDecodeError:
                        continue
                for q in questions:
                    label = (q.get("label") or "").strip().lower()
                    if label and label not in seen_labels:
                        seen_labels.add(label)
                        prior_list.append(q)
                    if len(prior_list) >= 30:
                        break
                if len(prior_list) >= 30:
                    break

            if prior_list:
                formatted = "\n".join(
                    f"- [{q.get('type', 'unknown')}] {q.get('label', '')}"
                    for q in prior_list
                )
                prior_questions_section = (
                    f"\n\nPreviously generated questions for this client:\n{formatted}"
                )
    except Exception as e:
        logger.warning("Failed to fetch prior survey questions: %s", e)

    # Build title/description section if either was provided
    td_parts = []
    title = state.get("title", "")
    description = state.get("description", "")
    if title and title.strip():
        td_parts.append(f"Survey title: {title.strip()}")
    if description and description.strip():
        td_parts.append(f"Survey description: {description.strip()}")
    title_description_section = ""
    if td_parts:
        title_description_section = (
            "\n\nThe following title and/or description have been provided for this survey. "
            "Use them to guide the tone, scope, and focus of the questions you generate:\n"
            + "\n".join(td_parts)
        )

    return {
        **state,
        "context": context_section,
        "tenant_profile": tenant_profile,
        "profile_section": profile_section,
        "prior_questions": prior_questions_section,
        "title_description_section": title_description_section,
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

    llm = get_llm("context_analysis")

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
    question_types = state.get("question_types", ALL_QUESTION_TYPES)
    question_type_instructions = get_question_type_instructions(question_types)

    llm = get_llm("survey_generation")

    chain = SURVEY_GENERATION_PROMPT | llm | StrOutputParser()

    try:
        raw_output = chain.invoke({
            "request": state["request"],
            "context_analysis": state.get("context_analysis", ""),
            "context_section": state.get("context", ""),
            "profile_section": state.get("profile_section", ""),
            "question_type_instructions": question_type_instructions,
            "prior_questions_section": state.get("prior_questions", ""),
            "title_description_section": state.get("title_description_section", ""),
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

    # Normalize each question to the required schema per type
    normalized = []
    for q in survey_data:
        qtype = q.get("type", "multiple_choice")
        base = {
            "id": q.get("id") if _is_valid_uuid(q.get("id")) else str(uuid.uuid4()),
            "type": qtype,
            "label": q.get("label") or q.get("text", ""),
            "required": bool(q.get("required", False)),
        }

        if qtype in ("multiple_choice", "checkbox"):
            base["options"] = q.get("options", [])

        elif qtype == "rating":
            base["min"] = q.get("min", 1)
            base["max"] = q.get("max", 5)
            base["lowLabel"] = q.get("lowLabel", "Poor")
            base["highLabel"] = q.get("highLabel", "Excellent")

        elif qtype == "ranking":
            base["items"] = q.get("items", [])

        elif qtype == "card_sort":
            # Ensure every item and category has a valid UUID id
            items = q.get("items", [])
            base["items"] = [
                {
                    "id": item.get("id") if _is_valid_uuid(item.get("id")) else str(uuid.uuid4()),
                    "label": item.get("label", ""),
                }
                for item in items
            ]
            categories = q.get("categories", [])
            base["categories"] = [
                {
                    "id": cat.get("id") if _is_valid_uuid(cat.get("id")) else str(uuid.uuid4()),
                    "label": cat.get("label", ""),
                }
                for cat in categories
            ]

        # short_text, long_text, yes_no, nps — no extra fields needed

        normalized.append(base)

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
    """Route based on output validation result. (Legacy — kept for reference.)"""
    if state.get("status") == "parse_error":
        return "fallback_output"
    return END


# ── Standalone generation functions ──────────────────────────────────────────


def generate_question(
    *,
    request: str,
    existing_questions: List[Dict[str, Any]],
    tenant_id: str,
    client_id: str,
    client_profile: Dict[str, Any] | None = None,
    question_types: List[str] | None = None,
    count: int = 3,
) -> Dict[str, Any]:
    """Generate question recommendations based on already-created survey questions.

    Retrieves KG context, analyses it, then asks the LLM to recommend new
    questions that complement the ones already in the survey.

    Returns dict with keys: recommendations (list), reasoning (str), status, error.
    """
    question_types = question_types or ALL_QUESTION_TYPES

    # ── retrieve context ──
    svc = SearchService(tenant_id=UUID(tenant_id), client_id=UUID(client_id))
    docs = svc.graph_search(request, top_k=10, hop_limit=1)
    context = ""
    if docs:
        context = "\n\n---\n\n".join(
            f"[Source {i + 1}]\n{doc.page_content}"
            for i, doc in enumerate(docs)
            if doc.page_content.strip()
        )

    # ── build profile ──
    profile_section = _build_profile_section(client_profile or {})

    # ── analyse context ──
    context_analysis = _run_context_analysis(
        request=request,
        context=context,
        client_profile=client_profile or {},
    )

    # ── format existing questions for prompt ──
    existing_text = json.dumps(existing_questions, indent=2) if existing_questions else "[]"

    # ── generate recommendations ──
    llm = get_llm("question_rec")
    chain = QUESTION_RECOMMENDATION_PROMPT | llm | StrOutputParser()

    try:
        raw = chain.invoke({
            "request": request,
            "count": str(count),
            "existing_questions_text": existing_text,
            "context_analysis": context_analysis,
            "context_section": f"\n\n{context}" if context else "",
            "profile_section": profile_section,
            "question_type_instructions": get_question_type_instructions(question_types),
        })
    except Exception as e:
        logger.exception("Question recommendation failed")
        return {"recommendations": [], "reasoning": "", "status": "failed", "error": str(e)}

    return _parse_recommendation_output(raw)


def generate_follow_up_survey(
    *,
    original_request: str,
    completed_questions: List[Dict[str, Any]],
    tenant_id: str,
    client_id: str,
    client_profile: Dict[str, Any] | None = None,
    question_types: List[str] | None = None,
    count: int = 5,
) -> Dict[str, Any]:
    """Generate follow-up survey questions from a completed survey.

    Takes the original survey questions (with optional response summaries) and
    produces a new set of questions that probe deeper into the findings.

    Returns dict with keys: questions (list), reasoning (str), status, error.
    """
    question_types = question_types or ALL_QUESTION_TYPES

    # ── retrieve context ──
    svc = SearchService(tenant_id=UUID(tenant_id), client_id=UUID(client_id))
    docs = svc.graph_search(original_request, top_k=10, hop_limit=1)
    context = ""
    if docs:
        context = "\n\n---\n\n".join(
            f"[Source {i + 1}]\n{doc.page_content}"
            for i, doc in enumerate(docs)
            if doc.page_content.strip()
        )

    # ── build profile ──
    profile_section = _build_profile_section(client_profile or {})

    # ── analyse context ──
    context_analysis = _run_context_analysis(
        request=original_request,
        context=context,
        client_profile=client_profile or {},
    )

    # ── format completed survey for prompt ──
    completed_text = _format_completed_survey(completed_questions)

    # ── generate follow-up ──
    llm = get_llm("follow_up")
    chain = FOLLOW_UP_SURVEY_PROMPT | llm | StrOutputParser()

    try:
        raw = chain.invoke({
            "request": original_request,
            "count": str(count),
            "completed_survey_text": completed_text,
            "context_analysis": context_analysis,
            "context_section": f"\n\n{context}" if context else "",
            "profile_section": profile_section,
            "question_type_instructions": get_question_type_instructions(question_types),
        })
    except Exception as e:
        logger.exception("Follow-up survey generation failed")
        return {"questions": [], "reasoning": "", "status": "failed", "error": str(e)}

    return _parse_recommendation_output(raw, key="questions")


# ── Shared helpers for new functions ─────────────────────────────────────────


def _build_profile_section(client_profile: Dict[str, Any]) -> str:
    """Build the profile section string from a client_profile dict."""
    parts = []
    if client_profile.get("industry"):
        parts.append(f"Industry: {client_profile['industry']}")
    if client_profile.get("headcount"):
        parts.append(f"Headcount: {client_profile['headcount']} employees")
    if client_profile.get("revenue"):
        parts.append(f"Revenue: {client_profile['revenue']}")
    if client_profile.get("company_name"):
        parts.append(f"Company: {client_profile['company_name']}")
    if client_profile.get("persona"):
        parts.append(f"Target persona: {client_profile['persona']}")
    demo = client_profile.get("demographic", {})
    if demo.get("age_range"):
        parts.append(f"Respondent age range: {demo['age_range']}")
    if demo.get("income_bracket"):
        parts.append(f"Income bracket: {demo['income_bracket']}")
    if demo.get("occupation"):
        parts.append(f"Respondent occupation: {demo['occupation']}")
    if demo.get("location"):
        parts.append(f"Location: {demo['location']}")
    if demo.get("language") and demo["language"] != "en":
        parts.append(f"Survey language: {demo['language']}")

    if parts:
        return f"\n\nOrganization profile:\n" + "\n".join(parts)
    return ""


def _build_questions_section(questions: List[Dict[str, Any]] | None) -> str:
    """Build a formatted questions section for title/description prompts."""
    if not questions:
        return ""
    formatted = "\n".join(
        f"- [{q.get('type', 'unknown')}] {q.get('label', '')}"
        for q in questions
    )
    return f"\n\nExisting survey questions:\n{formatted}"


def _run_context_analysis(
    request: str,
    context: str,
    client_profile: Dict[str, Any],
) -> str:
    """Run the LLM context-analysis step shared by both new functions."""
    parts = []
    for k, v in client_profile.items():
        if k != "demographic" and v:
            parts.append(f"{k}: {v}")
    tenant_profile = "\n".join(parts) if parts else "No profile provided."

    if not context.strip() and tenant_profile == "No profile provided.":
        return "No context or profile available. Generate general-purpose survey questions."

    llm = get_llm("context_analysis")
    chain = CONTEXT_ANALYSIS_PROMPT | llm | StrOutputParser()

    try:
        return chain.invoke({
            "tenant_profile": tenant_profile,
            "request": request,
            "context": context if context.strip() else "No knowledge base context available.",
        })
    except Exception as e:
        logger.exception("Context analysis failed")
        return f"Analysis unavailable: {e}. Proceed with general survey design."


def _format_completed_survey(questions: List[Dict[str, Any]]) -> str:
    """Format completed survey questions (with optional response_summary) for the prompt."""
    lines: List[str] = []
    for i, q in enumerate(questions, 1):
        parts = [f"Q{i}. [{q.get('type', 'unknown')}] {q.get('label', '')}"]
        if q.get("options"):
            parts.append(f"   Options: {', '.join(q['options'])}")
        if q.get("response_summary"):
            parts.append(f"   Response summary: {q['response_summary']}")
        lines.append("\n".join(parts))
    return "\n\n".join(lines) if lines else "No questions provided."


def _parse_recommendation_output(
    raw: str,
    key: str = "recommendations",
) -> Dict[str, Any]:
    """Parse LLM output that contains {reasoning, questions/recommendations}."""
    data = None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if match:
            try:
                data = json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

    if data is None:
        return {key: [], "reasoning": "", "status": "parse_error", "error": "Could not parse JSON"}

    reasoning = ""
    questions_raw: list = []

    if isinstance(data, dict):
        reasoning = data.get("reasoning", "")
        # Accept either the expected key or common alternatives
        questions_raw = data.get(key) or data.get("questions") or data.get("recommendations") or []
    elif isinstance(data, list):
        questions_raw = data

    # Normalise questions (assign UUIDs, enforce schema)
    normalized = []
    for q in questions_raw:
        qtype = q.get("type", "multiple_choice")
        base: Dict[str, Any] = {
            "id": q.get("id") if _is_valid_uuid(q.get("id")) else str(uuid.uuid4()),
            "type": qtype,
            "label": q.get("label") or q.get("text", ""),
            "required": bool(q.get("required", False)),
        }
        if qtype in ("multiple_choice", "checkbox"):
            base["options"] = q.get("options", [])
        elif qtype == "rating":
            base["min"] = q.get("min", 1)
            base["max"] = q.get("max", 5)
            base["lowLabel"] = q.get("lowLabel", "Poor")
            base["highLabel"] = q.get("highLabel", "Excellent")
        elif qtype == "ranking":
            base["items"] = q.get("items", [])
        elif qtype == "card_sort":
            items = q.get("items", [])
            base["items"] = [
                {"id": it.get("id") if _is_valid_uuid(it.get("id")) else str(uuid.uuid4()), "label": it.get("label", "")}
                for it in items
            ]
            categories = q.get("categories", [])
            base["categories"] = [
                {"id": cat.get("id") if _is_valid_uuid(cat.get("id")) else str(uuid.uuid4()), "label": cat.get("label", "")}
                for cat in categories
            ]
        normalized.append(base)

    return {key: normalized, "reasoning": reasoning, "status": "complete", "error": None}


# ── Title & description generation ────────────────────────────────────────────


def generate_title(
    *,
    request: str,
    tenant_id: str,
    client_id: str,
    client_profile: Dict[str, Any] | None = None,
    existing_questions: List[Dict[str, Any]] | None = None,
    description: str | None = None,
) -> Dict[str, Any]:
    """Generate a survey title based on business context.

    Retrieves KG context, analyses it, then asks the LLM to produce a concise
    survey title informed by the organization's profile and knowledge base.
    When existing questions or a description are provided they further inform
    the generated title.

    Returns dict with keys: title (str), status (str), error (str | None).
    """
    # ── retrieve context ──
    svc = SearchService(tenant_id=UUID(tenant_id), client_id=UUID(client_id))
    docs = svc.graph_search(request, top_k=10, hop_limit=1)
    context = ""
    if docs:
        context = "\n\n---\n\n".join(
            f"[Source {i + 1}]\n{doc.page_content}"
            for i, doc in enumerate(docs)
            if doc.page_content.strip()
        )

    # ── build profile ──
    profile_section = _build_profile_section(client_profile or {})

    # ── analyse context ──
    context_analysis = _run_context_analysis(
        request=request,
        context=context,
        client_profile=client_profile or {},
    )

    # ── build questions section ──
    questions_section = _build_questions_section(existing_questions)

    # ── build description section ──
    description_section = ""
    if description and description.strip():
        description_section = f"\n\nSurvey description: {description.strip()}"

    # ── generate title ──
    llm = get_llm("survey_title")
    chain = SURVEY_TITLE_PROMPT | llm | StrOutputParser()

    try:
        raw = chain.invoke({
            "request": request,
            "context_analysis": context_analysis,
            "profile_section": profile_section,
            "questions_section": questions_section,
            "description_section": description_section,
        })
    except Exception as e:
        logger.exception("Survey title generation failed")
        return {"title": "", "status": "failed", "error": str(e)}

    # Parse JSON response
    data = _parse_simple_json(raw)
    title = data.get("title", "").strip() if isinstance(data, dict) else ""

    if not title:
        return {"title": "", "status": "parse_error", "error": "Could not extract title from LLM output"}

    return {"title": title, "status": "complete", "error": None}


def generate_description(
    *,
    request: str,
    tenant_id: str,
    client_id: str,
    client_profile: Dict[str, Any] | None = None,
    title: str | None = None,
    existing_questions: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Generate a survey description based on title + context, or context alone.

    If a title is provided, the description complements it. If no title is
    provided, the description is derived solely from the business context.
    When existing questions are provided they further inform the description.

    Returns dict with keys: description (str), status (str), error (str | None).
    """
    # ── retrieve context ──
    svc = SearchService(tenant_id=UUID(tenant_id), client_id=UUID(client_id))
    docs = svc.graph_search(request, top_k=10, hop_limit=1)
    context = ""
    if docs:
        context = "\n\n---\n\n".join(
            f"[Source {i + 1}]\n{doc.page_content}"
            for i, doc in enumerate(docs)
            if doc.page_content.strip()
        )

    # ── build profile ──
    profile_section = _build_profile_section(client_profile or {})

    # ── analyse context ──
    context_analysis = _run_context_analysis(
        request=request,
        context=context,
        client_profile=client_profile or {},
    )

    # ── build title section ──
    title_section = ""
    if title and title.strip():
        title_section = f"Survey title: {title.strip()}\n\n"

    # ── build questions section ──
    questions_section = _build_questions_section(existing_questions)

    # ── generate description ──
    llm = get_llm("survey_description")
    chain = SURVEY_DESCRIPTION_PROMPT | llm | StrOutputParser()

    try:
        raw = chain.invoke({
            "request": request,
            "context_analysis": context_analysis,
            "profile_section": profile_section,
            "title_section": title_section,
            "questions_section": questions_section,
        })
    except Exception as e:
        logger.exception("Survey description generation failed")
        return {"description": "", "status": "failed", "error": str(e)}

    # Parse JSON response
    data = _parse_simple_json(raw)
    description = data.get("description", "").strip() if isinstance(data, dict) else ""

    if not description:
        return {"description": "", "status": "parse_error", "error": "Could not extract description from LLM output"}

    return {"description": description, "status": "complete", "error": None}


def generate_title_description_node(state: SurveyState) -> SurveyState:
    """Generate title and description from the completed survey questions.

    Runs after validate_output — uses the generated questions, context analysis,
    and profile to produce a title and description for the survey.
    """
    survey_json = state.get("survey", "[]")
    try:
        questions = json.loads(survey_json)
    except json.JSONDecodeError:
        questions = []

    request = state["request"]
    context_analysis = state.get("context_analysis", "")
    profile_section = state.get("profile_section", "")

    questions_section = _build_questions_section(questions)

    # ── generate title ──
    # If user provided a title use it; otherwise generate one
    user_title = state.get("title", "")
    generated_title = user_title if user_title and user_title.strip() else ""

    if not generated_title:
        description_for_title = state.get("description", "")
        description_section = ""
        if description_for_title and description_for_title.strip():
            description_section = f"\n\nSurvey description: {description_for_title.strip()}"

        llm_title = get_llm("survey_title")
        chain_title = SURVEY_TITLE_PROMPT | llm_title | StrOutputParser()
        try:
            raw_title = chain_title.invoke({
                "request": request,
                "context_analysis": context_analysis,
                "profile_section": profile_section,
                "questions_section": questions_section,
                "description_section": description_section,
            })
            data = _parse_simple_json(raw_title)
            generated_title = data.get("title", "").strip() if isinstance(data, dict) else ""
        except Exception as e:
            logger.warning("Title generation in workflow failed: %s", e)

    # ── generate description ──
    user_description = state.get("description", "")
    generated_description = user_description if user_description and user_description.strip() else ""

    if not generated_description:
        title_section = ""
        if generated_title:
            title_section = f"Survey title: {generated_title}\n\n"

        llm_desc = get_llm("survey_description")
        chain_desc = SURVEY_DESCRIPTION_PROMPT | llm_desc | StrOutputParser()
        try:
            raw_desc = chain_desc.invoke({
                "request": request,
                "context_analysis": context_analysis,
                "profile_section": profile_section,
                "title_section": title_section,
                "questions_section": questions_section,
            })
            data = _parse_simple_json(raw_desc)
            generated_description = data.get("description", "").strip() if isinstance(data, dict) else ""
        except Exception as e:
            logger.warning("Description generation in workflow failed: %s", e)

    return {
        **state,
        "generated_title": generated_title,
        "generated_description": generated_description,
    }


def generate_whole_survey(
    *,
    prompt: str,
    question_types: List[str] | None = None,
) -> Dict[str, Any]:
    """Generate a complete survey (title, description, questions) from a prompt alone.

    Skips KG retrieval and context analysis — the LLM generates everything
    directly from the user's free-text prompt.

    Returns dict with keys: title, description, questions (list), status, error.
    """
    question_types = question_types or ALL_QUESTION_TYPES
    question_type_instructions = get_question_type_instructions(question_types)

    # ── generate questions ──
    llm_gen = get_llm("survey_generation")
    chain = SURVEY_GENERATION_PROMPT | llm_gen | StrOutputParser()
    try:
        raw_output = chain.invoke({
            "request": prompt,
            "context_analysis": "",
            "context_section": "",
            "profile_section": "",
            "question_type_instructions": question_type_instructions,
            "prior_questions_section": "",
            "title_description_section": "",
        })
    except Exception as e:
        logger.exception("Whole survey generation failed")
        return {"title": "", "description": "", "questions": [], "status": "failed", "error": str(e)}

    # ── parse and normalize questions ──
    parsed = _parse_simple_json(raw_output)
    questions_raw = parsed if isinstance(parsed, list) else parsed.get("questions", []) if isinstance(parsed, dict) else []

    if not questions_raw:
        # Try direct JSON parse as array
        try:
            questions_raw = json.loads(raw_output)
        except json.JSONDecodeError:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_output)
            if match:
                try:
                    questions_raw = json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass

    if isinstance(questions_raw, dict) and "questions" in questions_raw:
        questions_raw = questions_raw["questions"]

    if not isinstance(questions_raw, list):
        questions_raw = []

    # Normalize questions
    normalized = []
    for q in questions_raw:
        qtype = q.get("type", "multiple_choice")
        base: Dict[str, Any] = {
            "id": q.get("id") if _is_valid_uuid(q.get("id")) else str(uuid.uuid4()),
            "type": qtype,
            "label": q.get("label") or q.get("text", ""),
            "required": bool(q.get("required", False)),
        }
        if qtype in ("multiple_choice", "checkbox"):
            base["options"] = q.get("options", [])
        elif qtype == "rating":
            base["min"] = q.get("min", 1)
            base["max"] = q.get("max", 5)
            base["lowLabel"] = q.get("lowLabel", "Poor")
            base["highLabel"] = q.get("highLabel", "Excellent")
        elif qtype == "ranking":
            base["items"] = q.get("items", [])
        elif qtype == "card_sort":
            items = q.get("items", [])
            base["items"] = [
                {"id": it.get("id") if _is_valid_uuid(it.get("id")) else str(uuid.uuid4()), "label": it.get("label", "")}
                for it in items
            ]
            categories = q.get("categories", [])
            base["categories"] = [
                {"id": cat.get("id") if _is_valid_uuid(cat.get("id")) else str(uuid.uuid4()), "label": cat.get("label", "")}
                for cat in categories
            ]
        normalized.append(base)

    # ── build questions section for title/description ──
    questions_section = _build_questions_section(normalized)

    # ── generate title ──
    title = ""
    llm_title = get_llm("survey_title")
    chain_title = SURVEY_TITLE_PROMPT | llm_title | StrOutputParser()
    try:
        raw_title = chain_title.invoke({
            "request": prompt,
            "context_analysis": "",
            "profile_section": "",
            "questions_section": questions_section,
            "description_section": "",
        })
        data = _parse_simple_json(raw_title)
        title = data.get("title", "").strip() if isinstance(data, dict) else ""
    except Exception as e:
        logger.warning("Title generation in generate_whole failed: %s", e)

    # ── generate description ──
    description = ""
    llm_desc = get_llm("survey_description")
    chain_desc = SURVEY_DESCRIPTION_PROMPT | llm_desc | StrOutputParser()
    try:
        raw_desc = chain_desc.invoke({
            "request": prompt,
            "context_analysis": "",
            "profile_section": "",
            "title_section": f"Survey title: {title}\n\n" if title else "",
            "questions_section": questions_section,
        })
        data = _parse_simple_json(raw_desc)
        description = data.get("description", "").strip() if isinstance(data, dict) else ""
    except Exception as e:
        logger.warning("Description generation in generate_whole failed: %s", e)

    return {
        "title": title,
        "description": description,
        "questions": normalized,
        "status": "complete",
        "error": None,
    }


def _parse_simple_json(raw: str) -> Dict[str, Any]:
    """Parse a simple JSON object from LLM output, with code-block fallback."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    return {}


# ── Graph ────────────────────────────────────────────────────────────────────

def route_on_validation_to_title(state: SurveyState) -> str:
    """Route based on output validation — go to title/description gen or fallback."""
    if state.get("status") == "parse_error":
        return "fallback_output"
    return "generate_title_description"


def build_survey_graph():
    """Build and compile the survey generation LangGraph."""
    graph = StateGraph(SurveyState)

    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("grade_context", grade_context)
    graph.add_node("build_prompt", build_prompt)
    graph.add_node("analyze_context", analyze_context)
    graph.add_node("generate_survey", generate_survey)
    graph.add_node("validate_output", validate_output)
    graph.add_node("generate_title_description", generate_title_description_node)
    graph.add_node("fallback_output", fallback_output)

    graph.set_entry_point("retrieve_context")

    graph.add_edge("retrieve_context", "grade_context")
    graph.add_conditional_edges("grade_context", route_on_context_confidence)
    graph.add_edge("build_prompt", "analyze_context")
    graph.add_edge("analyze_context", "generate_survey")
    graph.add_edge("generate_survey", "validate_output")
    graph.add_conditional_edges("validate_output", route_on_validation_to_title)
    graph.add_edge("generate_title_description", END)
    graph.add_edge("fallback_output", END)

    return graph.compile()
