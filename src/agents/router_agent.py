"""
src/agents/router_agent.py
----------------------------
Routing agent that classifies user intent with confidence scoring and delegates
to the appropriate sub-agent using a LangGraph StateGraph.

Intent categories
-----------------
  retrieval  — user wants to search or ask a question about ingested content
  survey     — user wants to generate a survey based on context
  ingest     — user wants to add new content (docs/weblinks) to the system

Confidence routing
------------------
  If classification confidence < 0.60, retries once with a retry prompt.
  If still low after retry, returns a clarification request.

Usage
-----
    from src.agents.router_agent import build_router_agent

    agent = build_router_agent()
    result = agent.invoke({
        "input": "Generate a customer satisfaction survey based on our product docs",
        "tenant_id": "...",
        "client_id": "...",
    })
    print(result["output"])
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from src.agents.retrieval_agent import run_retrieval_agent
from src.agents.survey_agent import run_survey_agent
from src.prompts.router_prompts import (
    INTENT_CLASSIFICATION_PROMPT,
    INTENT_CLASSIFICATION_RETRY_PROMPT,
)

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.60


# ── State ────────────────────────────────────────────────────────────────────

class RouterState(TypedDict, total=False):
    input: str
    tenant_id: str
    client_id: str
    client_profile: Dict[str, Any]
    intent: str
    intent_confidence: float
    classification_attempt: int
    output: str
    sources: List[Dict[str, Any]]
    error: Optional[str]


# ── Nodes ────────────────────────────────────────────────────────────────────

def classify_intent(state: RouterState) -> RouterState:
    """Use LLM to classify the user's intent with a confidence score."""
    attempt = state.get("classification_attempt", 0) + 1

    if attempt == 1:
        prompt = INTENT_CLASSIFICATION_PROMPT
        invoke_vars = {"input": state["input"]}
    else:
        prompt = INTENT_CLASSIFICATION_RETRY_PROMPT
        invoke_vars = {
            "input": state["input"],
            "previous_intent": state.get("intent", "unknown"),
            "previous_confidence": str(state.get("intent_confidence", 0.0)),
        }

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    chain = prompt | llm | StrOutputParser()

    try:
        raw = chain.invoke(invoke_vars)
        parsed = json.loads(raw.strip())
        intent = parsed.get("intent", "unknown").lower()
        confidence = float(parsed.get("confidence", 0.0))
    except (json.JSONDecodeError, Exception) as e:
        logger.error("Intent classification failed: %s", e)
        intent = "unknown"
        confidence = 0.0

    # Normalize
    if intent not in ("retrieval", "survey", "ingest", "unknown"):
        intent = "retrieval"  # default to retrieval

    logger.info(
        "Classified intent: %r (confidence=%.2f, attempt=%d) for input: %r",
        intent, confidence, attempt, state["input"][:80],
    )

    return {
        **state,
        "intent": intent,
        "intent_confidence": confidence,
        "classification_attempt": attempt,
    }


def grade_intent(state: RouterState) -> RouterState:
    """Grade the intent classification confidence for routing."""
    return state


def handle_retrieval(state: RouterState) -> RouterState:
    """Delegate to the retrieval agent."""
    try:
        result = run_retrieval_agent(
            query=state["input"],
            tenant_id=state["tenant_id"],
            client_id=state["client_id"],
            client_profile=state.get("client_profile"),
        )
        return {**state, "output": result["answer"], "sources": result.get("sources", [])}
    except Exception as e:
        logger.exception("Retrieval agent failed")
        return {**state, "output": f"Retrieval failed: {e}", "error": str(e)}


def handle_survey(state: RouterState) -> RouterState:
    """Delegate to the survey generation agent."""
    try:
        result = run_survey_agent(
            request=state["input"],
            tenant_id=state["tenant_id"],
            client_id=state["client_id"],
            client_profile=state.get("client_profile"),
        )
        return {**state, "output": result["survey"]}
    except Exception as e:
        logger.exception("Survey agent failed")
        return {**state, "output": f"Survey generation failed: {e}", "error": str(e)}


def handle_ingest(state: RouterState) -> RouterState:
    """For ingest intents, direct to the /ingest or /context endpoints."""
    return {
        **state,
        "output": (
            "To ingest new content, use the /ingest/file or /ingest/web endpoints, "
            "or the /context/build endpoint for a full pipeline run. "
            "I can help you search and generate surveys from existing content."
        ),
    }


def handle_unknown(state: RouterState) -> RouterState:
    """Handle unrecognized intents."""
    return {
        **state,
        "output": (
            "I'm not sure what you're asking. I can:\n"
            "- Search your knowledge base and answer questions\n"
            "- Generate surveys based on your content\n"
            "- Help you understand your ingested documents\n\n"
            "Please rephrase your request."
        ),
    }


def handle_clarification(state: RouterState) -> RouterState:
    """Handle low-confidence classification after retry."""
    return {
        **state,
        "output": (
            "I'm not quite sure what you're looking for. Could you clarify? I can:\n"
            "- Search your knowledge base and answer questions\n"
            "- Generate surveys based on your content\n"
            "- Help you understand your ingested documents\n\n"
            f"(I classified your request as '{state.get('intent', 'unknown')}' "
            f"with {state.get('intent_confidence', 0.0):.0%} confidence)"
        ),
    }


# ── Routing ──────────────────────────────────────────────────────────────────

def route_by_intent(state: RouterState) -> str:
    """Route to the handler matching the classified intent."""
    intent = state.get("intent", "unknown")
    return {
        "retrieval": "handle_retrieval",
        "survey": "handle_survey",
        "ingest": "handle_ingest",
        "unknown": "handle_unknown",
    }.get(intent, "handle_unknown")


def route_on_intent_confidence(state: RouterState) -> str:
    """Route based on classification confidence. Retry once if low."""
    confidence = state.get("intent_confidence", 0.0)
    attempt = state.get("classification_attempt", 1)

    if confidence < CONFIDENCE_THRESHOLD and attempt < 2:
        return "classify_intent"  # retry
    if confidence < CONFIDENCE_THRESHOLD:
        return "handle_clarification"  # fallback after retry
    return route_by_intent(state)


# ── Graph ────────────────────────────────────────────────────────────────────

def build_router_agent():
    """Build and compile the router agent LangGraph."""
    graph = StateGraph(RouterState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("grade_intent", grade_intent)
    graph.add_node("handle_retrieval", handle_retrieval)
    graph.add_node("handle_survey", handle_survey)
    graph.add_node("handle_ingest", handle_ingest)
    graph.add_node("handle_unknown", handle_unknown)
    graph.add_node("handle_clarification", handle_clarification)

    graph.set_entry_point("classify_intent")

    graph.add_edge("classify_intent", "grade_intent")
    graph.add_conditional_edges("grade_intent", route_on_intent_confidence)
    graph.add_edge("handle_retrieval", END)
    graph.add_edge("handle_survey", END)
    graph.add_edge("handle_ingest", END)
    graph.add_edge("handle_unknown", END)
    graph.add_edge("handle_clarification", END)

    return graph.compile()
