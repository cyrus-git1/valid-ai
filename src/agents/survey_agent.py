"""
src/agents/survey_agent.py
----------------------------
Survey generation sub-agent — thin wrapper around the survey_workflow LangGraph.

Delegates all logic to src/workflows/survey_workflow.py while preserving the
original function signature for backward compatibility with router_agent.py.

Usage
-----
    from src.agents.survey_agent import run_survey_agent

    result = run_survey_agent(
        request="Create a customer satisfaction survey about our product line",
        tenant_id="...",
        client_id="...",
        client_profile={"industry": "automotive", "demographic": {"age_range": "25-45"}},
    )
    print(result["survey"])
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.workflows.survey_workflow import build_survey_graph

logger = logging.getLogger(__name__)


def run_survey_agent(
    request: str,
    tenant_id: str,
    client_id: str,
    client_profile: Optional[Dict[str, Any]] = None,
    model: str = "gpt-4o-mini",
    top_k: int = 10,
    hop_limit: int = 1,
) -> Dict[str, Any]:
    """
    Generate a survey. Delegates to the survey_workflow LangGraph.

    Returns dict with keys: survey (JSON string), context_used (int), topic
    """
    graph = build_survey_graph()

    result = graph.invoke({
        "request": request,
        "tenant_id": tenant_id,
        "client_id": client_id,
        "client_profile": client_profile or {},
        "question_types": ["multiple_choice"],
    })

    return {
        "survey": result.get("survey", "[]"),
        "context_used": result.get("context_used", 0),
        "topic": request,
    }
