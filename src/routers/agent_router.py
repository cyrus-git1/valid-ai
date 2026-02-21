"""
/agent router
--------------
Agent interaction endpoint — routes user queries through the LangGraph
routing agent which classifies intent and delegates to sub-agents.

POST /agent/query  — Send a query through the routing agent
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.agents.router_agent import build_router_agent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agent", tags=["agent"])


class AgentQueryRequest(BaseModel):
    tenant_id: UUID
    client_id: UUID
    input: str = Field(..., description="User query or request")
    client_profile: Optional[Dict[str, Any]] = None


class AgentQueryResponse(BaseModel):
    intent: str
    output: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: Optional[float] = None


@router.post("/query", response_model=AgentQueryResponse)
def agent_query(req: AgentQueryRequest) -> AgentQueryResponse:
    """
    Send a query through the routing agent.

    The routing agent classifies the user's intent and delegates to the
    appropriate sub-agent:
      - retrieval: search the knowledge base and generate answers
      - survey: generate a survey based on context from the KG
      - ingest: guidance on adding new content

    Returns the agent's response along with the classified intent.
    """
    try:
        agent = build_router_agent()
        result = agent.invoke({
            "input": req.input,
            "tenant_id": str(req.tenant_id),
            "client_id": str(req.client_id),
            "client_profile": req.client_profile,
        })
    except Exception as e:
        logger.exception("Agent query failed")
        raise HTTPException(status_code=500, detail=f"Agent query failed: {e}")

    return AgentQueryResponse(
        intent=result.get("intent", "unknown"),
        output=result.get("output", ""),
        sources=result.get("sources", []),
        confidence=result.get("intent_confidence"),
    )
