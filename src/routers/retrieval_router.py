"""
/retrieve router
-----------------
Dedicated retrieval/RAG endpoint. Skips intent classification entirely —
if this endpoint is called, a retrieval MUST be performed.

POST /retrieve/ask  — RAG question answering via the retrieval agent
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.agents.retrieval_agent import run_retrieval_agent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/retrieve", tags=["retrieve"])


# ── Request / Response ────────────────────────────────────────────────────────

class RetrievalRequest(BaseModel):
    tenant_id: UUID
    client_id: UUID
    query: str = Field(..., description="Question to answer from the knowledge base")
    client_profile: Optional[Dict[str, Any]] = None
    top_k: int = Field(default=5, ge=1, le=50)
    hop_limit: int = Field(default=1, ge=0, le=3)


class SourceItem(BaseModel):
    node_id: Optional[str] = None
    document_id: Optional[str] = None
    chunk_index: Optional[int] = None
    similarity_score: Optional[float] = None
    source: Optional[str] = None
    content_preview: Optional[str] = None


class RetrievalResponse(BaseModel):
    answer: str
    sources: List[SourceItem] = Field(default_factory=list)
    confidence: float = 0.0


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/ask", response_model=RetrievalResponse)
def retrieve_ask(req: RetrievalRequest) -> RetrievalResponse:
    """
    RAG question answering. No intent classification — this endpoint IS the intent.

    Runs the retrieval agent directly:
      graph-expanded search → confidence check → LLM answer generation

    Accepts JSON, returns JSON.
    """
    try:
        result = run_retrieval_agent(
            query=req.query,
            tenant_id=str(req.tenant_id),
            client_id=str(req.client_id),
            client_profile=req.client_profile,
            top_k=req.top_k,
            hop_limit=req.hop_limit,
        )
    except Exception as e:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    sources = [
        SourceItem(**s) for s in result.get("sources", [])
    ]

    return RetrievalResponse(
        answer=result.get("answer", ""),
        sources=sources,
        confidence=result.get("confidence", 0.0),
    )
