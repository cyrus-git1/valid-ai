"""
src/agents/retrieval_agent.py
-------------------------------
Retrieval sub-agent — handles knowledge base search and RAG question answering.

Uses the KGRetrieverService for graph-expanded retrieval and SearchService
for LLM-powered answer generation.

This agent is called by the router agent when intent is classified as "retrieval".

Usage
-----
    from src.agents.retrieval_agent import run_retrieval_agent

    result = run_retrieval_agent(
        query="What is the refund policy?",
        tenant_id="...",
        client_id="...",
    )
    print(result["answer"])
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from src.prompts.retrieval_prompts import RAG_ANSWER_PROMPT
from src.services.search_service import SearchService

logger = logging.getLogger(__name__)


def run_retrieval_agent(
    query: str,
    tenant_id: str,
    client_id: str,
    client_profile: Optional[Dict[str, Any]] = None,
    model: str = "gpt-4o-mini",
    top_k: int = 5,
    hop_limit: int = 1,
) -> Dict[str, Any]:
    """
    Run the retrieval agent.

    Steps:
      1. Graph-expanded retrieval via SearchService
      2. Confidence check (similarity threshold)
      3. Build context from retrieved documents
      4. Generate answer with client profile awareness
      5. Return answer + sources

    Returns dict with keys: answer, sources, confidence
    """
    svc = SearchService(
        tenant_id=UUID(tenant_id),
        client_id=UUID(client_id),
    )

    # Step 1: Retrieve
    docs = svc.graph_search(
        query,
        top_k=top_k,
        hop_limit=hop_limit,
    )

    if not docs:
        return {
            "answer": "I couldn't find any relevant information to answer your question.",
            "sources": [],
            "confidence": 0.0,
        }

    # Step 2: Confidence check
    top_score = docs[0].metadata.get("similarity_score", 0.0)
    if top_score < 0.60:
        return {
            "answer": "I couldn't find information relevant enough to answer confidently. "
                      "Try rephrasing your question.",
            "sources": _format_sources(docs),
            "confidence": top_score,
        }

    # Step 3: Build context
    context = "\n\n---\n\n".join(
        f"[Source {i + 1}]\n{doc.page_content}"
        for i, doc in enumerate(docs)
        if doc.page_content.strip()
    )

    # Step 4: Generate with profile awareness
    profile_section = ""
    if client_profile:
        parts = []
        if client_profile.get("industry"):
            parts.append(f"Industry: {client_profile['industry']}")
        if client_profile.get("headcount"):
            parts.append(f"Company size: {client_profile['headcount']} employees")
        demo = client_profile.get("demographic", {})
        if demo.get("age_range"):
            parts.append(f"Target audience age: {demo['age_range']}")
        if demo.get("occupation"):
            parts.append(f"Audience occupation: {demo['occupation']}")
        if demo.get("location"):
            parts.append(f"Location: {demo['location']}")
        if parts:
            profile_section = "\n\nClient profile:\n" + "\n".join(parts)

    llm = ChatOpenAI(
        model=model,
        temperature=0,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    chain = RAG_ANSWER_PROMPT | llm | StrOutputParser()

    try:
        answer = chain.invoke({
            "context": context,
            "question": query,
            "profile_section": profile_section,
        })
    except Exception as e:
        logger.exception("LLM generation failed in retrieval agent")
        return {
            "answer": f"Generation failed: {e}",
            "sources": _format_sources(docs),
            "confidence": top_score,
        }

    return {
        "answer": answer,
        "sources": _format_sources(docs),
        "confidence": top_score,
    }


def _format_sources(docs: List[Document]) -> List[Dict[str, Any]]:
    """Convert Documents to serializable source dicts."""
    return [
        {
            "node_id": d.metadata.get("node_id"),
            "document_id": d.metadata.get("document_id"),
            "chunk_index": d.metadata.get("chunk_index"),
            "similarity_score": d.metadata.get("similarity_score"),
            "source": d.metadata.get("source"),
            "content_preview": d.page_content[:200],
        }
        for d in docs
    ]
