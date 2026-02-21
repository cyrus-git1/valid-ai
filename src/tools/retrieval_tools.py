"""
src/tools/retrieval_tools.py
------------------------------
LangChain tools that agents use to interact with the KG retriever and services.
"""
from __future__ import annotations

import os
from typing import Optional
from uuid import UUID

from langchain_core.tools import tool

from src.services.kg_retriever_service import KGRetrieverService
from src.services.search_service import SearchService


@tool
def semantic_search(query: str, tenant_id: str, client_id: str, top_k: int = 5) -> str:
    """Search the knowledge graph using vector similarity only.
    Use this for quick, direct lookups where you need the most similar content."""
    svc = SearchService(
        tenant_id=UUID(tenant_id),
        client_id=UUID(client_id),
    )
    docs = svc.semantic_search(query, top_k=top_k)
    if not docs:
        return "No results found."
    return "\n\n---\n\n".join(
        f"[Score: {d.metadata.get('similarity_score', 'N/A')}] {d.page_content[:500]}"
        for d in docs
    )


@tool
def graph_search(query: str, tenant_id: str, client_id: str, top_k: int = 5, hop_limit: int = 1) -> str:
    """Search the knowledge graph with graph expansion — finds related content
    that may not be directly similar to the query but is structurally connected.
    Use this for broader context gathering."""
    svc = SearchService(
        tenant_id=UUID(tenant_id),
        client_id=UUID(client_id),
    )
    docs = svc.graph_search(query, top_k=top_k, hop_limit=hop_limit)
    if not docs:
        return "No results found."
    return "\n\n---\n\n".join(
        f"[{d.metadata.get('source', 'unknown')} | Score: {d.metadata.get('similarity_score', 'N/A')}] {d.page_content[:500]}"
        for d in docs
    )


@tool
def ask_knowledge_base(question: str, tenant_id: str, client_id: str) -> str:
    """Ask the knowledge base a question and get a generated answer with sources.
    This runs the full RAG pipeline: retrieval + LLM generation."""
    svc = SearchService(
        tenant_id=UUID(tenant_id),
        client_id=UUID(client_id),
    )
    answer, docs = svc.ask(question)
    source_info = ""
    if docs:
        source_info = "\n\nSources:\n" + "\n".join(
            f"- {d.metadata.get('document_id', 'unknown')} (chunk {d.metadata.get('chunk_index', '?')})"
            for d in docs[:5]
        )
    return answer + source_info
