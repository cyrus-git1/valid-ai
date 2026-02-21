"""
src/workflows/rag_workflow.py
-------------------------------
LangGraph RAG workflow: query → retrieve → grade → generate.

Implements confidence-gated retrieval with automatic retry on low scores.

Usage
-----
    from src.workflows.rag_workflow import build_rag_graph

    app = build_rag_graph(tenant_id="...", client_id="...")
    result = app.invoke({"question": "What is our refund policy?"})
    print(result["answer"])
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, TypedDict
from uuid import UUID

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from src.prompts.retrieval_prompts import RAG_ANSWER_PROMPT
from src.services.kg_retriever_service import KGRetrieverService

logger = logging.getLogger(__name__)


# ── State ────────────────────────────────────────────────────────────────────

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


# ── Nodes ────────────────────────────────────────────────────────────────────

def retrieve(state: RAGState) -> RAGState:
    """Retrieve documents from the KG using graph-expanded search."""
    attempt = state.get("attempt", 0) + 1
    top_k = 5 if attempt == 1 else 10
    hop_limit = 1 if attempt == 1 else 2

    retriever = KGRetrieverService.from_env(
        tenant_id=UUID(state["tenant_id"]),
        client_id=UUID(state["client_id"]),
        top_k=top_k,
        hop_limit=hop_limit,
    )

    docs = retriever.invoke(state["question"])

    top_sim = 0.0
    if docs:
        top_sim = docs[0].metadata.get("similarity_score", 0.0)

    return {
        **state,
        "documents": docs,
        "top_similarity": top_sim,
        "attempt": attempt,
    }


def grade_documents(state: RAGState) -> RAGState:
    """Grade retrieval quality based on similarity scores."""
    top_sim = state.get("top_similarity", 0.0)
    return {**state, "confidence": top_sim}


def build_context(state: RAGState) -> RAGState:
    """Build context string from retrieved documents."""
    docs = state.get("documents", [])
    context = "\n\n---\n\n".join(
        f"[Source {i + 1}]\n{doc.page_content}"
        for i, doc in enumerate(docs)
        if doc.page_content.strip()
    )
    return {**state, "context": context}


def generate(state: RAGState) -> RAGState:
    """Generate answer from context using LLM."""
    model = state.get("model", "gpt-4o-mini")
    profile_ctx = ""
    profile = state.get("client_profile", {})
    if profile:
        parts = []
        if profile.get("industry"):
            parts.append(f"Industry: {profile['industry']}")
        if profile.get("headcount"):
            parts.append(f"Company size: {profile['headcount']} employees")
        demo = profile.get("demographic", {})
        if demo.get("age_range"):
            parts.append(f"Target audience age: {demo['age_range']}")
        if demo.get("occupation"):
            parts.append(f"Audience occupation: {demo['occupation']}")
        if parts:
            profile_ctx = "\n\nClient profile:\n" + "\n".join(parts)

    llm = ChatOpenAI(
        model=model,
        temperature=0,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    chain = RAG_ANSWER_PROMPT | llm | StrOutputParser()

    try:
        answer = chain.invoke({
            "context": state.get("context", ""),
            "question": state["question"],
            "profile_section": profile_ctx,
        })
    except Exception as e:
        logger.exception("LLM generation failed")
        answer = f"Generation failed: {e}"

    return {**state, "answer": answer}


def no_results(state: RAGState) -> RAGState:
    """Handle case where no relevant results were found."""
    return {
        **state,
        "answer": "I couldn't find information relevant enough to answer confidently. "
                  "Try rephrasing your question.",
    }


# ── Routing ──────────────────────────────────────────────────────────────────

def route_on_confidence(state: RAGState) -> str:
    """Route based on retrieval confidence."""
    confidence = state.get("confidence", 0.0)
    attempt = state.get("attempt", 1)

    if confidence < 0.60 and attempt < 2:
        return "retrieve"  # retry with broader search
    if confidence < 0.60:
        return "no_results"
    return "build_context"


# ── Graph ────────────────────────────────────────────────────────────────────

def build_rag_graph() -> StateGraph:
    """Build and compile the RAG LangGraph."""
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("build_context", build_context)
    graph.add_node("generate", generate)
    graph.add_node("no_results", no_results)

    graph.set_entry_point("retrieve")

    graph.add_edge("retrieve", "grade_documents")
    graph.add_conditional_edges("grade_documents", route_on_confidence)
    graph.add_edge("build_context", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("no_results", END)

    return graph.compile()
