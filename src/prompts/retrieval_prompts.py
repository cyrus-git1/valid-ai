"""
src/prompts/retrieval_prompts.py
----------------------------------
Prompt templates for retrieval / RAG answer generation.

Used by both:
  - src/agents/retrieval_agent.py
  - src/workflows/rag_workflow.py
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

# ── RAG Answer Generation ───────────────────────────────────────────────────

RAG_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Answer the user's question using ONLY the "
        "context provided below. If the context does not contain enough information "
        "to answer confidently, say so — do not make things up."
        "{profile_section}\n\n"
        "Context:\n{context}",
    ),
    ("human", "{question}"),
])
