"""
src/services/search_service.py
-------------------------------
RAG chain logic — builds context from retrieved documents and calls the LLM.
Works alongside KGRetrieverService which handles the retrieval step.

Import
------
    from src.services.search_service import SearchService
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional
from uuid import UUID

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.services.kg_retriever_service import KGRetrieverService

logger = logging.getLogger(__name__)


class SearchService:
    """
    Wraps the KGRetrieverService and adds LLM answer generation for /search/ask.

    Usage
    -----
        svc = SearchService(tenant_id=..., client_id=...)

        # Vector search only
        docs = svc.semantic_search("What is the return policy?", top_k=5)

        # Graph-expanded search
        docs = svc.graph_search("What is the return policy?", top_k=5, hop_limit=1)

        # Full RAG
        answer, docs = svc.ask("What is the return policy?")
    """

    def __init__(
        self,
        tenant_id: UUID,
        client_id: UUID,
        openai_api_key: Optional[str] = None,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        embed_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o-mini",
    ):
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.llm_model = llm_model
        self._api_key = openai_api_key or os.environ["OPENAI_API_KEY"]
        self._sb_url = supabase_url or os.environ["SUPABASE_URL"]
        self._sb_key = supabase_key or os.environ["SUPABASE_SERVICE_KEY"]
        self._embed_model = embed_model

    def _build_retriever(
        self,
        top_k: int,
        hop_limit: int,
        max_neighbours: int = 3,
        min_edge_weight: float = 0.75,
    ) -> KGRetrieverService:
        return KGRetrieverService(
            supabase_url=self._sb_url,
            supabase_key=self._sb_key,
            openai_api_key=self._api_key,
            tenant_id=self.tenant_id,
            client_id=self.client_id,
            top_k=top_k,
            hop_limit=hop_limit,
            max_neighbours=max_neighbours,
            min_edge_weight=min_edge_weight,
            embed_model=self._embed_model,
        )

    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Document]:
        """Pure vector search — no graph expansion."""
        retriever = self._build_retriever(top_k=top_k, hop_limit=0)
        return retriever.invoke(query)

    def graph_search(
        self,
        query: str,
        top_k: int = 5,
        hop_limit: int = 1,
        max_neighbours: int = 3,
        min_edge_weight: float = 0.75,
    ) -> List[Document]:
        """Vector search + graph expansion."""
        retriever = self._build_retriever(
            top_k=top_k,
            hop_limit=hop_limit,
            max_neighbours=max_neighbours,
            min_edge_weight=min_edge_weight,
        )
        return retriever.invoke(query)

    def ask(
        self,
        question: str,
        top_k: int = 5,
        hop_limit: int = 1,
        max_neighbours: int = 3,
        min_edge_weight: float = 0.75,
    ) -> tuple[str, List[Document]]:
        """
        Full RAG pipeline.

        Steps:
          1. Graph-expanded retrieval
          2. Build context string from retrieved documents
          3. Prompt LLM to answer from context only
          4. Return (answer, source_documents)

        Confidence routing
        ------------------
        If the top result's similarity_score is below 0.60, skip LLM and
        return a "no relevant information found" message directly. This avoids
        wasting LLM calls on queries where retrieval has clearly failed.
        """
        docs = self.graph_search(
            question,
            top_k=top_k,
            hop_limit=hop_limit,
            max_neighbours=max_neighbours,
            min_edge_weight=min_edge_weight,
        )

        if not docs:
            return "I couldn't find any relevant information to answer your question.", []

        # Confidence gate — skip LLM if best match is too weak
        top_score = docs[0].metadata.get("similarity_score", 1.0)
        if top_score < 0.60:
            logger.info("Low similarity score (%.3f) — skipping LLM generation.", top_score)
            return (
                "I couldn't find information relevant enough to answer confidently. "
                "Try rephrasing your question.",
                docs,
            )

        context = "\n\n---\n\n".join(
            f"[Source {i + 1}]\n{doc.page_content}"
            for i, doc in enumerate(docs)
            if doc.page_content.strip()
        )

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful assistant. Answer the user's question using ONLY the "
                "context provided below. If the context does not contain enough information "
                "to answer confidently, say so — do not make things up.\n\n"
                "Context:\n{context}",
            ),
            ("human", "{question}"),
        ])

        llm = ChatOpenAI(
            model=self.llm_model,
            temperature=0,
            api_key=self._api_key,
        )

        chain = prompt | llm | StrOutputParser()

        try:
            answer = chain.invoke({"context": context, "question": question})
        except Exception as e:
            logger.exception("LLM generation failed")
            raise RuntimeError(f"LLM generation failed: {e}") from e

        return answer, docs
