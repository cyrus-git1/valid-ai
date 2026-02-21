"""
Service layer for context summaries.

Handles CRUD via Supabase RPCs / direct table queries, and orchestrates
LLM-based summary generation using the tenant's knowledge graph context.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from supabase import Client

from src.services.search_service import SearchService

logger = logging.getLogger(__name__)

# ── LLM prompt for generating the context summary ────────────────────────────

CONTEXT_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert business analyst. You will be given a collection of "
        "knowledge-base excerpts belonging to a single tenant/client. Your job is "
        "to produce a concise, high-level context summary that captures:\n"
        "1. The tenant's primary industry and market positioning\n"
        "2. Key themes and topics present in their knowledge base\n"
        "3. Notable products, services, or offerings mentioned\n"
        "4. Target audience / customer profile indicators\n"
        "5. Any recurring concepts or terminology\n\n"
        "Also return a JSON array of topic tags (short strings) that categorize "
        "the content.\n\n"
        "{profile_section}"
        "Respond in the following JSON format:\n"
        '{{"summary": "...", "topics": ["topic1", "topic2", ...]}}'
    ),
    (
        "human",
        "Knowledge base excerpts:\n\n{context}\n\n"
        "Generate the context summary and topic tags."
    ),
])


class ContextSummaryService:
    """Manages context summary CRUD and LLM-powered generation."""

    def __init__(self, supabase: Client):
        self.sb = supabase

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_summary(
        self,
        *,
        tenant_id: UUID,
        client_id: UUID,
    ) -> Optional[Dict[str, Any]]:
        """Fetch the current context summary for a tenant+client, or None."""
        res = (
            self.sb.table("context_summaries")
            .select("*")
            .eq("tenant_id", str(tenant_id))
            .eq("client_id", str(client_id))
            .limit(1)
            .execute()
        )
        rows = res.data or []
        return rows[0] if rows else None

    # ── Upsert (via RPC) ─────────────────────────────────────────────────────

    def upsert_summary(
        self,
        *,
        tenant_id: UUID,
        client_id: UUID,
        summary: str,
        topics: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source_stats: Optional[Dict[str, Any]] = None,
    ) -> UUID:
        """Upsert a context summary row. Returns the row id."""
        res = self.sb.rpc(
            "upsert_context_summary",
            {
                "p_tenant_id": str(tenant_id),
                "p_client_id": str(client_id),
                "p_summary": summary,
                "p_topics": topics or [],
                "p_metadata": metadata or {},
                "p_source_stats": source_stats or {},
            },
        ).execute()
        return UUID(str(res.data))

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete_summary(
        self,
        *,
        tenant_id: UUID,
        client_id: UUID,
    ) -> bool:
        """Delete the context summary for a tenant+client. Returns True if deleted."""
        res = (
            self.sb.table("context_summaries")
            .delete()
            .eq("tenant_id", str(tenant_id))
            .eq("client_id", str(client_id))
            .execute()
        )
        return len(res.data or []) > 0

    # ── Generate (LLM) ───────────────────────────────────────────────────────

    def generate_summary(
        self,
        *,
        tenant_id: UUID,
        client_id: UUID,
        client_profile: Optional[Dict[str, Any]] = None,
        force_regenerate: bool = False,
        llm_model: str = "gpt-4o-mini",
    ) -> Dict[str, Any]:
        """
        Generate (or regenerate) a context summary using the tenant's KG.

        1. Check for existing summary (skip if exists and not force_regenerate)
        2. Retrieve KG context via SearchService.graph_search()
        3. Prompt LLM to produce summary + topic tags
        4. Upsert into context_summaries
        5. Return the full row

        Returns:
            dict with keys: summary_row, regenerated, status
        """
        # Check existing
        if not force_regenerate:
            existing = self.get_summary(tenant_id=tenant_id, client_id=client_id)
            if existing:
                return {
                    "summary_row": existing,
                    "regenerated": False,
                    "status": "complete",
                }

        # Retrieve context from KG
        search_svc = SearchService(
            tenant_id=tenant_id,
            client_id=client_id,
        )

        try:
            docs = search_svc.graph_search(
                query="overview of all topics and themes",
                top_k=15,
                hop_limit=1,
            )
        except Exception as e:
            logger.warning("KG retrieval failed for summary generation: %s", e)
            docs = []

        # Build context string
        if docs:
            context_str = "\n\n---\n\n".join(
                f"[Excerpt {i + 1}]\n{doc.page_content}"
                for i, doc in enumerate(docs)
            )
        else:
            context_str = "(No knowledge base content available yet.)"

        # Build profile section
        profile_section = ""
        if client_profile:
            parts = []
            if client_profile.get("industry"):
                parts.append(f"Industry: {client_profile['industry']}")
            if client_profile.get("headcount"):
                parts.append(f"Headcount: {client_profile['headcount']}")
            if client_profile.get("revenue"):
                parts.append(f"Revenue: {client_profile['revenue']}")
            if client_profile.get("persona"):
                parts.append(f"Target persona: {client_profile['persona']}")
            if parts:
                profile_section = (
                    "Known tenant profile:\n" + "\n".join(parts) + "\n\n"
                )

        # Call LLM
        llm = ChatOpenAI(model=llm_model, temperature=0)
        chain = CONTEXT_SUMMARY_PROMPT | llm | StrOutputParser()

        try:
            raw_output = chain.invoke({
                "context": context_str,
                "profile_section": profile_section,
            })

            # Parse JSON response
            parsed = json.loads(raw_output)
            summary_text = parsed.get("summary", raw_output)
            topics = parsed.get("topics", [])
        except json.JSONDecodeError:
            # LLM returned non-JSON — use raw text as summary
            summary_text = raw_output
            topics = []
        except Exception as e:
            logger.exception("LLM summary generation failed")
            summary_text = f"Summary generation failed: {e}"
            topics = []

        # Compute source stats
        source_stats = {
            "documents_retrieved": len(docs),
            "context_length": len(context_str),
        }

        # Upsert
        row_id = self.upsert_summary(
            tenant_id=tenant_id,
            client_id=client_id,
            summary=summary_text,
            topics=topics,
            metadata=client_profile or {},
            source_stats=source_stats,
        )

        # Fetch the full row to return
        row = self.get_summary(tenant_id=tenant_id, client_id=client_id)
        if row is None:
            row = {
                "id": str(row_id),
                "tenant_id": str(tenant_id),
                "client_id": str(client_id),
                "summary": summary_text,
                "topics": topics,
                "metadata": client_profile or {},
                "source_stats": source_stats,
            }

        return {
            "summary_row": row,
            "regenerated": True,
            "status": "complete",
        }
