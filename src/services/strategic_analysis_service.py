"""
src/services/strategic_analysis_service.py
--------------------------------------------
Convergent problem-solving analysis service.

Combines all tenant data sources into a single strategic analysis:
  - Vectorized chunks (via KG retriever)
  - Knowledge graph structure (graph-expanded retrieval)
  - Context summary (stored tenant summary + topics)
  - Client profile / company labels
  - External web search (Serper)

The analysis depth scales with the number of video transcripts
ingested for the tenant, enabling richer longitudinal insights
as more conversational data accumulates.

Supports three modes:
  - Single   — one focus query for one tenant+client
  - Batch    — multiple focus queries for the same tenant+client
  - All      — one focus query across every client under a tenant

Import
------
    from src.services.strategic_analysis_service import StrategicAnalysisService
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from supabase import Client

from src.prompts.strategic_analysis_prompts import (
    DEPTH_INSTRUCTIONS,
    STRATEGIC_ANALYSIS_PROMPT,
)
from src.services.context_summary_service import ContextSummaryService
from src.services.search_service import SearchService
from src.services.serper_service import SerperService

logger = logging.getLogger(__name__)


def _depth_tier(transcript_count: int) -> str:
    """Determine analysis depth tier from transcript count."""
    if transcript_count >= 10:
        return "deep"
    if transcript_count >= 4:
        return "comprehensive"
    if transcript_count >= 1:
        return "developing"
    return "foundational"


# ── Pre-fetched shared context ────────────────────────────────────────────────

@dataclass
class _SharedContext:
    """Data gathered once and reused across queries for the same tenant+client."""

    tenant_id: UUID
    client_id: UUID
    transcript_count: int = 0
    depth: str = "foundational"
    transcript_context: str = ""
    context_summary: str = ""
    transcript_chunks_retrieved: int = 0
    context_summary_available: bool = False


# ── Service ───────────────────────────────────────────────────────────────────

class StrategicAnalysisService:
    """Orchestrates convergent analysis across all tenant data sources."""

    def __init__(self, supabase: Client):
        self.sb = supabase

    # ── Data gathering helpers ────────────────────────────────────────────────

    def _count_transcripts(self, tenant_id: UUID, client_id: UUID) -> int:
        """Count documents with source_type 'vtt' for the tenant+client."""
        try:
            res = (
                self.sb.table("documents")
                .select("id", count="exact")
                .eq("tenant_id", str(tenant_id))
                .eq("client_id", str(client_id))
                .eq("source_type", "vtt")
                .execute()
            )
            return res.count or 0
        except Exception as e:
            logger.warning("Failed to count transcripts: %s", e)
            return 0

    def _get_transcript_chunks(
        self,
        tenant_id: UUID,
        client_id: UUID,
        limit: int = 30,
    ) -> List[Dict[str, Any]]:
        """Fetch chunks that belong to transcript (vtt) documents."""
        try:
            doc_res = (
                self.sb.table("documents")
                .select("id")
                .eq("tenant_id", str(tenant_id))
                .eq("client_id", str(client_id))
                .eq("source_type", "vtt")
                .execute()
            )
            doc_ids = [row["id"] for row in (doc_res.data or [])]
            if not doc_ids:
                return []

            chunk_res = (
                self.sb.table("chunks")
                .select("content, chunk_index, document_id, metadata")
                .eq("tenant_id", str(tenant_id))
                .in_("document_id", doc_ids)
                .order("chunk_index")
                .limit(limit)
                .execute()
            )
            return chunk_res.data or []
        except Exception as e:
            logger.warning("Failed to fetch transcript chunks: %s", e)
            return []

    def _list_client_ids(self, tenant_id: UUID) -> List[UUID]:
        """Return distinct client_ids that have at least one document for this tenant."""
        try:
            res = (
                self.sb.table("documents")
                .select("client_id")
                .eq("tenant_id", str(tenant_id))
                .execute()
            )
            seen: set[str] = set()
            client_ids: List[UUID] = []
            for row in (res.data or []):
                cid = row.get("client_id")
                if cid and cid not in seen:
                    seen.add(cid)
                    client_ids.append(UUID(cid))
            return client_ids
        except Exception as e:
            logger.warning("Failed to list client_ids for tenant %s: %s", tenant_id, e)
            return []

    def _build_profile_section(self, client_profile: Optional[Dict[str, Any]]) -> str:
        if not client_profile:
            return ""
        parts = []
        if client_profile.get("industry"):
            parts.append(f"Industry: {client_profile['industry']}")
        if client_profile.get("headcount"):
            parts.append(f"Headcount: {client_profile['headcount']}")
        demo = client_profile.get("demographic", {})
        if isinstance(demo, dict):
            if demo.get("age_range"):
                parts.append(f"Target age range: {demo['age_range']}")
            if demo.get("income_bracket"):
                parts.append(f"Income bracket: {demo['income_bracket']}")
            if demo.get("occupation"):
                parts.append(f"Occupation: {demo['occupation']}")
            if demo.get("location"):
                parts.append(f"Location: {demo['location']}")
        if not parts:
            return ""
        return "Company / Client Profile:\n" + "\n".join(parts) + "\n"

    # ── Shared context pre-fetch ──────────────────────────────────────────────

    def _gather_shared_context(
        self,
        tenant_id: UUID,
        client_id: UUID,
    ) -> _SharedContext:
        """
        Fetch transcript count, transcript chunks, and context summary once.
        These don't depend on the focus_query and can be reused across batch items.
        """
        transcript_count = self._count_transcripts(tenant_id, client_id)
        depth = _depth_tier(transcript_count)

        # Transcript chunks
        transcript_chunks = self._get_transcript_chunks(
            tenant_id, client_id,
            limit=15 + (transcript_count * 5),
        )
        if transcript_chunks:
            transcript_context = "\n\n---\n\n".join(
                f"[Transcript Excerpt {i + 1}] {c['content']}"
                for i, c in enumerate(transcript_chunks)
                if c.get("content", "").strip()
            )
        else:
            transcript_context = (
                "(No video transcript data available yet. "
                "Ingest .vtt transcripts to unlock deeper analysis.)"
            )

        # Context summary
        summary_svc = ContextSummaryService(self.sb)
        existing_summary = summary_svc.get_summary(
            tenant_id=tenant_id, client_id=client_id,
        )
        if existing_summary:
            context_summary = (
                f"Summary: {existing_summary.get('summary', 'N/A')}\n"
                f"Topics: {', '.join(existing_summary.get('topics', []))}"
            )
        else:
            context_summary = "(No context summary generated yet.)"

        return _SharedContext(
            tenant_id=tenant_id,
            client_id=client_id,
            transcript_count=transcript_count,
            depth=depth,
            transcript_context=transcript_context,
            context_summary=context_summary,
            transcript_chunks_retrieved=len(transcript_chunks),
            context_summary_available=existing_summary is not None,
        )

    # ── Core LLM call (operates on one focus_query) ──────────────────────────

    def _run_analysis(
        self,
        *,
        focus_query: str,
        shared: _SharedContext,
        client_profile: Optional[Dict[str, Any]],
        top_k: int,
        hop_limit: int,
        web_search_queries: Optional[List[str]],
        llm_model: str,
    ) -> Dict[str, Any]:
        """
        Execute the convergent analysis for a single focus_query using
        pre-fetched shared context.
        """

        # KG retrieval (query-specific)
        search_svc = SearchService(
            tenant_id=shared.tenant_id, client_id=shared.client_id,
        )
        try:
            kg_docs = search_svc.graph_search(
                focus_query, top_k=top_k, hop_limit=hop_limit,
            )
        except Exception as e:
            logger.warning("KG retrieval failed: %s", e)
            kg_docs = []

        kg_context = "\n\n---\n\n".join(
            f"[Chunk {i + 1}] {doc.page_content}"
            for i, doc in enumerate(kg_docs)
            if doc.page_content.strip()
        ) or "(No knowledge base chunks available.)"

        # Serper web search (query-specific)
        serper = SerperService()
        queries = list(web_search_queries or [])
        if not queries:
            industry = ""
            if client_profile and client_profile.get("industry"):
                industry = client_profile["industry"] + " "
            queries = [f"{industry}{focus_query}"]

        web_parts = []
        for q in queries[:3]:
            web_parts.append(serper.search_as_context(q, num_results=3))
        web_context = "\n\n".join(web_parts) if web_parts else "(No web search results.)"

        # Prompt inputs
        profile_section = self._build_profile_section(client_profile)
        depth_instructions = DEPTH_INSTRUCTIONS.get(
            shared.depth, DEPTH_INSTRUCTIONS["foundational"],
        )

        # LLM call
        llm = ChatOpenAI(model=llm_model, temperature=0.1)
        chain = STRATEGIC_ANALYSIS_PROMPT | llm | StrOutputParser()

        raw_output = chain.invoke({
            "focus_query": focus_query,
            "kg_context": kg_context,
            "context_summary": shared.context_summary,
            "transcript_context": shared.transcript_context,
            "transcript_count": shared.transcript_count,
            "web_context": web_context,
            "profile_section": profile_section,
            "depth_instructions": depth_instructions,
        })

        # Parse
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            logger.warning("LLM returned non-JSON — wrapping as executive summary.")
            parsed = {
                "executive_summary": raw_output,
                "convergent_themes": [],
                "action_points": [],
                "future_recommendations": [],
            }

        sources_used = {
            "kg_chunks_retrieved": len(kg_docs),
            "transcript_chunks_retrieved": shared.transcript_chunks_retrieved,
            "web_queries_executed": len(queries),
            "context_summary_available": shared.context_summary_available,
        }

        return {
            "tenant_id": str(shared.tenant_id),
            "client_id": str(shared.client_id),
            "focus_query": focus_query,
            "executive_summary": parsed.get("executive_summary", ""),
            "convergent_themes": parsed.get("convergent_themes", []),
            "action_points": parsed.get("action_points", []),
            "future_recommendations": parsed.get("future_recommendations", []),
            "analysis_depth": shared.depth,
            "transcript_count": shared.transcript_count,
            "sources_used": sources_used,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # ── Public: single ────────────────────────────────────────────────────────

    def generate_analysis(
        self,
        *,
        tenant_id: UUID,
        client_id: UUID,
        focus_query: str,
        client_profile: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        hop_limit: int = 1,
        web_search_queries: Optional[List[str]] = None,
        llm_model: str = "gpt-4o-mini",
    ) -> Dict[str, Any]:
        """Run the full convergent analysis pipeline for a single focus query."""
        logger.info(
            "Strategic analysis (single): tenant=%s client=%s",
            tenant_id, client_id,
        )
        shared = self._gather_shared_context(tenant_id, client_id)
        return self._run_analysis(
            focus_query=focus_query,
            shared=shared,
            client_profile=client_profile,
            top_k=top_k,
            hop_limit=hop_limit,
            web_search_queries=web_search_queries,
            llm_model=llm_model,
        )

    # ── Public: batch ─────────────────────────────────────────────────────────

    def generate_batch(
        self,
        *,
        tenant_id: UUID,
        client_id: UUID,
        focus_queries: List[str],
        client_profile: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        hop_limit: int = 1,
        web_search_queries: Optional[List[str]] = None,
        llm_model: str = "gpt-4o-mini",
    ) -> Dict[str, Any]:
        """
        Run convergent analysis for multiple focus queries against the same
        tenant+client. Shared context is gathered once and reused.
        """
        logger.info(
            "Strategic analysis (batch): tenant=%s client=%s queries=%d",
            tenant_id, client_id, len(focus_queries),
        )

        shared = self._gather_shared_context(tenant_id, client_id)
        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, str]] = []

        for query in focus_queries[:10]:  # cap at 10
            try:
                result = self._run_analysis(
                    focus_query=query,
                    shared=shared,
                    client_profile=client_profile,
                    top_k=top_k,
                    hop_limit=hop_limit,
                    web_search_queries=web_search_queries,
                    llm_model=llm_model,
                )
                results.append(result)
            except Exception as e:
                logger.warning("Batch query failed (%r): %s", query, e)
                errors.append({"focus_query": query, "error": str(e)})

        return {
            "tenant_id": str(tenant_id),
            "client_id": str(client_id),
            "total": len(focus_queries[:10]),
            "completed": len(results),
            "failed": len(errors),
            "results": results,
            "errors": errors,
        }

    # ── Public: all clients ───────────────────────────────────────────────────

    def generate_all(
        self,
        *,
        tenant_id: UUID,
        focus_query: str,
        client_profile: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        hop_limit: int = 1,
        web_search_queries: Optional[List[str]] = None,
        llm_model: str = "gpt-4o-mini",
    ) -> Dict[str, Any]:
        """
        Run the same focus query across every client_id that has data
        under this tenant_id.
        """
        client_ids = self._list_client_ids(tenant_id)
        logger.info(
            "Strategic analysis (all): tenant=%s clients=%d query=%r",
            tenant_id, len(client_ids), focus_query[:60],
        )

        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, str]] = []

        for cid in client_ids:
            try:
                shared = self._gather_shared_context(tenant_id, cid)
                result = self._run_analysis(
                    focus_query=focus_query,
                    shared=shared,
                    client_profile=client_profile,
                    top_k=top_k,
                    hop_limit=hop_limit,
                    web_search_queries=web_search_queries,
                    llm_model=llm_model,
                )
                results.append(result)
            except Exception as e:
                logger.warning(
                    "All-clients analysis failed for client %s: %s", cid, e,
                )
                errors.append({"client_id": str(cid), "error": str(e)})

        return {
            "tenant_id": str(tenant_id),
            "focus_query": focus_query,
            "total_clients": len(client_ids),
            "completed": len(results),
            "failed": len(errors),
            "results": results,
            "errors": errors,
        }
