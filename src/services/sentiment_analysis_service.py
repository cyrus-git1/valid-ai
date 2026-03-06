"""
src/services/sentiment_analysis_service.py
--------------------------------------------
Sentiment analysis of WebVTT transcript chunks.

Supports three execution modes:
  - Single   — one optional focus query for one tenant+client
  - Batch    — multiple focus queries for the same tenant+client (shared context)
  - All      — one focus query across every client under a tenant
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from supabase import Client

from src.prompts.sentiment_prompts import SENTIMENT_ANALYSIS_PROMPT
from src.services.context_summary_service import ContextSummaryService

logger = logging.getLogger(__name__)


# ── Shared context (pre-fetched once per tenant+client) ──────────────────────


@dataclass
class _SharedContext:
    tenant_id: UUID
    client_id: UUID
    transcript_count: int = 0
    transcript_chunks: List[Dict[str, Any]] = field(default_factory=list)
    transcript_context: str = ""
    context_summary: str = ""
    chunks_analysed: int = 0


# ── Service ──────────────────────────────────────────────────────────────────


class SentimentAnalysisService:
    """Orchestrates sentiment analysis of VTT transcript chunks."""

    def __init__(self, supabase: Client):
        self.sb = supabase

    # ── Data gathering helpers ────────────────────────────────────────────

    def _count_transcripts(self, tenant_id: UUID, client_id: UUID) -> int:
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
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Fetch chunks belonging to VTT transcript documents."""
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
        """Discover all unique client_ids that have documents under a tenant."""
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
            logger.warning("Failed to list client_ids: %s", e)
            return []

    def _build_profile_section(self, client_profile: Optional[Dict[str, Any]]) -> str:
        if not client_profile:
            return ""
        parts: List[str] = []
        if client_profile.get("industry"):
            parts.append(f"Industry: {client_profile['industry']}")
        if client_profile.get("headcount"):
            parts.append(f"Headcount: {client_profile['headcount']}")
        if client_profile.get("revenue"):
            parts.append(f"Revenue: {client_profile['revenue']}")
        if client_profile.get("company_name"):
            parts.append(f"Company: {client_profile['company_name']}")
        if client_profile.get("persona"):
            parts.append(f"Target persona: {client_profile['persona']}")
        demo = client_profile.get("demographic", {})
        if isinstance(demo, dict):
            for key in ("age_range", "income_bracket", "occupation", "location"):
                if demo.get(key):
                    parts.append(f"{key.replace('_', ' ').title()}: {demo[key]}")
        if not parts:
            return ""
        return "Company / Client Profile:\n" + "\n".join(parts) + "\n\n"

    # ── Shared context ────────────────────────────────────────────────────

    def _gather_shared_context(
        self,
        tenant_id: UUID,
        client_id: UUID,
        chunk_limit: int = 50,
    ) -> _SharedContext:
        """Fetch transcript chunks and context summary once for reuse."""
        transcript_count = self._count_transcripts(tenant_id, client_id)

        chunks = self._get_transcript_chunks(tenant_id, client_id, limit=chunk_limit)
        if chunks:
            transcript_context = "\n\n---\n\n".join(
                f"[Transcript Excerpt {i + 1}] {c['content']}"
                for i, c in enumerate(chunks)
                if c.get("content", "").strip()
            )
        else:
            transcript_context = (
                "(No video transcript data available. "
                "Ingest .vtt transcripts to enable sentiment analysis.)"
            )

        # Fetch existing context summary if available
        summary_svc = ContextSummaryService(self.sb)
        existing = summary_svc.get_summary(tenant_id=tenant_id, client_id=client_id)
        if existing:
            context_summary = (
                f"Summary: {existing.get('summary', 'N/A')}\n"
                f"Topics: {', '.join(existing.get('topics', []))}"
            )
        else:
            context_summary = "(No context summary generated yet.)"

        return _SharedContext(
            tenant_id=tenant_id,
            client_id=client_id,
            transcript_count=transcript_count,
            transcript_chunks=chunks,
            transcript_context=transcript_context,
            context_summary=context_summary,
            chunks_analysed=len(chunks),
        )

    # ── Core LLM call ─────────────────────────────────────────────────────

    def _run_analysis(
        self,
        *,
        shared: _SharedContext,
        focus_query: Optional[str],
        client_profile: Optional[Dict[str, Any]],
        llm_model: str,
    ) -> Dict[str, Any]:
        """Build prompt, call LLM, parse JSON output."""

        # Build focus instructions
        if focus_query:
            focus_instructions = (
                f"FOCUS AREA: {focus_query}\n"
                "Narrow your sentiment analysis to aspects related to this focus "
                "area. Still report overall sentiment, but weight themes and quotes "
                "toward content relevant to the focus area.\n\n"
            )
        else:
            focus_instructions = ""

        profile_section = self._build_profile_section(client_profile)

        llm = ChatOpenAI(model=llm_model, temperature=0.1)
        chain = SENTIMENT_ANALYSIS_PROMPT | llm | StrOutputParser()

        raw_output = chain.invoke({
            "focus_instructions": focus_instructions,
            "profile_section": profile_section,
            "transcript_count": str(shared.transcript_count),
            "chunk_count": str(shared.chunks_analysed),
            "transcript_context": shared.transcript_context,
            "context_summary": shared.context_summary,
        })

        # Parse JSON (with markdown code-block fallback)
        parsed = None
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_output)
            if match:
                try:
                    parsed = json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass

        if parsed is None:
            logger.warning("LLM returned non-JSON — wrapping as summary")
            parsed = {
                "overall_sentiment": {"positive": 0.33, "negative": 0.33, "neutral": 0.34},
                "dominant_sentiment": "neutral",
                "themes": [],
                "notable_quotes": [],
                "summary": raw_output,
            }

        return {
            "tenant_id": str(shared.tenant_id),
            "client_id": str(shared.client_id),
            "overall_sentiment": parsed.get("overall_sentiment", {}),
            "dominant_sentiment": parsed.get("dominant_sentiment", "neutral"),
            "themes": parsed.get("themes", []),
            "notable_quotes": parsed.get("notable_quotes", []),
            "summary": parsed.get("summary", ""),
            "transcript_count": shared.transcript_count,
            "chunks_analysed": shared.chunks_analysed,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # ── Public: single (from VTT content) ───────────────────────────────

    def generate_from_vtt(
        self,
        *,
        tenant_id: UUID,
        survey_id: UUID,
        vtt_content: str,
        llm_model: str = "gpt-4o-mini",
    ) -> Dict[str, Any]:
        """Run sentiment analysis on raw WebVTT content (no DB fetch)."""
        logger.info(
            "Sentiment analysis (vtt): tenant=%s survey=%s",
            tenant_id, survey_id,
        )

        transcript_context = vtt_content.strip() or (
            "(No transcript content provided.)"
        )

        llm = ChatOpenAI(model=llm_model, temperature=0.1)
        chain = SENTIMENT_ANALYSIS_PROMPT | llm | StrOutputParser()

        raw_output = chain.invoke({
            "focus_instructions": "",
            "profile_section": "",
            "transcript_count": "1",
            "chunk_count": "1",
            "transcript_context": transcript_context,
            "context_summary": "(Not applicable — raw VTT provided.)",
        })

        parsed = None
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_output)
            if match:
                try:
                    parsed = json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass

        if parsed is None:
            logger.warning("LLM returned non-JSON — wrapping as summary")
            parsed = {
                "overall_sentiment": {"positive": 0.33, "negative": 0.33, "neutral": 0.34},
                "dominant_sentiment": "neutral",
                "themes": [],
                "notable_quotes": [],
                "summary": raw_output,
            }

        return {
            "tenant_id": str(tenant_id),
            "survey_id": str(survey_id),
            "overall_sentiment": parsed.get("overall_sentiment", {}),
            "dominant_sentiment": parsed.get("dominant_sentiment", "neutral"),
            "themes": parsed.get("themes", []),
            "notable_quotes": parsed.get("notable_quotes", []),
            "summary": parsed.get("summary", ""),
            "transcript_count": 1,
            "chunks_analysed": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # ── Public: single (legacy, DB-backed) ────────────────────────────────

    def generate_analysis(
        self,
        *,
        tenant_id: UUID,
        client_id: UUID,
        focus_query: Optional[str] = None,
        client_profile: Optional[Dict[str, Any]] = None,
        llm_model: str = "gpt-4o-mini",
        chunk_limit: int = 50,
    ) -> Dict[str, Any]:
        logger.info(
            "Sentiment analysis (single): tenant=%s client=%s focus=%r",
            tenant_id, client_id, focus_query,
        )
        shared = self._gather_shared_context(tenant_id, client_id, chunk_limit)
        return self._run_analysis(
            shared=shared,
            focus_query=focus_query,
            client_profile=client_profile,
            llm_model=llm_model,
        )

    # ── Public: batch ─────────────────────────────────────────────────────

    def generate_batch(
        self,
        *,
        tenant_id: UUID,
        client_id: UUID,
        focus_queries: List[str],
        client_profile: Optional[Dict[str, Any]] = None,
        llm_model: str = "gpt-4o-mini",
        chunk_limit: int = 50,
    ) -> Dict[str, Any]:
        logger.info(
            "Sentiment analysis (batch): tenant=%s client=%s queries=%d",
            tenant_id, client_id, len(focus_queries),
        )
        shared = self._gather_shared_context(tenant_id, client_id, chunk_limit)
        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, str]] = []

        for query in focus_queries[:10]:
            try:
                result = self._run_analysis(
                    shared=shared,
                    focus_query=query,
                    client_profile=client_profile,
                    llm_model=llm_model,
                )
                results.append(result)
            except Exception as e:
                logger.warning("Batch sentiment query failed (%r): %s", query, e)
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

    # ── Public: all clients ───────────────────────────────────────────────

    def generate_all(
        self,
        *,
        tenant_id: UUID,
        focus_query: Optional[str] = None,
        client_profile: Optional[Dict[str, Any]] = None,
        llm_model: str = "gpt-4o-mini",
        chunk_limit: int = 50,
    ) -> Dict[str, Any]:
        client_ids = self._list_client_ids(tenant_id)
        logger.info(
            "Sentiment analysis (all): tenant=%s clients=%d",
            tenant_id, len(client_ids),
        )
        results: List[Dict[str, Any]] = []
        errors: List[Dict[str, str]] = []

        for cid in client_ids:
            try:
                shared = self._gather_shared_context(tenant_id, cid, chunk_limit)
                result = self._run_analysis(
                    shared=shared,
                    focus_query=focus_query,
                    client_profile=client_profile,
                    llm_model=llm_model,
                )
                results.append(result)
            except Exception as e:
                logger.warning("All-clients sentiment failed for %s: %s", cid, e)
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
