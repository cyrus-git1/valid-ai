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

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.output_parsers import StrOutputParser
from supabase import Client

from src.prompts.sentiment_prompts import SENTIMENT_ANALYSIS_PROMPT
from src.services.base_service import BaseAnalysisService
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


class SentimentAnalysisService(BaseAnalysisService):
    """Orchestrates sentiment analysis of VTT transcript chunks."""

    _FALLBACK_PARSED = {
        "overall_sentiment": {"positive": 0.33, "negative": 0.33, "neutral": 0.34},
        "dominant_sentiment": "neutral",
        "themes": [],
        "notable_quotes": [],
        "summary": "",
    }

    def __init__(self, supabase: Optional[Client] = None):
        super().__init__(supabase)

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
        transcript_context = self._build_transcript_context(chunks)

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

        llm = self._create_llm(model=llm_model, temperature=0.1)
        chain = SENTIMENT_ANALYSIS_PROMPT | llm | StrOutputParser()

        raw_output = chain.invoke({
            "focus_instructions": focus_instructions,
            "profile_section": profile_section,
            "transcript_count": str(shared.transcript_count),
            "chunk_count": str(shared.chunks_analysed),
            "transcript_context": shared.transcript_context,
            "context_summary": shared.context_summary,
        })

        parsed = self._parse_llm_json(raw_output, fallback_keys=self._FALLBACK_PARSED)
        # If fallback was used, put raw_output into summary
        if "raw_output" in parsed:
            parsed["summary"] = parsed.pop("raw_output")

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

        transcript_context = vtt_content.strip() or "(No transcript content provided.)"

        llm = self._create_llm(model=llm_model, temperature=0.1)
        chain = SENTIMENT_ANALYSIS_PROMPT | llm | StrOutputParser()

        raw_output = chain.invoke({
            "focus_instructions": "",
            "profile_section": "",
            "transcript_count": "1",
            "chunk_count": "1",
            "transcript_context": transcript_context,
            "context_summary": "(Not applicable — raw VTT provided.)",
        })

        parsed = self._parse_llm_json(raw_output, fallback_keys=self._FALLBACK_PARSED)
        if "raw_output" in parsed:
            parsed["summary"] = parsed.pop("raw_output")

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
