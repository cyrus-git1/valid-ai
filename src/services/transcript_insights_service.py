"""
src/services/transcript_insights_service.py
----------------------------------------------
Summarise WebVTT transcripts and extract actionable insights
for improving the client's product or service.

Scoped by tenant_id + survey_id (documents linked to a survey).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.output_parsers import StrOutputParser
from supabase import Client

from src.prompts.transcript_insights_prompts import TRANSCRIPT_INSIGHTS_PROMPT
from src.services.base_service import BaseAnalysisService

logger = logging.getLogger(__name__)


class TranscriptInsightsService(BaseAnalysisService):
    """Generate a summary and actionable insights from VTT transcript chunks."""

    _FALLBACK_PARSED = {
        "summary": "",
        "actionable_insights": [],
    }

    def __init__(self, supabase: Optional[Client] = None):
        super().__init__(supabase)

    # ── Shared LLM pipeline ──────────────────────────────────────────────

    def _run_insights(
        self,
        *,
        transcript_context: str,
        transcript_count: int | str,
        chunk_count: int | str,
        llm_model: str,
    ) -> Dict[str, Any] | None:
        """Call LLM and parse result. Returns parsed dict or None on failure."""
        llm = self._create_llm(model=llm_model, temperature=0.15)
        chain = TRANSCRIPT_INSIGHTS_PROMPT | llm | StrOutputParser()

        raw_output = chain.invoke({
            "transcript_count": str(transcript_count),
            "chunk_count": str(chunk_count),
            "transcript_context": transcript_context,
        })

        parsed = self._parse_llm_json(raw_output, fallback_keys=self._FALLBACK_PARSED)
        if "raw_output" in parsed:
            parsed["summary"] = parsed.pop("raw_output")

        return parsed

    # ── Public: from raw VTT content ─────────────────────────────────────

    def generate_from_vtt(
        self,
        *,
        tenant_id: UUID,
        survey_id: UUID,
        vtt_content: str,
        llm_model: str = "gpt-4o-mini",
    ) -> Dict[str, Any]:
        """Summarise raw WebVTT content and extract actionable insights (no DB fetch)."""
        logger.info(
            "Transcript insights (vtt): tenant=%s survey=%s", tenant_id, survey_id,
        )

        transcript_context = vtt_content.strip()
        if not transcript_context:
            return {
                "tenant_id": str(tenant_id),
                "survey_id": str(survey_id),
                "summary": "No transcript content provided.",
                "actionable_insights": [],
                "transcript_count": 0,
                "chunks_analysed": 0,
                "status": "complete",
                "error": None,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

        try:
            parsed = self._run_insights(
                transcript_context=transcript_context,
                transcript_count=1,
                chunk_count=1,
                llm_model=llm_model,
            )
        except Exception as e:
            logger.exception("Transcript insights LLM call failed")
            return {
                "tenant_id": str(tenant_id),
                "survey_id": str(survey_id),
                "summary": "",
                "actionable_insights": [],
                "transcript_count": 1,
                "chunks_analysed": 1,
                "status": "failed",
                "error": str(e),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

        return {
            "tenant_id": str(tenant_id),
            "survey_id": str(survey_id),
            "summary": parsed.get("summary", ""),
            "actionable_insights": parsed.get("actionable_insights", []),
            "transcript_count": 1,
            "chunks_analysed": 1,
            "status": "complete",
            "error": None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # ── Core generation (legacy, DB-backed) ───────────────────────────────

    def generate(
        self,
        *,
        tenant_id: UUID,
        survey_id: UUID,
        llm_model: str = "gpt-4o-mini",
        chunk_limit: int = 60,
    ) -> Dict[str, Any]:
        """Summarise transcripts and extract actionable insights.

        Returns dict with keys: summary, actionable_insights, transcript_count,
        chunks_analysed, status, error, generated_at.
        """
        logger.info(
            "Transcript insights: tenant=%s survey=%s", tenant_id, survey_id,
        )

        chunks = self._get_transcript_chunks(
            tenant_id, survey_id, limit=chunk_limit,
        )

        # Count transcript documents
        transcript_count = self._count_transcripts(tenant_id, survey_id)

        if not chunks:
            return {
                "tenant_id": str(tenant_id),
                "survey_id": str(survey_id),
                "summary": "No VTT transcript data found for this tenant and survey.",
                "actionable_insights": [],
                "transcript_count": 0,
                "chunks_analysed": 0,
                "status": "complete",
                "error": None,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

        transcript_context = self._build_transcript_context(chunks)

        try:
            parsed = self._run_insights(
                transcript_context=transcript_context,
                transcript_count=transcript_count,
                chunk_count=len(chunks),
                llm_model=llm_model,
            )
        except Exception as e:
            logger.exception("Transcript insights LLM call failed")
            return {
                "tenant_id": str(tenant_id),
                "survey_id": str(survey_id),
                "summary": "",
                "actionable_insights": [],
                "transcript_count": transcript_count,
                "chunks_analysed": len(chunks),
                "status": "failed",
                "error": str(e),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }

        return {
            "tenant_id": str(tenant_id),
            "survey_id": str(survey_id),
            "summary": parsed.get("summary", ""),
            "actionable_insights": parsed.get("actionable_insights", []),
            "transcript_count": transcript_count,
            "chunks_analysed": len(chunks),
            "status": "complete",
            "error": None,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
