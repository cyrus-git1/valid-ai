"""
src/services/transcript_insights_service.py
----------------------------------------------
Summarise WebVTT transcripts and extract actionable insights
for improving the client's product or service.

Scoped by tenant_id + survey_id (documents linked to a survey).
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from supabase import Client

from src.prompts.transcript_insights_prompts import TRANSCRIPT_INSIGHTS_PROMPT

logger = logging.getLogger(__name__)


class TranscriptInsightsService:
    """Generate a summary and actionable insights from VTT transcript chunks."""

    def __init__(self, supabase: Client):
        self.sb = supabase

    # ── Data retrieval ────────────────────────────────────────────────────

    def _get_transcript_chunks(
        self,
        tenant_id: UUID,
        survey_id: UUID,
        limit: int = 60,
    ) -> tuple[List[Dict[str, Any]], int]:
        """Fetch VTT chunks scoped to a tenant + survey.

        Returns (chunks, transcript_doc_count).
        """
        try:
            doc_res = (
                self.sb.table("documents")
                .select("id")
                .eq("tenant_id", str(tenant_id))
                .eq("client_id", str(survey_id))
                .eq("source_type", "vtt")
                .execute()
            )
            doc_ids = [row["id"] for row in (doc_res.data or [])]
            if not doc_ids:
                return [], 0

            chunk_res = (
                self.sb.table("chunks")
                .select("content, chunk_index, document_id, metadata")
                .eq("tenant_id", str(tenant_id))
                .in_("document_id", doc_ids)
                .order("chunk_index")
                .limit(limit)
                .execute()
            )
            return chunk_res.data or [], len(doc_ids)
        except Exception as e:
            logger.warning("Failed to fetch transcript chunks: %s", e)
            return [], 0

    # ── Core generation ───────────────────────────────────────────────────

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

        chunks, transcript_count = self._get_transcript_chunks(
            tenant_id, survey_id, limit=chunk_limit,
        )

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

        # Build transcript context string
        transcript_context = "\n\n---\n\n".join(
            f"[Excerpt {i + 1}] {c['content']}"
            for i, c in enumerate(chunks)
            if c.get("content", "").strip()
        )

        # Call LLM
        llm = ChatOpenAI(model=llm_model, temperature=0.15)
        chain = TRANSCRIPT_INSIGHTS_PROMPT | llm | StrOutputParser()

        try:
            raw_output = chain.invoke({
                "transcript_count": str(transcript_count),
                "chunk_count": str(len(chunks)),
                "transcript_context": transcript_context,
            })
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
            logger.warning("LLM returned non-JSON — using raw text as summary")
            parsed = {
                "summary": raw_output,
                "actionable_insights": [],
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
