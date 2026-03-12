"""
src/services/base_service.py
------------------------------
Shared base class for analysis services that work with Supabase,
LLM calls, and transcript data.

Eliminates duplicated helper methods across SentimentAnalysisService,
StrategicAnalysisService, TranscriptInsightsService, and ContextSummaryService.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_openai import ChatOpenAI
from supabase import Client

logger = logging.getLogger(__name__)


class BaseAnalysisService:
    """Base class providing shared DB queries, LLM creation, and JSON parsing."""

    def __init__(self, supabase: Optional[Client] = None):
        self.sb = supabase

    def _require_supabase(self) -> Client:
        """Raise if Supabase client is not available."""
        if self.sb is None:
            raise RuntimeError(
                f"{self.__class__.__name__} requires a Supabase client for this operation."
            )
        return self.sb

    # ── LLM ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _create_llm(model: str = "gpt-4o-mini", temperature: float = 0.1) -> ChatOpenAI:
        """Create a ChatOpenAI instance with consistent defaults."""
        return ChatOpenAI(model=model, temperature=temperature)

    # ── JSON parsing ──────────────────────────────────────────────────────────

    @staticmethod
    def _parse_llm_json(
        raw_output: str,
        fallback_keys: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Parse LLM output as JSON, with markdown code-block fallback.

        If parsing fails entirely, returns a dict with fallback_keys
        plus a 'raw_output' key containing the original text.
        """
        # Try direct JSON parse
        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_output)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Fallback
        logger.warning("LLM returned non-JSON — using fallback structure")
        result = dict(fallback_keys or {})
        result["raw_output"] = raw_output
        return result

    # ── Transcript data queries ───────────────────────────────────────────────

    def _count_transcripts(self, tenant_id: UUID, client_id: UUID) -> int:
        """Count VTT documents for a tenant+client."""
        sb = self._require_supabase()
        try:
            res = (
                sb.table("documents")
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
        sb = self._require_supabase()
        try:
            doc_res = (
                sb.table("documents")
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
                sb.table("chunks")
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
        sb = self._require_supabase()
        try:
            res = (
                sb.table("documents")
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

    # ── Profile formatting ────────────────────────────────────────────────────

    @staticmethod
    def _build_profile_section(client_profile: Optional[Dict[str, Any]]) -> str:
        """Format a client profile dict into a text section for LLM prompts."""
        if not client_profile:
            return ""
        parts: List[str] = []
        for key in ("industry", "headcount", "revenue", "company_name", "persona"):
            if client_profile.get(key):
                label = key.replace("_", " ").title()
                parts.append(f"{label}: {client_profile[key]}")
        demo = client_profile.get("demographic", {})
        if isinstance(demo, dict):
            for key in ("age_range", "income_bracket", "occupation", "location"):
                if demo.get(key):
                    parts.append(f"{key.replace('_', ' ').title()}: {demo[key]}")
        if not parts:
            return ""
        return "Company / Client Profile:\n" + "\n".join(parts) + "\n\n"

    # ── Transcript context building ───────────────────────────────────────────

    def _build_transcript_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format transcript chunks into a context string for LLM prompts."""
        if not chunks:
            return (
                "(No video transcript data available. "
                "Ingest .vtt transcripts to enable analysis.)"
            )
        return "\n\n---\n\n".join(
            f"[Transcript Excerpt {i + 1}] {c['content']}"
            for i, c in enumerate(chunks)
            if c.get("content", "").strip()
        )
