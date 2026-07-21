"""
src/services/base_service.py
------------------------------
Shared base class for analysis services that work with Supabase
and LLM calls.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_openai import ChatOpenAI
from supabase import Client

from src.config.llm import LLMConfig

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

    @staticmethod
    def _create_llm(model: str = LLMConfig.DEFAULT, temperature: float = 0.1) -> ChatOpenAI:
        """Create a ChatOpenAI instance with consistent defaults."""
        return ChatOpenAI(model=model, temperature=temperature)

    @staticmethod
    def _parse_llm_json(
        raw_output: str,
        fallback_keys: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Parse LLM output as JSON, with markdown code-block fallback."""
        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            pass

        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_output)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        logger.warning("LLM returned non-JSON; using fallback structure")
        result = dict(fallback_keys or {})
        result["raw_output"] = raw_output
        return result

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
