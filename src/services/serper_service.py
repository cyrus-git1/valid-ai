"""
src/services/serper_service.py
-------------------------------
Lightweight wrapper around the Serper.dev Google Search API.

Provides quick web search results to supplement internal knowledge
graph context with real-time external information.

Import
------
    from src.services.serper_service import SerperService

    svc = SerperService()
    results = svc.search("latest trends in automotive customer experience")
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

SERPER_ENDPOINT = "https://google.serper.dev/search"


class SerperService:
    """Thin wrapper for Serper.dev Google Search API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("SERPER_API_KEY", "")
        if not self.api_key:
            logger.warning(
                "SERPER_API_KEY is not set. Web search will return empty results."
            )

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def search(
        self,
        query: str,
        num_results: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Run a Google search via Serper and return simplified results.

        Returns a list of dicts with keys: title, link, snippet.
        Returns empty list if the API key is missing or the call fails.
        """
        if not self.is_configured:
            logger.debug("Serper not configured — skipping web search.")
            return []

        try:
            resp = httpx.post(
                SERPER_ENDPOINT,
                headers={
                    "X-API-KEY": self.api_key,
                    "Content-Type": "application/json",
                },
                json={"q": query, "num": num_results},
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("Serper search failed: %s", e)
            return []

        results: List[Dict[str, Any]] = []
        for item in data.get("organic", [])[:num_results]:
            results.append({
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            })

        logger.debug("Serper returned %d results for query: %r", len(results), query[:60])
        return results

    def search_as_context(
        self,
        query: str,
        num_results: int = 5,
    ) -> str:
        """
        Run a search and format results as a context string for LLM prompts.
        """
        results = self.search(query, num_results=num_results)
        if not results:
            return "(No web search results available.)"

        lines = []
        for i, r in enumerate(results, 1):
            lines.append(
                f"[Web Result {i}] {r['title']}\n"
                f"URL: {r['link']}\n"
                f"{r['snippet']}"
            )
        return "\n\n".join(lines)
