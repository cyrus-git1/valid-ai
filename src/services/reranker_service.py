"""
Cross-encoder reranker.

Sits between the retriever's candidate pool (e.g. top-50 from hybrid search)
and what the API returns (e.g. top-10). A good reranker scores
query/document pairs jointly, catching relevance signals that bi-encoders
miss because they encode query and doc separately.

Implementation:
  - Primary: Cohere rerank API (rerank-multilingual-v3.0) when COHERE_API_KEY
    is set. Cheap, fast, no infra.
  - Fallback: identity (no-op) — returns candidates in their existing order.

The fallback means missing-key never breaks search. The router/service
layer reads `is_available()` to decide whether to bother sending the call.

Future: pluggable backends (HuggingFace text-embeddings-inference, local
bge-reranker, etc.). Keep the interface stable — one `rerank(query, docs)`.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from src.logging_config import get_logger

logger = get_logger(__name__)

_COHERE_MODEL = os.environ.get("COHERE_RERANK_MODEL", "rerank-multilingual-v3.0")
_RERANK_TIMEOUT_S = float(os.environ.get("RERANK_TIMEOUT_SECONDS", "5"))


class RerankerService:
    """Stateless reranker. Construct once per process; call rerank() per query."""

    def __init__(self) -> None:
        self._cohere_client = None
        api_key = os.environ.get("COHERE_API_KEY")
        if api_key:
            try:
                import cohere  # type: ignore
                self._cohere_client = cohere.Client(api_key, timeout=_RERANK_TIMEOUT_S)
                logger.info("RerankerService: cohere backend ready (model=%s)", _COHERE_MODEL)
            except ImportError:
                logger.warning("RerankerService: COHERE_API_KEY set but `cohere` package not installed — falling back to no-op")
            except Exception as e:
                logger.warning("RerankerService: cohere init failed: %s — falling back to no-op", e)
        else:
            # The previously-silent case: no key at all → reranking is OFF and the
            # "cross-encoder rerank" degrades to hybrid+MMR order. Make that
            # visible at startup instead of disappearing without a trace.
            logger.warning(
                "RerankerService: COHERE_API_KEY not set — reranking DISABLED "
                "(retrieval returns hybrid+MMR order, no cross-encoder pass)"
            )

    def is_available(self) -> bool:
        return self._cohere_client is not None

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        *,
        text_field: str = "description",
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return candidates reordered by reranker score. Adds 'rerank_score'
        to each returned item.

        - `text_field`: which key in each candidate dict holds the text to
          rerank against. Defaults to 'description' (KG node descriptions);
          callers ranking chunks should pass 'content'.
        - `top_n`: cap the returned list. Defaults to len(candidates).
        """
        if not candidates or not query:
            return candidates
        if not self.is_available():
            return candidates[: top_n or len(candidates)]

        texts: List[str] = []
        for c in candidates:
            t = c.get(text_field) or c.get("name") or c.get("content") or ""
            texts.append(str(t)[:2000])  # cohere has a token budget; truncate safely

        try:
            resp = self._cohere_client.rerank(  # type: ignore[union-attr]
                query=query,
                documents=texts,
                model=_COHERE_MODEL,
                top_n=min(top_n or len(candidates), len(candidates)),
            )
        except Exception as e:
            logger.warning("rerank call failed (returning original order): %s", e)
            return candidates[: top_n or len(candidates)]

        # Cohere returns list of objects with .index and .relevance_score
        out: List[Dict[str, Any]] = []
        for r in resp.results:
            c = dict(candidates[r.index])
            c["rerank_score"] = float(r.relevance_score)
            out.append(c)
        return out


# Module-level singleton (idempotent — backing client is initialised once)
_singleton: Optional[RerankerService] = None


def get_reranker() -> RerankerService:
    global _singleton
    if _singleton is None:
        _singleton = RerankerService()
    return _singleton
