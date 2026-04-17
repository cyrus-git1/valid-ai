"""
cache_service.py
----------------
Thin typed wrapper over Redis for application-level caching.

Cache namespaces:
  embed:<sha256>           - cached query embeddings
  search:<tenant>:<scope>:* - hot search result sets (scoped)
  ctxsum:<tenant>:<client> - context summary cache

Fail-open on Redis errors — callers should treat the cache as a best-effort
optimisation, never a source of truth.
"""
from __future__ import annotations

import json
import os
from typing import Any, Iterable, Optional

from src.logging_config import get_logger

logger = get_logger(__name__)


class CacheService:
    """Tiny Redis-backed cache. Safe to use when Redis is unavailable."""

    def __init__(self, redis_client=None):
        self._r = redis_client if redis_client is not None else self._connect()

    def _connect(self):
        try:
            import redis
            url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
            r = redis.from_url(url, decode_responses=True)
            r.ping()
            return r
        except Exception:
            return None

    @property
    def available(self) -> bool:
        return self._r is not None

    # ── Generic ops ─────────────────────────────────────────────────────────

    def get(self, key: str) -> Optional[Any]:
        if not self._r:
            return None
        try:
            raw = self._r.get(key)
            return json.loads(raw) if raw else None
        except Exception as e:
            logger.debug("cache_get_failed", key=key, error=str(e))
            return None

    def set(self, key: str, value: Any, ttl: int) -> None:
        if not self._r:
            return
        try:
            self._r.setex(key, ttl, json.dumps(value, default=str))
        except Exception as e:
            logger.debug("cache_set_failed", key=key, error=str(e))

    def delete(self, key: str) -> None:
        if not self._r:
            return
        try:
            self._r.delete(key)
        except Exception:
            pass

    def delete_by_prefix(self, prefix: str) -> int:
        """Delete every key matching prefix*. Uses SCAN to avoid blocking."""
        if not self._r:
            return 0
        try:
            n = 0
            for key in self._iter_keys(f"{prefix}*"):
                self._r.delete(key)
                n += 1
            return n
        except Exception as e:
            logger.debug("cache_prefix_delete_failed", prefix=prefix, error=str(e))
            return 0

    def _iter_keys(self, pattern: str) -> Iterable[str]:
        cursor = 0
        while True:
            cursor, batch = self._r.scan(cursor=cursor, match=pattern, count=200)
            for k in batch:
                yield k
            if cursor == 0:
                break

    # ── Search-specific helpers ─────────────────────────────────────────────

    @staticmethod
    def search_key(*, tenant_id: str, client_id: Optional[str], payload_hash: str) -> str:
        scope = client_id or "all"
        return f"search:{tenant_id}:{scope}:{payload_hash}"

    def invalidate_search(self, tenant_id: str, client_id: Optional[str] = None) -> int:
        """Invalidate scoped search cache on memory_state bump."""
        if client_id:
            prefix = f"search:{tenant_id}:{client_id}:"
        else:
            prefix = f"search:{tenant_id}:"
        return self.delete_by_prefix(prefix)

    def invalidate_context_summary(self, tenant_id: str, client_id: str) -> None:
        self.delete(f"ctxsum:{tenant_id}:{client_id}")
