"""
API key verification service.

Looks up an API key by its sha256 hash, returns (tenant_id, key_id, scopes).
Uses Redis as a read-through cache (60s TTL) to avoid hitting Supabase on
every request. Falls through to the DB when Redis is unavailable.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Optional

from supabase import Client

from src.logging_config import get_logger

logger = get_logger(__name__)

_CACHE_TTL_SECONDS = 60
_TOUCH_DEBOUNCE_SECONDS = 60


class ApiKeyError(Exception):
    """Base for auth failures that the middleware can convert to 401."""


class ApiKeyNotFound(ApiKeyError):
    pass


class ApiKeyRevoked(ApiKeyError):
    pass


class ApiKeyExpired(ApiKeyError):
    pass


def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def key_prefix(raw_key: str) -> str:
    return raw_key[:8]


def _get_redis():
    try:
        import redis
        url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        r = redis.from_url(url, decode_responses=True)
        r.ping()
        return r
    except Exception:
        return None


class ApiKeyService:
    def __init__(self, supabase: Client):
        self.sb = supabase
        self._redis = _get_redis()

    def verify(self, raw_key: str) -> dict[str, Any]:
        """Return {tenant_id, key_id, scopes} or raise ApiKeyError."""
        if not raw_key:
            raise ApiKeyNotFound("Missing API key")

        h = hash_key(raw_key)

        # Cache hit?
        cached = self._cache_get(h)
        if cached is not None:
            if cached.get("status") == "revoked":
                raise ApiKeyRevoked("API key revoked")
            if cached.get("expired"):
                raise ApiKeyExpired("API key expired")
            self._maybe_touch(cached["key_id"])
            return cached

        # DB lookup
        try:
            res = (
                self.sb.table("api_keys")
                .select("id, tenant_id, scopes, status, expires_at")
                .eq("key_hash", h)
                .limit(1)
                .execute()
            )
        except Exception as e:
            logger.warning("api_key_lookup_failed", error=str(e))
            raise ApiKeyNotFound("Lookup failed") from e

        rows = res.data or []
        if not rows:
            # Negative-cache briefly to blunt brute force on random tokens
            self._cache_set(h, {"status": "not_found"}, ttl=10)
            raise ApiKeyNotFound("Unknown API key")

        row = rows[0]
        expired = False
        if row.get("expires_at"):
            from datetime import datetime, timezone
            try:
                exp = datetime.fromisoformat(row["expires_at"].replace("Z", "+00:00"))
                expired = exp <= datetime.now(timezone.utc)
            except Exception:
                pass

        entry = {
            "key_id": row["id"],
            "tenant_id": row["tenant_id"],
            "scopes": row.get("scopes") or [],
            "status": row["status"],
            "expired": expired,
        }
        self._cache_set(h, entry, ttl=_CACHE_TTL_SECONDS)

        if row["status"] == "revoked":
            raise ApiKeyRevoked("API key revoked")
        if expired:
            raise ApiKeyExpired("API key expired")

        self._maybe_touch(entry["key_id"])
        return entry

    # ── Redis helpers ──────────────────────────────────────────────────────

    def _cache_key(self, key_hash: str) -> str:
        # Only first 16 chars of hash used in cache key — full hash stored in value not needed
        return f"apikey:{key_hash[:16]}"

    def _cache_get(self, key_hash: str) -> Optional[dict[str, Any]]:
        if self._redis is None:
            return None
        try:
            raw = self._redis.get(self._cache_key(key_hash))
            if not raw:
                return None
            data = json.loads(raw)
            if data.get("status") == "not_found":
                # Force the exception path with NotFound
                raise ApiKeyNotFound("Unknown API key")
            return data
        except ApiKeyNotFound:
            raise
        except Exception:
            return None

    def _cache_set(self, key_hash: str, value: dict[str, Any], ttl: int) -> None:
        if self._redis is None:
            return
        try:
            self._redis.setex(self._cache_key(key_hash), ttl, json.dumps(value))
        except Exception:
            pass

    def _maybe_touch(self, key_id: str) -> None:
        """Bump last_used_at at most once per minute per key."""
        if self._redis is None:
            # Skip touch without Redis to avoid hammering the DB
            return
        debounce_key = f"apikey:touch:{key_id}"
        try:
            # SET NX EX — only set if not set, with expiry
            got = self._redis.set(debounce_key, str(int(time.time())), nx=True, ex=_TOUCH_DEBOUNCE_SECONDS)
            if not got:
                return
        except Exception:
            return
        try:
            self.sb.rpc("touch_api_key", {"p_key_id": key_id}).execute()
        except Exception:
            pass
