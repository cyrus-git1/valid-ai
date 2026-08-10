"""
TenantPlanService — looks up a tenant's subscription tier with Redis caching.

Fail-open: on any error, returns 'free' (the most restrictive plan). This
biases toward rejecting overly large requests under failure rather than
letting them through.
"""
from __future__ import annotations

import os
from typing import Optional

from supabase import Client

from src.logging_config import get_logger

logger = get_logger(__name__)

_CACHE_TTL_SECONDS = 300  # 5 minutes


def _get_redis():
    try:
        import redis
        url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        r = redis.from_url(url, decode_responses=True)
        r.ping()
        return r
    except Exception:
        return None


class TenantPlanService:
    def __init__(self, supabase: Client):
        self.sb = supabase
        self._redis = _get_redis()

    def get_plan(self, tenant_id: str) -> str:
        """Return the tenant's plan name. Defaults to 'free' for unknown tenants."""
        if not tenant_id:
            return "free"

        tenant_id = str(tenant_id)
        cached = self._cache_get(tenant_id)
        if cached is not None:
            return cached

        plan = "free"
        try:
            res = (
                self.sb.table("tenant_plans")
                .select("plan")
                .eq("tenant_id", tenant_id)
                .limit(1)
                .execute()
            )
            rows = res.data or []
            if rows and rows[0].get("plan"):
                plan = rows[0]["plan"]
        except Exception as e:
            logger.warning("tenant_plan_lookup_failed tenant=%s error=%s", tenant_id, e)
            plan = "free"

        self._cache_set(tenant_id, plan)
        return plan

    def set_plan(self, tenant_id: str, plan: str, notes: Optional[str] = None) -> None:
        """Upsert the tenant's plan via RPC and invalidate the cache."""
        try:
            self.sb.rpc("upsert_tenant_plan", {
                "p_tenant_id": str(tenant_id),
                "p_plan": plan,
                "p_notes": notes,
            }).execute()
            self._cache_invalidate(str(tenant_id))
        except Exception as e:
            logger.warning("tenant_plan_set_failed tenant=%s error=%s", tenant_id, e)
            raise

    # ── Redis helpers ──────────────────────────────────────────────────────

    def _cache_key(self, tenant_id: str) -> str:
        return f"tenant_plan:{tenant_id}"

    def _cache_get(self, tenant_id: str) -> Optional[str]:
        if self._redis is None:
            return None
        try:
            raw = self._redis.get(self._cache_key(tenant_id))
            return raw if raw else None
        except Exception:
            return None

    def _cache_set(self, tenant_id: str, plan: str) -> None:
        if self._redis is None:
            return
        try:
            self._redis.setex(self._cache_key(tenant_id), _CACHE_TTL_SECONDS, plan)
        except Exception:
            pass

    def _cache_invalidate(self, tenant_id: str) -> None:
        if self._redis is None:
            return
        try:
            self._redis.delete(self._cache_key(tenant_id))
        except Exception:
            pass


# ── Embedding-token quota (Option B) ────────────────────────────────────────

class EmbeddingQuotaService:
    """Tracks daily embedding-token consumption per tenant in Redis.

    Reject-then-charge: callers do `check_and_consume(tenant_id, estimated)`
    before the OpenAI call. If the quota would be exceeded, raises. After the
    OpenAI call returns the real token usage, callers reconcile via
    `reconcile_actual(tenant_id, estimated, actual)`.

    If Redis is unavailable, behaves as no-op (fail-open — token quota is a
    cost-control, not a security boundary).
    """

    def __init__(self, plan_service: TenantPlanService):
        self.plan_service = plan_service
        self._redis = _get_redis()

    def _today_key(self, tenant_id: str) -> str:
        from datetime import datetime, timezone
        ymd = datetime.now(timezone.utc).strftime("%Y%m%d")
        return f"embed_tokens:{tenant_id}:{ymd}"

    def _quota_for(self, tenant_id: str) -> int:
        from src.config.plan_limits import get_limit
        plan = self.plan_service.get_plan(tenant_id)
        return get_limit(plan, "daily_embedding_tokens")

    def current_usage(self, tenant_id: str) -> int:
        if self._redis is None:
            return 0
        try:
            v = self._redis.get(self._today_key(tenant_id))
            return int(v) if v else 0
        except Exception:
            return 0

    def check_and_consume(self, tenant_id: str, token_estimate: int) -> None:
        """Charge `token_estimate` against the daily bucket. Raises HTTPException(429)
        if it would exceed quota. No-op if Redis unavailable.
        """
        if not tenant_id or token_estimate <= 0:
            return
        if self._redis is None:
            return  # fail-open; cost-control, not security

        quota = self._quota_for(tenant_id)
        key = self._today_key(tenant_id)
        try:
            new_usage = self._redis.incrby(key, token_estimate)
            # 25 hour TTL so the bucket auto-cleans after the UTC day rolls over
            self._redis.expire(key, 60 * 60 * 25)
        except Exception as e:
            logger.warning("embedding_quota_redis_error tenant=%s error=%s", tenant_id, e)
            return

        if new_usage > quota:
            # Undo the charge so a retry tomorrow can succeed
            try:
                self._redis.decrby(key, token_estimate)
            except Exception:
                pass
            from fastapi import HTTPException
            plan = self.plan_service.get_plan(tenant_id)
            raise HTTPException(
                status_code=429,
                detail={
                    "code": "quota.embedding_tokens_exceeded",
                    "plan": plan,
                    "daily_limit": quota,
                    "used_today": new_usage - token_estimate,
                    "requested": token_estimate,
                    "message": (
                        f"Daily embedding-token quota exceeded for plan '{plan}' "
                        f"({quota:,} tokens/day). Upgrade or retry tomorrow."
                    ),
                },
            )

    def reconcile_actual(self, tenant_id: str, estimated: int, actual: int) -> None:
        """Adjust the running total when the actual token usage differs from estimate."""
        if not tenant_id or self._redis is None:
            return
        delta = actual - estimated
        if delta == 0:
            return
        try:
            key = self._today_key(tenant_id)
            self._redis.incrby(key, delta)
            self._redis.expire(key, 60 * 60 * 25)
        except Exception:
            pass


def estimate_tokens(texts: list[str]) -> int:
    """Rough estimate: 4 chars per token. Pessimistic enough for pre-flight checks."""
    return sum(max(1, len(t) // 4) for t in texts)
