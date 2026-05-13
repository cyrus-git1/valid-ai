"""Unit tests for plan_limits config and TenantPlanService."""
from __future__ import annotations

import pytest

from src.config.plan_limits import GLOBAL_MAX_BODY_BYTES, PLAN_LIMITS, VALID_PLANS, get_limit


def test_three_tiers_defined():
    assert set(VALID_PLANS) == {"free", "pro", "enterprise"}
    for tier in VALID_PLANS:
        assert tier in PLAN_LIMITS


def test_tier_keys_are_complete():
    required = {"max_body_bytes", "max_chunks_per_ingest", "daily_embedding_tokens"}
    for tier, limits in PLAN_LIMITS.items():
        assert required.issubset(limits.keys()), f"{tier} missing keys"


def test_tiers_are_monotonic():
    # Each tier should be at least as permissive as the previous
    for key in ("max_body_bytes", "max_chunks_per_ingest", "daily_embedding_tokens"):
        assert PLAN_LIMITS["free"][key] <= PLAN_LIMITS["pro"][key] <= PLAN_LIMITS["enterprise"][key]


def test_get_limit_falls_back_to_free_for_unknown_plan():
    free_max = PLAN_LIMITS["free"]["max_body_bytes"]
    assert get_limit("nonexistent", "max_body_bytes") == free_max


def test_get_limit_unknown_key_raises_keyerror():
    # The function falls back to free-tier values, but only for keys that exist
    # in the free tier. An unknown key isn't in any tier — should KeyError.
    with pytest.raises(KeyError):
        get_limit("free", "nonexistent_key")


def test_global_ceiling_is_at_least_enterprise_max():
    assert GLOBAL_MAX_BODY_BYTES >= PLAN_LIMITS["enterprise"]["max_body_bytes"]


def test_free_tier_caps_are_tight():
    # Free tier should be tight enough to push customers toward upgrades
    assert PLAN_LIMITS["free"]["max_body_bytes"] <= 50 * 1024 * 1024
    assert PLAN_LIMITS["free"]["max_chunks_per_ingest"] <= 1000


# ── EmbeddingQuotaService ──────────────────────────────────────────────────


class _FakeRedis:
    """In-memory Redis stub for quota tests."""

    def __init__(self):
        self.store: dict[str, int] = {}

    def get(self, key):
        v = self.store.get(key)
        return str(v) if v is not None else None

    def incrby(self, key, n):
        self.store[key] = self.store.get(key, 0) + n
        return self.store[key]

    def decrby(self, key, n):
        self.store[key] = self.store.get(key, 0) - n
        return self.store[key]

    def expire(self, key, ttl):
        return True

    def ping(self):
        return True


class _FakePlanService:
    def __init__(self, plan: str):
        self.plan = plan

    def get_plan(self, tenant_id: str) -> str:
        return self.plan


def test_check_and_consume_under_quota_succeeds():
    from src.services.tenant_plan_service import EmbeddingQuotaService
    svc = EmbeddingQuotaService(_FakePlanService("pro"))
    svc._redis = _FakeRedis()
    # Pro tier is 1M tokens/day; 1000 should pass
    svc.check_and_consume("tenant-1", 1000)
    assert svc.current_usage("tenant-1") == 1000


def test_check_and_consume_exceeds_quota_raises():
    from fastapi import HTTPException

    from src.services.tenant_plan_service import EmbeddingQuotaService
    svc = EmbeddingQuotaService(_FakePlanService("free"))
    svc._redis = _FakeRedis()
    # Free tier is 100k tokens/day; this should burn through it
    svc.check_and_consume("tenant-1", 99_000)
    with pytest.raises(HTTPException) as exc:
        svc.check_and_consume("tenant-1", 2000)
    assert exc.value.status_code == 429
    assert exc.value.detail["code"] == "quota.embedding_tokens_exceeded"
    # The failed charge should NOT have been kept (rollback)
    assert svc.current_usage("tenant-1") == 99_000


def test_check_and_consume_no_redis_fail_open():
    from src.services.tenant_plan_service import EmbeddingQuotaService
    svc = EmbeddingQuotaService(_FakePlanService("free"))
    svc._redis = None  # simulate Redis outage
    # Should not raise; fail-open for cost-control
    svc.check_and_consume("tenant-1", 999_999_999)


def test_reconcile_actual_adjusts_running_total():
    from src.services.tenant_plan_service import EmbeddingQuotaService
    svc = EmbeddingQuotaService(_FakePlanService("pro"))
    svc._redis = _FakeRedis()
    svc.check_and_consume("tenant-1", 1000)
    # OpenAI reported 1200 actual; reconcile adds 200 more
    svc.reconcile_actual("tenant-1", estimated=1000, actual=1200)
    assert svc.current_usage("tenant-1") == 1200


def test_estimate_tokens_rough_4_char_per_token():
    from src.services.tenant_plan_service import estimate_tokens
    assert estimate_tokens(["hello", "world"]) == 2  # 5//4=1 each, min 1
    assert estimate_tokens(["a" * 400]) == 100       # 400/4
