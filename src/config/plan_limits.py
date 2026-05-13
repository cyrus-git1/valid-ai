"""
Subscription tier limits.

Three tiers, applied at three enforcement points:
  - Body bytes: middleware (per-request)
  - Chunk count: ingest handlers (per-request)
  - Embedding tokens: ingest embedding helper (per-tenant per-day)

Limits chosen to be:
  - Free: enough for evaluation / single-document trials
  - Pro: enough for typical B2B SaaS production usage
  - Enterprise: generous; real customers contractually exceeding this
    should be upgraded explicitly, not silently throttled.

Override at runtime via PLAN_LIMITS_OVERRIDE_JSON env var if needed (a JSON
blob with the same shape). Useful for staging / testing.
"""
from __future__ import annotations

import json
import os
from typing import Dict


_DEFAULT_LIMITS: Dict[str, Dict[str, int]] = {
    "free": {
        # Body upload max (bytes). 10 MB.
        "max_body_bytes": 10 * 1024 * 1024,
        # Chunks per ingest request.
        "max_chunks_per_ingest": 500,
        # Embedding tokens per tenant per UTC day.
        "daily_embedding_tokens": 100_000,
    },
    "pro": {
        "max_body_bytes": 50 * 1024 * 1024,        # 50 MB
        "max_chunks_per_ingest": 2_000,
        "daily_embedding_tokens": 1_000_000,
    },
    "enterprise": {
        "max_body_bytes": 200 * 1024 * 1024,       # 200 MB
        "max_chunks_per_ingest": 10_000,
        "daily_embedding_tokens": 10_000_000,
    },
}


def _load_overrides() -> Dict[str, Dict[str, int]]:
    raw = os.environ.get("PLAN_LIMITS_OVERRIDE_JSON")
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return parsed


PLAN_LIMITS: Dict[str, Dict[str, int]] = {**_DEFAULT_LIMITS, **_load_overrides()}

# Hard ceiling — applied regardless of plan. Anything above this is
# rejected at the gateway level (mostly so a misconfigured enterprise
# plan can't silently allow gigabyte uploads).
GLOBAL_MAX_BODY_BYTES = int(
    os.environ.get("GLOBAL_MAX_BODY_BYTES", str(500 * 1024 * 1024))   # 500 MB
)


def get_limit(plan: str, key: str) -> int:
    """Return the limit for a plan/key pair, falling back to the free tier."""
    plan_limits = PLAN_LIMITS.get(plan) or PLAN_LIMITS["free"]
    return int(plan_limits.get(key, PLAN_LIMITS["free"][key]))


VALID_PLANS = ("free", "pro", "enterprise")
