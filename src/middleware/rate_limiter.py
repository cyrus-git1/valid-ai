"""
Redis-backed rate limiter for the data plane.

Sliding window per (tenant, route-group). Tenant is read from
request.state.tenant_id (set by AuthMiddleware) — cannot be spoofed.

Each route group has its own (limit, window_seconds) so heavy endpoints
(large ingests) can have longer minimum intervals than light endpoints.

Fail-open if Redis is unreachable.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from src.logging_config import get_logger

logger = get_logger(__name__)

# Per-route-group limits: (max_requests, window_seconds) per tenant.
#
# Heavy ingests need longer minimum intervals — a processed-web batch can take
# minutes of server work, so 1 every 15 minutes gives the worker pool room to
# actually finish. A normal processed ingest is 90s.
ROUTE_LIMITS: dict[str, tuple[int, int]] = {
    "/ingest/processed-web":    (1,  900),    # 1 / 15 min
    "/ingest/processed":        (1,  90),     # 1 / 90 s  (most-specific first; see _match_limit)
    "/ingest/jobs":             (300, 60),    # status polling — generous, once per second per tenant avg
    "/search/graph":            (100, 60),
    "/data/documents/delete":   (30,  60),
    "/data/documents":          (30,  60),
    "/data/context/summary":    (60,  60),
    "/data/document-titles":    (120, 60),
    "/admin":                   (30,  60),
}

DEFAULT_LIMIT = int(os.environ.get("RATE_LIMIT_DEFAULT", "120"))
DEFAULT_WINDOW_SECONDS = 60

_BYPASS = {"/", "/admin/health", "/docs", "/openapi.json", "/redoc"}


def _get_redis():
    try:
        import redis
        url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        r = redis.from_url(url, decode_responses=True)
        r.ping()
        return r
    except Exception:
        return None


def _match_limit(path: str) -> tuple[int, int, str]:
    """Longest-prefix match. Returns (limit, window_seconds, matched_prefix)."""
    best: tuple[str, int, int] = ("", DEFAULT_LIMIT, DEFAULT_WINDOW_SECONDS)
    for prefix, (limit, window) in ROUTE_LIMITS.items():
        if path.startswith(prefix) and len(prefix) > len(best[0]):
            best = (prefix, limit, window)
    return best[1], best[2], best[0]


class RateLimiterMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: Any):
        super().__init__(app)
        self.enabled = os.environ.get("RATE_LIMIT_ENABLED", "true").lower() == "true"
        self._redis = _get_redis() if self.enabled else None

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        if not self.enabled or not self._redis:
            return await call_next(request)

        path = request.url.path
        if path in _BYPASS or request.method == "OPTIONS":
            return await call_next(request)

        tenant_id = getattr(request.state, "tenant_id", None)
        if not tenant_id:
            # Unauthenticated requests are rejected by AuthMiddleware; this is
            # a belt-and-suspenders no-op.
            return await call_next(request)

        limit, window_seconds, matched_prefix = _match_limit(path)
        # Group key uses the matched prefix so distinct rules stay in distinct
        # buckets (otherwise /ingest/processed-web would share a counter with
        # /ingest/processed).
        group = matched_prefix.strip("/") or "default"
        key = f"ratelimit:{tenant_id}:{group}"

        try:
            allowed, remaining, reset_at = self._check(key, limit, window_seconds)
        except Exception as e:
            logger.debug("rate_limiter_redis_error", error=str(e))
            return await call_next(request)

        request_id = getattr(request.state, "request_id", "-")
        if not allowed:
            logger.warning(
                "rate_limit_exceeded",
                group=group,
                limit=limit,
                window_seconds=window_seconds,
            )
            body = {
                "error": {
                    "code": "rate_limit.exceeded",
                    "message": "Rate limit exceeded",
                    "limit": limit,
                    "window": f"{window_seconds}s",
                    "retry_after": reset_at,
                    "request_id": request_id,
                }
            }
            return Response(
                content=json.dumps(body),
                status_code=429,
                media_type="application/json",
                headers={
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Window": f"{window_seconds}s",
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset_at),
                    "Retry-After": str(reset_at),
                    "X-Request-ID": request_id,
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Window"] = f"{window_seconds}s"
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_at)
        return response

    def _check(self, key: str, limit: int, window_seconds: int) -> tuple[bool, int, int]:
        now = time.time()
        window_start = now - window_seconds
        pipe = self._redis.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zcard(key)
        pipe.zadd(key, {f"{now}:{id(pipe)}": now})
        pipe.expire(key, window_seconds + 1)
        results = pipe.execute()
        current_count = results[1]
        remaining = max(0, limit - current_count - 1)
        reset_at = int(window_seconds)
        if current_count >= limit:
            return False, 0, reset_at
        return True, remaining, reset_at
