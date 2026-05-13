"""
BodySizeMiddleware — enforces per-plan request body size limits.

Two checks:
  1. Global ceiling (env-tunable, default 500 MB). Applied to every request
     regardless of auth state. Protects against pure DoS uploads.
  2. Per-plan limit. Applied after AuthMiddleware has set
     `request.state.tenant_id`. Looks up the tenant's subscription plan and
     enforces the plan's `max_body_bytes` from PLAN_LIMITS.

Only checks Content-Length — the body is never read. Cheap pre-flight.

GET/DELETE/OPTIONS/HEAD requests bypass the check entirely.

Bypass scope: callers with the `admin` scope skip the per-plan check (still
subject to the global ceiling). Designed for service-account bulk ops.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from src.config.plan_limits import GLOBAL_MAX_BODY_BYTES, get_limit
from src.logging_config import get_logger

logger = get_logger(__name__)

_BYPASS_METHODS = {"GET", "DELETE", "OPTIONS", "HEAD"}
_BYPASS_PATHS = {"/", "/admin/health", "/docs", "/openapi.json", "/redoc"}


def _error(status: int, code: str, message: str, request_id: str, **extra: Any) -> Response:
    body = {"error": {"code": code, "message": message, "request_id": request_id, **extra}}
    return Response(
        content=json.dumps(body),
        status_code=status,
        media_type="application/json",
        headers={"X-Request-ID": request_id},
    )


class BodySizeMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: Any):
        super().__init__(app)
        # Lazy-init service so we don't open Supabase connections at import time
        self._plan_svc: Optional[Any] = None

    def _plan_service(self):
        if self._plan_svc is None:
            from src.services.tenant_plan_service import TenantPlanService
            from src.supabase.supabase_client import get_supabase
            self._plan_svc = TenantPlanService(get_supabase())
        return self._plan_svc

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        if request.method in _BYPASS_METHODS or request.url.path in _BYPASS_PATHS:
            return await call_next(request)

        cl_header = request.headers.get("content-length")
        if not cl_header:
            # No content-length (chunked transfer). Let it through; the
            # global ceiling can't be checked without reading the body.
            return await call_next(request)

        try:
            content_length = int(cl_header)
        except (TypeError, ValueError):
            return await call_next(request)

        request_id = getattr(request.state, "request_id", "-")

        # 1. Global ceiling
        if content_length > GLOBAL_MAX_BODY_BYTES:
            logger.warning(
                "body_size.global_exceeded",
                content_length=content_length,
                global_max=GLOBAL_MAX_BODY_BYTES,
            )
            return _error(
                413,
                "payload.too_large",
                f"Request body exceeds global maximum ({GLOBAL_MAX_BODY_BYTES} bytes).",
                request_id,
                content_length=content_length,
                global_max=GLOBAL_MAX_BODY_BYTES,
            )

        # 2. Per-plan limit (requires auth context)
        tenant_id = getattr(request.state, "tenant_id", None)
        scopes = getattr(request.state, "scopes", []) or []
        if not tenant_id:
            # No tenant context — auth middleware either bypassed or rejected.
            # Don't apply per-plan check; either auth will reject or it's a
            # public route already past us.
            return await call_next(request)

        # Admin scope skips the per-plan check (still subject to global)
        if "admin" in scopes:
            return await call_next(request)

        try:
            plan = self._plan_service().get_plan(str(tenant_id))
        except Exception as e:
            logger.warning("body_size.plan_lookup_failed", error=str(e))
            plan = "free"

        plan_max = get_limit(plan, "max_body_bytes")
        if content_length > plan_max:
            logger.warning(
                "body_size.plan_exceeded",
                tenant_id=str(tenant_id),
                plan=plan,
                content_length=content_length,
                plan_max=plan_max,
            )
            return _error(
                413,
                "payload.too_large_for_plan",
                f"Request body ({content_length:,} bytes) exceeds your '{plan}' plan limit "
                f"({plan_max:,} bytes). Upgrade your plan or split the request.",
                request_id,
                plan=plan,
                content_length=content_length,
                plan_max=plan_max,
            )

        return await call_next(request)
