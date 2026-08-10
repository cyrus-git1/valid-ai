"""
Auth middleware — verifies X-API-Key header and binds tenant/key identity.

Reads X-API-Key, resolves via ApiKeyService (Redis-cached), and attaches
tenant_id/key_id/scopes to request.state. Also enforces that any tenant_id
passed in request body/query must match the key's tenant — prevents tenant
confusion attacks.

Bypass paths (no auth required): /, /admin/health, /docs, /openapi.json, /redoc
"""
from __future__ import annotations

import json
import re
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from src.logging_config import bind_context, get_logger
from src.services.api_key_service import (
    ApiKeyError,
    ApiKeyExpired,
    ApiKeyNotFound,
    ApiKeyRevoked,
    ApiKeyService,
)
from src.db.supabase_client import get_supabase

logger = get_logger(__name__)

_BYPASS_EXACT = {"/", "/admin/health", "/docs", "/openapi.json", "/redoc"}
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


def _json_error(status: int, message: str, request_id: str, code: str) -> Response:
    body = {"error": {"code": code, "message": message, "request_id": request_id}}
    return Response(
        content=json.dumps(body),
        status_code=status,
        media_type="application/json",
        headers={"X-Request-ID": request_id},
    )


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: Any):
        super().__init__(app)
        self._service: ApiKeyService | None = None

    def _svc(self) -> ApiKeyService:
        if self._service is None:
            self._service = ApiKeyService(get_supabase())
        return self._service

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        path = request.url.path
        request_id = getattr(request.state, "request_id", "-")

        if path in _BYPASS_EXACT or request.method == "OPTIONS":
            return await call_next(request)

        raw_key = request.headers.get("X-API-Key")
        if not raw_key:
            logger.warning("auth_missing_key")
            return _json_error(401, "Missing X-API-Key header", request_id, "auth.missing_key")

        try:
            entry = self._svc().verify(raw_key)
        except ApiKeyNotFound:
            logger.warning("auth_unknown_key")
            return _json_error(401, "Invalid API key", request_id, "auth.invalid_key")
        except ApiKeyRevoked:
            logger.warning("auth_revoked_key")
            return _json_error(401, "API key revoked", request_id, "auth.revoked_key")
        except ApiKeyExpired:
            logger.warning("auth_expired_key")
            return _json_error(401, "API key expired", request_id, "auth.expired_key")
        except ApiKeyError as e:
            logger.warning("auth_error", error=str(e))
            return _json_error(401, "Authentication failed", request_id, "auth.failed")

        tenant_id = entry["tenant_id"]
        key_id = entry["key_id"]
        scopes = entry.get("scopes", [])

        request.state.tenant_id = tenant_id
        request.state.key_id = key_id
        request.state.scopes = scopes
        bind_context(tenant_id=tenant_id, key_id=key_id)

        # Enforce tenant match: any tenant_id in query params must equal the key's tenant
        q_tenant = request.query_params.get("tenant_id")
        if q_tenant and q_tenant.lower() != str(tenant_id).lower():
            logger.warning("auth_tenant_mismatch_query", claimed=q_tenant)
            return _json_error(403, "Tenant mismatch", request_id, "auth.tenant_mismatch")

        # Body check (best-effort, POST/PATCH/PUT with JSON)
        if request.method in ("POST", "PATCH", "PUT") and "application/json" in (request.headers.get("content-type") or ""):
            body_tenant = await self._peek_body_tenant(request)
            if body_tenant and body_tenant.lower() != str(tenant_id).lower():
                logger.warning("auth_tenant_mismatch_body", claimed=body_tenant)
                return _json_error(403, "Tenant mismatch", request_id, "auth.tenant_mismatch")

        return await call_next(request)

    async def _peek_body_tenant(self, request: Request) -> str | None:
        """Read body once, cache on request for downstream handlers."""
        try:
            body = await request.body()
        except Exception:
            return None
        if not body:
            return None

        # Make the body readable again by downstream handlers
        async def _receive():
            return {"type": "http.request", "body": body, "more_body": False}
        request._receive = _receive  # type: ignore[attr-defined]

        try:
            payload = json.loads(body)
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        tid = payload.get("tenant_id")
        if not tid or not isinstance(tid, str):
            return None
        if not _UUID_RE.match(tid):
            return None
        return tid
