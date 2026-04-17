"""
Request ID middleware — assigns a correlation ID to every request.

Reads X-Request-ID header if present, otherwise generates a UUID.
Binds to structlog contextvars so every log line in the request
lifecycle carries the request_id. Echoes back in response headers.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from src.logging_config import bind_context, clear_context, get_logger

logger = get_logger(__name__)


class RequestIdMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: Any):
        super().__init__(app)

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        # Fresh context per request
        clear_context()
        bind_context(
            request_id=request_id,
            path=request.url.path,
            method=request.method,
        )

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            logger.exception("request_failed", duration_ms=duration_ms)
            raise

        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        response.headers["X-Request-ID"] = request_id

        # Only log non-health requests to reduce noise
        if request.url.path not in ("/", "/admin/health"):
            logger.info(
                "request_completed",
                status_code=response.status_code,
                duration_ms=duration_ms,
            )

        return response
