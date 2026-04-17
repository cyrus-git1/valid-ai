"""
Centralized exception handlers for the data plane.

Produces a uniform error envelope:
  {"error": {"code": "...", "message": "...", "request_id": "..."}}

- HTTPException: status preserved, code derived from detail or "http.{status}"
- Validation errors: 422, code "validation.failed"
- Anything else: 500, code "internal.error", full traceback logged but
  not leaked in the response body.
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.logging_config import get_logger

logger = get_logger(__name__)


def _envelope(code: str, message: str | Any, request_id: str, **extra: Any) -> dict:
    return {"error": {"code": code, "message": message, "request_id": request_id, **extra}}


def register(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exc(request: Request, exc: StarletteHTTPException):
        request_id = getattr(request.state, "request_id", "-")
        code = f"http.{exc.status_code}"
        logger.info("http_exception", status_code=exc.status_code, detail=str(exc.detail))
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(code, exc.detail, request_id),
            headers={"X-Request-ID": request_id},
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation(request: Request, exc: RequestValidationError):
        request_id = getattr(request.state, "request_id", "-")
        logger.info("validation_error", errors=exc.errors())
        return JSONResponse(
            status_code=422,
            content=_envelope("validation.failed", "Request validation failed", request_id, errors=exc.errors()),
            headers={"X-Request-ID": request_id},
        )

    @app.exception_handler(Exception)
    async def handle_unhandled(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", "-")
        logger.exception("unhandled_exception", error_type=type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content=_envelope("internal.error", "Internal server error", request_id),
            headers={"X-Request-ID": request_id},
        )
