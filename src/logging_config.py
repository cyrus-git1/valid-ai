"""
Structured logging setup using structlog.

Produces JSON log lines to stdout with auto-injected context:
  - request_id, tenant_id, key_id (bound per-request by middleware)
  - path, method, status_code, duration_ms (bound by middleware)

Replaces stdlib logging.basicConfig in main.py.
"""
from __future__ import annotations

import logging
import os
import sys

import structlog


def configure_logging(level: str | int = "INFO") -> None:
    """Configure structlog + stdlib logging to emit JSON to stdout."""
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Route stdlib logging through structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # Pretty-print in dev, JSON in prod
    use_json = os.environ.get("LOG_FORMAT", "json").lower() == "json"
    if use_json:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None):
    """Return a structlog logger. Use this in place of logging.getLogger."""
    return structlog.get_logger(name)


def bind_context(**kwargs) -> None:
    """Bind key/value pairs to the current async-context for all subsequent logs."""
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """Clear all context variables (call at end of request)."""
    structlog.contextvars.clear_contextvars()
