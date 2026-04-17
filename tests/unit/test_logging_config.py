"""Smoke tests for structured logging configuration."""
from __future__ import annotations

import json
import logging

import structlog

from src.logging_config import bind_context, clear_context, configure_logging, get_logger


def _reconfigure_capturing(capsys):
    # Wipe any existing handlers (configure_logging is a no-op if they're already set)
    logging.getLogger().handlers.clear()
    configure_logging("INFO")


def test_json_logs_include_context(capsys, monkeypatch):
    monkeypatch.setenv("LOG_FORMAT", "json")
    _reconfigure_capturing(capsys)
    log = get_logger("test")
    clear_context()
    bind_context(request_id="req-xyz", tenant_id="t-1")
    log.info("hello", count=1)

    captured = capsys.readouterr()
    combined = (captured.out + captured.err).strip().splitlines()
    assert combined, "expected at least one log line"
    parsed = None
    for line in reversed(combined):
        try:
            parsed = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    assert parsed is not None
    assert parsed["event"] == "hello"
    assert parsed["request_id"] == "req-xyz"
    assert parsed["tenant_id"] == "t-1"
    assert parsed["count"] == 1


def test_clear_context_removes_vars(capsys, monkeypatch):
    monkeypatch.setenv("LOG_FORMAT", "json")
    _reconfigure_capturing(capsys)
    log = get_logger("test")
    bind_context(request_id="req-a")
    clear_context()
    log.info("second")
    captured = capsys.readouterr()
    combined = (captured.out + captured.err).strip().splitlines()
    parsed = None
    for line in reversed(combined):
        try:
            parsed = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    assert parsed is not None
    assert "request_id" not in parsed
