"""
ingest_service.py
-----------------
Worker-callable wrappers around the heavy ingest pipeline in
src.routers.ingest_router. The router delegates the real work to
functions here so an arq worker can invoke them without holding
an HTTP request.

The router's `_execute_*` functions still live in that module so we
don't have to move hundreds of lines of tightly coupled helpers.
Here we just adapt dict payloads to those functions.
"""
from __future__ import annotations

from typing import Any, Dict

from src.logging_config import get_logger

logger = get_logger(__name__)


def run_processed_ingest(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a /ingest/processed job from a JSON-serializable payload."""
    from src.routers.ingest_router import (
        ProcessedDocumentRequest,
        _execute_processed_ingest,
    )
    req = ProcessedDocumentRequest.model_validate(payload)
    result = _execute_processed_ingest(req)
    return result.model_dump()


def run_processed_web_ingest(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a /ingest/processed-web job from a JSON-serializable payload."""
    from src.routers.ingest_router import (
        ProcessedWebRequest,
        _execute_processed_web_ingest,
    )
    req = ProcessedWebRequest.model_validate(payload)
    result = _execute_processed_web_ingest(req)
    return result.model_dump()
