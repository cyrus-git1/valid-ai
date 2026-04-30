"""
PII redaction at output time.

Strategy (Option 1 — Redact at output, not at storage):
  - chunks.content stays as the original plaintext
  - chunks.pii_annotations carries [{type, span: [start, end], alias, ...}]
  - Whenever chunk content leaves the data plane, we substitute the spans
    with their aliases unless the caller has the `pii:reveal` scope.

The agent's PII detector is the single source of truth for what counts as
PII and what alias a given span maps to. The data plane just applies what
the agent produced.

Audit:
  - When `pii:reveal` is used to bypass redaction, the request is already
    auditable via audit_log (the request itself is captured). We don't
    audit every chunk read — that would generate one row per chunk per
    response and overwhelm the log. Audit at request-level via the
    standard audit_log entry the route handler writes.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


REVEAL_SCOPE = "pii:reveal"


def apply_redaction(text: str, annotations: Optional[List[Dict[str, Any]]]) -> str:
    """Substitute PII spans with their aliases. Idempotent for empty/missing input.

    Annotations format (set by the agent's PII detector):
        [{"type": "person", "span": [start, end], "alias": "SUBJ_a3f1"}, ...]

    Spans are character offsets. Processed back-to-front so earlier offsets
    aren't invalidated by length changes.
    """
    if not annotations or not text:
        return text

    valid: List[tuple[int, int, str]] = []
    for a in annotations:
        if not isinstance(a, dict):
            continue
        span = a.get("span")
        alias = a.get("alias")
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            continue
        if not alias:
            continue
        try:
            start = int(span[0])
            end = int(span[1])
        except (TypeError, ValueError):
            continue
        if 0 <= start < end <= len(text):
            valid.append((start, end, str(alias)))

    if not valid:
        return text

    valid.sort(key=lambda t: t[0], reverse=True)
    out = text
    for start, end, alias in valid:
        out = out[:start] + alias + out[end:]
    return out


def caller_can_reveal(request) -> bool:
    """True if the caller has been granted the `pii:reveal` scope."""
    if request is None:
        return False
    state = getattr(request, "state", None)
    if state is None:
        return False
    scopes = getattr(state, "scopes", None) or []
    return REVEAL_SCOPE in scopes


def maybe_redact(
    text: Optional[str],
    annotations: Optional[List[Dict[str, Any]]],
    *,
    request=None,
    force_redact: bool = False,
) -> Optional[str]:
    """Apply redaction unless caller can reveal (or force_redact wins).

    - request=None: redact (no caller context = err on the side of redaction)
    - caller has pii:reveal scope: return original
    - force_redact=True: always redact, ignore scope
    - default: redact
    """
    if text is None:
        return None
    if force_redact:
        return apply_redaction(text, annotations)
    if caller_can_reveal(request):
        return text
    return apply_redaction(text, annotations)


def redact_chunk_rows(
    rows: List[Dict[str, Any]],
    *,
    request=None,
    content_field: str = "content",
    annotations_field: str = "pii_annotations",
    force_redact: bool = False,
) -> List[Dict[str, Any]]:
    """Mutate-and-return a list of chunk-shaped dicts with redacted content.

    Skips rows that don't have either field. Returns the same list reference
    for convenience but values are mutated in place.
    """
    for row in rows:
        if not isinstance(row, dict):
            continue
        text = row.get(content_field)
        ann = row.get(annotations_field)
        if text is None:
            continue
        row[content_field] = maybe_redact(
            text, ann, request=request, force_redact=force_redact
        )
    return rows
