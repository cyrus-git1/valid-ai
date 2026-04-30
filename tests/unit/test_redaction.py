"""Unit tests for the PII redaction service."""
from __future__ import annotations

from types import SimpleNamespace

from src.services.redaction import (
    REVEAL_SCOPE,
    apply_redaction,
    caller_can_reveal,
    maybe_redact,
    redact_chunk_rows,
)


# ── apply_redaction ─────────────────────────────────────────────────────────


def test_apply_redaction_basic_substitution():
    text = "Jane Smith met with Acme on Tuesday."
    annotations = [
        {"type": "person", "span": [0, 10], "alias": "SUBJ_a3f1"},
        {"type": "org", "span": [20, 24], "alias": "ORG_b9c2"},
    ]
    out = apply_redaction(text, annotations)
    assert out == "SUBJ_a3f1 met with ORG_b9c2 on Tuesday."


def test_apply_redaction_preserves_text_when_no_annotations():
    text = "Nothing sensitive here."
    assert apply_redaction(text, None) == text
    assert apply_redaction(text, []) == text


def test_apply_redaction_empty_text():
    assert apply_redaction("", [{"type": "x", "span": [0, 1], "alias": "A"}]) == ""


def test_apply_redaction_handles_overlapping_back_to_front():
    # Two non-overlapping spans where naive front-to-back order would break offsets
    text = "Alice and Bob talked."
    annotations = [
        {"type": "person", "span": [0, 5], "alias": "P1"},
        {"type": "person", "span": [10, 13], "alias": "P2"},
    ]
    out = apply_redaction(text, annotations)
    assert out == "P1 and P2 talked."


def test_apply_redaction_skips_invalid_annotations():
    text = "Hello world"
    annotations = [
        {"type": "x", "span": "not-a-list", "alias": "A"},   # bad span
        {"type": "x", "span": [0, 5]},                       # missing alias
        {"type": "x", "span": [100, 200], "alias": "OOR"},   # out of range
        {"type": "x", "span": [5, 0], "alias": "INV"},       # reversed
        "not-a-dict",
        {"type": "x", "span": [0, 5], "alias": "OK"},        # this one survives
    ]
    out = apply_redaction(text, annotations)
    assert out == "OK world"


# ── caller_can_reveal ───────────────────────────────────────────────────────


def _req_with_scopes(scopes):
    return SimpleNamespace(state=SimpleNamespace(scopes=scopes))


def test_caller_can_reveal_true_when_scope_present():
    assert caller_can_reveal(_req_with_scopes(["read", REVEAL_SCOPE])) is True


def test_caller_can_reveal_false_without_scope():
    assert caller_can_reveal(_req_with_scopes(["read", "write"])) is False


def test_caller_can_reveal_false_for_none_request():
    assert caller_can_reveal(None) is False


def test_caller_can_reveal_false_when_state_missing():
    assert caller_can_reveal(SimpleNamespace()) is False


# ── maybe_redact ────────────────────────────────────────────────────────────


def test_maybe_redact_redacts_when_no_request():
    out = maybe_redact("Jane",
                       [{"type": "person", "span": [0, 4], "alias": "P"}],
                       request=None)
    assert out == "P"


def test_maybe_redact_returns_raw_when_caller_can_reveal():
    req = _req_with_scopes(["read", REVEAL_SCOPE])
    out = maybe_redact("Jane",
                       [{"type": "person", "span": [0, 4], "alias": "P"}],
                       request=req)
    assert out == "Jane"


def test_maybe_redact_force_redact_overrides_reveal():
    req = _req_with_scopes(["read", REVEAL_SCOPE])
    out = maybe_redact("Jane",
                       [{"type": "person", "span": [0, 4], "alias": "P"}],
                       request=req,
                       force_redact=True)
    assert out == "P"


def test_maybe_redact_passes_through_none():
    assert maybe_redact(None, []) is None


# ── redact_chunk_rows ───────────────────────────────────────────────────────


def test_redact_chunk_rows_mutates_in_place():
    rows = [
        {"id": 1, "content": "Jane talked to Bob",
         "pii_annotations": [{"type": "person", "span": [0, 4], "alias": "P1"}]},
        {"id": 2, "content": "Plain text", "pii_annotations": []},
    ]
    out = redact_chunk_rows(rows, request=None)
    assert out is rows  # same object
    assert rows[0]["content"] == "P1 talked to Bob"
    assert rows[1]["content"] == "Plain text"


def test_redact_chunk_rows_skips_when_caller_can_reveal():
    rows = [
        {"id": 1, "content": "Jane",
         "pii_annotations": [{"type": "person", "span": [0, 4], "alias": "P1"}]},
    ]
    redact_chunk_rows(rows, request=_req_with_scopes([REVEAL_SCOPE]))
    assert rows[0]["content"] == "Jane"
