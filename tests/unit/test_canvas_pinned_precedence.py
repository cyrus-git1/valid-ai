"""The pinned-block guard in `upsert_canvas_block` must cover the claim, not just the name.

Migration 66 wrote the governance contract into its own header — "an agent refresh NEVER
overwrites a pinned statement; it only updates evidenced / evidence_refs / divergence" — and
then guarded exactly one of the fields that promise covers. `v_name` was held back for a
pinned block; `v_props` was built identically either way, so an agent upsert still wrote
`stated`, `status`, `confidence` and `source` over a human's pinned block. Migration 78 fixes
it.

WHY THIS TEST IS A STATIC CHECK. The logic lives in plpgsql, and this repo has no database in
its test path — every service test mocks Supabase. The two honest options were a Python
reimplementation of the function (which would assert that my copy of the rule matches itself,
the vacuous-test shape) or a structural check on the artifact that actually ships. This is the
second. It cannot prove the function behaves correctly at runtime; it CAN prove that no
caller-supplied claim field is written into a pinned block without consulting the guard, which
is the regression class — someone adds a property to `v_props` and null-coalesces it like its
neighbours, and the protection quietly stops covering it.

Reads whichever migration most recently defines the function, so it keeps testing the live
definition rather than pinning migration 78 forever.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_MIGRATIONS = Path(__file__).resolve().parents[2] / "supabase" / "migrations"

# Fields the contract permits an agent refresh to update on a PINNED block. Evidence is the
# whole point of a refresh, and `divergence` is how it registers disagreement without
# overwriting what the human asserted.
_EVIDENCE = {"evidenced", "divergence", "evidence_refs"}

# Derived from the call's own arguments or from the guard itself — never a claim the caller
# asserts about the block's content, so they need no protection.
_DERIVED = {"external_ref", "block_key", "scope", "pinned"}


def _latest_definition() -> str:
    """The body of the most recent migration that defines `upsert_canvas_block`."""
    hits = sorted(
        p for p in _MIGRATIONS.glob("*.sql")
        if "create or replace function public.upsert_canvas_block" in p.read_text(encoding="utf-8")
    )
    assert hits, "no migration defines upsert_canvas_block"
    return hits[-1].read_text(encoding="utf-8")


def _props_block(sql: str) -> str:
    """The `v_props := jsonb_build_object( ... );` assignment."""
    m = re.search(r"v_props\s*:=\s*jsonb_build_object\((.*?)\n\s*\);", sql, re.S)
    assert m, "could not locate the v_props assignment"
    return m.group(1)


def _split_top_level(body: str) -> list[str]:
    """Split `jsonb_build_object`'s argument list on its own commas only.

    Naively splitting on quoted keys finds the ones NESTED inside the values too — the
    `'source'` in `coalesce(v_existing->>'source', 'human')` reads exactly like a key and
    shreds the entry boundaries. Walking parens and quotes is the only way to tell an argument
    separator from a comma inside one.
    """
    parts, buf, depth, quoted = [], [], 0, False
    for ch in body:
        if quoted:
            buf.append(ch)
            if ch == "'":
                quoted = False
            continue
        if ch == "'":
            quoted = True
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    if "".join(buf).strip():
        parts.append("".join(buf).strip())
    return parts


def _entries(block: str) -> dict[str, str]:
    """Map each `'key', <expr>` pair to its expression, ignoring `--` comment lines."""
    body = "\n".join(ln for ln in block.splitlines() if not ln.strip().startswith("--"))
    args = _split_top_level(body)
    assert len(args) % 2 == 0, f"odd argument count in jsonb_build_object: {len(args)}"
    out: dict[str, str] = {}
    for key, expr in zip(args[0::2], args[1::2]):
        m = re.fullmatch(r"'([a-z_]+)'", key.strip())
        assert m, f"expected a quoted literal key, got {key!r}"
        out[m.group(1)] = expr.strip()
    return out


def test_the_guard_exists_at_all():
    sql = _latest_definition()
    assert "v_protect" in sql, (
        "the pinned branch no longer computes a protection flag — migration 66's behaviour "
        "(guarding only the statement) is back"
    )


def test_every_claim_field_consults_the_guard():
    """The regression class. A new property added to `v_props` and coalesced like its
    neighbours would silently fall outside the protection."""
    entries = _entries(_props_block(_latest_definition()))
    assert entries, "parsed no entries out of v_props"

    unguarded = [
        key for key, expr in entries.items()
        if key not in _EVIDENCE and key not in _DERIVED and "v_protect" not in expr
    ]
    assert not unguarded, (
        "these v_props fields are written on a PINNED block without consulting v_protect, so "
        f"an agent upsert overwrites a human's: {sorted(unguarded)}. Either guard them with "
        "`case when v_protect then <existing> else ... end`, or — if the field really is "
        "evidence an agent refresh should update — add it to _EVIDENCE here with the reason."
    )


@pytest.mark.parametrize("field", sorted(["stated", "source", "status", "confidence"]))
def test_the_four_fields_the_original_bug_leaked(field):
    """Named individually so a failure says which one came unstuck, and so deleting a guard
    can't pass by also deleting the field."""
    entries = _entries(_props_block(_latest_definition()))
    assert field in entries, f"v_props no longer writes {field!r}"
    assert "v_protect" in entries[field], (
        f"{field!r} is written unconditionally on a pinned block. This is the original bug: a "
        "block ends up displaying the human's sentence while reporting itself agent-authored"
    )


def test_evidence_still_refreshes_on_a_pinned_block():
    """The guard must not become a blanket freeze. If an agent refresh cannot write evidence,
    a pinned block can never be challenged and `divergence` becomes unreachable — which is
    worse than the bug, because it looks like agreement."""
    entries = _entries(_props_block(_latest_definition()))
    for field in sorted(_EVIDENCE):
        assert field in entries, f"v_props no longer writes {field!r}"
        assert "v_protect" not in entries[field], (
            f"{field!r} is now frozen on a pinned block. A pinned statement that cannot accrue "
            "contrary evidence is unfalsifiable — keep this one null-coalesced"
        )


def test_the_embedding_is_held_back_for_a_protected_block():
    """The caller embeds `body.statement`. On a protected upsert the statement is discarded but
    the embedding computed from it is not, which would leave the block searchable as text it
    does not contain."""
    sql = _latest_definition()
    m = re.search(r"embedding\s*=\s*(.*?),\n", sql, re.S)
    assert m, "could not locate the embedding assignment in the UPDATE"
    assert "v_protect" in m.group(1), (
        "a protected block takes the incoming embedding, which was computed from the agent's "
        "statement rather than the human's — the vector and the name disagree"
    )
