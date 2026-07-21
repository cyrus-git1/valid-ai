"""Unit tests for Step 4 — write-time entity reconciliation in _link_entities.

Exercises the reconcile DECISION at the HTTP-adjacent layer with a fake Supabase
and stubbed embeddings: does a near-duplicate route onto the survivor, does a
miss mint a new node, does an exact-key hit stay a normal upsert. The actual
similarity behaviour (real embeddings + pgvector) is the integration checklist.
"""
from __future__ import annotations

import os
from uuid import UUID

import pytest

os.environ.setdefault("SUPABASE_URL", "http://localhost:54321")
os.environ.setdefault("SUPABASE_SERVICE_KEY", "dummy")
os.environ.setdefault("OPENAI_API_KEY", "sk-test")

from src.routers import ingest_router as ing
from src.routers.ingest_router import EntityItem, _link_entities

TENANT = UUID("00000000-0000-0000-0000-00000000bbbb")
CLIENT = UUID("00000000-0000-0000-0000-00000000cccc")


class _Exec:
    def __init__(self, data): self._data = data
    def execute(self):
        class R: ...
        r = R(); r.data = self._data; return r


class _Table:
    # chunk-node lookups in the mentions loop — return nothing so no edges form
    def select(self, *a, **k): return self
    def eq(self, *a, **k): return self
    def limit(self, *a, **k): return self
    def execute(self):
        class R: ...
        r = R(); r.data = []; return r


class _FakeSB:
    def __init__(self, nearest_rows):
        self.nearest_rows = nearest_rows
        self.upsert_node_calls = []
        self._n = 0

    def rpc(self, name, params):
        if name == "nearest_entity_candidate":
            return _Exec(self.nearest_rows)
        if name == "upsert_kg_node":
            self.upsert_node_calls.append(params)
            self._n += 1
            return _Exec(f"00000000-0000-0000-0000-0000000000{self._n:02d}")
        return _Exec(None)   # upsert_kg_edge etc.

    def table(self, name):
        return _Table()


@pytest.fixture(autouse=True)
def _stub_embeddings(monkeypatch):
    monkeypatch.setattr(ing, "_embed_in_batches", lambda texts, **kw: [[0.1] * 1536 for _ in texts])


def _run(nearest_rows, ent):
    sb = _FakeSB(nearest_rows)
    _link_entities(
        sb, tenant_id=TENANT, client_id=CLIENT,
        entities=[ent], chunk_ids=["chunk-1"], chunk_texts=["some text"],
    )
    return sb


def test_reconcile_routes_onto_survivor():
    # A near-duplicate of a different wording exists → route onto it.
    survivor_key = "entity:survivor-node-key"
    sb = _run(
        [{"node_key": survivor_key, "name": "Acme", "description": "ORG: Acme",
          "seen_count": 1, "status": "pending_linking", "similarity": 0.96}],
        EntityItem(name="Acme Corporation", type="ORG"),
    )
    assert len(sb.upsert_node_calls) == 1
    call = sb.upsert_node_calls[0]
    assert call["p_node_key"] == survivor_key          # routed onto survivor, not a new key
    assert call["p_name"] == "Acme"                    # survivor label preserved (no drift)
    assert call["p_embedding"] is None                 # survivor embedding kept
    assert call["p_properties"] == {}                  # no-op merge — survivor props untouched


def test_miss_mints_new_node():
    # No candidate above the floor → mint at the deterministic string key.
    sb = _run([], EntityItem(name="Globex", type="ORG"))
    call = sb.upsert_node_calls[0]
    assert call["p_node_key"] == f"entity:{TENANT}:{CLIENT}:org:globex"
    assert call["p_embedding"] is not None             # embedded (new node)
    assert call["p_properties"]["entity_type"] == "ORG"


def test_exact_key_hit_is_not_a_reconcile():
    # The nearest above-floor node is THIS entity's own key (re-ingest) → normal upsert.
    ent = EntityItem(name="Initech", type="ORG")
    own_key = f"entity:{TENANT}:{CLIENT}:org:initech"
    sb = _run(
        [{"node_key": own_key, "name": "Initech", "description": "ORG: Initech",
          "seen_count": 3, "status": "active", "similarity": 1.0}],
        ent,
    )
    call = sb.upsert_node_calls[0]
    assert call["p_node_key"] == own_key
    assert call["p_embedding"] is not None             # treated as normal upsert, not a reconcile
    assert call["p_properties"]["entity_type"] == "ORG"


def test_reconcile_passes_type_and_threshold_to_rpc(monkeypatch):
    captured = {}
    sb = _FakeSB([])
    real_rpc = sb.rpc
    def spy(name, params):
        if name == "nearest_entity_candidate":
            captured.update(params)
        return real_rpc(name, params)
    sb.rpc = spy
    _link_entities(sb, tenant_id=TENANT, client_id=CLIENT,
                   entities=[EntityItem(name="Umbrella", type="ORG")],
                   chunk_ids=["c1"], chunk_texts=["t"])
    assert captured["p_entity_type"] == "ORG"          # same-type guard
    assert captured["p_min_similarity"] == ing._ENTITY_RECONCILE_THRESHOLD
    assert len(captured["p_embedding"]) == 1536