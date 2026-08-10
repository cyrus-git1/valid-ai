"""Unit tests: evidence quote on observations:by-ids + /data/survey-outputs read."""
from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

os.environ["AUTH_ENABLED"] = "true"
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("CORS_ORIGINS", "http://test.local")

TENANT = "00000000-0000-0000-0000-00000000bbbb"
STUDY = "00000000-0000-0000-0000-0000000000a1"


class _Chain:
    def __init__(self, rows): self.rows = rows
    def select(self, *a, **k): return self
    def eq(self, *a, **k): return self
    def gte(self, *a, **k): return self
    def order(self, *a, **k): return self
    def limit(self, *a, **k): return self
    def execute(self):
        class R: pass
        r = R(); r.data = self.rows; return r


class _FakeSB:
    def __init__(self):
        self._rpc: Dict[str, Any] = {}
        self._tables: Dict[str, List[dict]] = {}
    def set_rpc(self, n, d): self._rpc[n] = d
    def set_table(self, n, rows): self._tables[n] = rows
    def rpc(self, name, params):
        ret = self._rpc.get(name, [])
        class R: pass
        r = R(); r.data = ret
        class E:
            def __init__(s, r): s._r = r
            def execute(s): return s._r
        return E(r)
    def table(self, name): return _Chain(self._tables.get(name, []))


@pytest.fixture
def client(monkeypatch):
    from src.db import supabase_client as sbmod
    from src.routers import spine_router as spine_mod
    from src.routers import data_router as data_mod
    from src.middleware import auth as auth_mod
    fake = _FakeSB(); getter = lambda: fake
    for m in (sbmod, spine_mod, data_mod, auth_mod):
        monkeypatch.setattr(m, "get_supabase", getter)
    from src.services import api_key_service
    monkeypatch.setattr(api_key_service.ApiKeyService, "verify",
                        lambda self, k: {"key_id": "k", "tenant_id": TENANT, "scopes": ["read"], "status": "active", "expired": False})
    from src.main import app
    c = TestClient(app); c.headers.update({"X-API-Key": "dp_test"})
    return c, fake


def test_by_ids_returns_evidence_quote(client):
    c, fake = client
    fake.set_rpc("observations_by_ids", [
        {"node_id": "n1", "observation_id": "obs-1", "value": {"number": 0.6, "unit": "pct"}, "study_id": STUDY,
         "evidence": {"text": "I couldn't find the sign-out button", "speaker": "Speaker 1", "offset_ms": 125000}},
        {"node_id": "n2", "observation_id": "obs-2", "value": {"number": 8, "unit": "NPS"}, "study_id": STUDY,
         "evidence": None},
    ])
    r = c.post("/observations/by-ids", json={"tenant_id": TENANT, "ids": ["obs-1", "obs-2"], "study_ids": [STUDY]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["obs-1"]["evidence"]["text"] == "I couldn't find the sign-out button"
    assert body["obs-1"]["evidence"]["speaker"] == "Speaker 1"
    assert body["obs-1"]["evidence"]["offset_ms"] == 125000
    assert body["obs-2"]["evidence"] is None   # absent, not crashing


def test_survey_outputs_returns_questions(client):
    c, fake = client
    fake.set_table("survey_outputs", [
        {"id": "s1", "output_type": "survey",
         "questions": [{"question_id": "q1", "text": "How satisfied are you?", "type": "likert"}],
         "metadata": {"study_id": STUDY}, "created_at": "2026-07-22T00:00:00Z"},
    ])
    r = c.get(f"/data/survey-outputs?tenant_id={TENANT}&study_id={STUDY}")
    assert r.status_code == 200, r.text
    surveys = r.json()["surveys"]
    assert surveys[0]["study_id"] == STUDY
    q = surveys[0]["questions"][0]
    assert q["question_id"] == "q1" and q["text"] == "How satisfied are you?" and q["type"] == "likert"


def test_survey_outputs_study_filter_excludes(client):
    c, fake = client
    fake.set_table("survey_outputs", [
        {"id": "s1", "output_type": "survey", "questions": [], "metadata": {"study_id": "other-study"}, "created_at": "x"},
    ])
    r = c.get(f"/data/survey-outputs?tenant_id={TENANT}&study_id={STUDY}")
    assert r.status_code == 200
    assert r.json()["surveys"] == []   # filtered out (different study)
