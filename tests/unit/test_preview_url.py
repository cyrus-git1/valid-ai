"""Unit tests for GET /data/documents/{id}/preview-url (signed URL for preview)."""
from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

os.environ["AUTH_ENABLED"] = "true"
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("CORS_ORIGINS", "http://test.local")
os.environ.setdefault("SUPABASE_URL", "https://proj.supabase.co")

TENANT = "00000000-0000-0000-0000-00000000bbbb"
CLIENT = "00000000-0000-0000-0000-00000000cccc"
DOC = "11111111-1111-4111-8111-111111111111"


class _DocQuery:
    def __init__(self, rows): self.rows = rows
    def select(self, *a, **k): return self
    def eq(self, *a, **k): return self
    def limit(self, *a, **k): return self
    def execute(self):
        class R: pass
        r = R(); r.data = self.rows; return r


class _Bucket:
    def __init__(self, ret): self.ret = ret; self.calls = []
    def create_signed_url(self, path, expires_in):
        self.calls.append((path, expires_in)); return self.ret


class _Storage:
    def __init__(self, ret): self.bucket = _Bucket(ret); self.names = []
    def from_(self, name): self.names.append(name); return self.bucket


class _FakeSB:
    def __init__(self, rows, signed):
        self.rows = rows
        self.storage = _Storage(signed)
    def table(self, name): return _DocQuery(self.rows)


def _mk_client(monkeypatch, rows, signed):
    from src.supabase import supabase_client as sbmod
    from src.routers import data_router as data_mod
    from src.middleware import auth as auth_mod
    fake = _FakeSB(rows, signed)
    getter = lambda: fake
    monkeypatch.setattr(sbmod, "get_supabase", getter)
    monkeypatch.setattr(data_mod, "get_supabase", getter)
    monkeypatch.setattr(auth_mod, "get_supabase", getter)
    from src.services import api_key_service
    monkeypatch.setattr(api_key_service.ApiKeyService, "verify",
                        lambda self, k: {"key_id": "k", "tenant_id": TENANT, "scopes": ["read"], "status": "active", "expired": False})
    from src.main import app
    c = TestClient(app); c.headers.update({"X-API-Key": "dp_test"})
    return c, fake


def test_bucket_doc_returns_signed_url(monkeypatch):
    rows = [{"id": DOC, "source_uri": f"bucket:pdf/{TENANT}/{CLIENT}/deck.pptx",
             "source_type": "PowerPoint", "title": "Q3 deck"}]
    c, fake = _mk_client(monkeypatch, rows, {"signedURL": "https://proj.supabase.co/storage/v1/object/sign/pdf/x?token=abc"})
    r = c.get(f"/data/documents/{DOC}/preview-url?tenant_id={TENANT}&client_id={CLIENT}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["url"].startswith("https://proj.supabase.co/storage/v1/object/sign/")
    assert body["filename"] == "deck.pptx"
    assert body["source_type"] == "PowerPoint"
    # signed the correct bucket + object path
    assert fake.storage.names == ["pdf"]
    assert fake.storage.bucket.calls[0][0] == f"{TENANT}/{CLIENT}/deck.pptx"


def test_relative_signed_url_gets_prefixed(monkeypatch):
    rows = [{"id": DOC, "source_uri": f"bucket:pdf/{TENANT}/{CLIENT}/report.docx", "source_type": "Docx", "title": "r"}]
    c, _ = _mk_client(monkeypatch, rows, {"signedURL": "/storage/v1/object/sign/pdf/x?token=abc"})
    r = c.get(f"/data/documents/{DOC}/preview-url?tenant_id={TENANT}&client_id={CLIENT}")
    assert r.status_code == 200
    expected = os.environ["SUPABASE_URL"].rstrip("/") + "/storage/v1/object/sign/pdf/x?token=abc"
    assert r.json()["url"] == expected


def test_web_doc_returns_url_directly(monkeypatch):
    rows = [{"id": DOC, "source_uri": "https://example.com/page", "source_type": "WebPage", "title": "Page"}]
    c, fake = _mk_client(monkeypatch, rows, {})
    r = c.get(f"/data/documents/{DOC}/preview-url?tenant_id={TENANT}&client_id={CLIENT}")
    assert r.status_code == 200
    assert r.json()["url"] == "https://example.com/page"
    assert fake.storage.names == []  # no signing for web docs


def test_not_found_404(monkeypatch):
    c, _ = _mk_client(monkeypatch, [], {})
    r = c.get(f"/data/documents/{DOC}/preview-url?tenant_id={TENANT}&client_id={CLIENT}")
    assert r.status_code == 404
