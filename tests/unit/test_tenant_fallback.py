"""Spine tenant resolution: body-tenant fallback when auth is off, enforce when on."""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from src.routers.spine_router import _check_tenant_match

TENANT = "00000000-0000-0000-0000-00000000bbbb"
OTHER = "00000000-0000-0000-0000-00000000dddd"


class _State:  # stand-in for request.state
    pass


class _FakeReq:
    def __init__(self, auth_tenant=None):
        self.state = _State()
        if auth_tenant is not None:
            self.state.tenant_id = auth_tenant


def test_auth_off_uses_body_tenant():
    # No auth context (AUTH_ENABLED=false) → trust the body tenant.
    assert _check_tenant_match(_FakeReq(), TENANT) == TENANT


def test_auth_off_requires_body_tenant():
    with pytest.raises(HTTPException) as e:
        _check_tenant_match(_FakeReq(), None)
    assert e.value.status_code == 400


def test_auth_on_matching_ok():
    # Auth on and body matches the key's tenant → ok.
    assert _check_tenant_match(_FakeReq(TENANT), TENANT) == TENANT


def test_auth_on_mismatch_403():
    # Auth on, body tenant != key tenant → 403 (key is authoritative).
    with pytest.raises(HTTPException) as e:
        _check_tenant_match(_FakeReq(TENANT), OTHER)
    assert e.value.status_code == 403
