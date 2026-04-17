"""Unit tests for CacheService."""
from __future__ import annotations

from src.services.cache_service import CacheService


class _InMemoryRedis:
    def __init__(self):
        self.store = {}

    def setex(self, key, ttl, value):
        self.store[key] = value

    def get(self, key):
        return self.store.get(key)

    def delete(self, *keys):
        for k in keys:
            self.store.pop(k, None)

    def scan(self, cursor=0, match=None, count=200):
        import fnmatch
        keys = [k for k in self.store if fnmatch.fnmatch(k, match or "*")]
        return 0, keys

    def ping(self):
        return True


def _svc() -> CacheService:
    return CacheService(redis_client=_InMemoryRedis())


def test_roundtrip_set_get():
    c = _svc()
    c.set("k1", {"a": 1}, ttl=60)
    assert c.get("k1") == {"a": 1}


def test_delete_by_prefix():
    c = _svc()
    c.set("search:t1:c1:hash1", {"x": 1}, 60)
    c.set("search:t1:c1:hash2", {"x": 2}, 60)
    c.set("search:t2:c1:hash3", {"x": 3}, 60)
    deleted = c.delete_by_prefix("search:t1:")
    assert deleted == 2
    assert c.get("search:t1:c1:hash1") is None
    assert c.get("search:t2:c1:hash3") == {"x": 3}


def test_search_key_shape():
    k = CacheService.search_key(tenant_id="T", client_id="C", payload_hash="abc")
    assert k == "search:T:C:abc"
    k2 = CacheService.search_key(tenant_id="T", client_id=None, payload_hash="abc")
    assert k2 == "search:T:all:abc"


def test_invalidate_search_client_scope():
    c = _svc()
    c.set("search:T:C1:h", {"x": 1}, 60)
    c.set("search:T:C2:h", {"x": 2}, 60)
    c.invalidate_search("T", client_id="C1")
    assert c.get("search:T:C1:h") is None
    assert c.get("search:T:C2:h") == {"x": 2}


def test_unavailable_cache_noops():
    c = CacheService(redis_client=None)
    c.set("x", 1, 60)  # no error
    assert c.get("x") is None
