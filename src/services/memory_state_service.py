from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from uuid import UUID

from supabase import Client

logger = logging.getLogger(__name__)


class MemoryStateService:
    """Tracks semantic-memory freshness/version for agent cache invalidation."""

    def __init__(self, supabase: Client):
        self.sb = supabase

    def bump(
        self,
        *,
        tenant_id: UUID,
        client_id: Optional[UUID],
        change_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        try:
            res = self.sb.rpc(
                "bump_memory_state",
                {
                    "p_tenant_id": str(tenant_id),
                    "p_client_id": str(client_id) if client_id else None,
                    "p_change_type": change_type,
                    "p_metadata": metadata or {},
                },
            ).execute()
            return int(res.data) if res.data is not None else None
        except Exception as e:
            logger.warning(
                "Memory state bump failed tenant=%s client=%s change_type=%s: %s",
                tenant_id,
                client_id,
                change_type,
                e,
            )
            return None

    def bump_dual(
        self,
        *,
        tenant_id: UUID,
        client_id: UUID,
        change_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Atomically bump both client-scoped and tenant-wide memory state.

        Also invalidates scoped search cache entries.
        Returns {"client_version": int, "tenant_version": int} or defaults on failure.
        """
        try:
            res = self.sb.rpc(
                "bump_memory_state_dual",
                {
                    "p_tenant_id": str(tenant_id),
                    "p_client_id": str(client_id),
                    "p_change_type": change_type,
                    "p_metadata": metadata or {},
                },
            ).execute()
            if res.data and isinstance(res.data, dict):
                versions = {
                    "client_version": int(res.data.get("client_version", 0)),
                    "tenant_version": int(res.data.get("tenant_version", 0)),
                }
            else:
                versions = {"client_version": 0, "tenant_version": 0}
        except Exception as e:
            logger.warning(
                "Memory state bump_dual failed tenant=%s client=%s change_type=%s: %s",
                tenant_id,
                client_id,
                change_type,
                e,
            )
            versions = {"client_version": 0, "tenant_version": 0}

        # Invalidate search cache at the tenant scope — covers both the
        # specific client and the tenant-wide scope (client_id=None) since
        # both are subsets of search:{tenant_id}:*.
        try:
            from src.services.cache_service import CacheService
            CacheService().invalidate_search(str(tenant_id))
        except Exception as e:
            logger.debug("search_cache_invalidation_failed: %s", e)

        return versions

    def get_state(self, *, tenant_id: UUID, client_id: Optional[UUID]) -> Dict[str, Any]:
        try:
            q = (
                self.sb.table("memory_state")
                .select("memory_version,last_ingested_at,last_changed_at,last_summary_at,metadata,updated_at")
                .eq("tenant_id", str(tenant_id))
            )
            if client_id is None:
                q = q.is_("client_id", "null")
            else:
                q = q.eq("client_id", str(client_id))
            res = q.limit(1).execute()
            rows = res.data or []
            if rows:
                return rows[0]
        except Exception as e:
            logger.warning("Memory state fetch failed tenant=%s client=%s: %s", tenant_id, client_id, e)
        return {
            "memory_version": 0,
            "last_ingested_at": None,
            "last_changed_at": None,
            "last_summary_at": None,
            "metadata": {},
            "updated_at": None,
        }
