-- 38_null_client_uniqueness.sql
-- Postgres treats NULL as distinct in unique constraints, so existing
-- constraints like (tenant_id, client_id, node_key) DO NOT prevent
-- duplicates when client_id is NULL. The entity upsert flow (tier 2/3)
-- intentionally allows client_id=NULL (tenant-wide entities), so without
-- this fix every upsert with NULL client_id INSERTs a fresh row instead
-- of updating the existing one.
--
-- Partial unique indexes cover the NULL case without disturbing the
-- existing constraint or existing data.

create unique index if not exists kg_nodes_node_key_null_client_uq
  on public.kg_nodes (tenant_id, node_key)
  where client_id is null;

create unique index if not exists kg_edges_null_client_uq
  on public.kg_edges (tenant_id, src_id, dst_id, rel_type)
  where client_id is null;
