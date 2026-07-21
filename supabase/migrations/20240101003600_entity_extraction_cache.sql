-- 36_entity_extraction_cache.sql
-- Tier-1 of the enriched-entity rollout: a durable key-value cache for LLM
-- extraction outputs. Lets valid-agents survive redeploys without re-paying
-- the LLM bill for previously-seen transcripts.
--
-- Keyed on sha256 of the transcript text — content-derived, so identical
-- transcripts across tenants share the same cache hit (intentional; the
-- entity payload doesn't carry tenant data, only labels and offsets).
--
-- TTL: 30 days. expires_at is computed by the writer (the data plane sets
-- it from a constant if not supplied) and the cleanup function purges
-- past-due rows.

create table if not exists public.entity_extraction_cache (
  transcript_sha256  text         primary key,
  tenant_id          uuid         not null,
  client_id          uuid,
  entities           jsonb        not null,
  model_name         text         not null,
  prompt_version     text         not null default 'v1',
  created_at         timestamptz  not null default now(),
  expires_at         timestamptz  not null default (now() + interval '30 days')
);

create index if not exists entity_cache_expires_idx
  on public.entity_extraction_cache (expires_at);

create index if not exists entity_cache_tenant_idx
  on public.entity_extraction_cache (tenant_id, created_at desc);


-- Cleanup: call from a daily cron or pg_cron. Returns rows deleted.
create or replace function public.cleanup_expired_entity_cache()
returns integer
language plpgsql
as $$
declare
  v_deleted integer;
begin
  delete from public.entity_extraction_cache
   where expires_at < now();
  get diagnostics v_deleted = row_count;
  return v_deleted;
end;
$$;
