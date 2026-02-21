-- 12_context_summaries.sql
-- Stores LLM-generated context summaries per tenant.
-- Each time context is built/rebuilt for a tenant+client, a new summary row
-- is created (or upserted) capturing the AI-generated overview of the tenant's
-- knowledge base (industry insights, key themes, content profile, etc.).

create table if not exists public.context_summaries (
  id          uuid        primary key default gen_random_uuid(),
  tenant_id   uuid        not null,
  client_id   uuid        not null,

  summary     text        not null,           -- LLM-generated context summary
  topics      jsonb       not null default '[]'::jsonb,  -- extracted topic tags
  metadata    jsonb       not null default '{}'::jsonb,   -- flexible extra fields

  source_stats jsonb      not null default '{}'::jsonb,   -- e.g. {"documents": 5, "nodes": 42, "chunks": 120}

  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now()
);

-- Natural key: one active summary per tenant+client
create unique index if not exists context_summaries_tenant_client_uq
  on public.context_summaries(tenant_id, client_id);

create index if not exists context_summaries_tenant_idx
  on public.context_summaries(tenant_id);

-- Auto-update updated_at via existing trigger function
do $$
begin
  if not exists (select 1 from pg_trigger where tgname = 'trg_context_summaries_set_updated_at') then
    create trigger trg_context_summaries_set_updated_at
    before update on public.context_summaries
    for each row execute function public.set_updated_at();
  end if;
end $$;


-- ── Upsert RPC ───────────────────────────────────────────────────────────────
-- Idempotent upsert on (tenant_id, client_id). On conflict, replaces the
-- summary, topics, metadata, and source_stats.

create or replace function public.upsert_context_summary(
  p_tenant_id    uuid,
  p_client_id    uuid,
  p_summary      text,
  p_topics       jsonb  default '[]'::jsonb,
  p_metadata     jsonb  default '{}'::jsonb,
  p_source_stats jsonb  default '{}'::jsonb
)
returns uuid
language plpgsql
as $$
declare
  v_id uuid;
begin
  insert into public.context_summaries (
    tenant_id, client_id, summary, topics, metadata, source_stats,
    created_at, updated_at
  )
  values (
    p_tenant_id, p_client_id, p_summary,
    coalesce(p_topics, '[]'::jsonb),
    coalesce(p_metadata, '{}'::jsonb),
    coalesce(p_source_stats, '{}'::jsonb),
    now(), now()
  )
  on conflict (tenant_id, client_id)
  do update set
    summary      = excluded.summary,
    topics       = excluded.topics,
    metadata     = coalesce(public.context_summaries.metadata, '{}'::jsonb) || coalesce(excluded.metadata, '{}'::jsonb),
    source_stats = excluded.source_stats,
    updated_at   = now()
  returning id into v_id;

  return v_id;
end;
$$;
