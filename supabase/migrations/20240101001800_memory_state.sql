-- 18_memory_state.sql
-- Tracks durable semantic-memory freshness for tenant/client and tenant-wide scopes.

create table if not exists public.memory_state (
  id uuid primary key default gen_random_uuid(),
  tenant_id uuid not null,
  client_id uuid,

  memory_version bigint not null default 0,
  last_ingested_at timestamptz,
  last_changed_at timestamptz not null default now(),
  last_summary_at timestamptz,

  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create unique index if not exists memory_state_tenant_client_uq
  on public.memory_state (tenant_id, client_id);

create index if not exists memory_state_tenant_idx
  on public.memory_state (tenant_id);

do $$
begin
  if not exists (select 1 from pg_trigger where tgname = 'trg_memory_state_set_updated_at') then
    create trigger trg_memory_state_set_updated_at
    before update on public.memory_state
    for each row execute function public.set_updated_at();
  end if;
end $$;

create or replace function public.bump_memory_state(
  p_tenant_id uuid,
  p_client_id uuid default null,
  p_change_type text default 'mutation',
  p_metadata jsonb default '{}'::jsonb
)
returns bigint
language plpgsql
as $$
declare
  v_version bigint;
begin
  insert into public.memory_state (
    tenant_id, client_id, memory_version, last_ingested_at, last_changed_at, metadata, created_at, updated_at
  )
  values (
    p_tenant_id,
    p_client_id,
    1,
    case when p_change_type = 'ingest' then now() else null end,
    now(),
    coalesce(p_metadata, '{}'::jsonb),
    now(),
    now()
  )
  on conflict (tenant_id, client_id)
  do update set
    memory_version = public.memory_state.memory_version + 1,
    last_ingested_at = case
      when p_change_type = 'ingest' then now()
      else public.memory_state.last_ingested_at
    end,
    last_changed_at = now(),
    metadata = coalesce(public.memory_state.metadata, '{}'::jsonb) || coalesce(p_metadata, '{}'::jsonb),
    updated_at = now()
  returning memory_version into v_version;

  return v_version;
end;
$$;

create or replace function public.mark_summary_fresh(
  p_tenant_id uuid,
  p_client_id uuid,
  p_metadata jsonb default '{}'::jsonb
)
returns bigint
language plpgsql
as $$
declare
  v_version bigint;
begin
  insert into public.memory_state (
    tenant_id, client_id, memory_version, last_changed_at, last_summary_at, metadata, created_at, updated_at
  )
  values (
    p_tenant_id,
    p_client_id,
    0,
    now(),
    now(),
    coalesce(p_metadata, '{}'::jsonb),
    now(),
    now()
  )
  on conflict (tenant_id, client_id)
  do update set
    last_summary_at = now(),
    metadata = coalesce(public.memory_state.metadata, '{}'::jsonb) || coalesce(p_metadata, '{}'::jsonb),
    updated_at = now()
  returning memory_version into v_version;

  return v_version;
end;
$$;
