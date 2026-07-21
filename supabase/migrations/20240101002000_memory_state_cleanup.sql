-- 20_memory_state_cleanup.sql
-- 1. Append-only change log for agent audit trails
-- 2. Fix metadata accumulation (overwrite instead of merge)
-- 3. Atomic dual-scope bump (client + tenant in one call)
-- 4. Drop mark_summary_fresh (decoupled from data plane)

-- ── Change log table ────────────────────────────────────────────────────────

create table if not exists public.memory_change_log (
  id           uuid primary key default gen_random_uuid(),
  tenant_id    uuid not null,
  client_id    uuid,
  change_type  text not null,
  memory_version bigint not null,
  metadata     jsonb not null default '{}'::jsonb,
  created_at   timestamptz not null default now()
);

create index if not exists memory_change_log_tenant_client_idx
  on public.memory_change_log (tenant_id, client_id, created_at desc);

create index if not exists memory_change_log_tenant_idx
  on public.memory_change_log (tenant_id, created_at desc);


-- ── Fix bump_memory_state: overwrite metadata, log change ───────────────────

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
    metadata = coalesce(p_metadata, '{}'::jsonb),
    updated_at = now()
  returning memory_version into v_version;

  -- Append to change log
  insert into public.memory_change_log (tenant_id, client_id, change_type, memory_version, metadata)
  values (p_tenant_id, p_client_id, p_change_type, v_version, coalesce(p_metadata, '{}'::jsonb));

  return v_version;
end;
$$;


-- ── Atomic dual bump (client + tenant in one transaction) ───────────────────

create or replace function public.bump_memory_state_dual(
  p_tenant_id uuid,
  p_client_id uuid,
  p_change_type text default 'mutation',
  p_metadata jsonb default '{}'::jsonb
)
returns jsonb
language plpgsql
as $$
declare
  v_client_version bigint;
  v_tenant_version bigint;
begin
  -- Bump client-scoped row
  insert into public.memory_state (
    tenant_id, client_id, memory_version, last_ingested_at, last_changed_at, metadata, created_at, updated_at
  )
  values (
    p_tenant_id, p_client_id, 1,
    case when p_change_type = 'ingest' then now() else null end,
    now(), coalesce(p_metadata, '{}'::jsonb), now(), now()
  )
  on conflict (tenant_id, client_id)
  do update set
    memory_version = public.memory_state.memory_version + 1,
    last_ingested_at = case when p_change_type = 'ingest' then now() else public.memory_state.last_ingested_at end,
    last_changed_at = now(),
    metadata = coalesce(p_metadata, '{}'::jsonb),
    updated_at = now()
  returning memory_version into v_client_version;

  -- Bump tenant-wide row
  insert into public.memory_state (
    tenant_id, client_id, memory_version, last_ingested_at, last_changed_at, metadata, created_at, updated_at
  )
  values (
    p_tenant_id, null, 1,
    case when p_change_type = 'ingest' then now() else null end,
    now(), coalesce(p_metadata, '{}'::jsonb), now(), now()
  )
  on conflict (tenant_id, client_id)
  do update set
    memory_version = public.memory_state.memory_version + 1,
    last_ingested_at = case when p_change_type = 'ingest' then now() else public.memory_state.last_ingested_at end,
    last_changed_at = now(),
    metadata = coalesce(p_metadata, '{}'::jsonb),
    updated_at = now()
  returning memory_version into v_tenant_version;

  -- Single change log entry (client-scoped version)
  insert into public.memory_change_log (tenant_id, client_id, change_type, memory_version, metadata)
  values (p_tenant_id, p_client_id, p_change_type, v_client_version, coalesce(p_metadata, '{}'::jsonb));

  return jsonb_build_object('client_version', v_client_version, 'tenant_version', v_tenant_version);
end;
$$;


-- ── Drop mark_summary_fresh (no longer called) ─────────────────────────────

drop function if exists public.mark_summary_fresh(uuid, uuid, jsonb);
