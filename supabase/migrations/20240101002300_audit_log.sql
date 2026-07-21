-- 23_audit_log.sql
-- Append-only audit trail: who did what to which resource, when.
-- Every state-changing endpoint INSERTs a row here.

create table if not exists public.audit_log (
  id             uuid primary key default gen_random_uuid(),
  tenant_id      uuid not null,
  key_id         uuid,                        -- api_keys.id (nullable for system actions)
  request_id     text not null,
  action         text not null,               -- e.g. 'document.delete', 'ingest.processed'
  resource_type  text,                        -- 'document' | 'api_key' | 'context_summary' | ...
  resource_id    text,
  status         text not null,               -- 'success' | 'failure'
  metadata       jsonb not null default '{}'::jsonb,
  source_ip      text,
  created_at     timestamptz not null default now(),

  constraint audit_log_status_check check (status in ('success', 'failure'))
);

create index if not exists audit_log_tenant_created_idx
  on public.audit_log (tenant_id, created_at desc);

create index if not exists audit_log_tenant_action_idx
  on public.audit_log (tenant_id, action, created_at desc);

create index if not exists audit_log_resource_idx
  on public.audit_log (resource_type, resource_id);

create index if not exists audit_log_request_idx
  on public.audit_log (request_id);
