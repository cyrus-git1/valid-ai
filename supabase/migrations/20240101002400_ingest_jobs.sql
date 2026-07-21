-- 24_ingest_jobs.sql
-- Durable status table for async ingest jobs.
-- arq keeps hot job state in Redis; we mirror to Postgres for queryability
-- and to survive Redis flushes.

create table if not exists public.ingest_jobs (
  id             uuid primary key default gen_random_uuid(),
  tenant_id      uuid not null,
  client_id      uuid,
  job_type       text not null,
  status         text not null default 'queued',
  request_id     text,
  key_id         uuid,
  payload_hash   text,
  document_id    uuid,
  result         jsonb,
  error          text,
  enqueued_at    timestamptz not null default now(),
  started_at     timestamptz,
  completed_at   timestamptz,

  constraint ingest_jobs_status_check
    check (status in ('queued','running','complete','failed')),
  constraint ingest_jobs_type_check
    check (job_type in ('processed','processed-web'))
);

create index if not exists ingest_jobs_tenant_client_idx
  on public.ingest_jobs (tenant_id, client_id, enqueued_at desc);

create index if not exists ingest_jobs_status_idx
  on public.ingest_jobs (status, enqueued_at);

create index if not exists ingest_jobs_payload_hash_idx
  on public.ingest_jobs (tenant_id, payload_hash);
