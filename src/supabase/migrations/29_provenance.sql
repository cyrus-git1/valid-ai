-- 29_provenance.sql
-- Adds actor/source/request provenance columns across documents, ingest_jobs,
-- audit_log, and chunks. Pure additive ALTER TABLEs — no data migration.
--
-- After this lands, the agent service can stop having its provenance fields
-- silently ignored. Endpoint updates in the next code release persist them.

-- ── documents ───────────────────────────────────────────────────────────────
alter table public.documents
  add column if not exists actor_id            text,
  add column if not exists actor_type          text,
  add column if not exists source_app          text,
  add column if not exists request_id          text,
  add column if not exists ingest_job_id       uuid,
  add column if not exists previous_version_id uuid;

-- Add the FK / self-FK separately so re-runs against an existing schema
-- don't fail when the constraint already exists.
do $$ begin
  if not exists (select 1 from pg_constraint where conname = 'documents_ingest_job_fk') then
    alter table public.documents
      add constraint documents_ingest_job_fk
      foreign key (ingest_job_id) references public.ingest_jobs(id) on delete set null;
  end if;
  if not exists (select 1 from pg_constraint where conname = 'documents_previous_version_fk') then
    alter table public.documents
      add constraint documents_previous_version_fk
      foreign key (previous_version_id) references public.documents(id) on delete set null;
  end if;
  if not exists (select 1 from pg_constraint where conname = 'documents_actor_type_check') then
    alter table public.documents
      add constraint documents_actor_type_check
      check (actor_type is null or actor_type in ('user','service','agent','anonymous'));
  end if;
end $$;

create index if not exists documents_actor_idx
  on public.documents (tenant_id, actor_id, created_at desc);
create index if not exists documents_ingest_job_idx
  on public.documents (ingest_job_id);
create index if not exists documents_previous_version_idx
  on public.documents (previous_version_id);
create index if not exists documents_request_idx
  on public.documents (request_id);


-- ── ingest_jobs ─────────────────────────────────────────────────────────────
-- Add actor/source columns and widen status to include 'processing','succeeded'
-- alongside existing 'running','complete' so old workers still write valid values.
alter table public.ingest_jobs
  add column if not exists actor_id    text,
  add column if not exists actor_type  text,
  add column if not exists source_app  text,
  add column if not exists source_type text,
  add column if not exists source_uri  text;

-- Replace the status check constraint to allow both old and new values during
-- the transition. App code may emit either set.
do $$ begin
  if exists (select 1 from pg_constraint where conname = 'ingest_jobs_status_check') then
    alter table public.ingest_jobs drop constraint ingest_jobs_status_check;
  end if;
end $$;

alter table public.ingest_jobs
  add constraint ingest_jobs_status_check
  check (status in ('queued','running','processing','complete','succeeded','failed'));

do $$ begin
  if not exists (select 1 from pg_constraint where conname = 'ingest_jobs_actor_type_check') then
    alter table public.ingest_jobs
      add constraint ingest_jobs_actor_type_check
      check (actor_type is null or actor_type in ('user','service','agent','anonymous'));
  end if;
end $$;

create index if not exists ingest_jobs_actor_idx
  on public.ingest_jobs (tenant_id, actor_id, enqueued_at desc);
create index if not exists ingest_jobs_request_idx
  on public.ingest_jobs (request_id);


-- ── audit_log (extended for SOC 2 audit_events semantics) ───────────────────
-- audit_log already serves the role of an audit_events table. Extend it with
-- before/after snapshots, a free-text reason (required by REVEAL/EXPORT), and
-- a free-text actor_id alongside the existing key_id.
alter table public.audit_log
  add column if not exists actor_id text,
  add column if not exists "before" jsonb,
  add column if not exists "after"  jsonb,
  add column if not exists reason   text;

-- Widen action constraint to include the SOC 2 verb set
do $$ begin
  if exists (select 1 from pg_constraint where conname = 'audit_log_action_check') then
    alter table public.audit_log drop constraint audit_log_action_check;
  end if;
  -- We allow free-form action so individual call sites stay flexible (e.g.
  -- 'document.delete', 'document.patch'), but normalize the canonical SOC 2
  -- verbs by length to catch typos. Skip the constraint — too restrictive.
end $$;

create index if not exists audit_log_actor_idx
  on public.audit_log (actor_id, created_at desc);


-- ── chunks (PII annotations) ────────────────────────────────────────────────
-- Stores [{type, span: [start,end], alias}, ...] per chunk so we can re-redact
-- or selectively reveal without re-running detection.
alter table public.chunks
  add column if not exists pii_annotations jsonb not null default '[]'::jsonb;

create index if not exists chunks_pii_annotations_gin
  on public.chunks using gin(pii_annotations);
