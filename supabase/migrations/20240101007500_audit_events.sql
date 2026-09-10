-- 75_audit_events.sql — durable audit history for /audit/survey.
--
-- Replaces an in-process dict in valid-agents that claimed to give SOC 2 auditors "the
-- full sequence of audit decisions on a study" and could not: it was a module-level
-- dict emptied by every deploy, expiring on time.monotonic() so its "30-day TTL" meant
-- 30 days of process uptime.
--
-- The write is the half with a deadline. History is the one thing that cannot be
-- backfilled: if an audit period is ever covered, the writes had to be durable DURING
-- it. See valid-agents/docs/vera-durable-audit-history-scope.md.
--
-- Three settled decisions are encoded here:
--   * retention is ONE YEAR, enforced by deleting whole rows past the window rather
--     than by a TTL that quietly forgets;
--   * scope is /audit/survey results only for now;
--   * a study's audit history SURVIVES the study's deletion — which is what makes it a
--     record, and why the actor is stored pseudonymously (see actor_ref).

create table if not exists public.audit_events (
  id              uuid primary key default gen_random_uuid(),
  tenant_id       uuid not null,                         -- an ORGANISATION, not a person
  client_ref      text,                                  -- hashed: see actor_ref below
  study_id        uuid,                                  -- NOT a foreign key: see below

  occurred_at     timestamptz not null default now(),    -- wall clock, never monotonic

  -- Attribution. `actor_ref` is a keyed hash of the actor id, computed by the WRITER
  -- before it ever crosses the wire — never a raw user id. It keeps everything the
  -- record is for (two rows by the same person are provably the same person; a known
  -- person is findable by hashing their id and searching) and drops only the ability to
  -- read a name out of the table. That is the half a right-to-erasure request is about,
  -- and dropping it is what lets this row outlive the study.
  --
  -- `client_ref` above is hashed for the SAME reason, and here it is not optional: on
  -- this platform the client id IS the authenticated user's id, so a raw one would sit
  -- in the next column and hand back exactly what actor_ref withholds. It could not even
  -- be scrubbed afterwards — this table refuses UPDATE, by design. An append-only table
  -- holding a raw user id can never satisfy an erasure request, so it must never hold
  -- one. tenant_id stays raw: an organisation is not a person, and scoping needs it.
  actor_ref       text,
  request_id      text,

  -- What was audited, and what the audit concluded.
  goal_version    int,
  alignment_score numeric(4,3),
  quality_score   numeric(4,3),
  status          text not null default 'complete'
                    check (status in ('complete', 'partial', 'failed', 'empty')),
  degraded        text[] not null default '{}',          -- analyzers that did not run
  finding_counts  jsonb  not null default '{}'::jsonb,   -- by severity and by check

  -- sha256 of the audited questions. Answers "was this the same survey?" — the question
  -- an auditor actually asks — without storing a copy of the survey per audit, which
  -- would grow without bound and duplicate what the study already holds.
  payload_digest  text
);

-- study_id is deliberately NOT a foreign key. The record outlives the study by design,
-- so a cascade or a restrict would either destroy the history or block the deletion.
create index if not exists audit_events_scope_idx  on public.audit_events (tenant_id, occurred_at desc);
create index if not exists audit_events_client_idx on public.audit_events (tenant_id, client_ref);
create index if not exists audit_events_study_idx  on public.audit_events (study_id, occurred_at desc);
create index if not exists audit_events_actor_idx  on public.audit_events (tenant_id, actor_ref);


-- ─── Append-only ─────────────────────────────────────────────────────────────
-- An audit trail that can be rewritten is not one. UPDATE is refused outright; DELETE is
-- refused unless the row is past the retention window, which is what lets the retention
-- job do its work without opening a general delete path.

create or replace function public.tg_audit_events_append_only()
  returns trigger
  language plpgsql
as $$
begin
  if TG_OP = 'UPDATE' then
    raise exception 'audit_events is append-only: row % cannot be updated', OLD.id
      using errcode = 'restrict_violation';
  end if;

  if TG_OP = 'DELETE' then
    if OLD.occurred_at > now() - interval '1 year' then
      raise exception
        'audit_events is append-only: row % is inside the 1-year retention window', OLD.id
        using errcode = 'restrict_violation';
    end if;
    return OLD;                              -- past retention: the purge may proceed
  end if;

  return NEW;
end;
$$;

drop trigger if exists audit_events_append_only on public.audit_events;
create trigger audit_events_append_only
  before update or delete on public.audit_events
  for each row
  execute function public.tg_audit_events_append_only();


-- ─── Write ───────────────────────────────────────────────────────────────────

create or replace function public.audit_event_record(
  p_tenant_id       uuid,
  p_client_ref      text    default null,
  p_study_id        uuid    default null,
  p_actor_ref       text    default null,
  p_request_id      text    default null,
  p_goal_version    int     default null,
  p_alignment_score numeric default null,
  p_quality_score   numeric default null,
  p_status          text    default 'complete',
  p_degraded        text[]  default '{}',
  p_finding_counts  jsonb   default '{}'::jsonb,
  p_payload_digest  text    default null
)
returns uuid
language sql
as $$
  insert into public.audit_events (
    tenant_id, client_ref, study_id, actor_ref, request_id, goal_version,
    alignment_score, quality_score, status, degraded, finding_counts, payload_digest
  )
  values (
    p_tenant_id, p_client_ref, p_study_id, p_actor_ref, p_request_id, p_goal_version,
    p_alignment_score, p_quality_score, coalesce(p_status, 'complete'),
    coalesce(p_degraded, '{}'), coalesce(p_finding_counts, '{}'::jsonb), p_payload_digest
  )
  returning id;
$$;


-- ─── Read ────────────────────────────────────────────────────────────────────
-- p_study_id null => the tenant-wide feed; set => that study's history.

create or replace function public.audit_events_by_scope(
  p_tenant_id  uuid,
  p_client_ref text default null,
  p_study_id   uuid default null,
  p_limit      int  default 100
)
returns table (
  id              uuid,
  study_id        uuid,
  occurred_at     timestamptz,
  actor_ref       text,
  request_id      text,
  goal_version    int,
  alignment_score numeric,
  quality_score   numeric,
  status          text,
  degraded        text[],
  finding_counts  jsonb,
  payload_digest  text
)
language sql
stable
as $$
  select e.id, e.study_id, e.occurred_at, e.actor_ref, e.request_id, e.goal_version,
         e.alignment_score, e.quality_score, e.status, e.degraded, e.finding_counts,
         e.payload_digest
  from public.audit_events e
  where e.tenant_id = p_tenant_id
    and (p_client_ref is null or e.client_ref = p_client_ref)
    and (p_study_id  is null or e.study_id  = p_study_id)
  order by e.occurred_at desc
  limit coalesce(p_limit, 100);
$$;


-- ─── Retention ───────────────────────────────────────────────────────────────
-- Deliberately a function a scheduled job calls, not a TTL. A TTL that forgets is how
-- the store this replaces ended up claiming a retention it did not have; a purge that
-- runs, reports how many rows it removed, and can be seen not running is auditable.

create or replace function public.audit_events_purge_expired()
returns integer
language plpgsql
as $$
declare
  v_deleted integer;
begin
  delete from public.audit_events
   where occurred_at <= now() - interval '1 year';
  get diagnostics v_deleted = row_count;
  return v_deleted;
end;
$$;
