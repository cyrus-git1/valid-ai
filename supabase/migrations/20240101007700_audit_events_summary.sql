-- 77_audit_events_summary.sql — carry the audit's own conclusion on the row.
--
-- Found while pointing the read endpoints at this table (valid-agents H4). The row had
-- every NUMBER a trend view needs and none of the WORDS: scores, counts, degraded
-- analyzers, but not the one-line summary the audit itself produced. A durable trend
-- would have rendered as a line chart with nothing to read.
--
-- Bounded and cheap, which is why it belongs here and `full_result` still does not. The
-- summary is the audit's conclusion — a sentence. The full result is a copy of the survey
-- and its findings, per audit, forever; that is a different thing and it stays out (see
-- 75_audit_events.sql on payload_digest).

alter table public.audit_events
  add column if not exists summary text;


-- Both RPCs are recreated rather than altered: postgres cannot change a function's
-- return type in place, and `audit_events_by_scope` gains a column.

drop function if exists public.audit_event_record(
  uuid, text, uuid, text, text, int, numeric, numeric, text, text[], jsonb, text
);

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
  p_payload_digest  text    default null,
  p_summary         text    default null
)
returns uuid
language sql
as $$
  insert into public.audit_events (
    tenant_id, client_ref, study_id, actor_ref, request_id, goal_version,
    alignment_score, quality_score, status, degraded, finding_counts, payload_digest,
    summary
  )
  values (
    p_tenant_id, p_client_ref, p_study_id, p_actor_ref, p_request_id, p_goal_version,
    p_alignment_score, p_quality_score, coalesce(p_status, 'complete'),
    coalesce(p_degraded, '{}'), coalesce(p_finding_counts, '{}'::jsonb), p_payload_digest,
    p_summary
  )
  returning id;
$$;


drop function if exists public.audit_events_by_scope(uuid, text, uuid, int);

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
  payload_digest  text,
  summary         text
)
language sql
stable
as $$
  select e.id, e.study_id, e.occurred_at, e.actor_ref, e.request_id, e.goal_version,
         e.alignment_score, e.quality_score, e.status, e.degraded, e.finding_counts,
         e.payload_digest, e.summary
  from public.audit_events e
  where e.tenant_id = p_tenant_id
    and (p_client_ref is null or e.client_ref = p_client_ref)
    and (p_study_id  is null or e.study_id  = p_study_id)
  order by e.occurred_at desc
  limit coalesce(p_limit, 100);
$$;
