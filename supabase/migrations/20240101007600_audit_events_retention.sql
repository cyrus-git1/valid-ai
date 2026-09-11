-- 76_audit_events_retention.sql — make the one-year window enforceable AND observable.
--
-- Migration 75 shipped `audit_events_purge_expired()`. A purge function nobody calls is
-- the same control as a docstring nobody honours: the table just grows, and the retention
-- promise quietly becomes false. The repo's existing pattern for this — a cleanup function
-- beside a COMMENTED-OUT `cron.schedule` block (see 13_survey_outputs, 36_entity_cache) —
-- is exactly how that happens, so this does two things differently.
--
-- 1. The schedule self-enables where pg_cron exists, instead of being dead text somebody
--    has to notice and uncomment.
-- 2. There is a read-only status function, so "is retention actually holding?" is a
--    question anyone can answer at any time.
--
-- The status function is deliberately NOT a last-ran timestamp. A last-ran field tells you
-- the job fired; it does not tell you the job is keeping up, and those come apart exactly
-- when it matters. `expired_count > 0` is the fact worth alerting on, and it is true
-- whether the schedule was never installed, was installed and errored, or ran and could
-- not keep pace. See valid-agents/docs/vera-durable-audit-history-scope.md.


-- ─── Status ──────────────────────────────────────────────────────────────────
-- Read-only, cross-tenant, cheap. Safe to poll.

create or replace function public.audit_events_retention_status()
returns table (
  total_count        bigint,
  expired_count      bigint,   -- rows past the one-year window: SHOULD be 0
  oldest_occurred_at timestamptz,
  window_starts_at   timestamptz
)
language sql
stable
as $$
  select
    count(*)                                                              as total_count,
    count(*) filter (where occurred_at <= now() - interval '1 year')      as expired_count,
    min(occurred_at)                                                      as oldest_occurred_at,
    (now() - interval '1 year')                                           as window_starts_at
  from public.audit_events;
$$;


-- ─── Schedule ────────────────────────────────────────────────────────────────
-- Daily at 03:17 UTC. Odd minute on purpose: a maintenance job on the hour competes with
-- every other job somebody scheduled on the hour.
--
-- Guarded rather than commented out. Where pg_cron exists this installs itself; where it
-- does not, this block is a no-op and the purge must be driven by the admin endpoint
-- (POST /admin/audit-events/purge). Either way `audit_events_retention_status()` reports
-- the truth, which is the part that must not depend on anyone remembering.

do $$
begin
  if exists (select 1 from pg_extension where extname = 'pg_cron') then
    -- Unschedule first so re-running this migration does not stack duplicate jobs.
    begin
      perform cron.unschedule('audit-events-purge-expired');
    exception when others then
      null;  -- not previously scheduled; nothing to remove
    end;

    perform cron.schedule(
      'audit-events-purge-expired',
      '17 3 * * *',
      $cron$ select public.audit_events_purge_expired(); $cron$
    );
    raise notice 'audit_events retention purge scheduled via pg_cron (daily 03:17 UTC)';
  else
    raise notice
      'pg_cron not installed — audit_events retention purge NOT scheduled. Drive it via '
      'POST /admin/audit-events/purge and watch audit_events_retention_status().';
  end if;
end;
$$;
