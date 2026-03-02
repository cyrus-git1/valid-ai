-- 13_survey_outputs.sql
-- Stores generated survey outputs (full surveys, question recommendations,
-- follow-up surveys) scoped to tenant + client.
-- Rows auto-expire after 7 days via the expires_at column.

create table if not exists public.survey_outputs (
  id          uuid        primary key default gen_random_uuid(),
  tenant_id   uuid        not null,
  client_id   uuid        not null,

  output_type text        not null,           -- 'survey' | 'recommendation' | 'follow_up'
  request     text        not null default '',-- original survey request / description
  questions   jsonb       not null default '[]'::jsonb,  -- generated questions array
  reasoning   text,                           -- LLM explanation (recommendations / follow-ups)
  metadata    jsonb       not null default '{}'::jsonb,

  created_at  timestamptz not null default now(),
  expires_at  timestamptz not null default (now() + interval '7 days')
);

create index if not exists survey_outputs_tenant_client_idx
  on public.survey_outputs(tenant_id, client_id);

create index if not exists survey_outputs_expires_idx
  on public.survey_outputs(expires_at);


-- ── Cleanup function ────────────────────────────────────────────────────────
-- Deletes all survey_outputs rows past their expires_at.
-- Call via pg_cron or application-level scheduled task.

create or replace function public.cleanup_expired_survey_outputs()
returns integer
language plpgsql
as $$
declare
  deleted_count integer;
begin
  delete from public.survey_outputs
  where expires_at < now();

  get diagnostics deleted_count = row_count;
  return deleted_count;
end;
$$;


-- ── Schedule auto-cleanup (requires pg_cron extension) ──────────────────────
-- Uncomment the lines below if pg_cron is available in your Supabase project.
-- This runs cleanup once per hour.
--
-- select cron.schedule(
--   'cleanup-expired-survey-outputs',
--   '0 * * * *',
--   $$ select public.cleanup_expired_survey_outputs(); $$
-- );
