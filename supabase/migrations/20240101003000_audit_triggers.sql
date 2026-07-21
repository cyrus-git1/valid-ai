-- 30_audit_triggers.sql
-- SOC 2 audit triggers: every CREATE/UPDATE/DELETE on the canonical resource
-- tables auto-writes an audit_log row.
--
-- Design notes:
--   - `before` and `after` carry the full row JSON snapshot at write time.
--     We do NOT compute a structural diff in the trigger to keep the hot-path
--     cheap; a downstream worker can diff if needed.
--   - We use SECURITY DEFINER so app-role writes can populate audit_log even
--     when the writing role lacks INSERT on audit_log directly.
--   - actor_id, request_id, and reason come from per-session GUCs that the
--     application sets at the start of each request (see set_audit_context).
--
-- Tables audited: documents, chunks, kg_nodes, survey_outputs.
-- (kg_edges is high-volume; auditing those would 2x ingest write cost. Skipped.)


-- ── Per-request context plumbing ────────────────────────────────────────────
-- The application sets these GUCs at the start of each request. They flow
-- into every audit_log row written by the triggers below.
--
-- Usage from Python:
--   sb.rpc("set_audit_context", {
--     "p_actor_id":   request.state.actor_id,
--     "p_request_id": request.state.request_id,
--     "p_reason":     None,                       -- or a value for REVEAL/EXPORT
--   }).execute()
--
-- These are set per-transaction (LOCAL) so they don't leak across connections.
create or replace function public.set_audit_context(
  p_actor_id   text default null,
  p_request_id text default null,
  p_reason     text default null
)
returns void
language plpgsql
as $$
begin
  perform set_config('app.audit.actor_id',   coalesce(p_actor_id,   ''), true);
  perform set_config('app.audit.request_id', coalesce(p_request_id, ''), true);
  perform set_config('app.audit.reason',     coalesce(p_reason,     ''), true);
end;
$$;


-- Helper: read a GUC, return NULL when empty / unset.
create or replace function public._audit_guc(p_key text)
returns text
language plpgsql
stable
as $$
declare
  v text;
begin
  begin
    v := current_setting(p_key, true);
  exception when others then
    v := null;
  end;
  if v = '' then
    return null;
  end if;
  return v;
end;
$$;


-- ── The trigger function ────────────────────────────────────────────────────
-- One generic trigger reused by every audited table. It picks the action from
-- TG_OP and snapshots row data via row_to_json.
create or replace function public._audit_row()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
declare
  v_action       text;
  v_resource     text := tg_argv[0];      -- 'document' | 'chunk' | 'kg_node' | 'survey_output'
  v_resource_id  text;
  v_tenant_id    uuid;
  v_before       jsonb;
  v_after        jsonb;
begin
  -- Map TG_OP to canonical action verbs. Updates that change `status` are
  -- specifically labelled STATUS_CHANGE so reviewers can filter quickly.
  if tg_op = 'INSERT' then
    v_action := 'CREATE';
    v_after  := to_jsonb(new);
    v_before := null;
    v_resource_id := new.id::text;
    v_tenant_id   := new.tenant_id;
  elsif tg_op = 'UPDATE' then
    v_before := to_jsonb(old);
    v_after  := to_jsonb(new);
    -- Detect status change vs generic update
    if (to_jsonb(old) ? 'status') and (to_jsonb(new) ? 'status')
       and (to_jsonb(old) ->> 'status') is distinct from (to_jsonb(new) ->> 'status') then
      v_action := 'STATUS_CHANGE';
    else
      v_action := 'UPDATE';
    end if;
    v_resource_id := new.id::text;
    v_tenant_id   := new.tenant_id;
  elsif tg_op = 'DELETE' then
    v_action := 'DELETE';
    v_before := to_jsonb(old);
    v_after  := null;
    v_resource_id := old.id::text;
    v_tenant_id   := old.tenant_id;
  end if;

  -- Best-effort write — never block the underlying mutation
  begin
    insert into public.audit_log (
      tenant_id, key_id, actor_id, request_id, action,
      resource_type, resource_id, status, "before", "after", reason, source_ip
    ) values (
      v_tenant_id,
      null,
      public._audit_guc('app.audit.actor_id'),
      coalesce(public._audit_guc('app.audit.request_id'), gen_random_uuid()::text),
      v_action,
      v_resource,
      v_resource_id,
      'success',
      v_before,
      v_after,
      public._audit_guc('app.audit.reason'),
      null
    );
  exception when others then
    -- Don't propagate — audit failure must not block the actual mutation.
    null;
  end;

  if tg_op = 'DELETE' then
    return old;
  end if;
  return new;
end;
$$;


-- ── Wire triggers to each audited table ─────────────────────────────────────
do $$ begin
  if not exists (select 1 from pg_trigger where tgname = 'trg_audit_documents') then
    create trigger trg_audit_documents
      after insert or update or delete on public.documents
      for each row execute function public._audit_row('document');
  end if;
  if not exists (select 1 from pg_trigger where tgname = 'trg_audit_chunks') then
    create trigger trg_audit_chunks
      after insert or update or delete on public.chunks
      for each row execute function public._audit_row('chunk');
  end if;
  if not exists (select 1 from pg_trigger where tgname = 'trg_audit_kg_nodes') then
    create trigger trg_audit_kg_nodes
      after insert or update or delete on public.kg_nodes
      for each row execute function public._audit_row('kg_node');
  end if;
  if not exists (select 1 from pg_trigger where tgname = 'trg_audit_survey_outputs') then
    create trigger trg_audit_survey_outputs
      after insert or update or delete on public.survey_outputs
      for each row execute function public._audit_row('survey_output');
  end if;
end $$;


-- ── Append-only enforcement on audit_log ───────────────────────────────────
-- Block UPDATE and DELETE from any non-superuser role. Superusers (including
-- the postgres owner) can still operate on the table for retention sweeps.
create or replace function public._audit_log_no_modify()
returns trigger
language plpgsql
as $$
begin
  if pg_has_role(current_user, 'rds_superuser', 'usage')
     or current_user = 'postgres'
     or current_user = 'supabase_admin' then
    return new;
  end if;
  raise exception 'audit_log is append-only (action=%, user=%)', tg_op, current_user;
end;
$$;

do $$ begin
  if not exists (select 1 from pg_trigger where tgname = 'trg_audit_log_no_update') then
    create trigger trg_audit_log_no_update
      before update on public.audit_log
      for each row execute function public._audit_log_no_modify();
  end if;
  if not exists (select 1 from pg_trigger where tgname = 'trg_audit_log_no_delete') then
    create trigger trg_audit_log_no_delete
      before delete on public.audit_log
      for each row execute function public._audit_log_no_modify();
  end if;
end $$;
