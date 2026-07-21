-- 35_audit_client_id.sql
-- Adds client_id to audit_log so "everything that touched client X" is an
-- indexed query, not a JSON extraction.
--
-- Replaces the _audit_row trigger to pluck client_id from the row being
-- audited. Legacy rows keep NULL client_id (audit_log is append-only, so a
-- backfill UPDATE is blocked by design — that's the trade-off for tamper
-- resistance).

-- ── 1. Column + index ───────────────────────────────────────────────────────
alter table public.audit_log
  add column if not exists client_id uuid;

create index if not exists audit_log_client_created_idx
  on public.audit_log (tenant_id, client_id, created_at desc)
  where client_id is not null;


-- ── 2. Replace _audit_row to capture client_id ──────────────────────────────
create or replace function public._audit_row()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
declare
  v_action       text;
  v_resource     text := tg_argv[0];
  v_resource_id  text;
  v_tenant_id    uuid;
  v_client_id    uuid;
  v_before       jsonb;
  v_after        jsonb;
begin
  if tg_op = 'INSERT' then
    v_action := 'CREATE';
    v_after  := to_jsonb(new);
    v_before := null;
    v_resource_id := new.id::text;
    v_tenant_id   := new.tenant_id;
    begin v_client_id := new.client_id; exception when others then v_client_id := null; end;
  elsif tg_op = 'UPDATE' then
    v_before := to_jsonb(old);
    v_after  := to_jsonb(new);
    if (to_jsonb(old) ? 'status') and (to_jsonb(new) ? 'status')
       and (to_jsonb(old) ->> 'status') is distinct from (to_jsonb(new) ->> 'status') then
      v_action := 'STATUS_CHANGE';
    else
      v_action := 'UPDATE';
    end if;
    v_resource_id := new.id::text;
    v_tenant_id   := new.tenant_id;
    begin v_client_id := new.client_id; exception when others then v_client_id := null; end;
  elsif tg_op = 'DELETE' then
    v_action := 'DELETE';
    v_before := to_jsonb(old);
    v_after  := null;
    v_resource_id := old.id::text;
    v_tenant_id   := old.tenant_id;
    begin v_client_id := old.client_id; exception when others then v_client_id := null; end;
  end if;

  begin
    insert into public.audit_log (
      tenant_id, client_id, key_id, actor_id, request_id, action,
      resource_type, resource_id, status, "before", "after", reason, source_ip
    ) values (
      v_tenant_id,
      v_client_id,
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
    null;
  end;

  if tg_op = 'DELETE' then
    return old;
  end if;
  return new;
end;
$$;
