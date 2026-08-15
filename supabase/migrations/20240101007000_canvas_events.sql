-- 70_canvas_events.sql — the canvas history layer (change log).
--
-- An append-only log of status/confidence transitions on CanvasBlock and
-- Hypothesis nodes, so the boards can show "what changed" and confidence
-- momentum. Written by a trigger with a WHEN clause scoped to the two node
-- types (so ordinary observation/concept writes never fire the function body),
-- and read by canvas_events_by_scope. No RPC rewrites.

create table if not exists public.canvas_events (
  id           uuid primary key default gen_random_uuid(),
  tenant_id    uuid not null,
  client_id    uuid,
  study_id     uuid,
  entity_type  text not null check (entity_type in ('canvas_block', 'hypothesis')),
  entity_key   text,                                   -- block_key or hypothesis external_id
  field        text not null check (field in ('status', 'confidence')),
  from_value   text,                                   -- null on create
  to_value     text,
  at           timestamptz not null default now()
);

create index if not exists canvas_events_scope_idx  on public.canvas_events (tenant_id, at desc);
create index if not exists canvas_events_entity_idx on public.canvas_events (tenant_id, entity_type, entity_key);
create index if not exists canvas_events_study_idx  on public.canvas_events (study_id);


-- ─── Trigger: log status + confidence transitions ────────────────────────────

create or replace function public.tg_canvas_status_event()
  returns trigger
  language plpgsql
as $$
declare
  v_etype text := case when NEW.type::text = 'CanvasBlock' then 'canvas_block' else 'hypothesis' end;
  v_key   text := coalesce(NEW.properties->>'block_key', NEW.properties->'external_ref'->>'id');
begin
  if TG_OP = 'INSERT'
     or (OLD.properties->>'status') is distinct from (NEW.properties->>'status') then
    insert into public.canvas_events (tenant_id, client_id, study_id, entity_type, entity_key, field, from_value, to_value)
    values (NEW.tenant_id, NEW.client_id, NEW.study_id, v_etype, v_key, 'status',
            case when TG_OP = 'INSERT' then null else OLD.properties->>'status' end,
            NEW.properties->>'status');
  end if;

  if TG_OP = 'UPDATE'
     and (OLD.properties->>'confidence') is distinct from (NEW.properties->>'confidence') then
    insert into public.canvas_events (tenant_id, client_id, study_id, entity_type, entity_key, field, from_value, to_value)
    values (NEW.tenant_id, NEW.client_id, NEW.study_id, v_etype, v_key, 'confidence',
            OLD.properties->>'confidence', NEW.properties->>'confidence');
  end if;

  return NEW;
end;
$$;

drop trigger if exists canvas_status_event on public.kg_nodes;
create trigger canvas_status_event
  after insert or update on public.kg_nodes
  for each row
  when (NEW.type::text in ('CanvasBlock', 'Hypothesis'))
  execute function public.tg_canvas_status_event();


-- ─── Read RPC ────────────────────────────────────────────────────────────────
-- p_study_id null => all events for the tenant (org-level feed); set => that study.

create or replace function public.canvas_events_by_scope(
  p_tenant_id  uuid,
  p_client_id  uuid,
  p_study_id   uuid default null,
  p_limit      int  default 100
)
returns table (
  entity_type  text,
  entity_key   text,
  field        text,
  from_value   text,
  to_value     text,
  study_id     uuid,
  at           timestamptz
)
language sql
stable
as $$
  select e.entity_type, e.entity_key, e.field, e.from_value, e.to_value, e.study_id, e.at
  from public.canvas_events e
  where e.tenant_id = p_tenant_id
    and (p_client_id is null and e.client_id is null
         or p_client_id is not null and e.client_id = p_client_id)
    and (p_study_id is null or e.study_id = p_study_id)
  order by e.at desc
  limit coalesce(p_limit, 100);
$$;
