-- 71_canvas_impact_links.sql — the Impact-ledger link (change -> hypothesis/block).
--
-- A shipped change ("Bundled analytics into the tier") linked to the hypothesis
-- or block it was meant to move. The Impact-ledger board joins this to the
-- change log (verdict movement) and spine trend (sentiment movement). User-
-- supplied via the UI; this is just the store.

create table if not exists public.canvas_impact_links (
  id                      uuid primary key default gen_random_uuid(),
  tenant_id               uuid not null,
  client_id               uuid,
  study_id                uuid,
  change_text             text not null,
  shipped_at              date,
  hypothesis_external_id  text,
  block_key               text,
  created_at              timestamptz not null default now(),
  updated_at              timestamptz not null default now()
);

create index if not exists canvas_impact_links_scope_idx
  on public.canvas_impact_links (tenant_id, created_at desc);


create or replace function public.upsert_impact_link(
  p_tenant_id               uuid,
  p_client_id               uuid,
  p_change_text             text,
  p_study_id                uuid default null,
  p_shipped_at              date default null,
  p_hypothesis_external_id  text default null,
  p_block_key               text default null,
  p_id                      uuid default null
)
returns uuid
language plpgsql
as $$
declare v_id uuid;
begin
  if p_id is not null then
    update public.canvas_impact_links
       set change_text = p_change_text, shipped_at = p_shipped_at,
           hypothesis_external_id = p_hypothesis_external_id, block_key = p_block_key,
           study_id = p_study_id, updated_at = now()
     where id = p_id and tenant_id = p_tenant_id
    returning id into v_id;
    if v_id is not null then
      return v_id;
    end if;
  end if;
  insert into public.canvas_impact_links (
    tenant_id, client_id, study_id, change_text, shipped_at,
    hypothesis_external_id, block_key
  ) values (
    p_tenant_id, p_client_id, p_study_id, p_change_text, p_shipped_at,
    p_hypothesis_external_id, p_block_key
  )
  returning id into v_id;
  return v_id;
end;
$$;


create or replace function public.impact_links_by_scope(
  p_tenant_id  uuid,
  p_client_id  uuid,
  p_study_id   uuid default null
)
returns table (
  id                      uuid,
  change_text             text,
  shipped_at              date,
  hypothesis_external_id  text,
  block_key               text,
  study_id                uuid,
  created_at              timestamptz
)
language sql
stable
as $$
  select l.id, l.change_text, l.shipped_at, l.hypothesis_external_id, l.block_key,
         l.study_id, l.created_at
  from public.canvas_impact_links l
  where l.tenant_id = p_tenant_id
    and (p_client_id is null and l.client_id is null
         or p_client_id is not null and l.client_id = p_client_id)
    and (p_study_id is null or l.study_id = p_study_id)
  order by l.created_at desc;
$$;
