-- 33_studies.sql
-- Introduces `studies` as a first-class scope alongside (tenant, client).
--
-- Domain model:
--   tenant_id  = the company (one per customer)
--   client_id  = who uploaded a row (provenance signal, not a scope)
--   study_id   = a research project / unit of work within a tenant
--
-- Additive only: existing rows get study_id = NULL and behave exactly as
-- today. Filtering by study_id is opt-in on the search RPC (migration 34).

-- ── 1. studies table ───────────────────────────────────────────────────────
create table if not exists public.studies (
  id          uuid primary key default gen_random_uuid(),
  tenant_id   uuid not null,
  name        text not null,
  objective   text,
  focus       text,
  status      text not null default 'active'
              check (status in ('active','paused','complete','archived')),
  metadata    jsonb not null default '{}'::jsonb,
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now()
);

create index if not exists studies_tenant_idx
  on public.studies (tenant_id, status);

create index if not exists studies_tenant_name_idx
  on public.studies (tenant_id, lower(name));

-- updated_at trigger (re-uses the shared touch function from migration 08)
do $$
begin
  if exists (select 1 from pg_proc where proname = 'touch_updated_at') then
    if not exists (
      select 1 from pg_trigger
      where tgname = 'studies_touch_updated_at'
    ) then
      execute $trg$
        create trigger studies_touch_updated_at
          before update on public.studies
          for each row execute function public.touch_updated_at();
      $trg$;
    end if;
  end if;
end $$;


-- ── 2. study_id on documents ───────────────────────────────────────────────
alter table public.documents
  add column if not exists study_id uuid
    references public.studies(id) on delete set null;

create index if not exists documents_study_idx
  on public.documents (tenant_id, study_id)
  where study_id is not null;


-- ── 3. study_id on kg_edges (so hop expansion can stay study-scoped) ───────
alter table public.kg_edges
  add column if not exists study_id uuid;

create index if not exists kg_edges_study_idx
  on public.kg_edges (tenant_id, study_id)
  where study_id is not null;


-- ── 4. study_id on chunks (denormalised for fast filter in search) ─────────
alter table public.chunks
  add column if not exists study_id uuid;

create index if not exists chunks_study_idx
  on public.chunks (tenant_id, study_id)
  where study_id is not null;


-- ── 5. study_id on kg_nodes (denormalised, same reason as chunks) ──────────
alter table public.kg_nodes
  add column if not exists study_id uuid;

create index if not exists kg_nodes_study_idx
  on public.kg_nodes (tenant_id, study_id)
  where study_id is not null;
