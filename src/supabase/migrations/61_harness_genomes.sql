-- 61_harness_genomes.sql
-- Harness genome storage, moved from the AGENT layer's Supabase into the data
-- plane so the agent no longer needs a direct DB client for this.
--
-- Genomes are GLOBAL deployment config (harness optimization state), NOT tenant
-- data — there is no tenant_id. Keyed by (step_name, version); at most one
-- active version per step. The /genomes router is admin-scoped.
--
-- `create table if not exists` is a no-op if this table already exists (e.g. the
-- agent and data plane share a Supabase project) — in that case no data copy is
-- needed. If they are separate projects, copy rows over after applying.

create table if not exists public.harness_genomes (
  id                    uuid primary key default gen_random_uuid(),
  step_name             text not null,
  version               int  not null,
  is_active             boolean not null default false,
  parent_version        int,
  manager_prompt        text default '',
  rubric                jsonb default '[]'::jsonb,
  score_threshold       float4 default 0.7,
  max_retries           int default 2,
  agent_system_prompt   text default '',
  output_format_prompt  text default '',
  optimization_notes    text default '',
  test_score            float4,
  test_details          jsonb default '{}'::jsonb,
  created_at            timestamptz default now(),
  unique (step_name, version)
);

-- At most one active genome per step (enforced at the DB, not just the app).
create unique index if not exists harness_genomes_one_active
  on public.harness_genomes(step_name) where is_active;

create index if not exists harness_genomes_step_idx
  on public.harness_genomes(step_name);

-- Atomic activate: set exactly the target version active for a step (all others
-- off). A null p_version deactivates every version for the step (revert to
-- hardcoded defaults) without leaving a window with two active rows.
create or replace function public.set_active_genome(p_step text, p_version int)
returns void
language sql
as $$
  update public.harness_genomes
     set is_active = (p_version is not null and version = p_version)
   where step_name = p_step;
$$;

notify pgrst, 'reload schema';
