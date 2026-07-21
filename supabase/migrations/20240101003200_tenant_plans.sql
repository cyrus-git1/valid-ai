-- 32_tenant_plans.sql
-- Subscription tier per tenant. Three tiers (free, pro, enterprise) gate:
--   - Per-request body byte limit (Option A)
--   - Per-request chunk count cap (Option C)
--   - Per-tenant daily embedding token quota (Option B)
--
-- Defaults to 'free' for any tenant without an explicit row.

create table if not exists public.tenant_plans (
  tenant_id   uuid primary key,
  plan        text not null default 'free'
              check (plan in ('free','pro','enterprise')),
  notes       text,
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now()
);

create index if not exists tenant_plans_plan_idx on public.tenant_plans (plan);

-- updated_at auto-bump trigger
do $$ begin
  if not exists (select 1 from pg_trigger where tgname='trg_tenant_plans_set_updated_at') then
    create trigger trg_tenant_plans_set_updated_at before update on public.tenant_plans
      for each row execute function public.set_updated_at();
  end if;
end $$;

-- Upsert helper
create or replace function public.upsert_tenant_plan(
  p_tenant_id uuid,
  p_plan      text default 'free',
  p_notes     text default null
) returns void
language plpgsql
as $$
begin
  insert into public.tenant_plans (tenant_id, plan, notes)
  values (p_tenant_id, p_plan, p_notes)
  on conflict (tenant_id) do update set
    plan = excluded.plan,
    notes = coalesce(excluded.notes, public.tenant_plans.notes),
    updated_at = now();
end;
$$;
