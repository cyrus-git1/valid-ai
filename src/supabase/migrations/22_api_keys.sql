-- 22_api_keys.sql
-- Per-tenant API keys for authentication against the data plane.
-- Raw keys are never stored; only sha256 hashes. key_prefix is the
-- first 8 chars of the raw key for safe display/logging.

create table if not exists public.api_keys (
  id            uuid primary key default gen_random_uuid(),
  tenant_id     uuid not null,
  key_hash      text not null unique,
  key_prefix    text not null,
  name          text not null,
  scopes        text[] not null default '{}'::text[],
  status        text not null default 'active',
  last_used_at  timestamptz,
  expires_at    timestamptz,
  created_at    timestamptz not null default now(),
  revoked_at    timestamptz,

  constraint api_keys_status_check check (status in ('active', 'revoked'))
);

create index if not exists api_keys_tenant_status_idx
  on public.api_keys (tenant_id, status);

-- key_hash already has UNIQUE constraint which creates an index.

-- Lightweight helper to bump last_used_at without contention
create or replace function public.touch_api_key(p_key_id uuid)
returns void
language sql
as $$
  update public.api_keys
     set last_used_at = now()
   where id = p_key_id;
$$;
