-- 20240101007200_context_corrections.sql
-- Tenant-scoped CONTEXT CORRECTIONS (a.k.a. overrides): durable, non-destructive
-- "going forward" fixes applied where context is READ, never by editing ingested
-- document text. E.g. term_replace 'Valid Technologies' -> 'NewName', or a pure
-- 'disregard' that drops a term. Vera proposes; only written after user confirms.
--
-- Applied at read time (context-summary generation + retrieval post-process), so
-- answers reflect the correction immediately while source docs stay untouched.
--
-- RLS enabled with NO policies → service-role (data plane) only; the app/agent
-- reach it through the API. Ends with NOTIFY pgrst 'reload schema'.

create table if not exists public.context_corrections (
  id          uuid primary key default gen_random_uuid(),
  tenant_id   uuid not null,
  client_id   uuid,
  kind        text not null check (kind in ('term_replace', 'disregard')),
  from_term   text not null,
  to_term     text,                                    -- null for 'disregard'
  note        text,
  applies_to  jsonb not null default '"all"'::jsonb,   -- "all" or ["<document_id>", ...]
  status      text  not null default 'active',
  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now(),
  -- a term_replace must carry a replacement; a disregard must not require one
  constraint context_corrections_to_term_ck
    check (kind <> 'term_replace' or (to_term is not null and length(btrim(to_term)) > 0))
);

create index if not exists context_corrections_tenant_idx
  on public.context_corrections (tenant_id, client_id)
  where status = 'active';

alter table public.context_corrections enable row level security;
-- (no policies → service_role only; all other access goes through the API)


notify pgrst, 'reload schema';
