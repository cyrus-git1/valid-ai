-- 19_document_ranking_fields.sql
-- Adds first-class document ranking fields for recency and explicit boosts.

alter table public.documents
  add column if not exists source_timestamp timestamptz,
  add column if not exists is_pinned boolean not null default false,
  add column if not exists is_canonical boolean not null default false,
  add column if not exists status text not null default 'active';

do $$
begin
  if not exists (
    select 1
    from pg_constraint
    where conname = 'documents_status_check'
  ) then
    alter table public.documents
      add constraint documents_status_check
      check (status in ('active', 'draft', 'deprecated', 'archived'));
  end if;
end $$;

create index if not exists documents_ranking_scope_idx
  on public.documents(tenant_id, client_id, status);

create index if not exists documents_source_timestamp_idx
  on public.documents(tenant_id, client_id, source_timestamp desc nulls last);

create index if not exists documents_pinned_idx
  on public.documents(tenant_id, client_id, is_pinned, is_canonical);
