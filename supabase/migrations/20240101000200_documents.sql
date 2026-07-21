-- 02_documents.sql
create table if not exists public.documents (
  id uuid primary key default gen_random_uuid(),
  tenant_id uuid not null,
  client_id uuid,

  source_type text not null,
  source_uri text,
  title text,

  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create unique index if not exists documents_tenant_source_uq
  on public.documents(tenant_id, client_id, source_uri);

create index if not exists documents_tenant_idx on public.documents(tenant_id);
create index if not exists documents_client_idx on public.documents(tenant_id, client_id);
create index if not exists documents_metadata_gin on public.documents using gin(metadata);
