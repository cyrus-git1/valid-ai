-- 03_chunks.sql
create table if not exists public.chunks (
  id uuid primary key default gen_random_uuid(),
  tenant_id uuid not null,
  document_id uuid not null references public.documents(id) on delete cascade,

  chunk_index int not null,
  page_start int,
  page_end int,

  content text not null,
  content_tokens int,

  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),

  embedding vector(1536)
);

create unique index if not exists chunks_doc_chunk_idx_uq
  on public.chunks(tenant_id, document_id, chunk_index);

create index if not exists chunks_tenant_doc_idx on public.chunks(tenant_id, document_id);
create index if not exists chunks_metadata_gin on public.chunks using gin(metadata);

create index if not exists chunks_embedding_hnsw
  on public.chunks using hnsw (embedding vector_cosine_ops);
