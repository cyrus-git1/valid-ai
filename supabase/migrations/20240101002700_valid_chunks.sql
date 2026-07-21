-- 27_valid_chunks.sql
-- Dedicated chunks table for the valid-docs bucket.
-- Same structure as the main chunks table but scoped to Valid's own
-- document corpus. Documents referenced here are stored in the
-- 'valid-docs' Supabase storage bucket.

create table if not exists public.valid_chunks (
  id              uuid primary key default gen_random_uuid(),
  tenant_id       uuid not null,
  document_id     uuid not null references public.documents(id) on delete cascade,
  chunk_index     int not null,
  page_start      int,
  page_end        int,
  content         text not null,
  content_tokens  int,
  metadata        jsonb not null default '{}'::jsonb,
  created_at      timestamptz not null default now(),
  embedding       vector(1536),
  embedding_model text not null default 'text-embedding-3-small'
);

create unique index if not exists valid_chunks_doc_chunk_idx_uq
  on public.valid_chunks(tenant_id, document_id, chunk_index);

create index if not exists valid_chunks_tenant_doc_idx
  on public.valid_chunks(tenant_id, document_id);

create index if not exists valid_chunks_metadata_gin
  on public.valid_chunks using gin(metadata);

create index if not exists valid_chunks_embedding_hnsw
  on public.valid_chunks using hnsw (embedding vector_cosine_ops);

create index if not exists valid_chunks_embedding_model_idx
  on public.valid_chunks(tenant_id, embedding_model);


-- Upsert RPC for valid_chunks (same pattern as upsert_chunk)
create or replace function public.upsert_valid_chunk(
  p_tenant_id       uuid,
  p_document_id     uuid,
  p_chunk_index     int,
  p_page_start      int default null,
  p_page_end        int default null,
  p_content         text default null,
  p_content_tokens  int default null,
  p_metadata        jsonb default '{}'::jsonb,
  p_embedding       vector(1536) default null,
  p_embedding_model text default 'text-embedding-3-small'
) returns uuid language plpgsql as $$
declare v_id uuid;
begin
  insert into public.valid_chunks (
    tenant_id, document_id, chunk_index, page_start, page_end,
    content, content_tokens, metadata, embedding, embedding_model, created_at
  ) values (
    p_tenant_id, p_document_id, p_chunk_index, p_page_start, p_page_end,
    p_content, p_content_tokens, coalesce(p_metadata, '{}'::jsonb),
    p_embedding, p_embedding_model, now()
  ) on conflict (tenant_id, document_id, chunk_index) do update set
    page_start = coalesce(excluded.page_start, public.valid_chunks.page_start),
    page_end = coalesce(excluded.page_end, public.valid_chunks.page_end),
    content = coalesce(excluded.content, public.valid_chunks.content),
    content_tokens = coalesce(excluded.content_tokens, public.valid_chunks.content_tokens),
    metadata = coalesce(public.valid_chunks.metadata, '{}'::jsonb) || coalesce(excluded.metadata, '{}'::jsonb),
    embedding = coalesce(excluded.embedding, public.valid_chunks.embedding),
    embedding_model = coalesce(excluded.embedding_model, public.valid_chunks.embedding_model)
  returning id into v_id;
  return v_id;
end; $$;
