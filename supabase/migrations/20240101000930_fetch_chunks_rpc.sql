-- 09b_fetch_chunks_rpc.sql
-- Bug 3 fix: replaces the two-step Python approach (fetch doc ids, then chunks)
-- with a single server-side JOIN. This avoids the Supabase .in_() limit that
-- breaks when a client has more than ~1000 documents.
--
-- Run this after 09_upsert_rpc.sql.

create or replace function public.fetch_chunks_with_embeddings(
  p_tenant_id  uuid,
  p_client_id  uuid default null,
  p_document_id uuid default null,
  p_limit      int  default 500,
  p_offset     int  default 0
)
returns table (
  id           uuid,
  document_id  uuid,
  chunk_index  int,
  content      text,
  embedding    vector(1536),
  metadata     jsonb
)
language sql
stable
as $$
  select
    c.id,
    c.document_id,
    c.chunk_index,
    c.content,
    c.embedding,
    c.metadata
  from public.chunks c
  join public.documents d on d.id = c.document_id
  where d.tenant_id  = p_tenant_id
    and (p_client_id is null or d.client_id = p_client_id)
    and (p_document_id is null or c.document_id = p_document_id)
    and c.embedding is not null
  order by c.document_id, c.chunk_index
  limit  p_limit
  offset p_offset;
$$;
