-- 11_search_kg_nodes_rpc.sql
-- Vector similarity search over kg_nodes using pgvector cosine distance.
-- Scoped by tenant + client; only returns active nodes with embeddings.

create or replace function public.search_kg_nodes(
  p_tenant_id uuid,
  p_client_id uuid,
  p_embedding vector(1536),
  p_top_k     int default 5
)
returns table (
  id          uuid,
  node_key    text,
  name        text,
  description text,
  properties  jsonb,
  type        artifact_type,
  similarity  float4
)
language sql
stable
as $$
  select
    n.id,
    n.node_key,
    n.name,
    n.description,
    n.properties,
    n.type,
    (1 - (n.embedding <=> p_embedding))::float4 as similarity
  from public.kg_nodes n
  where n.tenant_id = p_tenant_id
    and n.client_id = p_client_id
    and n.status    = 'active'
    and n.embedding is not null
  order by n.embedding <=> p_embedding
  limit p_top_k;
$$;
