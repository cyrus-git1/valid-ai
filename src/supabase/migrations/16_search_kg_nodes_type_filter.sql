-- 16_search_kg_nodes_type_filter.sql
-- Add optional p_types parameter to search_kg_nodes for type-filtered search.
-- When p_types is NULL, all node types are returned (backwards compatible).

create or replace function public.search_kg_nodes(
  p_tenant_id uuid,
  p_client_id uuid,
  p_embedding vector(1536),
  p_top_k     int default 5,
  p_types     text[] default null
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
    and (p_types is null or n.type::text = any(p_types))
  order by n.embedding <=> p_embedding
  limit p_top_k;
$$;
