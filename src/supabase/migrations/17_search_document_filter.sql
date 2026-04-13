-- 17_search_document_filter.sql
-- Add optional p_document_ids parameter to search_kg_nodes for document-scoped search.
-- When p_document_ids is NULL, all documents are searched (backwards compatible).
-- When provided, only nodes whose properties->>'document_id' matches are returned.

-- Drop all existing overloads to avoid ambiguity
DROP FUNCTION IF EXISTS public.search_kg_nodes(uuid, uuid, vector, integer);
DROP FUNCTION IF EXISTS public.search_kg_nodes(uuid, uuid, vector, integer, text[]);

create or replace function public.search_kg_nodes(
  p_tenant_id    uuid,
  p_client_id    uuid,
  p_embedding    vector(1536),
  p_top_k        int default 5,
  p_types        text[] default null,
  p_document_ids text[] default null
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
    and (p_document_ids is null or n.properties->>'document_id' = any(p_document_ids))
  order by n.embedding <=> p_embedding
  limit p_top_k;
$$;
