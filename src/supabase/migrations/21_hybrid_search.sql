-- 21_hybrid_search.sql
-- Hybrid search: blends vector cosine similarity with document-level ranking
-- signals (recency, pinned, canonical) and filters by document status.
--
-- Entity nodes (no document_id) bypass the document JOIN and get pure vector score.

DROP FUNCTION IF EXISTS public.search_kg_nodes(uuid, uuid, vector, integer, text[], text[]);

create or replace function public.search_kg_nodes(
  p_tenant_id      uuid,
  p_client_id      uuid         default null,
  p_embedding      vector(1536),
  p_top_k          int          default 5,
  p_types          text[]       default null,
  p_document_ids   text[]       default null,
  p_recency_weight float4       default 0.0,
  p_boost_pinned   boolean      default false,
  p_exclude_status text[]       default array['archived','deprecated']
)
returns table (
  id          uuid,
  node_key    text,
  name        text,
  description text,
  properties  jsonb,
  type        artifact_type,
  similarity  float4,
  final_score float4
)
language sql
stable
as $$
  with scored as (
    select
      n.id,
      n.node_key,
      n.name,
      n.description,
      n.properties,
      n.type,
      (1 - (n.embedding <=> p_embedding))::float4 as similarity,

      -- Recency: time-decay on source_timestamp (falls back to doc created_at)
      -- ~1.0 for today, ~0.5 for 1 day ago, ~0.1 for 9 days ago
      case when d.id is not null and p_recency_weight > 0 then
        (1.0 / (1.0 + extract(epoch from (now() - coalesce(d.source_timestamp, d.created_at))) / 86400.0))::float4
      else 0.0 end as recency_score,

      -- Pinned/canonical boosts (only when enabled and document exists)
      case when p_boost_pinned and d.id is not null then
        (case when d.is_pinned then 0.05 else 0.0 end)
        + (case when d.is_canonical then 0.03 else 0.0 end)
      else 0.0 end as boost_score

    from public.kg_nodes n
    left join public.documents d
      on d.id = (n.properties->>'document_id')::uuid
      and d.tenant_id = p_tenant_id

    where n.tenant_id = p_tenant_id
      and (p_client_id is null or n.client_id = p_client_id)
      and n.status    = 'active'
      and n.embedding is not null
      and (p_types is null or n.type::text = any(p_types))
      and (p_document_ids is null or n.properties->>'document_id' = any(p_document_ids))
      -- Exclude documents with filtered statuses; entity nodes (d.id is null) pass through
      and (d.id is null or not (d.status = any(p_exclude_status)))
  )
  select
    s.id,
    s.node_key,
    s.name,
    s.description,
    s.properties,
    s.type,
    s.similarity,
    (
      (1.0 - p_recency_weight) * s.similarity
      + p_recency_weight * s.recency_score
      + s.boost_score
    )::float4 as final_score
  from scored s
  order by final_score desc
  limit p_top_k;
$$;
