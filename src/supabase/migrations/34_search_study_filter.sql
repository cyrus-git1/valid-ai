-- 34_search_study_filter.sql
-- Adds p_study_id to search_kg_nodes so retrieval can be scoped to a single
-- study. NULL = study-agnostic (current behaviour, fully backward-compatible).
--
-- Also adds an optional p_cross_study_entities flag: when true, hop expansion
-- callers can choose to bridge entities across studies for the same tenant.
-- Default false → strict study isolation when p_study_id is set.

drop function if exists public.search_kg_nodes(
  uuid, uuid, vector, int, text[], text[], float4, boolean, text[], text, text[]
) cascade;

create or replace function public.search_kg_nodes(
  p_tenant_id              uuid,
  p_client_id              uuid         default null,
  p_embedding              vector(1536) default null,
  p_top_k                  int          default 5,
  p_types                  text[]       default null,
  p_document_ids           text[]       default null,
  p_recency_weight         float4       default 0.0,
  p_boost_pinned           boolean      default false,
  p_exclude_status         text[]       default array['archived','deprecated'],
  p_embedding_model        text         default 'text-embedding-3-small',
  p_source_types           text[]       default null,
  p_study_id               uuid         default null,
  p_cross_study_entities   boolean      default false
)
returns table (
  id              uuid,
  node_key        text,
  name            text,
  description     text,
  properties      jsonb,
  type            artifact_type,
  source_type     text,
  embedding_model text,
  study_id        uuid,
  similarity      float4,
  final_score     float4
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
      d.source_type as doc_source_type,
      n.embedding_model,
      coalesce(d.study_id, n.study_id) as study_id,
      (1 - (n.embedding <=> p_embedding))::float4 as similarity,

      case when d.id is not null and p_recency_weight > 0 then
        (1.0 / (1.0 + extract(epoch from (now() - coalesce(d.source_timestamp, d.created_at))) / 86400.0))::float4
      else 0.0 end as recency_score,

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
      and n.embedding_model = p_embedding_model
      and (p_types is null or n.type::text = any(p_types))
      and (p_document_ids is null or n.properties->>'document_id' = any(p_document_ids))
      and (d.id is null or not (d.status = any(p_exclude_status)))
      and (p_source_types is null or (d.id is not null and d.source_type = any(p_source_types)))
      and (
        p_study_id is null
        or coalesce(d.study_id, n.study_id) = p_study_id
        or (p_cross_study_entities and n.type::text in ('Entity','Person','Organization','Concept'))
      )
  )
  select
    s.id,
    s.node_key,
    s.name,
    s.description,
    s.properties,
    s.type,
    s.doc_source_type as source_type,
    s.embedding_model,
    s.study_id,
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
