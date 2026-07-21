-- 44_seen_count_scoring.sql
-- STEP 1 of corroboration-based learning: consume seen_count in scoring.
--
-- seen_count increments on every re-upsert of a node (09_upsert_rpc.sql:37,
-- 37/39 entity RPCs, 43 spine RPCs) but was never read by any scoring or
-- traversal path. This migration makes search_kg_nodes (latest def: mig 34) and
-- hybrid_search_kg_nodes (latest def: mig 42) add a SMALL, BOUNDED corroboration
-- bonus so a node corroborated across many ingests outranks an otherwise-equal
-- node seen once.
--
-- Shape of the bonus (saturating, never dominates):
--     bonus = p_corroboration_weight * (seen_count / (seen_count + p_corroboration_k))
-- The fraction is in [0,1), so the bonus is hard-capped at p_corroboration_weight
-- (default 0.05). Cosine similarity is ~[0,1] and the RRF score is ~[0,0.03];
-- a 0.05 cap means corroboration breaks near-ties and nudges ranking, but cannot
-- override a clearly-more-similar result. K=5 means the bonus reaches half its
-- cap at seen_count=5.
--
-- Tunable / reversible: pass p_corroboration_weight => 0 to disable per-call; a
-- follow-up migration can re-drop+recreate to revert. Defaults are ON so the
-- behavior is active without any caller change (the retriever passes neither
-- new param and inherits the defaults).
--
-- Honest caveat: seen_count counts re-upserts of the SAME node_key, not distinct
-- independent sources. For string-keyed entities a re-mention across ingests does
-- bump it; but until Step 4 (write-time reconciliation) lands, two phrasings of
-- one concept are two keys and each accrues its own count. So this is a coarse
-- corroboration proxy, strengthened materially only once Step 4 routes mentions
-- onto a shared node.
--
-- Tenant isolation unchanged: every WHERE clause keeps `n.tenant_id =
-- p_tenant_id` and the client predicate exactly as before. No traversal change.
-- No retrieval-path Python change. Independently revertable.


-- ── search_kg_nodes (dense) ─────────────────────────────────────────────────
-- Drop the migration-34 signature (13 args) before recreating with 2 new
-- trailing params + seen_count in output, to avoid an overload (the prior
-- "RPC overload ambiguity" class of bug).
drop function if exists public.search_kg_nodes(
  uuid, uuid, vector, int, text[], text[], float4, boolean, text[], text,
  text[], uuid, boolean
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
  p_cross_study_entities   boolean      default false,
  p_corroboration_weight   float4       default 0.05,
  p_corroboration_k        float4       default 5.0
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
  seen_count      int,
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
      n.seen_count,
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
    s.seen_count,
    s.similarity,
    (
      (1.0 - p_recency_weight) * s.similarity
      + p_recency_weight * s.recency_score
      + s.boost_score
      -- corroboration bonus: saturating, capped at p_corroboration_weight
      + p_corroboration_weight
        * (s.seen_count::float4 / (s.seen_count::float4 + p_corroboration_k))
    )::float4 as final_score
  from scored s
  order by final_score desc
  limit p_top_k;
$$;


-- ── hybrid_search_kg_nodes (dense + BM25 RRF) ───────────────────────────────
-- Drop the migration-42 signature (16 args) before recreating with 2 new
-- trailing params + seen_count in output.
drop function if exists public.hybrid_search_kg_nodes(
  uuid, uuid, vector, text, int, text[], text[], float4, boolean, text[], text,
  text[], uuid, boolean, int, float4
) cascade;

create or replace function public.hybrid_search_kg_nodes(
  p_tenant_id              uuid,
  p_client_id              uuid          default null,
  p_embedding              vector(1536)  default null,
  p_query_text             text          default null,
  p_top_k                  int           default 10,
  p_types                  text[]        default null,
  p_document_ids           text[]        default null,
  p_recency_weight         float4        default 0.0,
  p_boost_pinned           boolean       default false,
  p_exclude_status         text[]        default array['archived','deprecated'],
  p_embedding_model        text          default 'text-embedding-3-small',
  p_source_types           text[]        default null,
  p_study_id               uuid          default null,
  p_cross_study_entities   boolean       default false,
  p_candidate_pool         int           default 50,
  p_sparse_weight          float4        default 1.0,
  p_corroboration_weight   float4        default 0.05,
  p_corroboration_k        float4        default 5.0
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
  seen_count      int,
  dense_rank      int,
  sparse_rank     int,
  rrf_score       float4,
  final_score     float4
)
language sql
stable
as $$
  with
  dense as (
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
      n.seen_count,
      (1 - (n.embedding <=> p_embedding))::float4 as similarity,
      case when d.id is not null and p_recency_weight > 0 then
        (1.0 / (1.0 + extract(epoch from (now() - coalesce(d.source_timestamp, d.created_at))) / 86400.0))::float4
      else 0.0 end as recency_score,
      case when p_boost_pinned and d.id is not null then
        (case when d.is_pinned then 0.05 else 0.0 end)
        + (case when d.is_canonical then 0.03 else 0.0 end)
      else 0.0 end as boost_score,
      row_number() over (
        order by (n.embedding <=> p_embedding) asc
      )::int as dense_rank
    from public.kg_nodes n
    left join public.documents d
      on d.id = (n.properties->>'document_id')::uuid
     and d.tenant_id = p_tenant_id
    where p_embedding is not null
      and n.tenant_id = p_tenant_id
      and (p_client_id is null or n.client_id = p_client_id)
      and n.status = 'active'
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
    order by n.embedding <=> p_embedding
    limit p_candidate_pool
  ),
  sparse as (
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
      n.seen_count,
      ts_rank_cd(c.content_tsv, plainto_tsquery('english', p_query_text))::float4 as bm25,
      row_number() over (
        order by ts_rank_cd(c.content_tsv, plainto_tsquery('english', p_query_text)) desc
      )::int as sparse_rank
    from public.chunks c
    join public.kg_nodes n
      on (n.properties->>'chunk_id')::uuid = c.id
     and n.tenant_id = c.tenant_id
    left join public.documents d
      on d.id = c.document_id
     and d.tenant_id = p_tenant_id
    where p_query_text is not null and length(trim(p_query_text)) > 0
      and c.tenant_id = p_tenant_id
      and c.content_tsv @@ plainto_tsquery('english', p_query_text)
      and (p_client_id is null or n.client_id = p_client_id)
      and n.status = 'active'
      and (p_types is null or n.type::text = any(p_types))
      and (p_document_ids is null or n.properties->>'document_id' = any(p_document_ids))
      and (d.id is null or not (d.status = any(p_exclude_status)))
      and (p_source_types is null or (d.id is not null and d.source_type = any(p_source_types)))
      and (
        p_study_id is null
        or coalesce(d.study_id, n.study_id) = p_study_id
      )
    order by bm25 desc
    limit p_candidate_pool
  ),
  fused as (
    select
      coalesce(d.id, s.id)                            as id,
      coalesce(d.node_key, s.node_key)                as node_key,
      coalesce(d.name, s.name)                        as name,
      coalesce(d.description, s.description)          as description,
      coalesce(d.properties, s.properties)            as properties,
      coalesce(d.type, s.type)                        as type,
      coalesce(d.doc_source_type, s.doc_source_type)  as source_type,
      coalesce(d.embedding_model, s.embedding_model)  as embedding_model,
      coalesce(d.study_id, s.study_id)                as study_id,
      coalesce(d.seen_count, s.seen_count)            as seen_count,
      d.dense_rank,
      s.sparse_rank,
      (
        coalesce(1.0 / (60 + d.dense_rank), 0.0)
        + p_sparse_weight * coalesce(1.0 / (60 + s.sparse_rank), 0.0)
      )::float4 as rrf_score,
      coalesce(d.similarity, 0.0)    as similarity,
      coalesce(d.recency_score, 0.0) as recency_score,
      coalesce(d.boost_score, 0.0)   as boost_score
    from dense d
    full outer join sparse s on s.id = d.id
  )
  select
    f.id,
    f.node_key,
    f.name,
    f.description,
    f.properties,
    f.type,
    f.source_type,
    f.embedding_model,
    f.study_id,
    f.seen_count,
    f.dense_rank,
    f.sparse_rank,
    f.rrf_score,
    (
      f.rrf_score
      + p_recency_weight * f.recency_score
      + f.boost_score
      -- corroboration bonus: saturating, capped at p_corroboration_weight
      + p_corroboration_weight
        * (coalesce(f.seen_count, 0)::float4 / (coalesce(f.seen_count, 0)::float4 + p_corroboration_k))
    )::float4 as final_score
  from fused f
  order by final_score desc
  limit p_top_k;
$$;
