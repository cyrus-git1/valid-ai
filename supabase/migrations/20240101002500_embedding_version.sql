-- 25_embedding_version.sql
-- Track which embedding model generated each embedding so a model upgrade
-- doesn't silently mix vector spaces. Existing rows default to the current
-- model (text-embedding-3-small).

alter table public.chunks
  add column if not exists embedding_model text not null default 'text-embedding-3-small';

alter table public.kg_nodes
  add column if not exists embedding_model text not null default 'text-embedding-3-small';

create index if not exists chunks_embedding_model_idx
  on public.chunks (tenant_id, embedding_model);

create index if not exists kg_nodes_embedding_model_idx
  on public.kg_nodes (tenant_id, embedding_model);

-- Update upsert RPCs to accept embedding_model
create or replace function public.upsert_kg_node(
  p_tenant_id uuid,
  p_client_id uuid,
  p_node_key text,
  p_type artifact_type,
  p_name text,
  p_description text default null,
  p_properties jsonb default '{}'::jsonb,
  p_embedding vector(1536) default null,
  p_status node_status default 'active',
  p_embedding_model text default 'text-embedding-3-small'
)
returns uuid
language plpgsql
as $$
declare
  v_id uuid;
begin
  insert into public.kg_nodes (
    tenant_id, client_id, node_key, type, name, description, properties, embedding,
    embedding_model, status, last_seen_at, seen_count, created_at, updated_at
  )
  values (
    p_tenant_id, p_client_id, p_node_key, p_type, p_name, p_description,
    coalesce(p_properties, '{}'::jsonb), p_embedding, p_embedding_model,
    p_status, now(), 1, now(), now()
  )
  on conflict (tenant_id, client_id, node_key)
  do update set
    type = excluded.type,
    name = excluded.name,
    description = excluded.description,
    properties = coalesce(public.kg_nodes.properties, '{}'::jsonb) || coalesce(excluded.properties, '{}'::jsonb),
    embedding = coalesce(excluded.embedding, public.kg_nodes.embedding),
    embedding_model = coalesce(excluded.embedding_model, public.kg_nodes.embedding_model),
    status = excluded.status,
    last_seen_at = now(),
    seen_count = public.kg_nodes.seen_count + 1,
    updated_at = now()
  returning id into v_id;

  return v_id;
end;
$$;

create or replace function public.upsert_chunk(
  p_tenant_id uuid,
  p_document_id uuid,
  p_chunk_index int,
  p_page_start int default null,
  p_page_end int default null,
  p_content text default null,
  p_content_tokens int default null,
  p_metadata jsonb default '{}'::jsonb,
  p_embedding vector(1536) default null,
  p_embedding_model text default 'text-embedding-3-small'
)
returns uuid
language plpgsql
as $$
declare
  v_id uuid;
begin
  insert into public.chunks (
    tenant_id, document_id, chunk_index, page_start, page_end,
    content, content_tokens, metadata, embedding, embedding_model, created_at
  )
  values (
    p_tenant_id, p_document_id, p_chunk_index, p_page_start, p_page_end,
    p_content, p_content_tokens, coalesce(p_metadata, '{}'::jsonb),
    p_embedding, p_embedding_model, now()
  )
  on conflict (tenant_id, document_id, chunk_index)
  do update set
    page_start = coalesce(excluded.page_start, public.chunks.page_start),
    page_end = coalesce(excluded.page_end, public.chunks.page_end),
    content = coalesce(excluded.content, public.chunks.content),
    content_tokens = coalesce(excluded.content_tokens, public.chunks.content_tokens),
    metadata = coalesce(public.chunks.metadata, '{}'::jsonb) || coalesce(excluded.metadata, '{}'::jsonb),
    embedding = coalesce(excluded.embedding, public.chunks.embedding),
    embedding_model = coalesce(excluded.embedding_model, public.chunks.embedding_model)
  returning id into v_id;

  return v_id;
end;
$$;

-- Update search RPC: add embedding_model filter so we never compare across
-- different vector spaces.
DROP FUNCTION IF EXISTS public.search_kg_nodes(
  uuid, uuid, vector, integer, text[], text[], float4, boolean, text[]
);

create or replace function public.search_kg_nodes(
  p_tenant_id       uuid,
  p_client_id       uuid         default null,
  p_embedding       vector(1536),
  p_top_k           int          default 5,
  p_types           text[]       default null,
  p_document_ids    text[]       default null,
  p_recency_weight  float4       default 0.0,
  p_boost_pinned    boolean      default false,
  p_exclude_status  text[]       default array['archived','deprecated'],
  p_embedding_model text         default 'text-embedding-3-small'
)
returns table (
  id              uuid,
  node_key        text,
  name            text,
  description     text,
  properties      jsonb,
  type            artifact_type,
  embedding_model text,
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
      n.embedding_model,
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
  )
  select
    s.id,
    s.node_key,
    s.name,
    s.description,
    s.properties,
    s.type,
    s.embedding_model,
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
