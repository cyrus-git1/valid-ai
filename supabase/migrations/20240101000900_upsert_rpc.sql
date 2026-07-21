-- 09_upsert_rpc.sql
create or replace function public.upsert_kg_node(
  p_tenant_id uuid,
  p_client_id uuid,
  p_node_key text,
  p_type artifact_type,
  p_name text,
  p_description text default null,
  p_properties jsonb default '{}'::jsonb,
  p_embedding vector(1536) default null,
  p_status node_status default 'active'
)
returns uuid
language plpgsql
as $$
declare
  v_id uuid;
begin
  insert into public.kg_nodes (
    tenant_id, client_id, node_key, type, name, description, properties, embedding,
    status, last_seen_at, seen_count, created_at, updated_at
  )
  values (
    p_tenant_id, p_client_id, p_node_key, p_type, p_name, p_description,
    coalesce(p_properties, '{}'::jsonb), p_embedding,
    p_status, now(), 1, now(), now()
  )
  on conflict (tenant_id, client_id, node_key)
  do update set
    type = excluded.type,
    name = excluded.name,
    description = excluded.description,
    properties = coalesce(public.kg_nodes.properties, '{}'::jsonb) || coalesce(excluded.properties, '{}'::jsonb),
    embedding = coalesce(excluded.embedding, public.kg_nodes.embedding),
    status = excluded.status,
    last_seen_at = now(),
    seen_count = public.kg_nodes.seen_count + 1,
    updated_at = now()
  returning id into v_id;

  return v_id;
end;
$$;

create or replace function public.upsert_kg_edge(
  p_tenant_id uuid,
  p_client_id uuid,
  p_src_id uuid,
  p_dst_id uuid,
  p_rel_type text,
  p_weight float4 default null,
  p_properties jsonb default '{}'::jsonb
)
returns uuid
language plpgsql
as $$
declare
  v_id uuid;
begin
  insert into public.kg_edges (
    tenant_id, client_id, src_id, dst_id, rel_type, weight, properties,
    is_active, last_seen_at, seen_count, created_at, updated_at
  )
  values (
    p_tenant_id, p_client_id, p_src_id, p_dst_id, p_rel_type, p_weight,
    coalesce(p_properties, '{}'::jsonb),
    true, now(), 1, now(), now()
  )
  on conflict (tenant_id, client_id, src_id, dst_id, rel_type)
  do update set
    weight = coalesce(excluded.weight, public.kg_edges.weight),
    properties = coalesce(public.kg_edges.properties, '{}'::jsonb) || coalesce(excluded.properties, '{}'::jsonb),
    is_active = true,
    last_seen_at = now(),
    seen_count = public.kg_edges.seen_count + 1,
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
  p_embedding vector(1536) default null
)
returns uuid
language plpgsql
as $$
declare
  v_id uuid;
begin
  insert into public.chunks (
    tenant_id, document_id, chunk_index, page_start, page_end,
    content, content_tokens, metadata, embedding, created_at
  )
  values (
    p_tenant_id, p_document_id, p_chunk_index, p_page_start, p_page_end,
    p_content, p_content_tokens, coalesce(p_metadata, '{}'::jsonb), p_embedding, now()
  )
  on conflict (tenant_id, document_id, chunk_index)
  do update set
    page_start = coalesce(excluded.page_start, public.chunks.page_start),
    page_end = coalesce(excluded.page_end, public.chunks.page_end),
    content = coalesce(excluded.content, public.chunks.content),
    content_tokens = coalesce(excluded.content_tokens, public.chunks.content_tokens),
    metadata = coalesce(public.chunks.metadata, '{}'::jsonb) || coalesce(excluded.metadata, '{}'::jsonb),
    embedding = coalesce(excluded.embedding, public.chunks.embedding)
  returning id into v_id;

  return v_id;
end;
$$;
