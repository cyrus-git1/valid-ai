-- ============================================================================
-- Supabase Database Initialization Script
-- ============================================================================
-- This script initializes the complete database schema for the Knowledge Graph RAG API.
-- Run this script in your Supabase SQL editor or via psql to set up the database.
--
-- Usage:
--   1. Connect to your Supabase project
--   2. Run this entire script in the SQL editor
--   3. Or run migrations individually in order (00-11)
--
-- ============================================================================

-- ============================================================================
-- 00. Extensions
-- ============================================================================
-- Enable required PostgreSQL extensions

create extension if not exists vector;
create extension if not exists pgcrypto;

-- ============================================================================
-- 01. Custom Types
-- ============================================================================

do $$
begin
  if not exists (select 1 from pg_type where typname = 'node_status') then
    create type node_status as enum ('active', 'pending_linking', 'archived');
  end if;

  if not exists (select 1 from pg_type where typname = 'artifact_type') then
    create type artifact_type as enum (
      'WebPage',
      'PDF',
      'Image',
      'PowerPoint',
      'Docx',
      'VideoTranscript',
      'ChatTranscript',
      'ChatSnapshot',
      'Chunk'
    );
  end if;

  if not exists (select 1 from pg_type where typname = 'relation_type') then
    create type relation_type as enum (
      'has_chunk',
      'derived_from',
      'references',
      'related_to',
      'supports',
      'duplicate_of'
    );
  end if;
end $$;

-- ============================================================================
-- 02. Documents Table
-- ============================================================================
-- Stores metadata about ingested documents (PDFs, DOCX, web pages)

create table if not exists public.documents (
  id uuid primary key default gen_random_uuid(),
  tenant_id uuid not null,
  client_id uuid,

  source_type text not null,
  source_uri text,
  title text,

  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists documents_tenant_idx on public.documents(tenant_id);
create index if not exists documents_client_idx on public.documents(tenant_id, client_id);
create index if not exists documents_metadata_gin on public.documents using gin(metadata);

-- ============================================================================
-- 03. Chunks Table
-- ============================================================================
-- Stores text chunks extracted from documents with embeddings

create table if not exists public.chunks (
  id uuid primary key default gen_random_uuid(),
  tenant_id uuid not null,
  document_id uuid not null references public.documents(id) on delete cascade,

  chunk_index int not null,
  page_start int,
  page_end int,

  content text not null,
  content_tokens int,

  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),

  embedding vector(1536)
);

create unique index if not exists chunks_doc_chunk_idx_uq
  on public.chunks(tenant_id, document_id, chunk_index);

create index if not exists chunks_tenant_doc_idx on public.chunks(tenant_id, document_id);
create index if not exists chunks_metadata_gin on public.chunks using gin(metadata);

create index if not exists chunks_embedding_hnsw
  on public.chunks using hnsw (embedding vector_cosine_ops);

-- ============================================================================
-- 04. Knowledge Graph Nodes Table
-- ============================================================================
-- Nodes in the knowledge graph representing concepts, chunks, or artifacts

create table if not exists public.kg_nodes (
  id uuid primary key default gen_random_uuid(),
  tenant_id uuid not null,
  client_id uuid,

  node_key text not null,

  type artifact_type not null,
  name text not null,
  description text,

  properties jsonb not null default '{}'::jsonb,
  embedding vector(1536),

  status node_status not null default 'active',

  last_seen_at timestamptz not null default now(),
  seen_count int not null default 0,

  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create unique index if not exists kg_nodes_node_key_uq
  on public.kg_nodes(tenant_id, client_id, node_key);

create index if not exists kg_nodes_tenant_idx on public.kg_nodes(tenant_id);
create index if not exists kg_nodes_client_idx on public.kg_nodes(tenant_id, client_id);
create index if not exists kg_nodes_type_idx on public.kg_nodes(tenant_id, type);
create index if not exists kg_nodes_status_idx on public.kg_nodes(tenant_id, status);
create index if not exists kg_nodes_last_seen_idx on public.kg_nodes(tenant_id, client_id, last_seen_at);

create index if not exists kg_nodes_props_gin on public.kg_nodes using gin(properties);
create index if not exists kg_nodes_embedding_hnsw
  on public.kg_nodes using hnsw (embedding vector_cosine_ops);

-- ============================================================================
-- 05. Knowledge Graph Edges Table
-- ============================================================================
-- Relationships between nodes in the knowledge graph

create table if not exists public.kg_edges (
  id uuid primary key default gen_random_uuid(),
  tenant_id uuid not null,
  client_id uuid,

  src_id uuid not null references public.kg_nodes(id) on delete cascade,
  dst_id uuid not null references public.kg_nodes(id) on delete cascade,

  rel_type text not null,
  weight float4,
  properties jsonb not null default '{}'::jsonb,

  is_active boolean not null default true,
  last_seen_at timestamptz not null default now(),
  seen_count int not null default 0,

  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),

  unique (tenant_id, client_id, src_id, dst_id, rel_type)
);

create index if not exists kg_edges_src_idx on public.kg_edges(tenant_id, client_id, src_id);
create index if not exists kg_edges_dst_idx on public.kg_edges(tenant_id, client_id, dst_id);
create index if not exists kg_edges_rel_idx on public.kg_edges(tenant_id, client_id, rel_type);
create index if not exists kg_edges_active_idx on public.kg_edges(tenant_id, client_id, is_active);
create index if not exists kg_edges_last_seen_idx on public.kg_edges(tenant_id, client_id, last_seen_at);
create index if not exists kg_edges_props_gin on public.kg_edges using gin(properties);

-- ============================================================================
-- 06. Node Evidence Table
-- ============================================================================
-- Links nodes to source chunks that provide evidence for the node

create table if not exists public.kg_node_evidence (
  id uuid primary key default gen_random_uuid(),
  tenant_id uuid not null,
  client_id uuid,

  node_id uuid not null references public.kg_nodes(id) on delete cascade,
  chunk_id uuid not null references public.chunks(id) on delete cascade,

  quote text,
  score float4,

  created_at timestamptz not null default now(),

  unique (tenant_id, client_id, node_id, chunk_id)
);

create index if not exists kg_node_evidence_node_idx on public.kg_node_evidence(tenant_id, client_id, node_id);
create index if not exists kg_node_evidence_chunk_idx on public.kg_node_evidence(tenant_id, client_id, chunk_id);

-- ============================================================================
-- 07. Edge Evidence Table
-- ============================================================================
-- Links edges to source chunks that provide evidence for the relationship

create table if not exists public.kg_edge_evidence (
  id uuid primary key default gen_random_uuid(),
  tenant_id uuid not null,
  client_id uuid,

  edge_id uuid not null references public.kg_edges(id) on delete cascade,
  chunk_id uuid not null references public.chunks(id) on delete cascade,

  quote text,
  score float4,

  created_at timestamptz not null default now(),

  unique (tenant_id, client_id, edge_id, chunk_id)
);

create index if not exists kg_edge_evidence_edge_idx on public.kg_edge_evidence(tenant_id, client_id, edge_id);
create index if not exists kg_edge_evidence_chunk_idx on public.kg_edge_evidence(tenant_id, client_id, chunk_id);

-- ============================================================================
-- 08. Updated At Triggers
-- ============================================================================
-- Automatically update updated_at timestamp on row updates

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

do $$
begin
  if not exists (select 1 from pg_trigger where tgname='trg_documents_set_updated_at') then
    create trigger trg_documents_set_updated_at
    before update on public.documents
    for each row execute function public.set_updated_at();
  end if;

  if not exists (select 1 from pg_trigger where tgname='trg_kg_nodes_set_updated_at') then
    create trigger trg_kg_nodes_set_updated_at
    before update on public.kg_nodes
    for each row execute function public.set_updated_at();
  end if;

  if not exists (select 1 from pg_trigger where tgname='trg_kg_edges_set_updated_at') then
    create trigger trg_kg_edges_set_updated_at
    before update on public.kg_edges
    for each row execute function public.set_updated_at();
  end if;
end $$;

-- ============================================================================
-- 09. Upsert RPC Functions
-- ============================================================================
-- Functions for upserting chunks, nodes, and edges

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

-- ============================================================================
-- 09b. Fetch Chunks RPC
-- ============================================================================
-- Server-side JOIN to fetch chunks with embeddings, avoiding Supabase .in_() limits

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

-- ============================================================================
-- 10. Pruning Functions
-- ============================================================================
-- Functions for archiving stale nodes/edges and trimming evidence

create or replace function public.prune_archive_stale_edges(
  p_tenant_id uuid,
  p_client_id uuid,
  p_stale_days int default 90
)
returns int
language sql
as $$
  with upd as (
    update public.kg_edges e
    set is_active = false,
        updated_at = now()
    where e.tenant_id = p_tenant_id
      and e.client_id = p_client_id
      and e.is_active = true
      and e.last_seen_at < now() - make_interval(days => p_stale_days)
    returning 1
  )
  select count(*)::int from upd;
$$;

create or replace function public.prune_archive_stale_nodes(
  p_tenant_id uuid,
  p_client_id uuid,
  p_stale_days int default 180,
  p_min_degree int default 3
)
returns int
language sql
as $$
  with degrees as (
    select
      n.id as node_id,
      count(e.id) as degree
    from public.kg_nodes n
    left join public.kg_edges e
      on e.tenant_id = n.tenant_id
     and e.client_id = n.client_id
     and e.is_active = true
     and (e.src_id = n.id or e.dst_id = n.id)
    where n.tenant_id = p_tenant_id
      and n.client_id = p_client_id
    group by n.id
  ),
  upd as (
    update public.kg_nodes n
    set status = 'archived',
        updated_at = now()
    from degrees d
    where n.id = d.node_id
      and n.tenant_id = p_tenant_id
      and n.client_id = p_client_id
      and n.status <> 'archived'
      and n.last_seen_at < now() - make_interval(days => p_stale_days)
      and d.degree < p_min_degree
    returning 1
  )
  select count(*)::int from upd;
$$;

create or replace function public.prune_trim_edge_evidence(
  p_tenant_id uuid,
  p_client_id uuid,
  p_keep_per_edge int default 5
)
returns int
language sql
as $$
  with ranked as (
    select
      ee.id,
      row_number() over (
        partition by ee.edge_id
        order by ee.score desc nulls last, ee.created_at desc
      ) as rn
    from public.kg_edge_evidence ee
    where ee.tenant_id = p_tenant_id
      and ee.client_id = p_client_id
  ),
  del as (
    delete from public.kg_edge_evidence ee
    using ranked r
    where ee.id = r.id
      and r.rn > p_keep_per_edge
    returning 1
  )
  select count(*)::int from del;
$$;

create or replace function public.prune_trim_node_evidence(
  p_tenant_id uuid,
  p_client_id uuid,
  p_keep_per_node int default 10
)
returns int
language sql
as $$
  with ranked as (
    select
      ne.id,
      row_number() over (
        partition by ne.node_id
        order by ne.score desc nulls last, ne.created_at desc
      ) as rn
    from public.kg_node_evidence ne
    where ne.tenant_id = p_tenant_id
      and ne.client_id = p_client_id
  ),
  del as (
    delete from public.kg_node_evidence ne
    using ranked r
    where ne.id = r.id
      and r.rn > p_keep_per_node
    returning 1
  )
  select count(*)::int from del;
$$;

create or replace function public.prune_kg(
  p_tenant_id uuid,
  p_client_id uuid,
  p_edge_stale_days int default 90,
  p_node_stale_days int default 180,
  p_min_degree int default 3,
  p_keep_edge_evidence int default 5,
  p_keep_node_evidence int default 10
)
returns jsonb
language plpgsql
as $$
declare
  v_edges_archived int;
  v_nodes_archived int;
  v_edge_evidence_deleted int;
  v_node_evidence_deleted int;
begin
  select public.prune_archive_stale_edges(p_tenant_id, p_client_id, p_edge_stale_days)
    into v_edges_archived;

  select public.prune_archive_stale_nodes(p_tenant_id, p_client_id, p_node_stale_days, p_min_degree)
    into v_nodes_archived;

  select public.prune_trim_edge_evidence(p_tenant_id, p_client_id, p_keep_edge_evidence)
    into v_edge_evidence_deleted;

  select public.prune_trim_node_evidence(p_tenant_id, p_client_id, p_keep_node_evidence)
    into v_node_evidence_deleted;

  return jsonb_build_object(
    'edges_archived', v_edges_archived,
    'nodes_archived', v_nodes_archived,
    'edge_evidence_deleted', v_edge_evidence_deleted,
    'node_evidence_deleted', v_node_evidence_deleted
  );
end;
$$;

-- ============================================================================
-- 11. Search RPC Function
-- ============================================================================
-- Vector similarity search over kg_nodes for semantic retrieval

create or replace function public.search_kg_nodes(
  p_tenant_id uuid,
  p_client_id uuid,
  p_embedding vector(1536),
  p_top_k int default 5
)
returns table (
  id uuid,
  node_key text,
  name text,
  description text,
  properties jsonb,
  type artifact_type,
  similarity float
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
    1 - (n.embedding <=> p_embedding) as similarity
  from public.kg_nodes n
  where n.tenant_id = p_tenant_id
    and n.client_id = p_client_id
    and n.status = 'active'
    and n.embedding is not null
  order by n.embedding <=> p_embedding
  limit p_top_k;
$$;

-- ============================================================================
-- Initialization Complete
-- ============================================================================
-- The database schema is now ready for use.
--
-- Next steps:
--   1. Create storage buckets (e.g., 'pdf' bucket for document storage)
--   2. Set up Row Level Security (RLS) policies if needed
--   3. Configure API keys and environment variables
--
-- Storage bucket creation (run in Supabase Storage UI or via API):
--   - Create bucket named 'pdf' (private/public as needed)
--
-- ============================================================================


