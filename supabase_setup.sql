-- ============================================================================
-- Supabase Database Setup — Knowledge Graph RAG API
-- ============================================================================
--
-- This script runs all migrations from src/supabase/migrations/ in order,
-- then adds a storage bucket and optional seed data for testing.
--
-- HOW TO USE:
--   Option A — Supabase SQL Editor (Dashboard → SQL Editor → New Query):
--     Paste the full contents of this file and click "Run".
--
--   Option B — psql (against your Supabase/Postgres connection string):
--     psql "$DATABASE_URL" -f supabase_setup.sql
--
--   Option C — Run migrations individually in numbered order:
--     psql "$DATABASE_URL" -f src/supabase/migrations/00_extensions.sql
--     psql "$DATABASE_URL" -f src/supabase/migrations/01_types.sql
--     ...through 11_search_kg_nodes_rpc.sql, then run the SEED + VERIFY
--     sections from this file.
--
-- MIGRATION FILES (source of truth for schema):
--   src/supabase/migrations/
--     00_extensions.sql          — pgvector + pgcrypto extensions
--     01_types.sql               — custom enums: node_status, artifact_type, relation_type
--     02_documents.sql           — documents table + indexes
--     03_chunks.sql              — chunks table + embedding HNSW index
--     04_kg_nodes.sql            — knowledge graph nodes + embedding HNSW index
--     05_kg_edges.sql            — knowledge graph edges + indexes
--     06_evidence_node.sql       — node ↔ chunk evidence links
--     07_evidence_edge.sql       — edge ↔ chunk evidence links
--     08_triggers_updated_at.sql — auto-update updated_at trigger function
--     09_upsert_rpc.sql          — upsert_kg_node, upsert_kg_edge, upsert_chunk RPCs
--     09b_fetch_chunks_rpc.sql   — fetch_chunks_with_embeddings RPC (server-side JOIN)
--     10_pruning.sql             — prune_kg + helper RPCs for stale node/edge cleanup
--     11_search_kg_nodes_rpc.sql — search_kg_nodes vector similarity RPC
--     12_context_summaries.sql   — context_summaries table + upsert RPC
-- ============================================================================


-- ############################################################################
-- MIGRATION 00: Extensions
-- ############################################################################

create extension if not exists vector;
create extension if not exists pgcrypto;


-- ############################################################################
-- MIGRATION 01: Custom Types
-- ############################################################################

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


-- ############################################################################
-- MIGRATION 02: Documents
-- ############################################################################

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


-- ############################################################################
-- MIGRATION 03: Chunks
-- ############################################################################

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


-- ############################################################################
-- MIGRATION 04: KG Nodes
-- ############################################################################

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


-- ############################################################################
-- MIGRATION 05: KG Edges
-- ############################################################################

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


-- ############################################################################
-- MIGRATION 06: Node Evidence
-- ############################################################################

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


-- ############################################################################
-- MIGRATION 07: Edge Evidence
-- ############################################################################

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


-- ############################################################################
-- MIGRATION 08: Triggers — auto-update updated_at
-- ############################################################################

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


-- ############################################################################
-- MIGRATION 09: Upsert RPCs (kg_node, kg_edge, chunk)
-- ############################################################################

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


-- ############################################################################
-- MIGRATION 09b: fetch_chunks_with_embeddings RPC
-- ############################################################################
-- Server-side JOIN that replaces the two-step Python approach (fetch doc ids,
-- then chunks), avoiding the Supabase .in_() limit for large client datasets.

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


-- ############################################################################
-- MIGRATION 10: Pruning RPCs
-- ############################################################################

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


-- ############################################################################
-- MIGRATION 11: search_kg_nodes RPC
-- ############################################################################

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


-- ############################################################################
-- MIGRATION 12: Context Summaries
-- ############################################################################

create table if not exists public.context_summaries (
  id          uuid        primary key default gen_random_uuid(),
  tenant_id   uuid        not null,
  client_id   uuid        not null,

  summary     text        not null,
  topics      jsonb       not null default '[]'::jsonb,
  metadata    jsonb       not null default '{}'::jsonb,

  source_stats jsonb      not null default '{}'::jsonb,

  created_at  timestamptz not null default now(),
  updated_at  timestamptz not null default now()
);

create unique index if not exists context_summaries_tenant_client_uq
  on public.context_summaries(tenant_id, client_id);

create index if not exists context_summaries_tenant_idx
  on public.context_summaries(tenant_id);

do $$
begin
  if not exists (select 1 from pg_trigger where tgname = 'trg_context_summaries_set_updated_at') then
    create trigger trg_context_summaries_set_updated_at
    before update on public.context_summaries
    for each row execute function public.set_updated_at();
  end if;
end $$;

create or replace function public.upsert_context_summary(
  p_tenant_id    uuid,
  p_client_id    uuid,
  p_summary      text,
  p_topics       jsonb  default '[]'::jsonb,
  p_metadata     jsonb  default '{}'::jsonb,
  p_source_stats jsonb  default '{}'::jsonb
)
returns uuid
language plpgsql
as $$
declare
  v_id uuid;
begin
  insert into public.context_summaries (
    tenant_id, client_id, summary, topics, metadata, source_stats,
    created_at, updated_at
  )
  values (
    p_tenant_id, p_client_id, p_summary,
    coalesce(p_topics, '[]'::jsonb),
    coalesce(p_metadata, '{}'::jsonb),
    coalesce(p_source_stats, '{}'::jsonb),
    now(), now()
  )
  on conflict (tenant_id, client_id)
  do update set
    summary      = excluded.summary,
    topics       = excluded.topics,
    metadata     = coalesce(public.context_summaries.metadata, '{}'::jsonb) || coalesce(excluded.metadata, '{}'::jsonb),
    source_stats = excluded.source_stats,
    updated_at   = now()
  returning id into v_id;

  return v_id;
end;
$$;


-- ############################################################################
-- STORAGE BUCKET
-- ############################################################################
-- Create the "pdf" bucket for document storage.
-- NOTE: On Supabase you can also create this via Dashboard → Storage → New bucket.

insert into storage.buckets (id, name, public)
values ('pdf', 'pdf', false)
on conflict (id) do nothing;


-- ############################################################################
-- ROW LEVEL SECURITY (optional — enable per table as needed)
-- ############################################################################
-- Uncomment if you want RLS. For service_role key usage (server-side), RLS is
-- bypassed automatically. These policies apply to anon/authenticated keys.
--
-- alter table documents         enable row level security;
-- alter table chunks            enable row level security;
-- alter table kg_nodes          enable row level security;
-- alter table kg_edges          enable row level security;
-- alter table kg_node_evidence  enable row level security;
-- alter table kg_edge_evidence  enable row level security;
--
-- create policy "Tenant isolation" on documents
--     for all using (tenant_id = auth.uid());
-- (Repeat for each table with appropriate policy)


-- ############################################################################
-- SEED DATA FOR TESTING
-- ############################################################################

do $$
declare
  v_tenant_id uuid := 'a0000000-0000-0000-0000-000000000001';
  v_client_id uuid := 'b0000000-0000-0000-0000-000000000001';
  v_doc_id    uuid := 'c0000000-0000-0000-0000-000000000001';
begin
  -- Insert a test document
  insert into public.documents (id, tenant_id, client_id, source_type, source_uri, title, metadata)
  values (
    v_doc_id,
    v_tenant_id,
    v_client_id,
    'pdf',
    'bucket:pdf/test_document.pdf',
    'Test Document',
    '{"file_type": "pdf", "file_name": "test_document.pdf"}'::jsonb
  )
  on conflict do nothing;

  raise notice '──────────────────────────────────────────────────────';
  raise notice 'Test tenant_id : %', v_tenant_id;
  raise notice 'Test client_id : %', v_client_id;
  raise notice 'Test doc_id    : %', v_doc_id;
  raise notice 'Use these UUIDs in your .env or API calls for testing.';
  raise notice '──────────────────────────────────────────────────────';
end;
$$;


-- ############################################################################
-- VERIFICATION — run these to confirm everything is set up
-- ############################################################################

-- Check tables exist
select table_name
from information_schema.tables
where table_schema = 'public'
  and table_name in (
    'documents', 'chunks', 'kg_nodes', 'kg_edges',
    'kg_node_evidence', 'kg_edge_evidence', 'context_summaries'
  )
order by table_name;

-- Check RPC functions exist
select routine_name
from information_schema.routines
where routine_schema = 'public'
  and routine_name in (
    'upsert_chunk', 'upsert_kg_node', 'upsert_kg_edge',
    'search_kg_nodes', 'fetch_chunks_with_embeddings',
    'prune_kg', 'prune_archive_stale_edges', 'prune_archive_stale_nodes',
    'prune_trim_edge_evidence', 'prune_trim_node_evidence',
    'upsert_context_summary'
  )
order by routine_name;

-- Check custom types exist
select typname from pg_type
where typname in ('node_status', 'artifact_type', 'relation_type');

-- Check pgvector extension
select extname, extversion from pg_extension where extname = 'vector';

-- Check storage bucket
select id, name, public from storage.buckets where id = 'pdf';
