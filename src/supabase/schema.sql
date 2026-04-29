-- ============================================================================
-- Valid Data Plane — Consolidated Schema (final state)
-- ============================================================================
-- Single-file idempotent script. Safe to run on a fresh database (after
-- reset.sql) or over an existing one to add any missing objects.
--
-- Reflects all migrations 00–26. The legacy context_summaries table and
-- upsert_context_summary / mark_summary_fresh RPCs are NOT created here —
-- summaries are stored as documents + chunks + kg_nodes.
--
-- Usage:
--   Fresh database:  psql -f reset.sql && psql -f schema.sql
--   Existing database: psql -f schema.sql   (idempotent, adds only what's missing)
--
-- Requires superuser / service-role for:
--   CREATE EXTENSION (vector, pgcrypto)
--   CREATE TYPE, CREATE TRIGGER
-- ============================================================================


-- ============================================================================
-- 1. EXTENSIONS
-- ============================================================================
create extension if not exists vector;
create extension if not exists pgcrypto;


-- ============================================================================
-- 2. ENUM TYPES
-- ============================================================================
do $$
begin
  if not exists (select 1 from pg_type where typname = 'node_status') then
    create type node_status as enum ('active', 'pending_linking', 'archived');
  end if;

  if not exists (select 1 from pg_type where typname = 'artifact_type') then
    create type artifact_type as enum (
      'WebPage', 'PDF', 'Image', 'PowerPoint', 'Docx',
      'VideoTranscript', 'ChatTranscript', 'ChatSnapshot',
      'Chunk', 'Entity',
      'ContextSummary', 'DocumentSummary', 'TopicSummary'
    );
  end if;

  if not exists (select 1 from pg_type where typname = 'relation_type') then
    create type relation_type as enum (
      'has_chunk', 'derived_from', 'references',
      'related_to', 'supports', 'duplicate_of'
    );
  end if;
end $$;


-- ============================================================================
-- 3. SHARED TRIGGER FUNCTION
-- ============================================================================
create or replace function public.set_updated_at()
returns trigger language plpgsql as $$
begin new.updated_at = now(); return new; end; $$;


-- ============================================================================
-- 4. TABLES
-- ============================================================================

-- ── documents ───────────────────────────────────────────────────────────────
create table if not exists public.documents (
  id                  uuid primary key default gen_random_uuid(),
  tenant_id           uuid not null,
  client_id           uuid,
  source_type         text not null,
  source_uri          text,
  title               text,
  source_timestamp    timestamptz,
  is_pinned           boolean not null default false,
  is_canonical        boolean not null default false,
  status              text not null default 'active'
                      check (status in ('active','draft','deprecated','archived','flagged')),
  metadata            jsonb not null default '{}'::jsonb,
  -- Provenance (migration 29)
  actor_id            text,
  actor_type          text check (actor_type is null or actor_type in ('user','service','agent','anonymous')),
  source_app          text,
  request_id          text,
  ingest_job_id       uuid,
  previous_version_id uuid references public.documents(id) on delete set null,
  created_at          timestamptz not null default now(),
  updated_at          timestamptz not null default now()
);

create unique index if not exists documents_tenant_source_uq    on public.documents(tenant_id, client_id, source_uri);
create index if not exists documents_tenant_idx                 on public.documents(tenant_id);
create index if not exists documents_client_idx                 on public.documents(tenant_id, client_id);
create index if not exists documents_metadata_gin               on public.documents using gin(metadata);
create index if not exists documents_ranking_scope_idx          on public.documents(tenant_id, client_id, status);
create index if not exists documents_source_timestamp_idx       on public.documents(tenant_id, client_id, source_timestamp desc nulls last);
create index if not exists documents_pinned_idx                 on public.documents(tenant_id, client_id, is_pinned, is_canonical);
create index if not exists documents_actor_idx                  on public.documents(tenant_id, actor_id, created_at desc);
create index if not exists documents_ingest_job_idx             on public.documents(ingest_job_id);
create index if not exists documents_previous_version_idx       on public.documents(previous_version_id);
create index if not exists documents_request_idx                on public.documents(request_id);

-- One canonical active summary per scope (ContextSummary / DocumentSummary / TopicSummary)
do $$ begin
  if not exists (select 1 from pg_indexes where schemaname='public' and indexname='documents_summary_scope_uq') then
    execute $idx$
      create unique index documents_summary_scope_uq on public.documents (
        tenant_id, client_id, source_type, (metadata->>'document_id'), (metadata->>'topic')
      ) where is_canonical = true and status = 'active'
        and source_type::text in ('ContextSummary','DocumentSummary','TopicSummary');
    $idx$;
  end if;
end $$;


-- ── chunks ──────────────────────────────────────────────────────────────────
create table if not exists public.chunks (
  id              uuid primary key default gen_random_uuid(),
  tenant_id       uuid not null,
  document_id     uuid not null references public.documents(id) on delete cascade,
  chunk_index     int not null,
  page_start      int,
  page_end        int,
  content         text not null,
  content_tokens  int,
  metadata        jsonb not null default '{}'::jsonb,
  pii_annotations jsonb not null default '[]'::jsonb,
  created_at      timestamptz not null default now(),
  embedding       vector(1536),
  embedding_model text not null default 'text-embedding-3-small'
);

create unique index if not exists chunks_doc_chunk_idx_uq    on public.chunks(tenant_id, document_id, chunk_index);
create index if not exists chunks_tenant_doc_idx             on public.chunks(tenant_id, document_id);
create index if not exists chunks_metadata_gin               on public.chunks using gin(metadata);
create index if not exists chunks_embedding_hnsw             on public.chunks using hnsw (embedding vector_cosine_ops);
create index if not exists chunks_embedding_model_idx        on public.chunks(tenant_id, embedding_model);
create index if not exists chunks_pii_annotations_gin        on public.chunks using gin(pii_annotations);


-- ── kg_nodes ────────────────────────────────────────────────────────────────
create table if not exists public.kg_nodes (
  id              uuid primary key default gen_random_uuid(),
  tenant_id       uuid not null,
  client_id       uuid,
  node_key        text not null,
  type            artifact_type not null,
  name            text not null,
  description     text,
  properties      jsonb not null default '{}'::jsonb,
  embedding       vector(1536),
  embedding_model text not null default 'text-embedding-3-small',
  status          node_status not null default 'active',
  last_seen_at    timestamptz not null default now(),
  seen_count      int not null default 0,
  created_at      timestamptz not null default now(),
  updated_at      timestamptz not null default now()
);

create unique index if not exists kg_nodes_node_key_uq        on public.kg_nodes(tenant_id, client_id, node_key);
create index if not exists kg_nodes_tenant_idx                on public.kg_nodes(tenant_id);
create index if not exists kg_nodes_client_idx                on public.kg_nodes(tenant_id, client_id);
create index if not exists kg_nodes_type_idx                  on public.kg_nodes(tenant_id, type);
create index if not exists kg_nodes_status_idx                on public.kg_nodes(tenant_id, status);
create index if not exists kg_nodes_last_seen_idx             on public.kg_nodes(tenant_id, client_id, last_seen_at);
create index if not exists kg_nodes_props_gin                 on public.kg_nodes using gin(properties);
create index if not exists kg_nodes_embedding_hnsw            on public.kg_nodes using hnsw (embedding vector_cosine_ops);
create index if not exists kg_nodes_embedding_model_idx       on public.kg_nodes(tenant_id, embedding_model);


-- ── kg_edges ────────────────────────────────────────────────────────────────
create table if not exists public.kg_edges (
  id           uuid primary key default gen_random_uuid(),
  tenant_id    uuid not null,
  client_id    uuid,
  src_id       uuid not null references public.kg_nodes(id) on delete cascade,
  dst_id       uuid not null references public.kg_nodes(id) on delete cascade,
  rel_type     text not null,
  weight       float4,
  properties   jsonb not null default '{}'::jsonb,
  is_active    boolean not null default true,
  last_seen_at timestamptz not null default now(),
  seen_count   int not null default 0,
  created_at   timestamptz not null default now(),
  updated_at   timestamptz not null default now(),
  unique (tenant_id, client_id, src_id, dst_id, rel_type)
);

create index if not exists kg_edges_src_idx       on public.kg_edges(tenant_id, client_id, src_id);
create index if not exists kg_edges_dst_idx       on public.kg_edges(tenant_id, client_id, dst_id);
create index if not exists kg_edges_rel_idx       on public.kg_edges(tenant_id, client_id, rel_type);
create index if not exists kg_edges_active_idx    on public.kg_edges(tenant_id, client_id, is_active);
create index if not exists kg_edges_last_seen_idx on public.kg_edges(tenant_id, client_id, last_seen_at);
create index if not exists kg_edges_props_gin     on public.kg_edges using gin(properties);


-- ── kg_node_evidence ────────────────────────────────────────────────────────
create table if not exists public.kg_node_evidence (
  id         uuid primary key default gen_random_uuid(),
  tenant_id  uuid not null,
  client_id  uuid,
  node_id    uuid not null references public.kg_nodes(id) on delete cascade,
  chunk_id   uuid not null references public.chunks(id)   on delete cascade,
  quote      text,
  score      float4,
  created_at timestamptz not null default now(),
  unique (tenant_id, client_id, node_id, chunk_id)
);

create index if not exists kg_node_evidence_node_idx  on public.kg_node_evidence(tenant_id, client_id, node_id);
create index if not exists kg_node_evidence_chunk_idx on public.kg_node_evidence(tenant_id, client_id, chunk_id);


-- ── kg_edge_evidence ────────────────────────────────────────────────────────
create table if not exists public.kg_edge_evidence (
  id         uuid primary key default gen_random_uuid(),
  tenant_id  uuid not null,
  client_id  uuid,
  edge_id    uuid not null references public.kg_edges(id) on delete cascade,
  chunk_id   uuid not null references public.chunks(id)   on delete cascade,
  quote      text,
  score      float4,
  created_at timestamptz not null default now(),
  unique (tenant_id, client_id, edge_id, chunk_id)
);

create index if not exists kg_edge_evidence_edge_idx  on public.kg_edge_evidence(tenant_id, client_id, edge_id);
create index if not exists kg_edge_evidence_chunk_idx on public.kg_edge_evidence(tenant_id, client_id, chunk_id);


-- ── survey_outputs ──────────────────────────────────────────────────────────
create table if not exists public.survey_outputs (
  id          uuid primary key default gen_random_uuid(),
  tenant_id   uuid not null,
  client_id   uuid not null,
  output_type text not null,
  request     text not null default '',
  questions   jsonb not null default '[]'::jsonb,
  reasoning   text,
  metadata    jsonb not null default '{}'::jsonb,
  created_at  timestamptz not null default now(),
  expires_at  timestamptz not null default (now() + interval '7 days')
);

create index if not exists survey_outputs_tenant_client_idx on public.survey_outputs(tenant_id, client_id);
create index if not exists survey_outputs_expires_idx       on public.survey_outputs(expires_at);


-- ── memory_state ────────────────────────────────────────────────────────────
create table if not exists public.memory_state (
  id               uuid primary key default gen_random_uuid(),
  tenant_id        uuid not null,
  client_id        uuid,
  memory_version   bigint not null default 0,
  last_ingested_at timestamptz,
  last_changed_at  timestamptz not null default now(),
  last_summary_at  timestamptz,
  metadata         jsonb not null default '{}'::jsonb,
  created_at       timestamptz not null default now(),
  updated_at       timestamptz not null default now()
);

create unique index if not exists memory_state_tenant_client_uq on public.memory_state(tenant_id, client_id);
create index if not exists memory_state_tenant_idx              on public.memory_state(tenant_id);


-- ── memory_change_log ───────────────────────────────────────────────────────
create table if not exists public.memory_change_log (
  id             uuid primary key default gen_random_uuid(),
  tenant_id      uuid not null,
  client_id      uuid,
  change_type    text not null,
  memory_version bigint not null,
  metadata       jsonb not null default '{}'::jsonb,
  created_at     timestamptz not null default now()
);

create index if not exists memory_change_log_tenant_client_idx on public.memory_change_log(tenant_id, client_id, created_at desc);
create index if not exists memory_change_log_tenant_idx        on public.memory_change_log(tenant_id, created_at desc);


-- ── api_keys ────────────────────────────────────────────────────────────────
create table if not exists public.api_keys (
  id            uuid primary key default gen_random_uuid(),
  tenant_id     uuid not null,
  key_hash      text not null unique,
  key_prefix    text not null,
  name          text not null,
  scopes        text[] not null default '{}'::text[],
  status        text not null default 'active',
  last_used_at  timestamptz,
  expires_at    timestamptz,
  created_at    timestamptz not null default now(),
  revoked_at    timestamptz,
  constraint api_keys_status_check check (status in ('active','revoked'))
);

create index if not exists api_keys_tenant_status_idx on public.api_keys(tenant_id, status);


-- ── audit_log ───────────────────────────────────────────────────────────────
create table if not exists public.audit_log (
  id            uuid primary key default gen_random_uuid(),
  tenant_id     uuid not null,
  key_id        uuid,
  actor_id      text,                                       -- migration 29
  request_id    text not null,
  action        text not null,
  resource_type text,
  resource_id   text,
  status        text not null,
  metadata      jsonb not null default '{}'::jsonb,
  "before"      jsonb,                                      -- migration 29
  "after"       jsonb,                                      -- migration 29
  reason        text,                                       -- migration 29 (REVEAL/EXPORT)
  source_ip     text,
  created_at    timestamptz not null default now(),
  constraint audit_log_status_check check (status in ('success','failure'))
);

create index if not exists audit_log_tenant_created_idx on public.audit_log(tenant_id, created_at desc);
create index if not exists audit_log_tenant_action_idx  on public.audit_log(tenant_id, action, created_at desc);
create index if not exists audit_log_resource_idx       on public.audit_log(resource_type, resource_id);
create index if not exists audit_log_request_idx        on public.audit_log(request_id);
create index if not exists audit_log_actor_idx          on public.audit_log(actor_id, created_at desc);


-- ── ingest_jobs ─────────────────────────────────────────────────────────────
create table if not exists public.ingest_jobs (
  id           uuid primary key default gen_random_uuid(),
  tenant_id    uuid not null,
  client_id    uuid,
  job_type     text not null,
  status       text not null default 'queued',
  request_id   text,
  key_id       uuid,
  payload_hash text,
  document_id  uuid,
  result       jsonb,
  error        text,
  -- Provenance (migration 29)
  actor_id     text,
  actor_type   text check (actor_type is null or actor_type in ('user','service','agent','anonymous')),
  source_app   text,
  source_type  text,
  source_uri   text,
  enqueued_at  timestamptz not null default now(),
  started_at   timestamptz,
  completed_at timestamptz,
  constraint ingest_jobs_status_check check (status in ('queued','running','processing','complete','succeeded','failed')),
  constraint ingest_jobs_type_check   check (job_type in ('processed','processed-web'))
);

create index if not exists ingest_jobs_tenant_client_idx on public.ingest_jobs(tenant_id, client_id, enqueued_at desc);
create index if not exists ingest_jobs_status_idx        on public.ingest_jobs(status, enqueued_at);
create index if not exists ingest_jobs_payload_hash_idx  on public.ingest_jobs(tenant_id, payload_hash);
create index if not exists ingest_jobs_actor_idx         on public.ingest_jobs(tenant_id, actor_id, enqueued_at desc);
create index if not exists ingest_jobs_request_idx       on public.ingest_jobs(request_id);


-- ============================================================================
-- 5. TRIGGERS (updated_at auto-bump)
-- ============================================================================
do $$ begin
  if not exists (select 1 from pg_trigger where tgname='trg_documents_set_updated_at') then
    create trigger trg_documents_set_updated_at before update on public.documents
    for each row execute function public.set_updated_at();
  end if;
  if not exists (select 1 from pg_trigger where tgname='trg_kg_nodes_set_updated_at') then
    create trigger trg_kg_nodes_set_updated_at before update on public.kg_nodes
    for each row execute function public.set_updated_at();
  end if;
  if not exists (select 1 from pg_trigger where tgname='trg_kg_edges_set_updated_at') then
    create trigger trg_kg_edges_set_updated_at before update on public.kg_edges
    for each row execute function public.set_updated_at();
  end if;
  if not exists (select 1 from pg_trigger where tgname='trg_memory_state_set_updated_at') then
    create trigger trg_memory_state_set_updated_at before update on public.memory_state
    for each row execute function public.set_updated_at();
  end if;
end $$;


-- ============================================================================
-- 6. RPCs — UPSERTS
-- ============================================================================

create or replace function public.upsert_kg_node(
  p_tenant_id uuid, p_client_id uuid, p_node_key text,
  p_type artifact_type, p_name text, p_description text default null,
  p_properties jsonb default '{}'::jsonb, p_embedding vector(1536) default null,
  p_status node_status default 'active', p_embedding_model text default 'text-embedding-3-small'
) returns uuid language plpgsql as $$
declare v_id uuid;
begin
  insert into public.kg_nodes (
    tenant_id, client_id, node_key, type, name, description, properties, embedding,
    embedding_model, status, last_seen_at, seen_count, created_at, updated_at
  ) values (
    p_tenant_id, p_client_id, p_node_key, p_type, p_name, p_description,
    coalesce(p_properties, '{}'::jsonb), p_embedding, p_embedding_model,
    p_status, now(), 1, now(), now()
  ) on conflict (tenant_id, client_id, node_key) do update set
    type=excluded.type, name=excluded.name, description=excluded.description,
    properties=coalesce(public.kg_nodes.properties,'{}'::jsonb)||coalesce(excluded.properties,'{}'::jsonb),
    embedding=coalesce(excluded.embedding, public.kg_nodes.embedding),
    embedding_model=coalesce(excluded.embedding_model, public.kg_nodes.embedding_model),
    status=excluded.status, last_seen_at=now(),
    seen_count=public.kg_nodes.seen_count+1, updated_at=now()
  returning id into v_id;
  return v_id;
end; $$;


create or replace function public.upsert_kg_edge(
  p_tenant_id uuid, p_client_id uuid, p_src_id uuid, p_dst_id uuid,
  p_rel_type text, p_weight float4 default null, p_properties jsonb default '{}'::jsonb
) returns uuid language plpgsql as $$
declare v_id uuid;
begin
  insert into public.kg_edges (
    tenant_id, client_id, src_id, dst_id, rel_type, weight, properties,
    is_active, last_seen_at, seen_count, created_at, updated_at
  ) values (
    p_tenant_id, p_client_id, p_src_id, p_dst_id, p_rel_type, p_weight,
    coalesce(p_properties, '{}'::jsonb), true, now(), 1, now(), now()
  ) on conflict (tenant_id, client_id, src_id, dst_id, rel_type) do update set
    weight=coalesce(excluded.weight, public.kg_edges.weight),
    properties=coalesce(public.kg_edges.properties,'{}'::jsonb)||coalesce(excluded.properties,'{}'::jsonb),
    is_active=true, last_seen_at=now(),
    seen_count=public.kg_edges.seen_count+1, updated_at=now()
  returning id into v_id;
  return v_id;
end; $$;


create or replace function public.upsert_chunk(
  p_tenant_id uuid, p_document_id uuid, p_chunk_index int,
  p_page_start int default null, p_page_end int default null,
  p_content text default null, p_content_tokens int default null,
  p_metadata jsonb default '{}'::jsonb, p_embedding vector(1536) default null,
  p_embedding_model text default 'text-embedding-3-small'
) returns uuid language plpgsql as $$
declare v_id uuid;
begin
  insert into public.chunks (
    tenant_id, document_id, chunk_index, page_start, page_end,
    content, content_tokens, metadata, embedding, embedding_model, created_at
  ) values (
    p_tenant_id, p_document_id, p_chunk_index, p_page_start, p_page_end,
    p_content, p_content_tokens, coalesce(p_metadata,'{}'::jsonb),
    p_embedding, p_embedding_model, now()
  ) on conflict (tenant_id, document_id, chunk_index) do update set
    page_start=coalesce(excluded.page_start, public.chunks.page_start),
    page_end=coalesce(excluded.page_end, public.chunks.page_end),
    content=coalesce(excluded.content, public.chunks.content),
    content_tokens=coalesce(excluded.content_tokens, public.chunks.content_tokens),
    metadata=coalesce(public.chunks.metadata,'{}'::jsonb)||coalesce(excluded.metadata,'{}'::jsonb),
    embedding=coalesce(excluded.embedding, public.chunks.embedding),
    embedding_model=coalesce(excluded.embedding_model, public.chunks.embedding_model)
  returning id into v_id;
  return v_id;
end; $$;


-- ============================================================================
-- 7. RPCs — SEARCH (hybrid scoring + source_type filter)
-- ============================================================================

create or replace function public.search_kg_nodes(
  p_tenant_id       uuid,
  p_client_id       uuid         default null,
  p_embedding       vector(1536) default null,
  p_top_k           int          default 5,
  p_types           text[]       default null,
  p_document_ids    text[]       default null,
  p_recency_weight  float4       default 0.0,
  p_boost_pinned    boolean      default false,
  p_exclude_status  text[]       default array['archived','deprecated'],
  p_embedding_model text         default 'text-embedding-3-small',
  p_source_types    text[]       default null
)
returns table (
  id uuid, node_key text, name text, description text, properties jsonb,
  type artifact_type, source_type text, embedding_model text,
  similarity float4, final_score float4
)
language sql stable as $$
  with scored as (
    select
      n.id, n.node_key, n.name, n.description, n.properties, n.type,
      d.source_type as doc_source_type, n.embedding_model,
      (1 - (n.embedding <=> p_embedding))::float4 as similarity,
      case when d.id is not null and p_recency_weight > 0 then
        (1.0/(1.0+extract(epoch from (now()-coalesce(d.source_timestamp,d.created_at)))/86400.0))::float4
      else 0.0 end as recency_score,
      case when p_boost_pinned and d.id is not null then
        (case when d.is_pinned then 0.05 else 0 end)+(case when d.is_canonical then 0.03 else 0 end)
      else 0.0 end as boost_score
    from public.kg_nodes n
    left join public.documents d
      on d.id=(n.properties->>'document_id')::uuid and d.tenant_id=p_tenant_id
    where n.tenant_id=p_tenant_id
      and (p_client_id is null or n.client_id=p_client_id)
      and n.status='active' and n.embedding is not null
      and n.embedding_model=p_embedding_model
      and (p_types is null or n.type::text=any(p_types))
      and (p_document_ids is null or n.properties->>'document_id'=any(p_document_ids))
      and (d.id is null or not(d.status=any(p_exclude_status)))
      and (p_source_types is null or (d.id is not null and d.source_type=any(p_source_types)))
  )
  select s.id, s.node_key, s.name, s.description, s.properties, s.type,
    s.doc_source_type as source_type, s.embedding_model, s.similarity,
    ((1.0-p_recency_weight)*s.similarity + p_recency_weight*s.recency_score + s.boost_score)::float4 as final_score
  from scored s order by final_score desc limit p_top_k;
$$;


create or replace function public.fetch_chunks_with_embeddings(
  p_tenant_id uuid, p_client_id uuid default null,
  p_document_id uuid default null, p_limit int default 500, p_offset int default 0
)
returns table (id uuid, document_id uuid, chunk_index int, content text, embedding vector(1536), metadata jsonb)
language sql stable as $$
  select c.id, c.document_id, c.chunk_index, c.content, c.embedding, c.metadata
  from public.chunks c join public.documents d on d.id=c.document_id
  where d.tenant_id=p_tenant_id
    and (p_client_id is null or d.client_id=p_client_id)
    and (p_document_id is null or c.document_id=p_document_id)
    and c.embedding is not null
  order by c.document_id, c.chunk_index limit p_limit offset p_offset;
$$;


-- ============================================================================
-- 8. RPCs — KG PRUNING + CLEANUP
-- ============================================================================

create or replace function public.prune_archive_stale_edges(
  p_tenant_id uuid, p_client_id uuid, p_stale_days int default 90
) returns int language sql as $$
  with upd as (
    update public.kg_edges set is_active=false, updated_at=now()
    where tenant_id=p_tenant_id and client_id=p_client_id and is_active=true
      and last_seen_at < now()-make_interval(days=>p_stale_days)
    returning 1
  ) select count(*)::int from upd;
$$;

create or replace function public.prune_archive_stale_nodes(
  p_tenant_id uuid, p_client_id uuid, p_stale_days int default 180, p_min_degree int default 3
) returns int language sql as $$
  with degrees as (
    select n.id as node_id, count(e.id) as degree
    from public.kg_nodes n
    left join public.kg_edges e on e.tenant_id=n.tenant_id and e.client_id=n.client_id
      and e.is_active=true and (e.src_id=n.id or e.dst_id=n.id)
    where n.tenant_id=p_tenant_id and n.client_id=p_client_id
    group by n.id
  ), upd as (
    update public.kg_nodes n set status='archived', updated_at=now()
    from degrees d where n.id=d.node_id
      and n.tenant_id=p_tenant_id and n.client_id=p_client_id
      and n.status<>'archived' and n.last_seen_at < now()-make_interval(days=>p_stale_days)
      and d.degree < p_min_degree
    returning 1
  ) select count(*)::int from upd;
$$;

create or replace function public.prune_trim_edge_evidence(
  p_tenant_id uuid, p_client_id uuid, p_keep_per_edge int default 5
) returns int language sql as $$
  with ranked as (
    select ee.id, row_number() over (partition by ee.edge_id order by ee.score desc nulls last, ee.created_at desc) as rn
    from public.kg_edge_evidence ee where ee.tenant_id=p_tenant_id and ee.client_id=p_client_id
  ), del as (
    delete from public.kg_edge_evidence ee using ranked r where ee.id=r.id and r.rn>p_keep_per_edge returning 1
  ) select count(*)::int from del;
$$;

create or replace function public.prune_trim_node_evidence(
  p_tenant_id uuid, p_client_id uuid, p_keep_per_node int default 10
) returns int language sql as $$
  with ranked as (
    select ne.id, row_number() over (partition by ne.node_id order by ne.score desc nulls last, ne.created_at desc) as rn
    from public.kg_node_evidence ne where ne.tenant_id=p_tenant_id and ne.client_id=p_client_id
  ), del as (
    delete from public.kg_node_evidence ne using ranked r where ne.id=r.id and r.rn>p_keep_per_node returning 1
  ) select count(*)::int from del;
$$;

create or replace function public.prune_kg(
  p_tenant_id uuid, p_client_id uuid,
  p_edge_stale_days int default 90, p_node_stale_days int default 180,
  p_min_degree int default 3, p_keep_edge_evidence int default 5, p_keep_node_evidence int default 10
) returns jsonb language plpgsql as $$
declare ea int; na int; eed int; ned int;
begin
  select prune_archive_stale_edges(p_tenant_id, p_client_id, p_edge_stale_days) into ea;
  select prune_archive_stale_nodes(p_tenant_id, p_client_id, p_node_stale_days, p_min_degree) into na;
  select prune_trim_edge_evidence(p_tenant_id, p_client_id, p_keep_edge_evidence) into eed;
  select prune_trim_node_evidence(p_tenant_id, p_client_id, p_keep_node_evidence) into ned;
  return jsonb_build_object('edges_archived',ea,'nodes_archived',na,'edge_evidence_deleted',eed,'node_evidence_deleted',ned);
end; $$;

create or replace function public.cleanup_orphaned_kg_nodes(
  p_tenant_id uuid, p_client_id uuid default null
) returns jsonb language plpgsql as $$
declare v int;
begin
  with orphaned as (
    select n.id from public.kg_nodes n left join public.kg_node_evidence ne on ne.node_id=n.id
    where n.tenant_id=p_tenant_id and (p_client_id is null or n.client_id=p_client_id)
    group by n.id having count(ne.id)=0
  ), del as (
    delete from public.kg_nodes n using orphaned o where n.id=o.id returning 1
  ) select count(*)::int into v from del;
  return jsonb_build_object('nodes_deleted', v);
end; $$;


-- ============================================================================
-- 9. RPCs — MEMORY STATE
-- ============================================================================

create or replace function public.bump_memory_state(
  p_tenant_id uuid, p_client_id uuid default null,
  p_change_type text default 'mutation', p_metadata jsonb default '{}'::jsonb
) returns bigint language plpgsql as $$
declare v bigint;
begin
  insert into public.memory_state (
    tenant_id, client_id, memory_version, last_ingested_at, last_changed_at, metadata, created_at, updated_at
  ) values (
    p_tenant_id, p_client_id, 1,
    case when p_change_type='ingest' then now() else null end,
    now(), coalesce(p_metadata,'{}'::jsonb), now(), now()
  ) on conflict (tenant_id, client_id) do update set
    memory_version=public.memory_state.memory_version+1,
    last_ingested_at=case when p_change_type='ingest' then now() else public.memory_state.last_ingested_at end,
    last_changed_at=now(), metadata=coalesce(p_metadata,'{}'::jsonb), updated_at=now()
  returning memory_version into v;
  insert into public.memory_change_log (tenant_id, client_id, change_type, memory_version, metadata)
  values (p_tenant_id, p_client_id, p_change_type, v, coalesce(p_metadata,'{}'::jsonb));
  return v;
end; $$;

create or replace function public.bump_memory_state_dual(
  p_tenant_id uuid, p_client_id uuid,
  p_change_type text default 'mutation', p_metadata jsonb default '{}'::jsonb
) returns jsonb language plpgsql as $$
declare cv bigint; tv bigint;
begin
  insert into public.memory_state (tenant_id, client_id, memory_version, last_ingested_at, last_changed_at, metadata, created_at, updated_at)
  values (p_tenant_id, p_client_id, 1, case when p_change_type='ingest' then now() else null end, now(), coalesce(p_metadata,'{}'::jsonb), now(), now())
  on conflict (tenant_id, client_id) do update set
    memory_version=public.memory_state.memory_version+1,
    last_ingested_at=case when p_change_type='ingest' then now() else public.memory_state.last_ingested_at end,
    last_changed_at=now(), metadata=coalesce(p_metadata,'{}'::jsonb), updated_at=now()
  returning memory_version into cv;

  insert into public.memory_state (tenant_id, client_id, memory_version, last_ingested_at, last_changed_at, metadata, created_at, updated_at)
  values (p_tenant_id, null, 1, case when p_change_type='ingest' then now() else null end, now(), coalesce(p_metadata,'{}'::jsonb), now(), now())
  on conflict (tenant_id, client_id) do update set
    memory_version=public.memory_state.memory_version+1,
    last_ingested_at=case when p_change_type='ingest' then now() else public.memory_state.last_ingested_at end,
    last_changed_at=now(), metadata=coalesce(p_metadata,'{}'::jsonb), updated_at=now()
  returning memory_version into tv;

  insert into public.memory_change_log (tenant_id, client_id, change_type, memory_version, metadata)
  values (p_tenant_id, p_client_id, p_change_type, cv, coalesce(p_metadata,'{}'::jsonb));
  return jsonb_build_object('client_version', cv, 'tenant_version', tv);
end; $$;


-- ============================================================================
-- 10. RPCs — HOUSEKEEPING
-- ============================================================================

create or replace function public.cleanup_expired_survey_outputs()
returns integer language plpgsql as $$
declare n integer;
begin delete from public.survey_outputs where expires_at < now(); get diagnostics n = row_count; return n; end; $$;

create or replace function public.touch_api_key(p_key_id uuid)
returns void language sql as $$
  update public.api_keys set last_used_at=now() where id=p_key_id;
$$;


-- ============================================================================
-- 11. SEED — bootstrap admin API key (edit before running)
-- ============================================================================
-- Generate key material in Python:
--   import secrets, hashlib
--   raw = "dp_" + secrets.token_urlsafe(32)
--   print("RAW:", raw)
--   print("HASH:", hashlib.sha256(raw.encode()).hexdigest())
--   print("PREFIX:", raw[:8])
--
-- Then uncomment and fill in:
--
-- insert into public.api_keys (tenant_id, key_hash, key_prefix, name, scopes, status)
-- values (
--   'YOUR-TENANT-UUID'::uuid,
--   'PASTE-SHA256-HEX-HERE',
--   'dp_XXXXX',
--   'bootstrap admin',
--   array['read','write','admin'],
--   'active'
-- ) on conflict (key_hash) do nothing;
