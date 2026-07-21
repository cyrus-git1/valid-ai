-- 28_valid_kg.sql
-- Dedicated knowledge graph for the valid-docs corpus.
-- Powers the client-facing landing page bot (sales conversion + meeting booking).
-- Completely isolated from the main KG (client research, analysis, internal content).

-- ── valid_kg_nodes ──────────────────────────────────────────────────────────
create table if not exists public.valid_kg_nodes (
  id              uuid primary key default gen_random_uuid(),
  node_key        text not null unique,
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

create index if not exists valid_kg_nodes_type_idx           on public.valid_kg_nodes(type);
create index if not exists valid_kg_nodes_status_idx         on public.valid_kg_nodes(status);
create index if not exists valid_kg_nodes_props_gin          on public.valid_kg_nodes using gin(properties);
create index if not exists valid_kg_nodes_embedding_hnsw     on public.valid_kg_nodes using hnsw (embedding vector_cosine_ops);
create index if not exists valid_kg_nodes_embedding_model_idx on public.valid_kg_nodes(embedding_model);


-- ── valid_kg_edges ──────────────────────────────────────────────────────────
create table if not exists public.valid_kg_edges (
  id           uuid primary key default gen_random_uuid(),
  src_id       uuid not null references public.valid_kg_nodes(id) on delete cascade,
  dst_id       uuid not null references public.valid_kg_nodes(id) on delete cascade,
  rel_type     text not null,
  weight       float4,
  properties   jsonb not null default '{}'::jsonb,
  is_active    boolean not null default true,
  last_seen_at timestamptz not null default now(),
  seen_count   int not null default 0,
  created_at   timestamptz not null default now(),
  updated_at   timestamptz not null default now(),
  unique (src_id, dst_id, rel_type)
);

create index if not exists valid_kg_edges_src_idx       on public.valid_kg_edges(src_id);
create index if not exists valid_kg_edges_dst_idx       on public.valid_kg_edges(dst_id);
create index if not exists valid_kg_edges_rel_idx       on public.valid_kg_edges(rel_type);
create index if not exists valid_kg_edges_active_idx    on public.valid_kg_edges(is_active);


-- ── valid_kg_node_evidence ──────────────────────────────────────────────────
create table if not exists public.valid_kg_node_evidence (
  id         uuid primary key default gen_random_uuid(),
  node_id    uuid not null references public.valid_kg_nodes(id) on delete cascade,
  chunk_id   uuid not null references public.valid_chunks(id)   on delete cascade,
  quote      text,
  score      float4,
  created_at timestamptz not null default now(),
  unique (node_id, chunk_id)
);

create index if not exists valid_kg_node_evidence_node_idx  on public.valid_kg_node_evidence(node_id);
create index if not exists valid_kg_node_evidence_chunk_idx on public.valid_kg_node_evidence(chunk_id);


-- ── Triggers ────────────────────────────────────────────────────────────────
do $$ begin
  if not exists (select 1 from pg_trigger where tgname='trg_valid_kg_nodes_set_updated_at') then
    create trigger trg_valid_kg_nodes_set_updated_at before update on public.valid_kg_nodes
    for each row execute function public.set_updated_at();
  end if;
  if not exists (select 1 from pg_trigger where tgname='trg_valid_kg_edges_set_updated_at') then
    create trigger trg_valid_kg_edges_set_updated_at before update on public.valid_kg_edges
    for each row execute function public.set_updated_at();
  end if;
end $$;


-- ── Upsert RPCs ─────────────────────────────────────────────────────────────

create or replace function public.upsert_valid_kg_node(
  p_node_key text, p_type artifact_type, p_name text,
  p_description text default null, p_properties jsonb default '{}'::jsonb,
  p_embedding vector(1536) default null, p_status node_status default 'active',
  p_embedding_model text default 'text-embedding-3-small'
) returns uuid language plpgsql as $$
declare v_id uuid;
begin
  insert into public.valid_kg_nodes (
    node_key, type, name, description, properties, embedding,
    embedding_model, status, last_seen_at, seen_count, created_at, updated_at
  ) values (
    p_node_key, p_type, p_name, p_description,
    coalesce(p_properties, '{}'::jsonb), p_embedding, p_embedding_model,
    p_status, now(), 1, now(), now()
  ) on conflict (node_key) do update set
    type=excluded.type, name=excluded.name, description=excluded.description,
    properties=coalesce(public.valid_kg_nodes.properties,'{}'::jsonb)||coalesce(excluded.properties,'{}'::jsonb),
    embedding=coalesce(excluded.embedding, public.valid_kg_nodes.embedding),
    embedding_model=coalesce(excluded.embedding_model, public.valid_kg_nodes.embedding_model),
    status=excluded.status, last_seen_at=now(),
    seen_count=public.valid_kg_nodes.seen_count+1, updated_at=now()
  returning id into v_id;
  return v_id;
end; $$;


create or replace function public.upsert_valid_kg_edge(
  p_src_id uuid, p_dst_id uuid, p_rel_type text,
  p_weight float4 default null, p_properties jsonb default '{}'::jsonb
) returns uuid language plpgsql as $$
declare v_id uuid;
begin
  insert into public.valid_kg_edges (
    src_id, dst_id, rel_type, weight, properties,
    is_active, last_seen_at, seen_count, created_at, updated_at
  ) values (
    p_src_id, p_dst_id, p_rel_type, p_weight,
    coalesce(p_properties, '{}'::jsonb), true, now(), 1, now(), now()
  ) on conflict (src_id, dst_id, rel_type) do update set
    weight=coalesce(excluded.weight, public.valid_kg_edges.weight),
    properties=coalesce(public.valid_kg_edges.properties,'{}'::jsonb)||coalesce(excluded.properties,'{}'::jsonb),
    is_active=true, last_seen_at=now(),
    seen_count=public.valid_kg_edges.seen_count+1, updated_at=now()
  returning id into v_id;
  return v_id;
end; $$;


-- ── Search RPC (vector search over valid's KG) ─────────────────────────────
-- Simpler than the main search_kg_nodes — no tenant/client scoping, no hybrid
-- ranking (sales bot doesn't need recency decay or pinned boosts).

create or replace function public.search_valid_kg_nodes(
  p_embedding       vector(1536) default null,
  p_top_k           int          default 5,
  p_types           text[]       default null,
  p_embedding_model text         default 'text-embedding-3-small'
)
returns table (
  id uuid, node_key text, name text, description text, properties jsonb,
  type artifact_type, embedding_model text, similarity float4
)
language sql stable as $$
  select
    n.id, n.node_key, n.name, n.description, n.properties, n.type,
    n.embedding_model,
    (1 - (n.embedding <=> p_embedding))::float4 as similarity
  from public.valid_kg_nodes n
  where n.status = 'active'
    and n.embedding is not null
    and n.embedding_model = p_embedding_model
    and (p_types is null or n.type::text = any(p_types))
  order by n.embedding <=> p_embedding
  limit p_top_k;
$$;
