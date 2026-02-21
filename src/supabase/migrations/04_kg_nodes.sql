-- 04_kg_nodes.sql
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
