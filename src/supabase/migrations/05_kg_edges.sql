-- 05_kg_edges.sql
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
