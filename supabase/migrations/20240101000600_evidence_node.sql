-- 06_evidence_node.sql
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
