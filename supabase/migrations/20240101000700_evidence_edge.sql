-- 07_evidence_edge.sql
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
