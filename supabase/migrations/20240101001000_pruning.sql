-- 10_pruning.sql
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
