-- 14_cleanup_orphaned_kg.sql
-- Delete KG nodes that have zero evidence remaining (orphaned after document deletion).
-- Edges cascade-delete via FK on kg_edges(src_id/dst_id → kg_nodes.id ON DELETE CASCADE).

create or replace function public.cleanup_orphaned_kg_nodes(
  p_tenant_id uuid,
  p_client_id uuid default null
)
returns jsonb
language plpgsql
as $$
declare
  v_nodes_deleted int;
begin
  with orphaned as (
    select n.id
    from public.kg_nodes n
    left join public.kg_node_evidence ne on ne.node_id = n.id
    where n.tenant_id = p_tenant_id
      and (p_client_id is null or n.client_id = p_client_id)
    group by n.id
    having count(ne.id) = 0
  ),
  del as (
    delete from public.kg_nodes n
    using orphaned o
    where n.id = o.id
    returning 1
  )
  select count(*)::int into v_nodes_deleted from del;

  return jsonb_build_object(
    'nodes_deleted', v_nodes_deleted
  );
end;
$$;
