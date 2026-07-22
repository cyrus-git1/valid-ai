-- 53_concept_relations.sql
-- Signed concept↔concept edges (durable Tensions) — the deferred spine item #3.
--
-- Cross-concept relations the agent can't derive from single-concept aggregates:
-- when two concepts CO-OCCUR (observations sharing a source.aggregate_id) and
-- their `direction`s consistently AGREE → 'supports'; consistently OPPOSE →
-- 'contradicts'. Persisted as normal kg_edges via upsert_kg_edge (mig 46), so
-- weight accumulates with each compute pass (weight = dominant co-occurrence
-- count). Concept identity is per-org; edges are tenant-level.
--
-- SEMANTICS (documented — tune as the agent calibrates):
--   * co-occurrence key = observation source.aggregate_id (same transcript/aggregate)
--   * only observations with direction in ('positive','negative') count
--   * pair (a,b) canonicalised as a < b (one edge per unordered pair per rel_type)
--   * aligned = same direction; opposing = differing direction
--   * write 'supports' if aligned >= opposing else 'contradicts'; weight = the
--     dominant count; require aligned+opposing >= p_min_cooccur
--
--   compute_concept_relations — batch pass (agent/cron triggers it), writes edges
--   concept_relations         — read: {src,dst,rel_type,weight}
--
-- Ends with NOTIFY pgrst 'reload schema'.

create or replace function public.compute_concept_relations(
  p_tenant_id   uuid,
  p_study_ids   uuid[] default null,
  p_min_cooccur int    default 2
)
returns jsonb
language plpgsql
as $$
declare
  r record;
  v_rel      text;
  v_strength int;
  v_written  int := 0;
begin
  for r in
    with obs as (
      select
        e.dst_id                                   as concept_id,
        o.properties->'source'->>'aggregate_id'    as agg,
        o.properties->>'direction'                 as direction
      from public.kg_nodes o
      join public.kg_edges e
        on e.src_id = o.id and e.rel_type = 'about_concept' and e.is_active
       and e.tenant_id = p_tenant_id
      where o.tenant_id = p_tenant_id
        and o.type::text = 'Observation'
        and o.status = 'active'
        and o.properties->'source'->>'aggregate_id' is not null
        and o.properties->>'direction' in ('positive', 'negative')
        and (p_study_ids is null or o.study_id = any(p_study_ids))
    ),
    pairs as (
      select
        o1.concept_id as a,
        o2.concept_id as b,
        count(*) filter (where o1.direction = o2.direction)  as aligned,
        count(*) filter (where o1.direction <> o2.direction) as opposing
      from obs o1
      join obs o2
        on o1.agg = o2.agg
       and o1.concept_id < o2.concept_id
      group by o1.concept_id, o2.concept_id
    )
    select a, b, aligned, opposing
    from pairs
    where aligned + opposing >= greatest(p_min_cooccur, 1)
  loop
    if r.aligned >= r.opposing then
      v_rel := 'supports';    v_strength := r.aligned;
    else
      v_rel := 'contradicts'; v_strength := r.opposing;
    end if;

    perform public.upsert_kg_edge(
      p_tenant_id, null, r.a, r.b, v_rel, v_strength::float4,
      jsonb_build_object('aligned', r.aligned, 'opposing', r.opposing,
                         'method', 'cooccurrence_direction')
    );
    v_written := v_written + 1;
  end loop;

  return jsonb_build_object('relations_written', v_written);
end;
$$;


create or replace function public.concept_relations(
  p_tenant_id uuid,
  p_study_ids uuid[] default null,
  p_rel_types text[] default array['supports', 'contradicts']
)
returns table (
  src_concept_id uuid,
  dst_concept_id uuid,
  rel_type       text,
  weight         float4
)
language sql
stable
as $$
  select e.src_id, e.dst_id, e.rel_type, e.weight
  from public.kg_edges e
  where e.tenant_id = p_tenant_id
    and e.is_active
    and e.rel_type = any(p_rel_types)
    -- concept-relation edges are tenant-level (study_id null); the filter is
    -- forward-compatible if edges are ever study-stamped.
    and (p_study_ids is null or e.study_id = any(p_study_ids) or e.study_id is null)
  order by e.weight desc nulls last;
$$;


notify pgrst, 'reload schema';
