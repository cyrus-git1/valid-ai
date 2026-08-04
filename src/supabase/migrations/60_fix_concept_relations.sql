-- 60_fix_concept_relations.sql
-- Corrects compute_concept_relations / concept_relations (mig 53) on the three
-- counts the spine audit flagged. Same signatures → `create or replace` suffices.
--
--   1. COUNTED OBSERVATION PAIRS, NOT DISTINCT SOURCES.
--      The old `pairs` CTE self-joined observations sharing an aggregate and did
--      count(*), so a single verbose transcript where concept A has 5 obs and B
--      has 4 produced 20 "co-occurrences" from ONE source — one loud interview
--      could manufacture a high-confidence relation. Fix: collapse each
--      (concept, source-aggregate) to ONE dominant-direction vote, then count
--      DISTINCT aggregates (sources).
--
--   2. "SAME DIRECTION IN SAME INTERVIEW => SUPPORTS" conflated co-mention with
--      logical agreement. Now each source contributes a single aligned/opposing
--      vote from the two concepts' DOMINANT direction in that source, not the
--      cross-product of their raw observations.
--
--   3. PER-STUDY READ WAS A NO-OP. Relation edges are written tenant-level
--      (study_id null) and concept_relations() had `... or e.study_id is null`,
--      always true — so a single-study view leaked every tenant-wide relation.
--      kg_edges can't be study-stamped without changing upsert_kg_edge's conflict
--      key (shared by ALL edge types), so instead we record the contributing
--      studies in properties.study_ids and filter on that. The leak clause is
--      removed.
--
-- MIGRATION NOTE: re-run compute_concept_relations after applying so existing
-- edges gain properties.study_ids. Old edges without it are simply ABSENT from
-- per-study reads until recomputed (they also carried the bogus counts, so
-- excluding them until a clean recompute is the desired behaviour).
--
-- The mock-based unit test (tests/unit/test_concept_relations.py) is unaffected:
-- signatures are unchanged and that test stubs the RPC rather than executing SQL.
-- SQL-level correctness needs the real-Postgres integration suite.

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
        o.study_id                                 as study_id,
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
    -- (1)+(2) ONE dominant-direction vote per (concept, source). A source that
    -- mentions a concept 10 times still contributes a single vote.
    concept_src as (
      select
        concept_id,
        agg,
        min(study_id)                            as study_id,   -- an aggregate belongs to one study
        mode() within group (order by direction) as direction   -- dominant direction in this source
      from obs
      group by concept_id, agg
    ),
    -- one aligned/opposing verdict per (pair, source)
    pair_src as (
      select
        a.concept_id as a,
        b.concept_id as b,
        a.study_id   as study_id,
        (a.direction = b.direction) as aligned
      from concept_src a
      join concept_src b
        on a.agg = b.agg and a.concept_id < b.concept_id
    ),
    -- (1) count DISTINCT sources (one pair_src row per source); (3) collect the
    -- studies that actually contributed to the relation.
    pairs as (
      select
        a, b,
        count(*) filter (where aligned)                                as aligned,
        count(*) filter (where not aligned)                            as opposing,
        array_agg(distinct study_id) filter (where study_id is not null) as study_ids
      from pair_src
      group by a, b
    )
    select a, b, aligned, opposing, study_ids
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
      jsonb_build_object(
        'aligned', r.aligned, 'opposing', r.opposing,
        'method', 'cooccurrence_direction_distinct_source',
        'study_ids', to_jsonb(coalesce(r.study_ids, array[]::uuid[]))
      )
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
    -- (3) HONEST per-study filter: match on the studies that actually
    -- contributed to the relation (recorded in properties.study_ids by
    -- compute_concept_relations). No more `or study_id is null` leak — a
    -- single-study view no longer returns tenant-wide relations.
    and (
      p_study_ids is null
      or (e.properties -> 'study_ids') ?| (
           select array_agg(x::text) from unnest(p_study_ids) as x
         )
    )
  order by e.weight desc nulls last;
$$;


notify pgrst, 'reload schema';
