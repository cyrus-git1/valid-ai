-- 52_spine_reads.sql
-- Tenant-scoped spine reads for the stateless agent layer. All are reads over the
-- existing graph (kg_nodes typed Observation/Concept joined by about_concept
-- edges) — NO new storage. tenant_id + study_id scoping is inherited from the
-- columns and filtered on every hop (defense in depth).
--
--   concepts_by_study     — distinct concepts with >=1 in-scope observation
--   observations_by_ids   — resolve observations by properties.observation_id
--   observations_rollup   — the scope cube (group-by aggregates) + top-N evidence
--
-- Ends with NOTIFY pgrst 'reload schema' so PostgREST sees the new functions
-- immediately (learned from the migration-51 PGRST202 incident).

-- ── 1. concepts_by_study ────────────────────────────────────────────────────
create or replace function public.concepts_by_study(
  p_tenant_id uuid,
  p_study_ids uuid[]  default null,
  p_client_id uuid    default null
)
returns table (
  concept_id uuid,
  label      text
)
language sql
stable
as $$
  select distinct
    e.dst_id                              as concept_id,
    c.properties->>'canonical_label'      as label
  from public.kg_nodes o
  join public.kg_edges e
    on e.src_id   = o.id
   and e.rel_type = 'about_concept'
   and e.is_active
   and e.tenant_id = p_tenant_id
  join public.kg_nodes c
    on c.id        = e.dst_id
   and c.type::text = 'Concept'
   and c.status    = 'active'
   and c.tenant_id = p_tenant_id
  where o.tenant_id = p_tenant_id
    and o.type::text = 'Observation'
    and o.status    = 'active'
    and (p_study_ids is null or o.study_id = any(p_study_ids))
    and (p_client_id is null or o.client_id = p_client_id);
$$;


-- ── 2. observations_by_ids ──────────────────────────────────────────────────
-- Same record shape as observations_by_concept; resolved by observation_id
-- (properties->>'observation_id', indexed in mig 43). Tenant-scoped.
create or replace function public.observations_by_ids(
  p_tenant_id        uuid,
  p_observation_ids  text[],
  p_study_ids        uuid[] default null
)
returns table (
  node_id            uuid,
  observation_id     text,
  nl_text            text,
  value              jsonb,
  modality           text,
  signal_type        text,
  direction          text,
  prevalence         jsonb,
  confidence         float4,
  reliability        jsonb,
  segment            jsonb,
  occurred_at        text,
  source             jsonb,
  study_id           uuid,
  evidence_chunk_ids uuid[]
)
language sql
stable
as $$
  select
    n.id                                  as node_id,
    n.properties->>'observation_id'       as observation_id,
    n.name                                as nl_text,
    n.properties->'value'                 as value,
    n.properties->>'modality'             as modality,
    n.properties->>'signal_type'          as signal_type,
    n.properties->>'direction'            as direction,
    n.properties->'prevalence'            as prevalence,
    (n.properties->>'confidence')::float4 as confidence,
    n.properties->'reliability'           as reliability,
    n.properties->'segment'               as segment,
    n.properties->>'occurred_at'          as occurred_at,
    n.properties->'source'                as source,
    n.study_id,
    (
      select array_agg(ev.chunk_id)
        from public.kg_node_evidence ev
       where ev.node_id = n.id
         and ev.tenant_id = p_tenant_id
    )                                     as evidence_chunk_ids
  from public.kg_nodes n
  where n.tenant_id = p_tenant_id
    and n.type::text = 'Observation'
    and n.status = 'active'
    and n.properties->>'observation_id' = any(p_observation_ids)
    and (p_study_ids is null or n.study_id = any(p_study_ids));
$$;


-- ── 3. observations_rollup ──────────────────────────────────────────────────
-- One call: the scope cube (concept × study × modality × persona × period ×
-- direction with obs_count + n_sum) plus top-N evidence observation_ids per
-- concept. Returns RAW direction counts + n_sum — the agent applies the
-- sign/threshold/divergence policy. occurred_at handled as text (ISO-8601 sorts
-- chronologically); period = 'YYYY-MM' prefix — no timestamp cast (robust to
-- missing/odd values). n_sum = sum of prevalence.n.
create or replace function public.observations_rollup(
  p_tenant_id     uuid,
  p_study_ids     uuid[] default null,
  p_range_from    text   default null,
  p_range_to      text   default null,
  p_top_evidence  int    default 5
)
returns jsonb
language sql
stable
as $$
  with obs as (
    select
      o.id,
      o.study_id,
      e.dst_id                                       as concept_id,
      c.properties->>'canonical_label'               as label,
      o.properties->>'observation_id'                as observation_id,
      o.properties->>'modality'                      as modality,
      o.properties->'segment'->>'persona'            as persona,
      o.properties->>'direction'                     as direction,
      coalesce((o.properties->'prevalence'->>'n')::int, 0) as n,
      o.properties->>'occurred_at'                   as occurred_at,
      left(o.properties->>'occurred_at', 7)          as period
    from public.kg_nodes o
    join public.kg_edges e
      on e.src_id = o.id and e.rel_type = 'about_concept'
     and e.is_active and e.tenant_id = p_tenant_id
    join public.kg_nodes c
      on c.id = e.dst_id and c.type::text = 'Concept'
     and c.status = 'active' and c.tenant_id = p_tenant_id
    where o.tenant_id = p_tenant_id
      and o.type::text = 'Observation'
      and o.status = 'active'
      and (p_study_ids is null or o.study_id = any(p_study_ids))
      and (p_range_from is null or o.properties->>'occurred_at' >= p_range_from)
      and (p_range_to   is null or o.properties->>'occurred_at' <  p_range_to)
  ),
  cube as (
    select
      concept_id,
      max(label)  as label,
      study_id,
      modality,
      persona,
      period,
      direction,
      count(*)::int as obs_count,
      sum(n)::int   as n_sum
    from obs
    group by concept_id, study_id, modality, persona, period, direction
  ),
  ev as (
    select concept_id,
           jsonb_agg(observation_id order by occurred_at desc nulls last)
             filter (where rn <= greatest(p_top_evidence, 1)) as ids
    from (
      select concept_id, observation_id, occurred_at,
             row_number() over (partition by concept_id
                                 order by occurred_at desc nulls last) as rn
      from obs
    ) t
    group by concept_id
  )
  select jsonb_build_object(
    'cube',     coalesce((select jsonb_agg(to_jsonb(cb)) from cube cb), '[]'::jsonb),
    'evidence', coalesce((select jsonb_object_agg(concept_id::text, ids) from ev), '{}'::jsonb)
  );
$$;


notify pgrst, 'reload schema';
