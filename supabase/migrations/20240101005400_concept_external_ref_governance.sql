-- 54_concept_external_ref_governance.sql
-- Theme-taxonomy governance: the graph is the vault for the per-org codebook.
--
-- Governed concepts are transcript_tags mirrored into the graph as Concept nodes
-- carrying properties.external_ref = tag_id (and node id = tag id). Candidates are
-- ordinary Concept nodes with no external_ref. The agent tells them apart purely
-- by external_ref, so:
--
--   R2 — the three concept reads now RETURN external_ref (the load-bearing change
--        that flips governance from "everything candidate" to live governed/candidate):
--          nearest_concepts, concepts_by_study, observations_rollup(cube rows)
--   R1 — mirror_tag_concept(tag_id, label, description, embedding): upsert a
--        governed Concept node keyed by the tag id (node id = tag id,
--        external_ref = tag id). Idempotent — refreshes label/description/embedding
--        on codebook edits; the app emits tags on write (app-entity pattern).
--   R3 — link_concept_tag(concept_id, tag_id): stamp external_ref on an EXISTING
--        candidate node in place (graduation), preserving its accumulated
--        about_concept observations.
--
-- Ends with NOTIFY pgrst 'reload schema'.

-- ── R2a: nearest_concepts + external_ref ────────────────────────────────────
create or replace function public.nearest_concepts(
  p_tenant_id        uuid,
  p_embedding        vector(1536),
  p_hints            text[]  default null,
  p_top_k            int     default 10,
  p_embedding_model  text    default 'text-embedding-3-small'
)
returns table (
  id               uuid,
  canonical_id     text,
  canonical_label  text,
  alias_set        jsonb,
  merge_confidence float4,
  external_ref     text,
  similarity       float4,
  final_score      float4
)
language sql
stable
as $$
  select
    n.id,
    n.properties->>'canonical_id'                 as canonical_id,
    n.properties->>'canonical_label'              as canonical_label,
    n.properties->'alias_set'                     as alias_set,
    (n.properties->>'merge_confidence')::float4   as merge_confidence,
    n.properties->>'external_ref'                 as external_ref,
    (1 - (n.embedding <=> p_embedding))::float4   as similarity,
    (
      (1 - (n.embedding <=> p_embedding))
      + case
          when p_hints is not null and (
            exists (select 1 from unnest(p_hints) h
                     where n.properties->>'canonical_label' ilike '%' || h || '%')
            or exists (select 1
                         from jsonb_array_elements_text(
                                coalesce(n.properties->'alias_set'->'members', '[]'::jsonb)) m,
                              unnest(p_hints) h
                        where m ilike '%' || h || '%')
          ) then 0.05 else 0.0 end
    )::float4 as final_score
  from public.kg_nodes n
  where n.tenant_id        = p_tenant_id
    and n.type::text       = 'Concept'
    and n.status           = 'active'
    and n.embedding is not null
    and n.embedding_model  = p_embedding_model
  order by final_score desc
  limit p_top_k;
$$;


-- ── R2b: concepts_by_study + external_ref ───────────────────────────────────
create or replace function public.concepts_by_study(
  p_tenant_id uuid,
  p_study_ids uuid[]  default null,
  p_client_id uuid    default null
)
returns table (
  concept_id   uuid,
  label        text,
  external_ref text
)
language sql
stable
as $$
  select distinct
    e.dst_id                              as concept_id,
    c.properties->>'canonical_label'      as label,
    c.properties->>'external_ref'         as external_ref
  from public.kg_nodes o
  join public.kg_edges e
    on e.src_id = o.id and e.rel_type = 'about_concept' and e.is_active
   and e.tenant_id = p_tenant_id
  join public.kg_nodes c
    on c.id = e.dst_id and c.type::text = 'Concept' and c.status = 'active'
   and c.tenant_id = p_tenant_id
  where o.tenant_id = p_tenant_id
    and o.type::text = 'Observation'
    and o.status    = 'active'
    and (p_study_ids is null or o.study_id = any(p_study_ids))
    and (p_client_id is null or o.client_id = p_client_id);
$$;


-- ── R2c: observations_rollup + external_ref per cube row ─────────────────────
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
      c.properties->>'external_ref'                  as external_ref,
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
      max(label)         as label,
      max(external_ref)  as external_ref,
      study_id, modality, persona, period, direction,
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
             row_number() over (partition by concept_id order by occurred_at desc nulls last) as rn
      from obs
    ) t
    group by concept_id
  )
  select jsonb_build_object(
    'cube',     coalesce((select jsonb_agg(to_jsonb(cb)) from cube cb), '[]'::jsonb),
    'evidence', coalesce((select jsonb_object_agg(concept_id::text, ids) from ev), '{}'::jsonb)
  );
$$;


-- ── R1: mirror a transcript_tag into the graph as a governed Concept ────────
-- node id = tag id, external_ref = tag id, node_key = 'concept:{tag_id}'.
-- Idempotent — refreshes on codebook edits. Returns { concept_id, external_ref }.
create or replace function public.mirror_tag_concept(
  p_tenant_id       uuid,
  p_client_id       uuid,
  p_tag_id          uuid,
  p_label           text,
  p_description     text         default null,
  p_embedding       vector(1536) default null,
  p_embedding_model text         default 'text-embedding-3-small'
)
returns jsonb
language plpgsql
as $$
declare
  v_node_key text := 'concept:' || p_tag_id::text;
  v_id       uuid;
  v_props    jsonb;
begin
  v_props := jsonb_build_object(
    'canonical_id',     p_tag_id::text,
    'canonical_label',  p_label,
    'external_ref',     p_tag_id::text,
    'alias_set',        jsonb_build_object('canonical', p_label, 'members', '[]'::jsonb),
    'merge_confidence', 1.0
  );

  select id into v_id
    from public.kg_nodes
   where tenant_id = p_tenant_id and node_key = v_node_key and type = 'Concept'
   limit 1;

  if v_id is null then
    insert into public.kg_nodes (
      id, tenant_id, client_id, node_key, type, name, description, properties,
      embedding, embedding_model, status, last_seen_at, seen_count, created_at, updated_at
    ) values (
      p_tag_id, p_tenant_id, p_client_id, v_node_key, 'Concept',
      p_label, coalesce(p_description, p_label), v_props,
      p_embedding, coalesce(p_embedding_model, 'text-embedding-3-small'),
      'active', now(), 1, now(), now()
    )
    returning id into v_id;
  else
    update public.kg_nodes
       set name         = p_label,
           description  = coalesce(p_description, p_label),
           properties   = coalesce(properties, '{}'::jsonb) || v_props,
           embedding    = coalesce(p_embedding, embedding),
           status       = 'active',
           last_seen_at = now(),
           seen_count   = seen_count + 1,
           updated_at   = now()
     where id = v_id;
  end if;

  return jsonb_build_object('concept_id', v_id, 'external_ref', p_tag_id::text, 'node_key', v_node_key);
end;
$$;


-- ── R3: stamp external_ref on an existing candidate concept (graduation) ────
-- Marks a candidate governed in place — preserves its accumulated observations.
create or replace function public.link_concept_tag(
  p_tenant_id  uuid,
  p_concept_id uuid,
  p_tag_id     uuid
)
returns jsonb
language plpgsql
as $$
declare v_rows int;
begin
  update public.kg_nodes
     set properties = coalesce(properties, '{}'::jsonb)
                      || jsonb_build_object('external_ref', p_tag_id::text),
         updated_at = now()
   where id = p_concept_id
     and tenant_id = p_tenant_id
     and type::text = 'Concept';
  get diagnostics v_rows = row_count;
  return jsonb_build_object('linked', v_rows > 0, 'concept_id', p_concept_id, 'external_ref', p_tag_id::text);
end;
$$;


notify pgrst, 'reload schema';
