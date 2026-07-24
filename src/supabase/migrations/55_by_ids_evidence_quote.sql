-- 55_by_ids_evidence_quote.sql
-- Return the evidence QUOTE TEXT on observations_by_ids, so /spine/hydrate turns
-- quote-cluster refs from scope-checked pointers into rendered quotes.
--
-- Per observation, the primary linked evidence (highest-scored kg_node_evidence
-- row) → { text, speaker?, offset_ms? }:
--   text      = coalesce(evidence.quote, transcript segment text, chunk content[:500])
--   speaker   = the linked chunk's first transcript segment speaker_label (if any)
--   offset_ms = that segment's start_seconds * 1000 (if any)
-- Null when the observation has no linked evidence (absent, not crashing).
--
-- Adding the `evidence` column changes the return type → drop before recreate.
-- Ends with NOTIFY pgrst 'reload schema'.

drop function if exists public.observations_by_ids(uuid, text[], uuid[]) cascade;

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
  evidence_chunk_ids uuid[],
  evidence           jsonb
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
       where ev.node_id = n.id and ev.tenant_id = p_tenant_id
    )                                     as evidence_chunk_ids,
    (
      select jsonb_build_object(
        'text',      coalesce(ev.quote, seg.text, left(ch.content, 500)),
        'speaker',   seg.speaker_label,
        'offset_ms', case when seg.start_seconds is not null
                          then (seg.start_seconds * 1000)::bigint else null end
      )
      from public.kg_node_evidence ev
      left join public.chunks ch
        on ch.id = ev.chunk_id and ch.tenant_id = p_tenant_id
      left join lateral (
        select ts.speaker_label, ts.start_seconds, ts.text
        from public.transcript_segments ts
        where ts.chunk_id = ev.chunk_id
        order by ts.start_seconds asc
        limit 1
      ) seg on true
      where ev.node_id = n.id and ev.tenant_id = p_tenant_id
      order by ev.score desc nulls last
      limit 1
    )                                     as evidence
  from public.kg_nodes n
  where n.tenant_id = p_tenant_id
    and n.type::text = 'Observation'
    and n.status = 'active'
    and n.properties->>'observation_id' = any(p_observation_ids)
    and (p_study_ids is null or n.study_id = any(p_study_ids));
$$;


notify pgrst, 'reload schema';
