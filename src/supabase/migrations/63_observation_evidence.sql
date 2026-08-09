-- 60_observation_evidence.sql
-- Persist observation EVIDENCE (verbatim respondent quote + chunk links) so chat
-- synth can ground answers at quote level, not just concept-label level.
--
-- BEFORE: upsert accepted a single evidence_chunk_id and stored kg_node_evidence
-- with quote = NULL; there was NO field for a verbatim quote. by-concept didn't
-- return evidence at all. So observations were label-only (evidence_text null,
-- evidence_chunk_ids []).
--
-- NOW:
--   upsert_observation  — accepts p_evidence {text, speaker?, offset_ms?} (stored
--        on the node as properties.evidence) AND p_evidence_chunk_ids uuid[]
--        (each linked via kg_node_evidence with quote = the verbatim text).
--        Back-compat: p_evidence_chunk_id still works (folded into the list).
--   observations_by_ids     — evidence PREFERS the stored quote, falls back to
--        chunk-derived (transcript segment / content).
--   observations_by_concept — now RETURNS the evidence quote (same coalesce),
--        alongside evidence_chunk_ids.
--
-- Ends with NOTIFY pgrst 'reload schema'.

-- ── upsert_observation: add evidence params (signature changes → drop first) ──
drop function if exists public.upsert_observation(
  uuid, uuid, text, text, jsonb, vector, text, uuid, uuid, uuid
);

create or replace function public.upsert_observation(
  p_tenant_id          uuid,
  p_client_id          uuid,
  p_observation_id     text,
  p_nl_text            text,
  p_properties         jsonb,
  p_embedding          vector(1536) default null,
  p_embedding_model    text         default 'text-embedding-3-small',
  p_study_id           uuid         default null,
  p_evidence_chunk_id  uuid         default null,
  p_concept_id         uuid         default null,
  p_evidence           jsonb        default null,
  p_evidence_chunk_ids uuid[]       default null
)
returns jsonb
language plpgsql
as $$
declare
  v_node_key        text := 'observation:' || p_observation_id;
  v_node_id         uuid;
  v_existing_emb    vector(1536);
  v_props           jsonb;
  v_evidence_linked boolean := false;
  v_concept_linked  boolean := false;
  v_edge_id         uuid;
  v_chunk_ids       uuid[];
  v_cid             uuid;
begin
  v_props := coalesce(p_properties, '{}'::jsonb)
             || jsonb_build_object('observation_id', p_observation_id);
  -- primary verbatim quote lives on the node so hydration never needs a chunk
  if p_evidence is not null then
    v_props := v_props || jsonb_build_object('evidence', p_evidence);
  end if;

  -- ── 1. Observation node (lookup or insert) ───────────────────────────────
  select id, embedding
    into v_node_id, v_existing_emb
    from public.kg_nodes
   where tenant_id = p_tenant_id
     and node_key  = v_node_key
     and type      = 'Observation'
     and (p_client_id is null and client_id is null
          or p_client_id is not null and client_id = p_client_id)
   limit 1;

  if v_node_id is null then
    insert into public.kg_nodes (
      tenant_id, client_id, node_key, type, name, description, properties,
      embedding, embedding_model, study_id, status,
      last_seen_at, seen_count, created_at, updated_at
    ) values (
      p_tenant_id, p_client_id, v_node_key, 'Observation',
      p_nl_text, p_nl_text, v_props,
      p_embedding, coalesce(p_embedding_model, 'text-embedding-3-small'),
      p_study_id, 'active', now(), 1, now(), now()
    )
    returning id into v_node_id;
  else
    update public.kg_nodes
       set name         = p_nl_text,
           description  = p_nl_text,
           properties   = coalesce(properties, '{}'::jsonb) || v_props,
           embedding    = coalesce(p_embedding, v_existing_emb),
           study_id     = coalesce(p_study_id, study_id),
           last_seen_at = now(),
           seen_count   = seen_count + 1,
           updated_at   = now()
     where id = v_node_id;
  end if;

  -- ── 2. observation -> evidence (kg_node_evidence), one row per chunk ──────
  -- Persist the verbatim quote onto each chunk link (was NULL before).
  v_chunk_ids := coalesce(
    p_evidence_chunk_ids,
    case when p_evidence_chunk_id is not null then array[p_evidence_chunk_id] else null end
  );
  if v_chunk_ids is not null then
    foreach v_cid in array v_chunk_ids loop
      if v_cid is not null
         and exists (select 1 from public.chunks
                      where id = v_cid and tenant_id = p_tenant_id) then
        if not exists (
          select 1 from public.kg_node_evidence
           where tenant_id = p_tenant_id
             and node_id   = v_node_id
             and chunk_id  = v_cid
             and (p_client_id is null and client_id is null
                  or p_client_id is not null and client_id = p_client_id)
        ) then
          insert into public.kg_node_evidence (
            tenant_id, client_id, node_id, chunk_id, quote, score, created_at
          ) values (
            p_tenant_id, p_client_id, v_node_id, v_cid, p_evidence->>'text', null, now()
          );
        else
          update public.kg_node_evidence
             set quote = coalesce(p_evidence->>'text', quote)
           where tenant_id = p_tenant_id
             and node_id   = v_node_id
             and chunk_id  = v_cid;
        end if;
        v_evidence_linked := true;
      end if;
    end loop;
  end if;

  -- ── 3. observation -> concept (kg_edges 'about_concept') + RE-POINT ──────
  if p_concept_id is not null
     and exists (select 1 from public.kg_nodes
                  where id = p_concept_id and tenant_id = p_tenant_id and type = 'Concept') then
    select id into v_edge_id
      from public.kg_edges
     where tenant_id = p_tenant_id
       and src_id    = v_node_id
       and dst_id    = p_concept_id
       and rel_type  = 'about_concept'
       and (p_client_id is null and client_id is null
            or p_client_id is not null and client_id = p_client_id)
     limit 1;

    if v_edge_id is null then
      insert into public.kg_edges (
        tenant_id, client_id, src_id, dst_id, rel_type, weight, properties,
        study_id, is_active, last_seen_at, seen_count, created_at, updated_at
      ) values (
        p_tenant_id, p_client_id, v_node_id, p_concept_id, 'about_concept',
        1.0, '{}'::jsonb, p_study_id, true, now(), 1, now(), now()
      );
    else
      update public.kg_edges
         set study_id     = coalesce(p_study_id, study_id),
             is_active    = true,
             last_seen_at = now(),
             seen_count   = seen_count + 1,
             updated_at   = now()
       where id = v_edge_id;
    end if;

    delete from public.kg_edge_evidence ee
     using public.kg_edges e
     where ee.edge_id   = e.id
       and e.tenant_id  = p_tenant_id
       and e.src_id     = v_node_id
       and e.rel_type   = 'about_concept'
       and e.dst_id    <> p_concept_id;
    delete from public.kg_edges
     where tenant_id = p_tenant_id
       and src_id    = v_node_id
       and rel_type  = 'about_concept'
       and dst_id   <> p_concept_id;

    v_concept_linked := true;
  end if;

  return jsonb_build_object(
    'observation_id',  p_observation_id,
    'node_id',         v_node_id,
    'evidence_linked', v_evidence_linked,
    'concept_linked',  v_concept_linked
  );
end;
$$;


-- ── observations_by_ids: prefer the stored quote, fall back to chunk-derived ─
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
    coalesce(
      n.properties->'evidence',
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
      )
    )                                     as evidence
  from public.kg_nodes n
  where n.tenant_id = p_tenant_id
    and n.type::text = 'Observation'
    and n.status = 'active'
    and n.properties->>'observation_id' = any(p_observation_ids)
    and (p_study_ids is null or n.study_id = any(p_study_ids));
$$;


-- ── observations_by_concept: RETURN the evidence quote (return type changes) ──
drop function if exists public.observations_by_concept(uuid, uuid, uuid[], text, text, text);

create or replace function public.observations_by_concept(
  p_tenant_id   uuid,
  p_concept_id  uuid,
  p_study_ids   uuid[]  default null,
  p_modality    text    default null,
  p_persona     text    default null,
  p_variant_key text    default null
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
    n.id                                         as node_id,
    n.properties->>'observation_id'              as observation_id,
    n.name                                       as nl_text,
    n.properties->'value'                        as value,
    n.properties->>'modality'                    as modality,
    n.properties->>'signal_type'                 as signal_type,
    n.properties->>'direction'                   as direction,
    n.properties->'prevalence'                   as prevalence,
    (n.properties->>'confidence')::float4        as confidence,
    n.properties->'reliability'                  as reliability,
    n.properties->'segment'                      as segment,
    n.properties->>'occurred_at'                 as occurred_at,
    n.properties->'source'                       as source,
    coalesce(n.study_id, e.study_id)             as study_id,
    (
      select array_agg(ev.chunk_id)
        from public.kg_node_evidence ev
       where ev.node_id = n.id and ev.tenant_id = p_tenant_id
    )                                            as evidence_chunk_ids,
    coalesce(
      n.properties->'evidence',
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
      )
    )                                            as evidence
  from public.kg_edges e
  join public.kg_nodes n
    on n.id = e.src_id
   and n.tenant_id = p_tenant_id
   and n.type::text = 'Observation'
   and n.status = 'active'
  where e.tenant_id = p_tenant_id
    and e.dst_id    = p_concept_id
    and e.rel_type  = 'about_concept'
    and e.is_active = true
    and (p_study_ids is null
         or coalesce(n.study_id, e.study_id) = any(p_study_ids))
    and (p_modality is null    or n.properties->>'modality' = p_modality)
    and (p_persona is null     or n.properties->'segment'->>'persona' = p_persona)
    and (p_variant_key is null or n.properties->'segment'->>'variant_key' = p_variant_key)
  order by n.properties->>'occurred_at' desc nulls last;
$$;


notify pgrst, 'reload schema';
