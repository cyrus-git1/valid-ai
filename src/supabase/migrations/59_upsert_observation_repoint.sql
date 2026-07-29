-- 59_upsert_observation_repoint.sql
-- RE-POINT on concept_id change (the mechanism the theme-split collapse depends on).
--
-- An observation is ABOUT EXACTLY ONE concept. The original upsert_observation
-- (mig 43 §3) only ensured an edge to the NEW concept_id existed — it left any
-- edge to a DIFFERENT concept in place. So re-emitting an observation with the
-- governed node id created a SECOND about_concept edge and the candidate twin
-- never emptied → retire-orphans could never collapse the split.
--
-- Fix: after binding the observation to p_concept_id, DELETE its about_concept
-- edges to any OTHER concept (+ their edge_evidence). Re-emitting onto the
-- governed node now MOVES the observation off the candidate, which then hits 0
-- observations and is retired. Also self-heals pre-existing double-edges.
--
-- (If an observation is ever meant to be about multiple concepts, this enforces
-- single-binding — but the spine's resolution emits exactly one concept_id, and
-- rollup/relations assume one about_concept edge per observation.)
--
-- Full recreate of upsert_observation; only §3 changes. Ends with schema reload.

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
  p_concept_id         uuid         default null
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
begin
  v_props := coalesce(p_properties, '{}'::jsonb)
             || jsonb_build_object('observation_id', p_observation_id);

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

  -- ── 2. observation -> evidence (kg_node_evidence) ────────────────────────
  if p_evidence_chunk_id is not null
     and exists (select 1 from public.chunks
                  where id = p_evidence_chunk_id and tenant_id = p_tenant_id) then
    if not exists (
      select 1 from public.kg_node_evidence
       where tenant_id = p_tenant_id
         and node_id   = v_node_id
         and chunk_id  = p_evidence_chunk_id
         and (p_client_id is null and client_id is null
              or p_client_id is not null and client_id = p_client_id)
    ) then
      insert into public.kg_node_evidence (
        tenant_id, client_id, node_id, chunk_id, quote, score, created_at
      ) values (
        p_tenant_id, p_client_id, v_node_id, p_evidence_chunk_id, null, null, now()
      );
    end if;
    v_evidence_linked := true;
  end if;

  -- ── 3. observation -> concept (kg_edges, rel_type 'about_concept') ───────
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

    -- RE-POINT: drop this observation's about_concept edges to any OTHER concept,
    -- so a re-emit onto the governed node empties the old candidate twin.
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


notify pgrst, 'reload schema';
