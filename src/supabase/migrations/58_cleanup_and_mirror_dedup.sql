-- 58_cleanup_and_mirror_dedup.sql
-- Formalize the manual dedup/cleanup sweeps into RPCs, and make the taxonomy
-- mirror idempotent on external_ref (so "Sync themes" can't re-fragment already
-- graduated themes).
--
--   retire_orphan_concepts(tenant)        → delete CANDIDATE concepts with 0 live
--        linked observations. Governed nodes (external_ref set) are PRESERVED —
--        a governed theme with 0 observations is a codebook mirror awaiting
--        resolution, not an orphan. Returns { retired, retired_labels }.
--   purge_study(tenant, study)            → delete a study's observations, then
--        retire the candidate concepts THOSE observations left orphaned.
--        Returns { observations_deleted, concepts_deleted }.
--   mirror_tag_concept(...)  [REPLACED]   → dedup on external_ref: if any concept
--        already carries external_ref = tag_id (e.g. a GRADUATED candidate whose
--        node id != tag_id), update THAT node in place instead of minting a second
--        governed node keyed concept:{tag_id}.
--
-- All idempotent. Ends with NOTIFY pgrst 'reload schema'.

-- ── #1 retire_orphan_concepts ───────────────────────────────────────────────
create or replace function public.retire_orphan_concepts(
  p_tenant_id uuid
)
returns jsonb
language plpgsql
as $$
declare
  v_ids    uuid[];
  v_labels text[];
begin
  select array_agg(c.id), array_agg(c.properties->>'canonical_label')
    into v_ids, v_labels
  from public.kg_nodes c
  where c.tenant_id = p_tenant_id
    and c.type::text = 'Concept'
    and c.status = 'active'
    and (c.properties->>'external_ref') is null                 -- preserve governed/codebook nodes
    and not exists (
      select 1
        from public.kg_edges e
        join public.kg_nodes o
          on o.id = e.src_id and o.status = 'active' and o.type::text = 'Observation'
       where e.dst_id = c.id
         and e.tenant_id = p_tenant_id
         and e.rel_type = 'about_concept'
         and e.is_active
    );

  if v_ids is null then
    return jsonb_build_object('retired', 0, 'retired_labels', '[]'::jsonb);
  end if;

  delete from public.kg_edge_evidence ee
   using public.kg_edges e
   where ee.edge_id = e.id
     and (e.src_id = any(v_ids) or e.dst_id = any(v_ids));
  delete from public.kg_edges
   where src_id = any(v_ids) or dst_id = any(v_ids);
  delete from public.kg_node_evidence
   where node_id = any(v_ids);
  delete from public.kg_nodes
   where id = any(v_ids) and tenant_id = p_tenant_id;

  return jsonb_build_object(
    'retired',        array_length(v_ids, 1),
    'retired_labels', to_jsonb(v_labels)
  );
end;
$$;


-- ── #2 purge_study ──────────────────────────────────────────────────────────
create or replace function public.purge_study(
  p_tenant_id uuid,
  p_study_id  uuid
)
returns jsonb
language plpgsql
as $$
declare
  v_obs              uuid[];
  v_linked           uuid[];
  v_dead             uuid[];
  v_obs_deleted      int := 0;
  v_concepts_deleted int := 0;
begin
  select array_agg(id) into v_obs
    from public.kg_nodes
   where tenant_id = p_tenant_id
     and type::text = 'Observation'
     and study_id = p_study_id;

  if v_obs is null then
    return jsonb_build_object('observations_deleted', 0, 'concepts_deleted', 0);
  end if;

  -- concepts these observations point at (checked for orphanhood after deletion)
  select array_agg(distinct e.dst_id) into v_linked
    from public.kg_edges e
   where e.src_id = any(v_obs)
     and e.tenant_id = p_tenant_id
     and e.rel_type = 'about_concept';

  -- delete the observations (edges + evidence + nodes)
  delete from public.kg_edge_evidence ee
   using public.kg_edges e
   where ee.edge_id = e.id
     and (e.src_id = any(v_obs) or e.dst_id = any(v_obs));
  delete from public.kg_edges
   where (src_id = any(v_obs) or dst_id = any(v_obs))
     and tenant_id = p_tenant_id;
  delete from public.kg_node_evidence
   where node_id = any(v_obs);
  delete from public.kg_nodes
   where id = any(v_obs) and tenant_id = p_tenant_id;
  get diagnostics v_obs_deleted = row_count;

  -- retire the candidate concepts THOSE observations left orphaned
  if v_linked is not null then
    select array_agg(c.id) into v_dead
    from public.kg_nodes c
    where c.id = any(v_linked)
      and c.tenant_id = p_tenant_id
      and c.type::text = 'Concept'
      and c.status = 'active'
      and (c.properties->>'external_ref') is null              -- preserve governed
      and not exists (
        select 1
          from public.kg_edges e
          join public.kg_nodes o
            on o.id = e.src_id and o.status = 'active' and o.type::text = 'Observation'
         where e.dst_id = c.id
           and e.tenant_id = p_tenant_id
           and e.rel_type = 'about_concept'
           and e.is_active
      );

    if v_dead is not null then
      delete from public.kg_edge_evidence ee
       using public.kg_edges e
       where ee.edge_id = e.id
         and (e.src_id = any(v_dead) or e.dst_id = any(v_dead));
      delete from public.kg_edges
       where src_id = any(v_dead) or dst_id = any(v_dead);
      delete from public.kg_node_evidence
       where node_id = any(v_dead);
      delete from public.kg_nodes
       where id = any(v_dead) and tenant_id = p_tenant_id;
      v_concepts_deleted := array_length(v_dead, 1);
    end if;
  end if;

  return jsonb_build_object(
    'observations_deleted', v_obs_deleted,
    'concepts_deleted',     v_concepts_deleted
  );
end;
$$;


-- ── #3 mirror_tag_concept — dedup on external_ref ───────────────────────────
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
  v_key      text;
  v_props    jsonb;
begin
  -- (a) DEDUP: an existing node already governed by this tag (e.g. a graduated
  -- candidate whose node id != tag_id). Update it in place — never mint a second.
  select id, node_key into v_id, v_key
    from public.kg_nodes
   where tenant_id = p_tenant_id
     and type = 'Concept'
     and properties->>'external_ref' = p_tag_id::text
   limit 1;

  if v_id is not null then
    update public.kg_nodes
       set name            = p_label,
           description     = coalesce(p_description, description),
           properties      = coalesce(properties, '{}'::jsonb)
                             || jsonb_build_object('external_ref', p_tag_id::text,
                                                   'canonical_label', p_label),
           embedding       = coalesce(p_embedding, embedding),
           embedding_model = coalesce(p_embedding_model, embedding_model),
           status          = 'active',
           last_seen_at    = now(),
           seen_count      = seen_count + 1,
           updated_at      = now()
     where id = v_id;
    return jsonb_build_object('concept_id', v_id, 'external_ref', p_tag_id::text, 'node_key', v_key);
  end if;

  -- (b) canonical mirror node keyed concept:{tag_id} (node id = tag id).
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


notify pgrst, 'reload schema';
