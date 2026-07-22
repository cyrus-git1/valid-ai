-- 51_recreate_spine_functions.sql
-- FIX: PGRST202 "Could not find the function public.nearest_concepts(...) in the
-- schema cache" — migrations 43 & 47 were marked applied but their CREATE
-- FUNCTIONs never took (the enum ADD VALUE + same-transaction use rolled the
-- rest back). The spine/app-entity RPCs were missing from the DB entirely, so
-- every /observations, /concepts, /app-entities call 500'd.
--
-- This re-creates the 7 functions (idempotent create-or-replace) and ensures the
-- enum values exist for runtime. SAFE in one transaction: the SQL functions use
-- `n.type::text = '...'` (no enum literal at create time) and the plpgsql bodies
-- only reference the enum values at call time — so no 55P04. No index predicates
-- with enum literals here (those were the trap); indexes are perf-only and can be
-- re-added separately with ::text predicates if needed.
-- Creating these functions is DDL, which triggers Supabase's PostgREST auto-reload.

do $$
begin
  alter type artifact_type add value if not exists 'Observation';
  alter type artifact_type add value if not exists 'Concept';
  alter type artifact_type add value if not exists 'AppEntity';
  alter type node_status   add value if not exists 'inactive';
exception when duplicate_object then null;
end $$;

-- ═══ functions from migration 43 (observation/concept spine) ═══
create or replace function public.upsert_observation(
  p_tenant_id          uuid,
  p_client_id          uuid,
  p_observation_id     text,
  p_nl_text            text,
  p_properties         jsonb,                        -- value + descriptive + source, verbatim
  p_embedding          vector(1536) default null,
  p_embedding_model    text         default 'text-embedding-3-small',
  p_study_id           uuid         default null,
  p_evidence_chunk_id  uuid         default null,
  p_concept_id         uuid         default null     -- resolved Concept kg_nodes.id
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
  -- Verbatim properties, with observation_id guaranteed present.
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
           -- verbatim re-send: new keys overwrite, nothing silently retained
           properties   = coalesce(properties, '{}'::jsonb) || v_props,
           -- re-embed only when a fresh embedding is supplied
           embedding    = coalesce(p_embedding, v_existing_emb),
           study_id     = coalesce(p_study_id, study_id),
           last_seen_at = now(),
           seen_count   = seen_count + 1,
           updated_at   = now()
     where id = v_node_id;
  end if;

  -- ── 2. observation -> evidence (kg_node_evidence) ────────────────────────
  -- Only when the chunk exists in this tenant (FK + tenant safety).
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
  -- Only when the target is a Concept node in this tenant.
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


-- ── 4. nearest_concepts RPC ─────────────────────────────────────────────────
-- Given an embedding (+ optional lexical hints), return the top-k candidate
-- Concepts WITHIN the caller's tenant, ranked by cosine similarity with a small
-- additive boost when a hint matches the canonical_label or an alias member.
-- Resolution LOGIC stays in the agent layer; this is just the candidate primitive.

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
    (1 - (n.embedding <=> p_embedding))::float4   as similarity,
    (
      (1 - (n.embedding <=> p_embedding))
      + case
          when p_hints is not null and (
            exists (
              select 1 from unnest(p_hints) h
               where n.properties->>'canonical_label' ilike '%' || h || '%'
            )
            or exists (
              select 1
                from jsonb_array_elements_text(
                       coalesce(n.properties->'alias_set'->'members', '[]'::jsonb)) m,
                     unnest(p_hints) h
               where m ilike '%' || h || '%'
            )
          ) then 0.05
          else 0.0
        end
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


-- ── 5. create_concept RPC ───────────────────────────────────────────────────
-- Mints a per-org canonical_id (random UUID) unless the caller supplies one
-- (idempotent resolve-first path). Inserts/updates the Concept node.
--
-- Redirect guard: if a supplied canonical_id resolves to a MERGED tombstone
-- (a deterministic-recreate of a retired id after a merge), follow merged_into
-- to the terminal survivor and return it with redirected=true. This path is
-- strictly NON-DESTRUCTIVE — the create payload (canonical_label / alias_set /
-- embedding) is NEVER applied to the survivor, so a stray recreate can't clobber
-- a live concept's label or grow its aliases as a side effect. Alias growth is
-- explicit, via the (future) feedback API only.
--
-- Returns: { concept_id, canonical_id, node_key, created, redirected }

create or replace function public.create_concept(
  p_tenant_id        uuid,
  p_client_id        uuid,
  p_canonical_label  text,
  p_aliases          text[]       default array[]::text[],
  p_embedding        vector(1536) default null,
  p_embedding_model  text         default 'text-embedding-3-small',
  p_merge_confidence float4       default 1.0,
  p_canonical_id     text         default null
)
returns jsonb
language plpgsql
as $$
declare
  v_canonical_id    text := coalesce(nullif(p_canonical_id, ''), gen_random_uuid()::text);
  v_node_key        text := 'concept:' || v_canonical_id;
  v_node_id         uuid;
  v_existing_emb    vector(1536);
  v_existing_status text;
  v_merged_into     text;
  v_found_canon     text;
  v_found_node_key  text;
  v_alias_set       jsonb := jsonb_build_object(
                              'canonical', p_canonical_label,
                              'members',   to_jsonb(coalesce(p_aliases, array[]::text[]))
                            );
  v_props           jsonb;
  v_created         boolean := false;
  v_redirected      boolean := false;
  v_guard           int := 0;
begin
  v_props := jsonb_build_object(
    'canonical_id',     v_canonical_id,
    'canonical_label',  p_canonical_label,
    'alias_set',        v_alias_set,
    'merge_confidence', p_merge_confidence
  );

  select id, embedding, status,
         properties->>'merged_into', properties->>'canonical_id', node_key
    into v_node_id, v_existing_emb, v_existing_status,
         v_merged_into, v_found_canon, v_found_node_key
    from public.kg_nodes
   where tenant_id = p_tenant_id
     and node_key  = v_node_key
     and type      = 'Concept'
     and (p_client_id is null and client_id is null
          or p_client_id is not null and client_id = p_client_id)
   limit 1;

  if v_node_id is null then
    -- Fresh concept: insert active.
    insert into public.kg_nodes (
      tenant_id, client_id, node_key, type, name, description, properties,
      embedding, embedding_model, status,
      last_seen_at, seen_count, created_at, updated_at
    ) values (
      p_tenant_id, p_client_id, v_node_key, 'Concept',
      p_canonical_label, p_canonical_label, v_props,
      p_embedding, coalesce(p_embedding_model, 'text-embedding-3-small'),
      'active', now(), 1, now(), now()
    )
    returning id into v_node_id;
    v_created := true;

  elsif v_existing_status = 'merged' and v_merged_into is not null then
    -- Tombstone: deterministic-recreate of a retired canonical_id. Follow the
    -- merged_into chain to the terminal survivor and return it. NON-DESTRUCTIVE:
    -- the survivor is never updated with the create payload.
    v_redirected     := true;
    v_found_node_key := v_merged_into;
    loop
      select id, status, properties->>'merged_into',
             properties->>'canonical_id', node_key
        into v_node_id, v_existing_status, v_merged_into, v_found_canon, v_found_node_key
        from public.kg_nodes
       where tenant_id = p_tenant_id
         and node_key  = v_found_node_key
         and type      = 'Concept'
         and (p_client_id is null and client_id is null
              or p_client_id is not null and client_id = p_client_id)
       limit 1;
      v_guard := v_guard + 1;
      exit when v_node_id is null
             or v_existing_status is distinct from 'merged'
             or v_merged_into is null
             or v_guard >= 16;
    end loop;
    if v_node_id is not null then
      v_canonical_id := coalesce(v_found_canon, v_canonical_id);
      v_node_key     := v_found_node_key;
    end if;

  elsif v_existing_status = 'merged' then
    -- Merged tombstone with no survivor pointer (shouldn't happen — merge always
    -- sets merged_into). Stay non-destructive: don't touch it, don't resurrect it.
    v_canonical_id := coalesce(v_found_canon, v_canonical_id);
    v_node_key     := coalesce(v_found_node_key, v_node_key);

  else
    -- Active concept: idempotent re-affirm (also the create-with-same-id path;
    -- updating label/alias/embedding here is intended for a live concept).
    update public.kg_nodes
       set name         = p_canonical_label,
           description  = p_canonical_label,
           properties   = coalesce(properties, '{}'::jsonb) || v_props,
           embedding    = coalesce(p_embedding, v_existing_emb),
           last_seen_at = now(),
           seen_count   = seen_count + 1,
           updated_at   = now()
     where id = v_node_id;
    v_canonical_id := coalesce(v_found_canon, v_canonical_id);
    v_node_key     := coalesce(v_found_node_key, v_node_key);
  end if;

  return jsonb_build_object(
    'concept_id',   v_node_id,
    'canonical_id', v_canonical_id,
    'node_key',     v_node_key,
    'created',      v_created,
    'redirected',   v_redirected
  );
end;
$$;


-- ── 6. merge_concepts RPC ───────────────────────────────────────────────────
-- Fold source concept into survivor: union alias_set (source canonical_label +
-- its members become survivor members), rewire about_concept edges to survivor
-- (dedup on unique-violation, like merge_entities), and mark source 'merged'
-- with merged_into pointing at the survivor's node_key. Preserves alias_set and
-- all observation edges. Idempotent. Tenant-isolated.
--
-- Returns: { merged, rewired_count, surviving_member_count }

create or replace function public.merge_concepts(
  p_tenant_id            uuid,
  p_surviving_concept_id uuid,
  p_source_concept_id    uuid
)
returns jsonb
language plpgsql
as $$
declare
  v_surviving_key   text;
  v_surviving_props jsonb;
  v_source_props    jsonb;
  v_members         jsonb;
  v_rewired         int := 0;
begin
  if p_source_concept_id = p_surviving_concept_id then
    raise exception 'cannot merge a concept into itself';
  end if;

  select node_key, properties into v_surviving_key, v_surviving_props
    from public.kg_nodes
   where id = p_surviving_concept_id and tenant_id = p_tenant_id and type = 'Concept';
  if v_surviving_key is null then
    raise exception 'surviving concept not found in tenant';
  end if;

  select properties into v_source_props
    from public.kg_nodes
   where id = p_source_concept_id and tenant_id = p_tenant_id and type = 'Concept';
  if v_source_props is null then
    raise exception 'source concept not found in tenant';
  end if;

  -- Union members: survivor members + source canonical_label + source members,
  -- de-duplicated, preserving order of first appearance.
  select jsonb_agg(distinct_member) into v_members
  from (
    select member as distinct_member
    from (
      select jsonb_array_elements_text(
               coalesce(v_surviving_props->'alias_set'->'members', '[]'::jsonb)) as member
      union
      select v_source_props->>'canonical_label'
      union
      select jsonb_array_elements_text(
               coalesce(v_source_props->'alias_set'->'members', '[]'::jsonb))
    ) u
    where member is not null
      and member <> coalesce(v_surviving_props->'alias_set'->>'canonical', '')
  ) d;

  update public.kg_nodes
     set properties = properties
                      || jsonb_build_object(
                           'alias_set',
                           jsonb_build_object(
                             'canonical', v_surviving_props->'alias_set'->>'canonical',
                             'members',   coalesce(v_members, '[]'::jsonb)
                           )
                         ),
         updated_at = now()
   where id = p_surviving_concept_id;

  -- Rewire about_concept edges (observation -> concept; concept is dst_id).
  begin
    update public.kg_edges
       set dst_id = p_surviving_concept_id,
           updated_at = now()
     where dst_id = p_source_concept_id
       and tenant_id = p_tenant_id
       and rel_type = 'about_concept';
    get diagnostics v_rewired = row_count;
  exception when unique_violation then
    -- A (src, surviving_dst, rel_type) already exists: drop the conflicting
    -- source-side edge, then rewire the rest.
    delete from public.kg_edges
     where dst_id = p_source_concept_id
       and tenant_id = p_tenant_id
       and rel_type = 'about_concept'
       and exists (
         select 1 from public.kg_edges e2
          where e2.tenant_id = public.kg_edges.tenant_id
            and e2.client_id is not distinct from public.kg_edges.client_id
            and e2.src_id    = public.kg_edges.src_id
            and e2.dst_id    = p_surviving_concept_id
            and e2.rel_type  = 'about_concept'
       );
    update public.kg_edges
       set dst_id = p_surviving_concept_id,
           updated_at = now()
     where dst_id = p_source_concept_id
       and tenant_id = p_tenant_id
       and rel_type = 'about_concept';
    get diagnostics v_rewired = row_count;
  end;

  -- Mark source merged.
  update public.kg_nodes
     set status     = 'merged',
         properties = properties || jsonb_build_object('merged_into', v_surviving_key),
         updated_at = now()
   where id = p_source_concept_id;

  return jsonb_build_object(
    'merged',                 true,
    'rewired_count',          v_rewired,
    'surviving_member_count', jsonb_array_length(coalesce(v_members, '[]'::jsonb))
  );
end;
$$;


-- ── 7. observations_by_concept RPC ──────────────────────────────────────────
-- Scoped fetch backing the agent layer's heatmap / drill-down: every observation
-- linked to a concept, filtered by tenant + optional study_ids + modality +
-- segment, returned with its verbatim value, prevalence, reliability, occurred_at,
-- source provenance, and the chunk_ids it cites as evidence. Tenant-scoped on
-- every hop.

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
  evidence_chunk_ids uuid[]
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
       where ev.node_id = n.id
         and ev.tenant_id = p_tenant_id
    )                                            as evidence_chunk_ids
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

-- ═══ functions from migration 47 (app entities) ═══
create or replace function public.upsert_app_entity(
  p_tenant_id        uuid,
  p_client_id        uuid,
  p_study_id         uuid,
  p_kind             text,
  p_external_id      text,
  p_text             text,
  p_embedding        vector(1536) default null,
  p_embedding_model  text         default 'text-embedding-3-small',
  p_status           text         default 'active'
)
returns jsonb
language plpgsql
as $$
declare
  v_node_key     text := 'app:' || p_kind || ':' || p_external_id;
  v_node_id      uuid;
  v_existing_emb vector(1536);
  v_created      boolean := false;
  v_props        jsonb;
begin
  v_props := jsonb_build_object(
    'external_ref', jsonb_build_object('kind', p_kind, 'id', p_external_id),
    'kind',         p_kind
  );

  select id, embedding
    into v_node_id, v_existing_emb
    from public.kg_nodes
   where tenant_id = p_tenant_id
     and node_key  = v_node_key
     and type      = 'AppEntity'
     and (p_client_id is null and client_id is null
          or p_client_id is not null and client_id = p_client_id)
   limit 1;

  if v_node_id is null then
    insert into public.kg_nodes (
      tenant_id, client_id, node_key, type, name, description, properties,
      embedding, embedding_model, study_id, status,
      last_seen_at, seen_count, created_at, updated_at
    ) values (
      p_tenant_id, p_client_id, v_node_key, 'AppEntity',
      p_text, p_text, v_props,
      p_embedding, coalesce(p_embedding_model, 'text-embedding-3-small'),
      p_study_id, coalesce(p_status, 'active')::node_status,
      now(), 1, now(), now()
    )
    returning id into v_node_id;
    v_created := true;
  else
    update public.kg_nodes
       set name         = p_text,
           description  = p_text,
           properties   = coalesce(properties, '{}'::jsonb) || v_props,
           -- re-embed only when a fresh embedding is supplied
           embedding    = coalesce(p_embedding, v_existing_emb),
           study_id     = coalesce(p_study_id, study_id),
           status       = coalesce(p_status, 'active')::node_status,
           last_seen_at = now(),
           seen_count   = seen_count + 1,
           updated_at   = now()
     where id = v_node_id;
  end if;

  return jsonb_build_object(
    'node_id',     v_node_id,
    'external_id', p_external_id,
    'kind',        p_kind,
    'created',     v_created
  );
end;
$$;


-- ── 4. nearest_app_entities RPC (mirror nearest_concepts, mig 43) ───────────
-- Returns app ids (never node text), ranked by cosine, scoped to study_ids and
-- (optionally) kinds. Tenant-scoped. status='active' excludes soft-deleted rows.
create or replace function public.nearest_app_entities(
  p_tenant_id        uuid,
  p_embedding        vector(1536),
  p_study_ids        uuid[]  default null,
  p_kinds            text[]  default null,
  p_top_k            int     default 20,
  p_embedding_model  text    default 'text-embedding-3-small'
)
returns table (
  external_id  text,
  kind         text,
  study_id     uuid,
  similarity   float4
)
language sql
stable
as $$
  select
    n.properties->'external_ref'->>'id'          as external_id,
    n.properties->>'kind'                        as kind,
    n.study_id,
    (1 - (n.embedding <=> p_embedding))::float4  as similarity
  from public.kg_nodes n
  where n.tenant_id        = p_tenant_id
    and n.type::text       = 'AppEntity'
    and n.status           = 'active'
    and n.embedding is not null
    and n.embedding_model  = p_embedding_model
    and (p_study_ids is null or n.study_id = any(p_study_ids))
    and (p_kinds is null or n.properties->>'kind' = any(p_kinds))
  order by n.embedding <=> p_embedding
  limit p_top_k;
$$;

notify pgrst, 'reload schema';
