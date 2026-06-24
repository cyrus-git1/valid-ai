-- 43_observation_concept_spine.sql
-- First Slice of the agent-layer memory spine: Observation nodes + a durable,
-- tenant-scoped Concept spine. Reuses kg_nodes / kg_edges exactly like the
-- Entity-KG layer (migrations 37/39) so embedding/HNSW, hybrid search, hop
-- expansion, per-hop tenant filters, audit triggers, and PII redaction all
-- work for the new types for free.
--
-- Storage model:
--   - Observation node:  kg_nodes row, type='Observation',
--                        node_key='observation:{observation_id}',
--                        name/description = nl_text,
--                        embedding = embed(nl_text) in the shared 1536-d space,
--                        properties = {observation_id, value{number,unit} (OPAQUE,
--                          stored & returned verbatim, never parsed), modality,
--                          signal_type, direction, prevalence{pct,n}, confidence,
--                          reliability{sample_n,method,quality_flags[],
--                          diarization_confidence}, segment{persona,variant_key},
--                          occurred_at, source{aggregate_id,call_id,input_hash,
--                          agent_version,evidence_ref}}
--   - Concept node:      kg_nodes row, type='Concept',
--                        node_key='concept:{canonical_id}',
--                        properties = {canonical_id, canonical_label,
--                          alias_set{canonical,members[]}, merge_confidence}
--   - observation->evidence:  kg_node_evidence row (node_id=obs, chunk_id=source)
--   - observation->concept:   kg_edges row, rel_type='about_concept',
--                             src_id=obs node, dst_id=concept node
--
-- Concept identity is PER-ORG: canonical_id is a freshly minted UUID at create
-- time (NOT a label-only hash — that was the latent cross-tenant leak). Keyed by
-- (tenant_id, node_key) it is unique within the tenant and never global.
--
-- Idempotency: observation upsert is keyed by observation_id (node_key) — a
-- re-send is an update-in-place, never a duplicate. concept create is idempotent
-- when the caller supplies an explicit canonical_id (the agent's resolve-first
-- flow on re-ingest).
--
-- Deferred (shaped to drop in additively, NOT built here): supersede-on-change
-- (status 'superseded' + versioned node_key), concept-feedback override
-- persistence, participant/question/study nodes as edge targets, study:context,
-- timeline fetch, purge cascade.

-- ── 1. Enum additions ───────────────────────────────────────────────────────
do $$
begin
  alter type artifact_type add value if not exists 'Observation';
  alter type artifact_type add value if not exists 'Concept';
exception
  when duplicate_object then null;
end $$;


-- ── 2. Fast lookup indexes (mirror kg_nodes_canonical_id_idx, mig 37) ───────
create index if not exists kg_nodes_observation_id_idx
  on public.kg_nodes ((properties->>'observation_id'))
  where type = 'Observation';

create index if not exists kg_nodes_concept_canonical_id_idx
  on public.kg_nodes ((properties->>'canonical_id'))
  where type = 'Concept';


-- ── 3. upsert_observation RPC ───────────────────────────────────────────────
-- One transactional call: upsert the Observation node, link it to its source
-- evidence chunk, and (when the agent has already resolved a concept) link it to
-- that concept. Idempotent on observation_id. Re-embeds only when a non-null
-- p_embedding is supplied (caller re-embeds on nl_text change).
--
-- Follows the explicit lookup -> insert-or-update shape of
-- upsert_entity_with_mention (mig 39) so client_id IS NULL is handled correctly
-- against the partial unique indexes from migration 38.
--
-- Returns: { observation_id, node_id, evidence_linked, concept_linked }

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
-- Returns: { concept_id, canonical_id, node_key, created }

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
  v_canonical_id text := coalesce(nullif(p_canonical_id, ''), gen_random_uuid()::text);
  v_node_key     text := 'concept:' || v_canonical_id;
  v_node_id      uuid;
  v_existing_emb vector(1536);
  v_alias_set    jsonb := jsonb_build_object(
                            'canonical', p_canonical_label,
                            'members',   to_jsonb(coalesce(p_aliases, array[]::text[]))
                          );
  v_props        jsonb;
  v_created      boolean := false;
begin
  v_props := jsonb_build_object(
    'canonical_id',     v_canonical_id,
    'canonical_label',  p_canonical_label,
    'alias_set',        v_alias_set,
    'merge_confidence', p_merge_confidence
  );

  select id, embedding into v_node_id, v_existing_emb
    from public.kg_nodes
   where tenant_id = p_tenant_id
     and node_key  = v_node_key
     and type      = 'Concept'
     and (p_client_id is null and client_id is null
          or p_client_id is not null and client_id = p_client_id)
   limit 1;

  if v_node_id is null then
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
  else
    update public.kg_nodes
       set name         = p_canonical_label,
           description  = p_canonical_label,
           properties   = coalesce(properties, '{}'::jsonb) || v_props,
           embedding    = coalesce(p_embedding, v_existing_emb),
           last_seen_at = now(),
           seen_count   = seen_count + 1,
           updated_at   = now()
     where id = v_node_id;
  end if;

  return jsonb_build_object(
    'concept_id',   v_node_id,
    'canonical_id', v_canonical_id,
    'node_key',     v_node_key,
    'created',      v_created
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
