-- 37_entity_kg.sql
-- Tier-2 of the enriched-entity rollout: Entity nodes + MENTIONS edges in
-- the KG. Reuses the existing kg_nodes / kg_edges infrastructure so that
-- search_graph, hop expansion, audit triggers, tenant filtering, and the
-- embedding HNSW index all work for entities for free.
--
-- Design:
--   - Entity node:   kg_nodes row, type='Entity',
--                    node_key='entity:{canonical_id}:{label}',
--                    properties={canonical_id, canonical_text, label,
--                                source_origin, total_mentions,
--                                distinct_session_count, sentiment_aggregate,
--                                merged_into}, embedding=embed(canonical_text+label)
--   - Session node:  kg_nodes row, type='Document',
--                    node_key='document:{document_id}',
--                    properties={document_id}
--                    (Session in the brief == Document in our model.)
--   - MENTIONS edge: kg_edges row, src_id=document_node, dst_id=entity_node,
--                    rel_type='mentions', weight=confidence,
--                    properties={cue_indices, sample_contexts, sentiment,
--                                count, source, confidence, extracted_at,
--                                extractor_version}
--
-- The brief mandates atomicity (no partial-success) for the upsert call.
-- That's why this lives as a server-side RPC rather than two supabase-py
-- table calls — the function runs in a single implicit transaction.

-- ── 1. Enum additions ───────────────────────────────────────────────────────
do $$
begin
  alter type artifact_type add value if not exists 'Document';
  alter type node_status   add value if not exists 'merged';
exception
  when duplicate_object then null;
end $$;


-- ── 2. Fast lookup index for canonical_id (label-agnostic) ─────────────────
-- node_key already encodes canonical_id but Postgres can't use it for
-- LIKE 'entity:%:abc:%' efficiently. An expression index over the JSON
-- property gives fast GET-by-canonical_id when label is omitted.
create index if not exists kg_nodes_canonical_id_idx
  on public.kg_nodes ((properties->>'canonical_id'))
  where type = 'Entity';


-- ── 3. upsert_entity_with_mention RPC ──────────────────────────────────────
-- One transactional call that handles every step from the brief:
--   1. Resolve session_id (== document_id) → kg_node, create if missing
--   2. Resolve canonical_id+label → Entity kg_node
--      - Follow merged_into pointer if status='merged'
--   3. Upsert Entity (preserve embedding if already set; only insert sets it)
--   4. Upsert MENTIONS edge with the per-mention payload
--   5. Recompute Entity aggregates from kg_edges (idempotent — calling twice
--      with the same payload is a no-op modulo last_seen_at)
--
-- Returns: { entity_id, redirected_from, document_node_id }
--
-- Caller (Python) is responsible for embedding the canonical_text+label
-- once per *new* entity. Subsequent updates pass p_embedding=null and the
-- existing embedding is preserved.

drop function if exists public.upsert_entity_with_mention(
  uuid, uuid, uuid, text, text, text, text, vector, jsonb, jsonb
) cascade;

create or replace function public.upsert_entity_with_mention(
  p_tenant_id           uuid,
  p_client_id           uuid,
  p_document_id         uuid,           -- the session / source document
  p_canonical_id        text,
  p_canonical_text      text,
  p_label               text,
  p_source_origin       text default 'spacy',
  p_embedding           vector(1536) default null,
  p_entity_properties   jsonb default '{}'::jsonb,    -- extra fields on the entity (sentiment_aggregate etc.)
  p_mention_properties  jsonb default '{}'::jsonb     -- per-mention payload on the edge
)
returns jsonb
language plpgsql
as $$
declare
  v_doc_node_id     uuid;
  v_entity_id       uuid;
  v_redirected_from uuid := null;
  v_existing        record;
  v_doc_name        text;
  v_node_key        text := 'entity:' || p_canonical_id || ':' || p_label;
  v_doc_node_key    text := 'document:' || p_document_id::text;
begin
  -- 1. Document/Session node ────────────────────────────────────────────────
  select title into v_doc_name
    from public.documents
   where id = p_document_id and tenant_id = p_tenant_id
   limit 1;

  if v_doc_name is null then
    v_doc_name := 'Document ' || left(p_document_id::text, 8);
  end if;

  insert into public.kg_nodes (
    tenant_id, client_id, node_key, type, name, description, properties,
    status, last_seen_at, seen_count, created_at, updated_at
  ) values (
    p_tenant_id, p_client_id, v_doc_node_key, 'Document',
    v_doc_name, null,
    jsonb_build_object('document_id', p_document_id::text),
    'active', now(), 1, now(), now()
  )
  on conflict (tenant_id, client_id, node_key) do update
    set last_seen_at = now(),
        seen_count   = public.kg_nodes.seen_count + 1,
        updated_at   = now()
  returning id into v_doc_node_id;

  -- 2. Entity node — follow merge pointer if any ────────────────────────────
  select id, status, properties into v_existing
    from public.kg_nodes
   where tenant_id = p_tenant_id
     and node_key  = v_node_key
     and type      = 'Entity'
   limit 1;

  if v_existing.id is not null and v_existing.status = 'merged' then
    v_redirected_from := v_existing.id;
    -- properties.merged_into holds the surviving entity's canonical_id+label
    declare
      v_surviving_key text := v_existing.properties->>'merged_into';
    begin
      if v_surviving_key is not null then
        select id into v_entity_id
          from public.kg_nodes
         where tenant_id = p_tenant_id
           and node_key  = v_surviving_key
           and type      = 'Entity'
         limit 1;
      end if;
    end;
  end if;

  -- 3. Upsert the entity (or the surviving one if we redirected) ────────────
  if v_entity_id is null then
    -- Normal path: create or update the entity at the requested key
    insert into public.kg_nodes (
      tenant_id, client_id, node_key, type, name, description, properties,
      embedding, status, last_seen_at, seen_count, created_at, updated_at
    ) values (
      p_tenant_id, p_client_id, v_node_key, 'Entity',
      p_canonical_text, p_label || ': ' || p_canonical_text,
      coalesce(p_entity_properties, '{}'::jsonb) || jsonb_build_object(
        'canonical_id',   p_canonical_id,
        'canonical_text', p_canonical_text,
        'label',          p_label,
        'source_origin',  p_source_origin
      ),
      p_embedding, 'active', now(), 1, now(), now()
    )
    on conflict (tenant_id, client_id, node_key) do update
      set name         = excluded.name,
          description  = excluded.description,
          -- merge new props onto existing; preserves merged_into etc.
          properties   = public.kg_nodes.properties || excluded.properties,
          -- preserve embedding if already set (the brief: "do NOT re-embed on update")
          embedding    = coalesce(public.kg_nodes.embedding, excluded.embedding),
          last_seen_at = now(),
          seen_count   = public.kg_nodes.seen_count + 1,
          updated_at   = now()
    returning id into v_entity_id;
  end if;

  -- 4. MENTIONS edge: Document → Entity ─────────────────────────────────────
  insert into public.kg_edges (
    tenant_id, client_id, src_id, dst_id, rel_type, weight, properties,
    is_active, last_seen_at, seen_count, created_at, updated_at
  ) values (
    p_tenant_id, p_client_id, v_doc_node_id, v_entity_id, 'mentions',
    coalesce((p_mention_properties->>'confidence')::float4, 1.0),
    coalesce(p_mention_properties, '{}'::jsonb) || jsonb_build_object(
      'extracted_at', now()::text
    ),
    true, now(), 1, now(), now()
  )
  on conflict (tenant_id, client_id, src_id, dst_id, rel_type) do update
    -- Last-writer-wins on properties; latest extraction reflects latest model
    set properties   = excluded.properties,
        weight       = coalesce(excluded.weight, public.kg_edges.weight),
        is_active    = true,
        last_seen_at = now(),
        seen_count   = public.kg_edges.seen_count + 1,
        updated_at   = now();

  -- 5. Recompute aggregates from edges (idempotent) ─────────────────────────
  perform public.recompute_entity_aggregates(v_entity_id);

  return jsonb_build_object(
    'entity_id',          v_entity_id,
    'redirected_from',    v_redirected_from,
    'document_node_id',   v_doc_node_id
  );
end;
$$;


-- ── 4. recompute_entity_aggregates helper ──────────────────────────────────
-- Computed from kg_edges rather than incremented. Keeps the upsert
-- idempotent and lets edge deletes (session removed) self-heal aggregates.

create or replace function public.recompute_entity_aggregates(p_entity_id uuid)
returns void
language plpgsql
as $$
declare
  v_total_mentions    int  := 0;
  v_distinct_sessions int  := 0;
  v_first_seen        timestamptz;
  v_last_seen         timestamptz;
  v_sent_mean         float4;
  v_sent_min          float4;
  v_sent_max          float4;
  v_pos               int  := 0;
  v_neg               int  := 0;
  v_neu               int  := 0;
  v_mix               int  := 0;
begin
  -- Sum per-edge counts and unique sessions
  select coalesce(sum( coalesce((properties->'sentiment'->>'mention_count')::int,
                                 coalesce((properties->>'count')::int, 1)) ), 0),
         count(distinct src_id),
         min(created_at),
         max(last_seen_at)
    into v_total_mentions, v_distinct_sessions, v_first_seen, v_last_seen
    from public.kg_edges
   where dst_id = p_entity_id
     and rel_type = 'mentions'
     and is_active = true;

  -- Sentiment aggregate across all edges
  select avg((properties->'sentiment'->>'compound_mean')::float4),
         min((properties->'sentiment'->>'compound_min')::float4),
         max((properties->'sentiment'->>'compound_max')::float4)
    into v_sent_mean, v_sent_min, v_sent_max
    from public.kg_edges
   where dst_id = p_entity_id
     and rel_type = 'mentions'
     and is_active = true
     and properties ? 'sentiment';

  -- Valence distribution
  select
    sum(case when properties->'sentiment'->>'valence' = 'positive' then 1 else 0 end),
    sum(case when properties->'sentiment'->>'valence' = 'negative' then 1 else 0 end),
    sum(case when properties->'sentiment'->>'valence' = 'neutral'  then 1 else 0 end),
    sum(case when properties->'sentiment'->>'valence' = 'mixed'    then 1 else 0 end)
    into v_pos, v_neg, v_neu, v_mix
    from public.kg_edges
   where dst_id = p_entity_id
     and rel_type = 'mentions'
     and is_active = true;

  update public.kg_nodes
     set properties = properties || jsonb_build_object(
           'total_mentions',         v_total_mentions,
           'distinct_session_count', v_distinct_sessions,
           'first_seen_at',          coalesce(v_first_seen, now())::text,
           'last_seen_at',           coalesce(v_last_seen,  now())::text,
           'sentiment_aggregate', jsonb_build_object(
             'compound_mean', v_sent_mean,
             'compound_min',  v_sent_min,
             'compound_max',  v_sent_max,
             'valence_distribution', jsonb_build_object(
               'positive', coalesce(v_pos, 0),
               'negative', coalesce(v_neg, 0),
               'neutral',  coalesce(v_neu, 0),
               'mixed',    coalesce(v_mix, 0)
             )
           )
         ),
         updated_at = now()
   where id = p_entity_id;
end;
$$;


-- ── 5. merge_entities RPC ──────────────────────────────────────────────────
-- Marks source as status='merged' with merged_into pointing at surviving
-- entity's node_key. Rewires all incoming MENTIONS edges to the survivor.
-- Recomputes the surviving entity's aggregates. Idempotent.

create or replace function public.merge_entities(
  p_tenant_id            uuid,
  p_source_entity_id     uuid,
  p_surviving_entity_id  uuid
)
returns jsonb
language plpgsql
as $$
declare
  v_surviving_key  text;
  v_source_status  text;
  v_rewired        int := 0;
begin
  -- Tenant isolation check
  if not exists (select 1 from public.kg_nodes
                  where id = p_source_entity_id and tenant_id = p_tenant_id) then
    raise exception 'source entity not found in tenant';
  end if;
  if not exists (select 1 from public.kg_nodes
                  where id = p_surviving_entity_id and tenant_id = p_tenant_id) then
    raise exception 'surviving entity not found in tenant';
  end if;
  if p_source_entity_id = p_surviving_entity_id then
    raise exception 'cannot merge an entity into itself';
  end if;

  select node_key into v_surviving_key
    from public.kg_nodes where id = p_surviving_entity_id;

  -- Rewire MENTIONS edges. If a duplicate would result (same src,
  -- same dst-after-rewire, same rel_type), the unique constraint catches
  -- it — handle by deleting the duplicate source-side edge.
  begin
    update public.kg_edges
       set dst_id = p_surviving_entity_id,
           updated_at = now()
     where dst_id = p_source_entity_id
       and tenant_id = p_tenant_id;
    get diagnostics v_rewired = row_count;
  exception when unique_violation then
    -- Conflict: same (src, surviving_dst, rel_type) already exists.
    -- Strategy: drop the source-side row, keep the existing surviving edge.
    delete from public.kg_edges
     where dst_id = p_source_entity_id
       and tenant_id = p_tenant_id
       and exists (
         select 1 from public.kg_edges e2
          where e2.tenant_id = public.kg_edges.tenant_id
            and e2.client_id is not distinct from public.kg_edges.client_id
            and e2.src_id    = public.kg_edges.src_id
            and e2.dst_id    = p_surviving_entity_id
            and e2.rel_type  = public.kg_edges.rel_type
       );
    -- Rewire the remaining non-conflicting ones
    update public.kg_edges
       set dst_id = p_surviving_entity_id,
           updated_at = now()
     where dst_id = p_source_entity_id
       and tenant_id = p_tenant_id;
    get diagnostics v_rewired = row_count;
  end;

  -- Mark source as merged
  update public.kg_nodes
     set status = 'merged',
         properties = properties || jsonb_build_object('merged_into', v_surviving_key),
         updated_at = now()
   where id = p_source_entity_id;

  -- Recompute survivor aggregates
  perform public.recompute_entity_aggregates(p_surviving_entity_id);

  return jsonb_build_object(
    'merged',         true,
    'redirect_count', v_rewired
  );
end;
$$;
