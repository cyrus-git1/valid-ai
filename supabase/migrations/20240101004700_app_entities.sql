-- 47_app_entities.sql
-- App-entity recall: make app objects (insights, quotes, sessions) semantically
-- recallable by their app id. A near-clone of the observation/concept spine
-- (migration 43) — same kg_nodes substrate, same embedding space, same
-- lookup-then-upsert idempotency, same tenant/study scoping. No new tables.
--
-- Storage model:
--   - AppEntity node: kg_nodes row, type='AppEntity',
--                     node_key='app:{kind}:{external_id}'  (idempotency),
--                     name/description = the embeddable text,
--                     embedding = embed(text) in the shared 1536-d space,
--                     study_id = the app study,
--                     status = 'active' | 'inactive'  (inactive = soft-delete),
--                     properties = { external_ref:{kind,id}, kind }
--   kind in {insight, quote, session}; external_id is the app's row id
--   (insights.id / transcript_highlights.id / calls.id).

-- ── 1. Enum additions ───────────────────────────────────────────────────────
do $$
begin
  alter type artifact_type add value if not exists 'AppEntity';
  -- 'inactive' is the soft-delete status target for app entities. Recall filters
  -- status='active', so 'inactive' rows are excluded. Nothing enumerates
  -- node_status exhaustively; the mig-45 trust-gate only checks archived/merged.
  alter type node_status   add value if not exists 'inactive';
exception
  when duplicate_object then null;
end $$;


-- ── 2. Idempotency/dedup index (mirror kg_nodes_observation_id_idx, mig 43) ──
create index if not exists kg_nodes_app_external_id_idx
  on public.kg_nodes ((properties->'external_ref'->>'id'))
  where type = 'AppEntity';


-- ── 3. upsert_app_entity RPC (mirror upsert_observation, mig 43) ────────────
-- Idempotent on (tenant_id, node_key, client_id). Re-embeds only when the caller
-- supplies a fresh embedding (the router skips embedding when text is unchanged).
-- Returns: { node_id, external_id, kind, created }
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
