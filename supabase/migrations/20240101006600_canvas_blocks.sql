-- 66_canvas_blocks.sql  (part 2 of 2) — indexes + RPCs.
--
-- Uses the CanvasBlock artifact_type added (and committed) in migration 65.
-- Grounding canvas persistence: store the twelve business-canvas blocks as
-- kg_nodes, a near-clone of app_entities (migration 47). No new tables.
--
-- Storage model:
--   - CanvasBlock node: kg_nodes row, type='CanvasBlock',
--       node_key = 'canvas:org:{block_key}'                (org-level, study_id null)
--                | 'canvas:study:{study_id}:{block_key}'   (study overlay)
--       name/description = the STATEMENT (embedded, so a block can relate to concepts),
--       study_id = the study (null = org canvas),
--       properties = { external_ref:{kind:'canvas_block', id:node_key},
--                      block_key, scope, stated, evidenced,
--                      source, status, confidence, pinned, divergence,
--                      evidence_refs:[{type,id,origin,stance,weight,study_id}] }
--
-- Governance: a human edit pins the block (source='human' => pinned=true). An
-- agent refresh (source='agent') NEVER overwrites a pinned statement — it only
-- updates evidenced / evidence_refs / divergence. Omitted fields preserve their
-- existing value (null-coalesced), so a partial agent refresh can't wipe the
-- stated/status a prior pass wrote.


-- ── 1. Scope-read indexes ───────────────────────────────────────────────────
create index if not exists kg_nodes_canvas_block_key_idx
  on public.kg_nodes ((properties->>'block_key'))
  where type = 'CanvasBlock';
create index if not exists kg_nodes_canvas_scope_idx
  on public.kg_nodes (tenant_id, study_id)
  where type = 'CanvasBlock';


-- ── 2. upsert_canvas_block RPC (mirror upsert_app_entity, mig 47) ────────────
-- Idempotent on (tenant_id, node_key, client_id). Re-embeds only when a fresh
-- embedding is supplied. Returns: { node_id, block_key, scope, created, pinned }
create or replace function public.upsert_canvas_block(
  p_tenant_id        uuid,
  p_client_id        uuid,
  p_study_id         uuid,
  p_block_key        text,
  p_statement        text         default null,
  p_stated           text         default null,
  p_evidenced        text         default null,
  p_source           text         default 'agent',
  p_status           text         default null,
  p_confidence       text         default null,
  p_pinned           boolean      default null,
  p_divergence       boolean      default null,
  p_evidence_refs    jsonb        default null,
  p_embedding        vector(1536) default null,
  p_embedding_model  text         default 'text-embedding-3-small'
)
returns jsonb
language plpgsql
as $$
declare
  v_scope        text := case when p_study_id is null then 'org' else 'study' end;
  v_node_key     text := case when p_study_id is null
                              then 'canvas:org:' || p_block_key
                              else 'canvas:study:' || p_study_id::text || ':' || p_block_key end;
  v_node_id      uuid;
  v_existing_emb vector(1536);
  v_existing     jsonb;
  v_existing_nm  text;
  v_pinned_prev  boolean := false;
  v_created      boolean := false;
  v_name         text;
  v_final_pinned boolean;
  v_props        jsonb;
begin
  select id, embedding, properties, name
    into v_node_id, v_existing_emb, v_existing, v_existing_nm
    from public.kg_nodes
   where tenant_id = p_tenant_id
     and node_key  = v_node_key
     and type      = 'CanvasBlock'
     and (p_client_id is null and client_id is null
          or p_client_id is not null and client_id = p_client_id)
   limit 1;

  v_pinned_prev := coalesce((v_existing->>'pinned')::boolean, false);

  -- Pinned + agent => preserve the human statement; only refresh evidence.
  if v_node_id is not null and v_pinned_prev and p_source = 'agent' then
    v_name         := v_existing_nm;
    v_final_pinned := true;
  else
    v_name         := coalesce(p_statement, v_existing_nm, p_stated);
    v_final_pinned := case when p_source = 'human' then true
                           else coalesce(p_pinned, v_pinned_prev, false) end;
  end if;

  v_props := jsonb_build_object(
    'external_ref',  jsonb_build_object('kind', 'canvas_block', 'id', v_node_key),
    'block_key',     p_block_key,
    'scope',         v_scope,
    'stated',        coalesce(p_stated,     v_existing->>'stated'),
    'evidenced',     coalesce(p_evidenced,  v_existing->>'evidenced'),
    'source',        p_source,
    'status',        coalesce(p_status,     v_existing->>'status',     'assumption'),
    'confidence',    coalesce(p_confidence, v_existing->>'confidence', 'low'),
    'pinned',        v_final_pinned,
    'divergence',    coalesce(p_divergence, (v_existing->>'divergence')::boolean, false),
    'evidence_refs', coalesce(p_evidence_refs, v_existing->'evidence_refs', '[]'::jsonb)
  );

  if v_node_id is null then
    insert into public.kg_nodes (
      tenant_id, client_id, node_key, type, name, description, properties,
      embedding, embedding_model, study_id, status,
      last_seen_at, seen_count, created_at, updated_at
    ) values (
      p_tenant_id, p_client_id, v_node_key, 'CanvasBlock',
      v_name, v_name, v_props,
      p_embedding, coalesce(p_embedding_model, 'text-embedding-3-small'),
      p_study_id, 'active'::node_status,
      now(), 1, now(), now()
    )
    returning id into v_node_id;
    v_created := true;
  else
    update public.kg_nodes
       set name         = coalesce(v_name, name),
           description  = coalesce(v_name, description),
           properties   = coalesce(properties, '{}'::jsonb) || v_props,
           embedding    = coalesce(p_embedding, v_existing_emb),
           study_id     = coalesce(p_study_id, study_id),
           last_seen_at = now(),
           seen_count   = seen_count + 1,
           updated_at   = now()
     where id = v_node_id;
  end if;

  return jsonb_build_object(
    'node_id',    v_node_id,
    'block_key',  p_block_key,
    'scope',      v_scope,
    'created',    v_created,
    'pinned',     v_final_pinned,
    'divergence', coalesce(p_divergence, (v_existing->>'divergence')::boolean, false)
  );
end;
$$;


-- ── 3. canvas_by_scope RPC — fetch all blocks for a scope (not nearest) ──────
-- study_id set => that study's overlay; null => the org-level canvas.
create or replace function public.canvas_by_scope(
  p_tenant_id  uuid,
  p_client_id  uuid,
  p_study_id   uuid default null
)
returns table (
  node_id        uuid,
  block_key      text,
  statement      text,
  stated         text,
  evidenced      text,
  source         text,
  status         text,
  confidence     text,
  pinned         boolean,
  divergence     boolean,
  study_id       uuid,
  evidence_refs  jsonb
)
language sql
stable
as $$
  select
    n.id,
    n.properties->>'block_key',
    n.name,
    n.properties->>'stated',
    n.properties->>'evidenced',
    n.properties->>'source',
    n.properties->>'status',
    n.properties->>'confidence',
    coalesce((n.properties->>'pinned')::boolean, false),
    coalesce((n.properties->>'divergence')::boolean, false),
    n.study_id,
    coalesce(n.properties->'evidence_refs', '[]'::jsonb)
  from public.kg_nodes n
  where n.tenant_id  = p_tenant_id
    and n.type::text = 'CanvasBlock'
    and n.status     = 'active'
    and (p_client_id is null and n.client_id is null
         or p_client_id is not null and n.client_id = p_client_id)
    and (p_study_id is not null and n.study_id = p_study_id
         or p_study_id is null and n.study_id is null);
$$;
