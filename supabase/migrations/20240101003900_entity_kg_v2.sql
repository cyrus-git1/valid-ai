-- 39_entity_kg_v2.sql
-- Rewrites upsert_entity_with_mention to handle client_id=NULL correctly.
--
-- Postgres ON CONFLICT can't target a partial unique index easily, so we
-- switch to explicit "lookup → insert OR update" inside the function. This
-- matches the partial unique indexes from migration 38 and works whether
-- client_id is NULL or set.

drop function if exists public.upsert_entity_with_mention(
  uuid, uuid, uuid, text, text, text, text, vector, jsonb, jsonb
) cascade;

create or replace function public.upsert_entity_with_mention(
  p_tenant_id           uuid,
  p_client_id           uuid,
  p_document_id         uuid,
  p_canonical_id        text,
  p_canonical_text      text,
  p_label               text,
  p_source_origin       text default 'spacy',
  p_embedding           vector(1536) default null,
  p_entity_properties   jsonb default '{}'::jsonb,
  p_mention_properties  jsonb default '{}'::jsonb
)
returns jsonb
language plpgsql
as $$
declare
  v_doc_node_id     uuid;
  v_entity_id       uuid;
  v_redirected_from uuid := null;
  v_existing_id     uuid;
  v_existing_status text;
  v_existing_props  jsonb;
  v_existing_emb    vector(1536);
  v_doc_name        text;
  v_node_key        text := 'entity:' || p_canonical_id || ':' || p_label;
  v_doc_node_key    text := 'document:' || p_document_id::text;
  v_edge_id         uuid;
  v_surviving_key   text;
begin
  -- ── 1. Document/Session node (lookup or insert) ──────────────────────────
  select id into v_doc_node_id
    from public.kg_nodes
   where tenant_id = p_tenant_id
     and node_key  = v_doc_node_key
     and type      = 'Document'
     and (p_client_id is null and client_id is null
          or p_client_id is not null and client_id = p_client_id)
   limit 1;

  if v_doc_node_id is null then
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
    returning id into v_doc_node_id;
  else
    update public.kg_nodes
       set last_seen_at = now(),
           seen_count   = seen_count + 1,
           updated_at   = now()
     where id = v_doc_node_id;
  end if;

  -- ── 2. Entity node (lookup; follow merge pointer) ────────────────────────
  select id, status, properties, embedding
    into v_existing_id, v_existing_status, v_existing_props, v_existing_emb
    from public.kg_nodes
   where tenant_id = p_tenant_id
     and node_key  = v_node_key
     and type      = 'Entity'
     and (p_client_id is null and client_id is null
          or p_client_id is not null and client_id = p_client_id)
   limit 1;

  if v_existing_id is not null and v_existing_status = 'merged' then
    v_redirected_from := v_existing_id;
    v_surviving_key   := v_existing_props->>'merged_into';
    if v_surviving_key is not null then
      select id, properties, embedding
        into v_existing_id, v_existing_props, v_existing_emb
        from public.kg_nodes
       where tenant_id = p_tenant_id
         and node_key  = v_surviving_key
         and type      = 'Entity'
         and (p_client_id is null and client_id is null
              or p_client_id is not null and client_id = p_client_id)
       limit 1;
      v_existing_status := 'active';
    end if;
  end if;

  -- ── 3. Insert or update the entity ───────────────────────────────────────
  if v_existing_id is null then
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
    returning id into v_entity_id;
  else
    v_entity_id := v_existing_id;
    update public.kg_nodes
       set name         = p_canonical_text,
           description  = p_label || ': ' || p_canonical_text,
           properties   = coalesce(properties, '{}'::jsonb)
                          || coalesce(p_entity_properties, '{}'::jsonb),
           -- preserve existing embedding (brief: do NOT re-embed on update)
           embedding    = coalesce(v_existing_emb, p_embedding),
           last_seen_at = now(),
           seen_count   = seen_count + 1,
           updated_at   = now()
     where id = v_entity_id;
  end if;

  -- ── 4. MENTIONS edge (lookup or insert) ──────────────────────────────────
  select id into v_edge_id
    from public.kg_edges
   where tenant_id = p_tenant_id
     and src_id    = v_doc_node_id
     and dst_id    = v_entity_id
     and rel_type  = 'mentions'
     and (p_client_id is null and client_id is null
          or p_client_id is not null and client_id = p_client_id)
   limit 1;

  if v_edge_id is null then
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
    );
  else
    update public.kg_edges
       set properties   = coalesce(p_mention_properties, '{}'::jsonb) || jsonb_build_object(
              'extracted_at', now()::text
           ),
           weight       = coalesce((p_mention_properties->>'confidence')::float4, weight),
           is_active    = true,
           last_seen_at = now(),
           seen_count   = seen_count + 1,
           updated_at   = now()
     where id = v_edge_id;
  end if;

  -- ── 5. Recompute aggregates ──────────────────────────────────────────────
  perform public.recompute_entity_aggregates(v_entity_id);

  return jsonb_build_object(
    'entity_id',          v_entity_id,
    'redirected_from',    v_redirected_from,
    'document_node_id',   v_doc_node_id
  );
end;
$$;
