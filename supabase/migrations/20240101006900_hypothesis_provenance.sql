-- 69_hypothesis_provenance.sql — origin tag + lifecycle timestamps on hypotheses.
--
-- Adds an `origin` (docs | spine | user | inferred) to properties, and surfaces
-- the kg_nodes lifecycle columns (created_at, updated_at, seen_count) through
-- hypotheses_by_scope so the app can tell where a hypothesis came from and when
-- it was added / last touched. seen_count = 1 means freshly added (never
-- re-linked). Function return shape changes, so both are dropped + recreated.

drop function if exists public.upsert_hypothesis(
  uuid, uuid, uuid, text, text, text, text, text, text, jsonb, jsonb, vector, text
);
drop function if exists public.hypotheses_by_scope(uuid, uuid, uuid);


create or replace function public.upsert_hypothesis(
  p_tenant_id        uuid,
  p_client_id        uuid,
  p_study_id         uuid,
  p_external_id      text,
  p_text             text         default null,
  p_block_key        text         default null,
  p_status           text         default null,
  p_confidence       text         default null,
  p_reasoning        text         default null,
  p_theme_ids        jsonb        default null,
  p_evidence_refs    jsonb        default null,
  p_embedding        vector(1536) default null,
  p_embedding_model  text         default 'text-embedding-3-small',
  p_origin           text         default null
)
returns jsonb
language plpgsql
as $$
declare
  v_scope        text := case when p_study_id is null then 'org' else 'study' end;
  v_node_key     text := case when p_study_id is null
                              then 'hyp:org:' || p_external_id
                              else 'hyp:study:' || p_study_id::text || ':' || p_external_id end;
  v_node_id      uuid;
  v_existing_emb vector(1536);
  v_existing     jsonb;
  v_existing_nm  text;
  v_created      boolean := false;
  v_name         text;
  v_props        jsonb;
begin
  select id, embedding, properties, name
    into v_node_id, v_existing_emb, v_existing, v_existing_nm
    from public.kg_nodes
   where tenant_id = p_tenant_id
     and node_key  = v_node_key
     and type      = 'Hypothesis'
     and (p_client_id is null and client_id is null
          or p_client_id is not null and client_id = p_client_id)
   limit 1;

  v_name := coalesce(p_text, v_existing_nm);

  v_props := jsonb_build_object(
    'external_ref',  jsonb_build_object('kind', 'hypothesis', 'id', p_external_id),
    'scope',         v_scope,
    'block_key',     coalesce(p_block_key,  v_existing->>'block_key'),
    'status',        coalesce(p_status,     v_existing->>'status',     'untested'),
    'confidence',    coalesce(p_confidence, v_existing->>'confidence', 'low'),
    'reasoning',     coalesce(p_reasoning,  v_existing->>'reasoning'),
    -- origin is set on first write and preserved on later re-links.
    'origin',        coalesce(v_existing->>'origin', p_origin, 'user'),
    'theme_ids',     coalesce(p_theme_ids,     v_existing->'theme_ids',     '[]'::jsonb),
    'evidence_refs', coalesce(p_evidence_refs, v_existing->'evidence_refs', '[]'::jsonb)
  );

  if v_node_id is null then
    insert into public.kg_nodes (
      tenant_id, client_id, node_key, type, name, description, properties,
      embedding, embedding_model, study_id, status,
      last_seen_at, seen_count, created_at, updated_at
    ) values (
      p_tenant_id, p_client_id, v_node_key, 'Hypothesis',
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
    'node_id',     v_node_id,
    'external_id', p_external_id,
    'scope',       v_scope,
    'created',     v_created
  );
end;
$$;


create or replace function public.hypotheses_by_scope(
  p_tenant_id  uuid,
  p_client_id  uuid,
  p_study_id   uuid default null
)
returns table (
  node_id        uuid,
  external_id    text,
  text           text,
  block_key      text,
  status         text,
  confidence     text,
  reasoning      text,
  origin         text,
  theme_ids      jsonb,
  evidence_refs  jsonb,
  study_id       uuid,
  seen_count     integer,
  created_at     timestamptz,
  updated_at     timestamptz
)
language sql
stable
as $$
  select
    n.id,
    n.properties->'external_ref'->>'id',
    n.name,
    n.properties->>'block_key',
    n.properties->>'status',
    n.properties->>'confidence',
    n.properties->>'reasoning',
    n.properties->>'origin',
    coalesce(n.properties->'theme_ids',     '[]'::jsonb),
    coalesce(n.properties->'evidence_refs', '[]'::jsonb),
    n.study_id,
    n.seen_count,
    n.created_at,
    n.updated_at
  from public.kg_nodes n
  where n.tenant_id  = p_tenant_id
    and n.type::text = 'Hypothesis'
    and n.status     = 'active'
    and (p_client_id is null and n.client_id is null
         or p_client_id is not null and n.client_id = p_client_id)
    and (p_study_id is not null and n.study_id = p_study_id
         or p_study_id is null and n.study_id is null);
$$;
