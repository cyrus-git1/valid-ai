-- 26_summary_types.sql
-- Folds context summaries into the unified documents+chunks+kg_nodes model.
--
-- Changes:
--   1. Adds ContextSummary, DocumentSummary, TopicSummary to artifact_type
--   2. Adds a partial unique index on documents enforcing one canonical
--      summary per (tenant, client, source_type, scope-ref)
--   3. Replaces search_kg_nodes with a version that accepts p_source_types
--      so agents can filter retrieval to summary chunks
--   4. One-shot backfill function copying rows from public.context_summaries
--      into documents + chunks + kg_nodes
--   5. Drops the legacy context_summaries table and upsert RPC AFTER backfill
--      runs successfully
--
-- IMPORTANT: the backfill is idempotent. Run it once, then re-run the drop
-- block to remove the legacy table.

-- ── 1. Enum additions ───────────────────────────────────────────────────────
do $$
begin
  alter type artifact_type add value if not exists 'ContextSummary';
  alter type artifact_type add value if not exists 'DocumentSummary';
  alter type artifact_type add value if not exists 'TopicSummary';
exception
  when duplicate_object then null;
end $$;


-- ── 2. Partial unique index on documents ────────────────────────────────────
-- One canonical, active summary per scope. Older summaries stay around with
-- is_canonical=false + status='deprecated' for history.
do $$
begin
  if not exists (
    select 1 from pg_indexes
    where schemaname = 'public'
      and indexname  = 'documents_summary_scope_uq'
  ) then
    execute $idx$
      create unique index documents_summary_scope_uq
        on public.documents (
          tenant_id,
          client_id,
          source_type,
          (metadata->>'document_id'),
          (metadata->>'topic')
        )
        where is_canonical = true
          and status = 'active'
          and source_type::text in ('ContextSummary','DocumentSummary','TopicSummary');
    $idx$;
  end if;
end $$;


-- ── 3. search_kg_nodes with p_source_types ─────────────────────────────────
-- Drop the most recent overload (matches migration 25 signature).
drop function if exists public.search_kg_nodes(
  uuid, uuid, vector, int, text[], text[], float4, boolean, text[], text
) cascade;

create or replace function public.search_kg_nodes(
  p_tenant_id       uuid,
  p_client_id       uuid         default null,
  p_embedding       vector(1536),
  p_top_k           int          default 5,
  p_types           text[]       default null,
  p_document_ids    text[]       default null,
  p_recency_weight  float4       default 0.0,
  p_boost_pinned    boolean      default false,
  p_exclude_status  text[]       default array['archived','deprecated'],
  p_embedding_model text         default 'text-embedding-3-small',
  p_source_types    text[]       default null
)
returns table (
  id              uuid,
  node_key        text,
  name            text,
  description     text,
  properties      jsonb,
  type            artifact_type,
  source_type     text,
  embedding_model text,
  similarity      float4,
  final_score     float4
)
language sql
stable
as $$
  with scored as (
    select
      n.id,
      n.node_key,
      n.name,
      n.description,
      n.properties,
      n.type,
      d.source_type as doc_source_type,
      n.embedding_model,
      (1 - (n.embedding <=> p_embedding))::float4 as similarity,

      case when d.id is not null and p_recency_weight > 0 then
        (1.0 / (1.0 + extract(epoch from (now() - coalesce(d.source_timestamp, d.created_at))) / 86400.0))::float4
      else 0.0 end as recency_score,

      case when p_boost_pinned and d.id is not null then
        (case when d.is_pinned then 0.05 else 0.0 end)
        + (case when d.is_canonical then 0.03 else 0.0 end)
      else 0.0 end as boost_score

    from public.kg_nodes n
    left join public.documents d
      on d.id = (n.properties->>'document_id')::uuid
     and d.tenant_id = p_tenant_id

    where n.tenant_id = p_tenant_id
      and (p_client_id is null or n.client_id = p_client_id)
      and n.status    = 'active'
      and n.embedding is not null
      and n.embedding_model = p_embedding_model
      and (p_types is null or n.type::text = any(p_types))
      and (p_document_ids is null or n.properties->>'document_id' = any(p_document_ids))
      and (d.id is null or not (d.status = any(p_exclude_status)))
      and (p_source_types is null or (d.id is not null and d.source_type = any(p_source_types)))
  )
  select
    s.id,
    s.node_key,
    s.name,
    s.description,
    s.properties,
    s.type,
    s.doc_source_type as source_type,
    s.embedding_model,
    s.similarity,
    (
      (1.0 - p_recency_weight) * s.similarity
      + p_recency_weight * s.recency_score
      + s.boost_score
    )::float4 as final_score
  from scored s
  order by final_score desc
  limit p_top_k;
$$;


-- ── 4. Backfill function ────────────────────────────────────────────────────
-- Copies every row from context_summaries into documents+chunks+kg_nodes.
-- Summary text gets stored in chunks.content; embedding is left NULL and
-- must be regenerated after migration (the worker handles this).
-- Idempotent: skips rows that already have a canonical ContextSummary doc
-- at the same (tenant, client).
create or replace function public.backfill_context_summaries_to_chunks()
returns table (migrated int, skipped int)
language plpgsql
as $$
declare
  v_migrated int := 0;
  v_skipped  int := 0;
  r          record;
  v_doc_id   uuid;
  v_chunk_id uuid;
begin
  -- Source table may already be dropped; be defensive.
  if not exists (select 1 from information_schema.tables
                 where table_schema='public' and table_name='context_summaries') then
    return query select 0, 0;
    return;
  end if;

  for r in
    select id, tenant_id, client_id, summary, topics, metadata, source_stats,
           created_at, updated_at
    from public.context_summaries
  loop
    -- Skip if a canonical ContextSummary already exists at this scope
    if exists (
      select 1 from public.documents
      where tenant_id = r.tenant_id
        and client_id = r.client_id
        and source_type::text = 'ContextSummary'
        and is_canonical = true
        and status = 'active'
    ) then
      v_skipped := v_skipped + 1;
      continue;
    end if;

    insert into public.documents (
      tenant_id, client_id, source_type, source_uri, title,
      source_timestamp, is_pinned, is_canonical, status,
      metadata, created_at, updated_at
    ) values (
      r.tenant_id,
      r.client_id,
      'ContextSummary',
      'summary:context:' || r.tenant_id::text || ':' || coalesce(r.client_id::text, 'tenant'),
      'Context Summary',
      r.updated_at,
      false,
      true,
      'active',
      coalesce(r.metadata, '{}'::jsonb)
        || jsonb_build_object(
          'topics',       coalesce(r.topics, '[]'::jsonb),
          'source_stats', coalesce(r.source_stats, '{}'::jsonb),
          'migrated_from_context_summaries', r.id
        ),
      r.created_at,
      r.updated_at
    )
    returning id into v_doc_id;

    insert into public.chunks (
      tenant_id, document_id, chunk_index, content, metadata, created_at,
      embedding, embedding_model
    ) values (
      r.tenant_id,
      v_doc_id,
      0,
      r.summary,
      jsonb_build_object(
        'source_type', 'ContextSummary',
        'tenant_id',   r.tenant_id::text,
        'client_id',   r.client_id::text,
        'document_id', v_doc_id::text
      ),
      r.created_at,
      null,                                  -- re-embedded by background worker
      'text-embedding-3-small'
    )
    returning id into v_chunk_id;

    -- Corresponding KG node
    insert into public.kg_nodes (
      tenant_id, client_id, node_key, type, name, description,
      properties, status, embedding_model, last_seen_at, seen_count,
      created_at, updated_at
    ) values (
      r.tenant_id,
      r.client_id,
      'summary:context:' || v_chunk_id::text,
      'ContextSummary',
      'Context Summary',
      left(r.summary, 80),
      jsonb_build_object(
        'chunk_id',    v_chunk_id::text,
        'document_id', v_doc_id::text,
        'tenant_id',   r.tenant_id::text,
        'client_id',   r.client_id::text
      ),
      'active',
      'text-embedding-3-small',
      r.created_at,
      1,
      r.created_at,
      r.updated_at
    )
    on conflict (tenant_id, client_id, node_key) do nothing;

    v_migrated := v_migrated + 1;
  end loop;

  return query select v_migrated, v_skipped;
end;
$$;


-- ── 5. Drop legacy (run MANUALLY after backfill + re-embed complete) ────────
-- Uncomment when ready.
--
-- begin;
--   drop table if exists public.context_summaries cascade;
--   drop function if exists public.upsert_context_summary(uuid, uuid, text, jsonb, jsonb, jsonb) cascade;
-- commit;
