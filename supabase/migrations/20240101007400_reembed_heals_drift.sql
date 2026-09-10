-- 20240101007400_reembed_heals_drift.sql
-- Phase 2 (heal): fold the drift classes reconcile DETECTS into what reembed
-- HEALS. Closes the loop: /admin/reconcile reports → /admin/reembed fixes.
--
-- Before: reembed only picked up NULL embeddings, and reembed_apply_batch filled
-- only null rows (safe for outage backfill, but couldn't heal a stale/mismatched
-- vector). Now:
--   reembed_candidates  → null_embedding OR model_mismatch OR content_drift
--        (identical selection to reconcile_candidates; gains p_target_model).
--   reembed_apply_batch → OVERWRITES the vector for the given ids (not just nulls)
--        and sets embedding_model. The stamp trigger (mig 7300) then refreshes
--        embedded_hash automatically because the embedding changed → drift clears.
--
-- Legacy rows with embedded_hash = null are NOT content_drift, so a heal pass does
-- NOT needlessly re-embed untouched history — only genuine null/mismatch/drift.
--
-- Ends with NOTIFY pgrst 'reload schema'.

-- reembed_candidates: broaden selection + add target-model param (signature
-- changes → drop the old (uuid, text[], int) overload first to avoid ambiguity).
drop function if exists public.reembed_candidates(uuid, text[], int);

create or replace function public.reembed_candidates(
  p_tenant_id    uuid,
  p_types        text[] default null,
  p_target_model text   default 'text-embedding-3-small',
  p_limit        int    default 500
)
returns table (id uuid, node_type text, embed_text text)
language sql
stable
as $$
  select
    n.id,
    n.type::text                                       as node_type,
    coalesce(nullif(btrim(n.name), ''), n.description) as embed_text
  from public.kg_nodes n
  where n.tenant_id = p_tenant_id
    and n.status = 'active'
    and (p_types is null or n.type::text = any(p_types))
    and coalesce(nullif(btrim(n.name), ''), n.description) is not null
    and (
      n.embedding is null
      or n.embedding_model is distinct from p_target_model
      or (n.embedded_hash is not null
          and n.embedded_hash <> public.kg_node_embed_fingerprint(n.name, n.description))
    )
  order by n.updated_at asc
  limit greatest(p_limit, 1);
$$;


-- reembed_apply_batch: overwrite the vector (heal), not fill-only. Same signature
-- → create or replace. The BEFORE trigger restamps embedded_hash on the change.
create or replace function public.reembed_apply_batch(
  p_tenant_id       uuid,
  p_ids             uuid[],
  p_embeddings      jsonb,        -- array of float arrays, aligned 1:1 to p_ids
  p_embedding_model text default 'text-embedding-3-small'
)
returns int
language plpgsql
as $$
declare
  v_count int := 0;
  i       int;
begin
  if p_ids is null or array_length(p_ids, 1) is null then
    return 0;
  end if;

  for i in 1 .. array_length(p_ids, 1) loop
    update public.kg_nodes
       set embedding       = ((p_embeddings -> (i - 1))::text)::vector,
           embedding_model = coalesce(p_embedding_model, 'text-embedding-3-small'),
           updated_at      = now()
     where id = p_ids[i]
       and tenant_id = p_tenant_id;   -- heal: overwrite null/stale/mismatched alike
    if found then
      v_count := v_count + 1;
    end if;
  end loop;

  return v_count;
end;
$$;


notify pgrst, 'reload schema';
