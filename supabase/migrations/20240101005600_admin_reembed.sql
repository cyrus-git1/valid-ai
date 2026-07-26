-- 56_admin_reembed.sql
-- Re-embed backfill: nodes written while the embedder was down (OpenAI quota
-- outage) landed with a NULL embedding. They're visible to edge-join reads
-- (rollup / by-study) but invisible to vector recall (nearest / search) until
-- an embedding exists. This pair of RPCs drives an operator-triggered sweep:
--
--   reembed_candidates(tenant, types?, limit)  → the null-embedding rows + the
--        text to embed (name, falling back to description). Oldest first so a
--        chunked backfill drains deterministically.
--   reembed_apply_batch(tenant, ids[], embeddings, model) → fill the embeddings
--        the caller computed, ONLY where still null (idempotent; safe to re-run,
--        never clobbers a concurrently-embedded row). Returns rows updated.
--
-- Embedding itself happens in the app layer (OpenAI); SQL only selects targets
-- and writes the vectors back. Ends with NOTIFY pgrst 'reload schema'.

create or replace function public.reembed_candidates(
  p_tenant_id uuid,
  p_types     text[] default null,
  p_limit     int    default 500
)
returns table (
  id         uuid,
  node_type  text,
  embed_text text
)
language sql
stable
as $$
  select
    n.id,
    n.type::text                              as node_type,
    coalesce(nullif(btrim(n.name), ''), n.description) as embed_text
  from public.kg_nodes n
  where n.tenant_id = p_tenant_id
    and n.embedding is null
    and n.status = 'active'
    and (p_types is null or n.type::text = any(p_types))
    and coalesce(nullif(btrim(n.name), ''), n.description) is not null
  order by n.updated_at asc
  limit greatest(p_limit, 1);
$$;


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
       and tenant_id = p_tenant_id
       and embedding is null;          -- idempotent: only fill nulls
    if found then
      v_count := v_count + 1;
    end if;
  end loop;

  return v_count;
end;
$$;


notify pgrst, 'reload schema';
