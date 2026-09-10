-- 20240101007300_reconcile_embed_drift.sql
-- Phase 1: DETECT embedding drift on kg_nodes (read-only reporting). No re-embed
-- here — this only makes drift observable.
--
-- The vector for a node can go stale relative to its text: upsert_observation /
-- mirror_tag_concept / create_concept update name/description in place but keep a
-- coalesced embedding when no fresh one is supplied (e.g. during an embed outage).
-- Nothing recorded WHAT text a vector was built from, so drift was invisible.
--
-- Mechanism: a BEFORE trigger stamps embedded_hash = md5(name+description) ONLY
-- when the embedding actually changes. So a text edit that keeps a stale vector
-- leaves the OLD hash → drift = (embedded_hash <> md5(current text)). The stamp
-- and the check use the SAME expression, so they're consistent by construction.
--
-- Rows never re-embedded since this ships have embedded_hash = null → treated as
-- "unknown", NOT drift (zero false positives on legacy rows). Chunks are NOT
-- covered: re-ingest is versioned (new doc + fresh chunks embedded together), so
-- a chunk vector is never stale relative to its own content.
--
-- Ends with NOTIFY pgrst 'reload schema'.

alter table public.kg_nodes add column if not exists embedded_hash text;
alter table public.kg_nodes add column if not exists embedded_at   timestamptz;


-- The canonical fingerprint of a node's embeddable text (name + description).
-- Used identically by the stamp trigger and the drift check.
create or replace function public.kg_node_embed_fingerprint(p_name text, p_description text)
returns text
language sql
immutable
as $$
  select md5(coalesce(p_name, '') || E'\n' || coalesce(p_description, ''));
$$;


create or replace function public.kg_nodes_stamp_embed_hash()
returns trigger
language plpgsql
as $$
begin
  -- Stamp only when the embedding is (re)set — never when text changes but the
  -- vector is coalesced/kept. That keeps the old hash so drift is detectable.
  if NEW.embedding is not null
     and (TG_OP = 'INSERT' or NEW.embedding is distinct from OLD.embedding) then
    NEW.embedded_hash := public.kg_node_embed_fingerprint(NEW.name, NEW.description);
    NEW.embedded_at   := now();
  end if;
  return NEW;
end;
$$;

drop trigger if exists trg_kg_nodes_stamp_embed_hash on public.kg_nodes;
create trigger trg_kg_nodes_stamp_embed_hash
  before insert or update on public.kg_nodes
  for each row execute function public.kg_nodes_stamp_embed_hash();


-- ── reconcile_report: drift counts for a tenant (read-only) ──────────────────
create or replace function public.reconcile_report(
  p_tenant_id     uuid,
  p_types         text[] default null,
  p_target_model  text   default 'text-embedding-3-small'
)
returns jsonb
language sql
stable
as $$
  with n as (
    select embedding, embedding_model, embedded_hash, name, description
    from public.kg_nodes
    where tenant_id = p_tenant_id
      and status = 'active'
      and (p_types is null or type::text = any(p_types))
  )
  select jsonb_build_object(
    'null_embedding',   count(*) filter (where embedding is null),
    'model_mismatch',   count(*) filter (
                          where embedding is not null
                            and embedding_model is distinct from p_target_model),
    'content_drift',    count(*) filter (
                          where embedding is not null
                            and embedding_model is not distinct from p_target_model
                            and embedded_hash is not null
                            and embedded_hash <> public.kg_node_embed_fingerprint(name, description)),
    'unknown_hash',     count(*) filter (where embedding is not null and embedded_hash is null),
    'total_vectorized', count(*) filter (where embedding is not null),
    'total_active',     count(*)
  )
  from n;
$$;


-- ── reconcile_candidates: the drifted rows (for a future heal pass) ──────────
create or replace function public.reconcile_candidates(
  p_tenant_id     uuid,
  p_types         text[] default null,
  p_target_model  text   default 'text-embedding-3-small',
  p_limit         int    default 500
)
returns table (id uuid, node_type text, reason text, embed_text text)
language sql
stable
as $$
  select
    n.id,
    n.type::text as node_type,
    case
      when n.embedding is null then 'null_embedding'
      when n.embedding_model is distinct from p_target_model then 'model_mismatch'
      else 'content_drift'
    end as reason,
    coalesce(nullif(btrim(n.name), ''), n.description) as embed_text
  from public.kg_nodes n
  where n.tenant_id = p_tenant_id
    and n.status = 'active'
    and (p_types is null or n.type::text = any(p_types))
    and (
      n.embedding is null
      or n.embedding_model is distinct from p_target_model
      or (n.embedded_hash is not null
          and n.embedded_hash <> public.kg_node_embed_fingerprint(n.name, n.description))
    )
  order by n.updated_at asc
  limit greatest(p_limit, 1);
$$;


notify pgrst, 'reload schema';
