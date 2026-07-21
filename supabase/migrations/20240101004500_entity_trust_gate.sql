-- 45_entity_trust_gate.sql
-- STEP 2 of corroboration-based learning: wire the provisional-trust gate.
--
-- node_status has a 'pending_linking' value (01_types.sql) that nothing ever
-- assigned — every write went straight to 'active'. This migration makes
-- freshly-extracted Entity nodes provisional until corroborated, and promotes
-- them to 'active' once seen_count crosses a threshold. Retrieval already
-- filters status='active' (search_kg_nodes, hybrid_search_kg_nodes,
-- _fetch_nodes_by_ids), so provisional entities are auto-excluded from both
-- seed search and hop expansion with NO change to the retrieval filter.
--
-- Implemented as ONE BEFORE INSERT/UPDATE trigger rather than edits to the
-- write RPCs, because:
--   * entities are created via three paths (upsert_kg_node from _link_entities,
--     upsert_entity_with_mention, and the generic RPC) — a trigger covers all;
--   * upsert_kg_node forces status = excluded.status ('active') on conflict, so
--     gating inside one RPC is defeated by another. The trigger is authoritative
--     after the RPC sets NEW, and re-asserts the gate.
--   * it keeps Step 3/4 (which edit those RPCs) un-entangled. Revert = drop trigger.
--
-- SCOPE: Entity nodes only. Chunk nodes are direct evidence (never gated).
-- Observation nodes are agent-authored. CONCEPT nodes are deliberately EXCLUDED:
-- a concept is created by the agent's resolver (create_concept) and read back
-- immediately via nearest_concepts (which filters status='active'); gating it
-- behind a corroboration count would hide a just-created concept from the very
-- resolve-first loop that created it. Extending the gate to concepts is a
-- separate decision (flagged to the caller), not assumed here.
--
-- Promotion-only semantics: never demote an already-active entity; never
-- override an explicit retirement ('archived' / 'merged'). seen_count starts at
-- 1 on insert, so the default threshold of 2 means "appears after a second
-- corroborating ingest."
--
-- Tenant isolation untouched (the trigger reads only NEW/OLD of the row being
-- written). Independently revertable: drop the trigger + function.

create or replace function public.kg_entity_trust_gate()
returns trigger
language plpgsql
as $$
declare
  -- TUNABLE: number of corroborating observations before an extracted entity is
  -- trusted (promoted from provisional to active). Default 2 = "seen at least
  -- twice." Raise for a stricter gate.
  v_promote_threshold constant int := 2;
begin
  -- Only extracted Entity nodes are gated.
  if NEW.type::text <> 'Entity' then
    return NEW;
  end if;

  if TG_OP = 'INSERT' then
    -- A freshly-extracted entity is provisional until corroborated. Respect an
    -- explicit non-active insert (e.g. a caller that already set 'archived').
    if NEW.status = 'active'
       and coalesce(NEW.seen_count, 0) < v_promote_threshold then
      NEW.status := 'pending_linking';
    end if;
    return NEW;
  end if;

  -- UPDATE: promotion-only. Act only while the node is still provisional, and
  -- never clobber an explicit retirement the writer is setting now.
  if TG_OP = 'UPDATE'
     and OLD.status = 'pending_linking'
     and NEW.status not in ('archived', 'merged') then
    if coalesce(NEW.seen_count, 0) >= v_promote_threshold then
      NEW.status := 'active';           -- corroborated → trusted, now retrievable
    else
      -- Still under threshold: keep provisional, overriding any RPC that forced
      -- status back to 'active' (e.g. upsert_kg_node's ON CONFLICT).
      NEW.status := 'pending_linking';
    end if;
  end if;

  return NEW;
end;
$$;

drop trigger if exists kg_entity_trust_gate_trg on public.kg_nodes;
create trigger kg_entity_trust_gate_trg
  before insert or update on public.kg_nodes
  for each row execute function public.kg_entity_trust_gate();
