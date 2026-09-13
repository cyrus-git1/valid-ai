-- 78_canvas_pinned_precedence.sql — make the pinned-block guard cover what
-- migration 66 said it covered.
--
-- Migration 66's own header states the governance contract:
--
--     "An agent refresh (source='agent') NEVER overwrites a pinned statement
--      — it only updates evidenced / evidence_refs / divergence."
--
-- The implementation guarded exactly one of those fields. `v_name` was held
-- back for a pinned block, but `v_props` was built the same way either way, so
-- an agent upsert against a pinned human block still wrote `stated`, `status`,
-- `confidence` and — unconditionally, not even null-coalesced — `source`.
--
-- That is reachable today, by the writer least equipped to notice. There are
-- two agent writers of canvas blocks. `refresh_canvas` checks `pinned` itself
-- and does the right thing (it withholds a new statement and raises
-- `divergence` instead). `generate_canvas` performs NO pinned check at all: it
-- builds blocks from scratch and upserts each with `statement=`, `stated=` and
-- a flat `status='assumption'`, relying entirely on this function to protect
-- the user's work.
--
-- So, today, a user edits a block (which pins it, source='human'), a later
-- re-grounding pass runs, and the row becomes:
--
--     statement  = the human's sentence      (protected — correct)
--     stated     = the agent's draft         (silently replaced)
--     status     = 'assumption'              (a validated block downgraded)
--     source     = 'agent'                   (the provenance now reads false)
--     pinned     = true
--
-- A block that displays a human's sentence while reporting itself as
-- agent-authored is worse than either half alone: the UI has no way to tell
-- the user which parts of their canvas are still theirs, and `pinned=true`
-- makes the row look protected while three of its fields were not.
--
-- The fix is the contract as written. On a pinned block an agent upsert
-- refreshes ONLY the evidence fields — `evidenced`, `evidence_refs`,
-- `divergence` — and preserves the claim and its provenance. Disagreement has
-- a channel already: `divergence=true` says "the evidence pushes against this"
-- without overwriting what the human asserted, and `refresh_canvas` already
-- computes it that way. A human upsert (source='human') is unaffected, and so
-- is every upsert against an unpinned block: both take the same `else` branch
-- they take today.
--
-- ONE TRADEOFF, MADE DELIBERATELY. `status` is frozen on a pinned block, and it
-- is not purely a claim: `refresh_canvas` derives it from evidence
-- (validated / refined / assumption, from has_support / has_counter), so
-- freezing it means a human-pinned claim cannot be shown as validated by
-- evidence that supports it. The reason it is frozen anyway is that this
-- function cannot tell the two agent writers apart — both arrive as
-- source='agent' — and the other one, `generate_canvas`, sends a flat
-- status='assumption' with no evidence behind it at all. Letting status through
-- to serve the first writer readmits the downgrade from the second.
--
-- Of the two failures, the frozen one is the safer and the more visible: the
-- user sees their own claim sitting at 'assumption' and can move it themselves
-- via /canvas/block, which already accepts `status`. The alternative fails
-- silently and in the direction of losing their work. If evidence-driven status
-- on pinned blocks turns out to matter, the fix is to let the evidence writer
-- say so — a `p_status_is_evidenced` flag, or a separate entry point — not to
-- reopen this branch.
--
-- Also holds the EMBEDDING back on a protected upsert. The caller embeds
-- `body.statement`, so an agent refresh carrying its own statement would store
-- the agent text's vector against the human's `name` — leaving the block
-- semantically searchable as text it does not contain.
--
-- Behaviour-neutral for every path except pinned+agent. No schema change; this
-- is a `create or replace` of the function only.

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
  v_protect      boolean := false;
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

  -- Pinned + agent => the block's CLAIM and PROVENANCE are the human's and stay
  -- the human's. Only the evidence fields below are refreshed.
  v_protect := v_node_id is not null and v_pinned_prev and p_source = 'agent';

  if v_protect then
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
    -- Claim + provenance: held for a protected block, null-coalesced otherwise.
    'stated',        case when v_protect then v_existing->>'stated'
                          else coalesce(p_stated, v_existing->>'stated') end,
    'source',        case when v_protect then coalesce(v_existing->>'source', 'human')
                          else p_source end,
    'status',        case when v_protect then coalesce(v_existing->>'status', 'assumption')
                          else coalesce(p_status, v_existing->>'status', 'assumption') end,
    'confidence',    case when v_protect then coalesce(v_existing->>'confidence', 'low')
                          else coalesce(p_confidence, v_existing->>'confidence', 'low') end,
    'pinned',        v_final_pinned,
    -- Evidence: an agent refresh updates these on a pinned block. That is the
    -- whole point of the refresh, and `divergence` is how it disagrees.
    'evidenced',     coalesce(p_evidenced,  v_existing->>'evidenced'),
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
           -- A protected block keeps its own vector: the incoming embedding was
           -- computed from the agent's statement, which is not what `name` holds.
           embedding    = case when v_protect then v_existing_emb
                               else coalesce(p_embedding, v_existing_emb) end,
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
