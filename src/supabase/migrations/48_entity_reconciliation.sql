-- 48_entity_reconciliation.sql
-- STEP 4 of corroboration-based learning: reconcile entities on write.
--
-- _link_entities (ingest_router.py) mints Entity nodes by a deterministic string
-- key, so two phrasings of one concept ("Acme" / "Acme Corp") become two nodes
-- and their corroboration never pools. This adds the read primitive the write
-- path uses to find an existing near-duplicate BEFORE minting.
--
-- Why a dedicated RPC instead of reusing search_kg_nodes (as the step sketch
-- suggested): search_kg_nodes hardcodes `status = 'active'`, but Step 2's trust
-- gate makes a first-mention entity 'pending_linking'. If reconciliation could
-- only see active nodes, the first two wordings of a concept (both provisional)
-- could never find each other, never pool, and never cross the gate — Step 4
-- would be defeated by Step 2. So this primitive considers BOTH active and
-- pending_linking entities. It also:
--   - restricts to the SAME entity_type (never merge a Person into an Org),
--   - applies the similarity floor in SQL (caller just takes row 0 if present),
--   - stays tenant + client scoped (isolation unchanged).
--
-- Reconciliation DECISION + conservatism (create-new-when-uncertain) live in the
-- caller; this only returns candidates at/above the floor. Independently
-- revertable: drop this function + revert the _link_entities change.

create or replace function public.nearest_entity_candidate(
  p_tenant_id       uuid,
  p_client_id       uuid,
  p_embedding       vector(1536),
  p_entity_type     text    default null,
  p_min_similarity  float4  default 0.93,
  p_top_k           int     default 1,
  p_embedding_model text    default 'text-embedding-3-small'
)
returns table (
  id          uuid,
  node_key    text,
  name        text,
  description text,
  properties  jsonb,
  seen_count  int,
  status      node_status,
  similarity  float4
)
language sql
stable
as $$
  select
    n.id,
    n.node_key,
    n.name,
    n.description,
    n.properties,
    n.seen_count,
    n.status,
    (1 - (n.embedding <=> p_embedding))::float4 as similarity
  from public.kg_nodes n
  where n.tenant_id       = p_tenant_id
    and (p_client_id is null or n.client_id = p_client_id)
    and n.type::text      = 'Entity'
    -- include provisional nodes so the first two wordings of a concept can pool
    and n.status in ('active', 'pending_linking')
    and n.embedding is not null
    and n.embedding_model = p_embedding_model
    and (p_entity_type is null or n.properties->>'entity_type' = p_entity_type)
    and (1 - (n.embedding <=> p_embedding)) >= p_min_similarity
  order by n.embedding <=> p_embedding
  limit greatest(p_top_k, 1);
$$;
