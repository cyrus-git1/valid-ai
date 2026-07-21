-- 46_edge_confidence_accumulation.sql
-- STEP 3 of corroboration-based learning: accumulate edge confidence.
--
-- upsert_kg_edge (09_upsert_rpc.sql) did last-writer-wins on weight:
--     weight = coalesce(excluded.weight, old)
-- so repeated co-observation of the same relationship never strengthened it, and
-- (worse) a re-ingest with a *lower* similarity silently WEAKENED a strong edge.
--
-- New rule — count-weighted saturating blend:
--     base   = greatest(strongest single-observation weight ever seen, this obs)
--     count  = seen_count + 1              (post-increment)
--     weight = base + CORR_CAP * (1 - 1/count)
--
-- Properties chosen over running-max or raw-count because:
--   * running max FAILS on fixed-weight edges (co_occurs/mentions/adjacency are
--     always 1.0 — max never grows, so a multi-ingest edge is not heavier);
--   * raw count is unbounded and erases the calibrated [0,1] similarity signal;
--   * the blend keeps `base` as a floor (never weakens — fixes the latent
--     weakening bug) and adds a bounded, diminishing-returns premium so repeated
--     co-observation strictly increases weight (1.0 -> 1.25 -> 1.33 -> ... -> 1.5).
--
-- `base` is persisted in properties.base_weight so weight is recomputed from
-- (base, count) every time — idempotent, no compounding drift.
--
-- Traversal is UNCHANGED: kg_retriever_service._get_neighbour_ids filters
-- `weight >= min_edge_weight` and orders by weight desc; it makes no upper-bound
-- assumption, so weights >1.0 are safe and simply rank corroborated edges first.
-- No search RPC reads edge weight. No check constraint on kg_edges.weight.
--
-- SCOPE: this changes upsert_kg_edge only (the ingest_router edge path:
-- adjacent_to / related_to / mentions / co_occurs). The entity router's own
-- 'mentions' edge write inside upsert_entity_with_mention (39) and the spine's
-- 'about_concept' edge write inside upsert_observation (43) have their OWN
-- (still last-writer-wins) edge logic and are NOT covered here — flagged, can be
-- extended in a parallel change if wanted.
--
-- Same signature as mig 09, so `create or replace` suffices (no drop).
-- Tenant isolation unchanged. Independently revertable (re-apply mig 09 body).

create or replace function public.upsert_kg_edge(
  p_tenant_id uuid,
  p_client_id uuid,
  p_src_id uuid,
  p_dst_id uuid,
  p_rel_type text,
  p_weight float4 default null,
  p_properties jsonb default '{}'::jsonb
)
returns uuid
language plpgsql
as $$
declare
  v_id uuid;
  -- TUNABLE: maximum corroboration premium added on top of the strongest single
  -- observation as count -> infinity. 0.5 keeps a saturated edge (base<=1) <=1.5.
  v_corr_cap constant float4 := 0.5;
begin
  insert into public.kg_edges (
    tenant_id, client_id, src_id, dst_id, rel_type, weight, properties,
    is_active, last_seen_at, seen_count, created_at, updated_at
  )
  values (
    p_tenant_id, p_client_id, p_src_id, p_dst_id, p_rel_type, p_weight,
    -- seed base_weight from the first observation so accumulation has a floor
    coalesce(p_properties, '{}'::jsonb)
      || case when p_weight is not null
              then jsonb_build_object('base_weight', p_weight)
              else '{}'::jsonb end,
    true, now(), 1, now(), now()
  )
  on conflict (tenant_id, client_id, src_id, dst_id, rel_type)
  do update set
    -- persist the running max of the raw (single-observation) strength
    properties = coalesce(public.kg_edges.properties, '{}'::jsonb)
                 || coalesce(excluded.properties, '{}'::jsonb)
                 || jsonb_build_object(
                      'base_weight',
                      greatest(
                        coalesce((public.kg_edges.properties->>'base_weight')::float4,
                                 public.kg_edges.weight, 0),
                        coalesce(excluded.weight, 0)
                      )
                    ),
    -- weight = base + saturating corroboration premium (uses post-increment count)
    weight = (
      greatest(
        coalesce((public.kg_edges.properties->>'base_weight')::float4,
                 public.kg_edges.weight, 0),
        coalesce(excluded.weight, 0)
      )
      + v_corr_cap * (1.0 - 1.0 / (public.kg_edges.seen_count + 1))
    )::float4,
    is_active = true,
    last_seen_at = now(),
    seen_count = public.kg_edges.seen_count + 1,
    updated_at = now()
  returning id into v_id;

  return v_id;
end;
$$;
