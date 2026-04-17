-- ============================================================================
-- Valid Data Plane — DESTRUCTIVE RESET
-- ============================================================================
-- Drops ALL tables, functions, types, and triggers. Run this to wipe the
-- database clean before running schema.sql to rebuild from scratch.
--
-- WARNING: This deletes ALL data. Back up first.
--
-- Usage:
--   psql -f reset.sql
--   psql -f schema.sql
-- ============================================================================

begin;

-- Tables (dependents before dependencies)
drop table if exists public.audit_log           cascade;
drop table if exists public.ingest_jobs         cascade;
drop table if exists public.api_keys            cascade;
drop table if exists public.memory_change_log   cascade;
drop table if exists public.memory_state        cascade;
drop table if exists public.survey_outputs      cascade;
drop table if exists public.context_summaries   cascade;  -- legacy, may not exist
drop table if exists public.kg_edge_evidence    cascade;
drop table if exists public.kg_node_evidence    cascade;
drop table if exists public.kg_edges            cascade;
drop table if exists public.kg_nodes            cascade;
drop table if exists public.chunks              cascade;
drop table if exists public.documents           cascade;

-- Functions (all known signatures)
drop function if exists public.search_kg_nodes            cascade;
drop function if exists public.bump_memory_state_dual     cascade;
drop function if exists public.bump_memory_state          cascade;
drop function if exists public.mark_summary_fresh         cascade;
drop function if exists public.upsert_context_summary     cascade;
drop function if exists public.upsert_kg_node             cascade;
drop function if exists public.upsert_kg_edge             cascade;
drop function if exists public.upsert_chunk               cascade;
drop function if exists public.fetch_chunks_with_embeddings cascade;
drop function if exists public.prune_kg                   cascade;
drop function if exists public.prune_archive_stale_edges  cascade;
drop function if exists public.prune_archive_stale_nodes  cascade;
drop function if exists public.prune_trim_edge_evidence   cascade;
drop function if exists public.prune_trim_node_evidence   cascade;
drop function if exists public.cleanup_orphaned_kg_nodes  cascade;
drop function if exists public.cleanup_expired_survey_outputs cascade;
drop function if exists public.touch_api_key              cascade;
drop function if exists public.set_updated_at             cascade;
drop function if exists public.backfill_context_summaries_to_chunks cascade;

-- Types
drop type if exists relation_type cascade;
drop type if exists artifact_type cascade;
drop type if exists node_status   cascade;

commit;
