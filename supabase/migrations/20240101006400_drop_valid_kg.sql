-- 64_drop_valid_kg.sql
-- Remove the retired Valid sales/demo bot's dedicated vector KG.
--
-- The bot is deleted in code — agent: /valid/stream + valid_agent / valid_tools /
-- valid_ingest_router; data plane: /ingest/valid + /search/valid. These DB
-- objects are now unreferenced by any running code.
--
-- ⚠ IRREVERSIBLE DATA DROP — deletes the valid_kg_* corpus. Apply deliberately.
-- To keep anything first: `select count(*) from public.valid_kg_nodes;`

drop function if exists public.search_valid_kg_nodes;
drop function if exists public.upsert_valid_kg_node;
drop function if exists public.upsert_valid_kg_edge;
drop function if exists public.upsert_valid_chunk;

drop table if exists public.valid_kg_node_evidence cascade;
drop table if exists public.valid_chunks           cascade;
drop table if exists public.valid_kg_edges         cascade;
drop table if exists public.valid_kg_nodes         cascade;

notify pgrst, 'reload schema';
