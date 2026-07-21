-- 41_bm25_chunks.sql
-- BM25-style sparse retrieval over chunks.content.
--
-- Postgres ships full-text search via tsvector + GIN. ts_rank_cd gives a
-- BM25-equivalent ranking. We add a generated column so the tsvector is
-- always in sync with content and indexed for O(log n) lookups.
--
-- Why this matters: dense embeddings consistently miss rare keywords,
-- product names, codes, acronyms — exactly the surface area qualitative
-- research has the most of. BM25 catches them; dense catches paraphrases.
-- Fusing them (next migration) closes both failure modes.

alter table public.chunks
  add column if not exists content_tsv tsvector
    generated always as (to_tsvector('english', coalesce(content, ''))) stored;

create index if not exists chunks_content_tsv_idx
  on public.chunks using gin (content_tsv);

-- Note: a composite (tenant_id, content_tsv) GIN would need the btree_gin
-- extension. Skipped — the planner already does tenant_id btree filter then
-- GIN scan on the global index above. Add btree_gin + composite later if
-- query plans show it'd help.
