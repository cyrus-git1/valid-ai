-- 65_canvas_blocks.sql  (part 1 of 2) — enum value ONLY.
--
-- A new enum value must be COMMITTED before it can be used, or Postgres raises
-- 55P04 "unsafe use of new value" (the DB was originally built by pasting SQL
-- out of band, so this never surfaced until `supabase db push` ran the whole
-- migration in one transaction). So the CanvasBlock artifact_type is added here
-- in its own migration/transaction; the indexes + RPCs that USE it live in
-- 20240101006600_canvas_blocks.sql, which the CLI applies as a later transaction.

do $$
begin
  alter type artifact_type add value if not exists 'CanvasBlock';
exception
  when duplicate_object then null;
end $$;
