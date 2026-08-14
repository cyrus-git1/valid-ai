-- 67_hypotheses.sql  (part 1 of 2) — enum value ONLY.
-- A new enum value must be committed before use (Postgres 55P04). The Hypothesis
-- artifact_type is added here; the indexes + RPCs that use it live in
-- 20240101006800_hypotheses.sql (a later transaction).
do $$
begin
  alter type artifact_type add value if not exists 'Hypothesis';
exception
  when duplicate_object then null;
end $$;
