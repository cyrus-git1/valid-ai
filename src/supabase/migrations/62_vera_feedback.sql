-- 62_vera_feedback.sql
-- Durable Vera chat feedback, moved off the AGENT layer's direct Supabase client
-- (Step 5, Bucket C). Written via POST /feedback. Best-effort — the agent keeps
-- an in-memory ring regardless, so a missing table never fails the request.
create table if not exists public.vera_feedback (
  id           uuid primary key default gen_random_uuid(),
  request_id   text,
  tenant_id    text,
  client_id    text,
  session_id   text,
  rating       text,          -- 'up' | 'down'
  comment      text,
  intent       text,
  message      text,
  response     text,
  created_at   timestamptz default now()
);
create index if not exists vera_feedback_tenant_idx on public.vera_feedback(tenant_id);

notify pgrst, 'reload schema';
