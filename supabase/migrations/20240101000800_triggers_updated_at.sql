-- 08_triggers_updated_at.sql
create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

do $$
begin
  if not exists (select 1 from pg_trigger where tgname='trg_documents_set_updated_at') then
    create trigger trg_documents_set_updated_at
    before update on public.documents
    for each row execute function public.set_updated_at();
  end if;

  if not exists (select 1 from pg_trigger where tgname='trg_kg_nodes_set_updated_at') then
    create trigger trg_kg_nodes_set_updated_at
    before update on public.kg_nodes
    for each row execute function public.set_updated_at();
  end if;

  if not exists (select 1 from pg_trigger where tgname='trg_kg_edges_set_updated_at') then
    create trigger trg_kg_edges_set_updated_at
    before update on public.kg_edges
    for each row execute function public.set_updated_at();
  end if;
end $$;
