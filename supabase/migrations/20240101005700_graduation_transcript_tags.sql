-- 57_graduation_transcript_tags.sql
-- Own the codebook in the data plane so the agent stops writing Supabase directly.
--
-- Context: the agent shares THIS project (its kg_* spine writes land here) but no
-- migration — agent's or ours — ever created `transcript_tags` or a
-- create_transcript_tag RPC. The agent's "direct graduation write" therefore had
-- no backing table. We create the codebook table here and expose the whole
-- graduation write server-side via graduate_concept(), called by POST /concepts/
-- graduate. The agent then deletes its direct RPC write.
--
-- graduate_concept(): (1) create-or-get the transcript_tags row, idempotent on
-- (tenant_id, lower(label)); (2) stamp external_ref = tag_id on the EXISTING
-- candidate Concept node in place (reuses the link-tag semantics — preserves the
-- node's accumulated about_concept observations); (3) return {tag_id, concept_id,
-- external_ref, created}.
--
-- RLS is enabled with NO policies: only the service role (the data plane) can
-- touch the table — everyone else must go through the API, which is the point.
-- Ends with NOTIFY pgrst 'reload schema'.

create table if not exists public.transcript_tags (
  id                uuid primary key default gen_random_uuid(),
  tenant_id         uuid not null,
  client_id         uuid,
  label             text not null,
  description       text,
  cluster_id        text,
  source_concept_id uuid,                       -- the candidate that graduated into this tag
  evidence_ids      jsonb not null default '[]'::jsonb,
  status            text  not null default 'active',
  created_at        timestamptz not null default now(),
  updated_at        timestamptz not null default now()
);

-- Idempotency key: one codebook entry per (tenant, case-insensitive label).
create unique index if not exists transcript_tags_tenant_label_uniq
  on public.transcript_tags (tenant_id, lower(label));

create index if not exists transcript_tags_tenant_idx
  on public.transcript_tags (tenant_id);

alter table public.transcript_tags enable row level security;
-- (no policies → service_role only; forces all other access through the API)


create or replace function public.graduate_concept(
  p_tenant_id    uuid,
  p_concept_id   uuid,
  p_label        text,
  p_description  text    default null,
  p_client_id    uuid    default null,
  p_cluster_id   text    default null,
  p_evidence_ids jsonb   default '[]'::jsonb
)
returns jsonb
language plpgsql
as $$
declare
  v_tag_id   uuid;
  v_created  boolean := false;
begin
  -- (1) create-or-get the codebook row, idempotent on (tenant, lower(label)).
  select id into v_tag_id
    from public.transcript_tags
   where tenant_id = p_tenant_id
     and lower(label) = lower(p_label)
   limit 1;

  if v_tag_id is null then
    insert into public.transcript_tags (
      tenant_id, client_id, label, description, cluster_id,
      source_concept_id, evidence_ids, status
    ) values (
      p_tenant_id, p_client_id, p_label, p_description, p_cluster_id,
      p_concept_id, coalesce(p_evidence_ids, '[]'::jsonb), 'active'
    )
    returning id into v_tag_id;
    v_created := true;
  else
    -- refresh descriptive fields on re-graduation (label is the stable key).
    update public.transcript_tags
       set description = coalesce(p_description, description),
           cluster_id  = coalesce(p_cluster_id, cluster_id),
           updated_at  = now()
     where id = v_tag_id;
  end if;

  -- (2) stamp external_ref on the existing candidate node in place (link-tag).
  update public.kg_nodes
     set properties = coalesce(properties, '{}'::jsonb)
                      || jsonb_build_object('external_ref', v_tag_id::text),
         updated_at = now()
   where id = p_concept_id
     and tenant_id = p_tenant_id
     and type::text = 'Concept';

  -- (3) return the codebook ref.
  return jsonb_build_object(
    'tag_id',       v_tag_id,
    'concept_id',   p_concept_id,
    'external_ref', v_tag_id::text,
    'created',      v_created
  );
end;
$$;


notify pgrst, 'reload schema';
