-- 40_transcripts_and_responses.sql
-- Adds:
--   1. 'WebVTT' source_type to artifact_type enum
--   2. transcript_metadata    — 1:1 sidecar to documents for transcript-only fields
--   3. transcript_segments    — speaker/timestamp/segment rows (many per document)
--   4. survey_responses       — structured per-participant answers
--   5. A convenience view `transcripts` joining the three transcript types
--      with their metadata, for UI ergonomics.
--
-- Design choice (locked in earlier this session): we do NOT physically
-- separate transcripts from documents. Discrimination via source_type +
-- a sidecar table preserves all existing search/summary/audit infrastructure
-- without fragmentation.

-- ── 1. Enum addition ────────────────────────────────────────────────────────
do $$
begin
  alter type artifact_type add value if not exists 'WebVTT';
exception
  when duplicate_object then null;
end $$;


-- ── 2. transcript_metadata (1:1 sidecar) ───────────────────────────────────
create table if not exists public.transcript_metadata (
  document_id        uuid        primary key references public.documents(id) on delete cascade,
  tenant_id          uuid        not null,
  client_id          uuid,
  study_id           uuid        references public.studies(id) on delete set null,

  recording_url      text,
  duration_seconds   int,
  participant_count  int,
  participants       jsonb       not null default '[]'::jsonb,  -- [{id, name, role, speaker_label}, ...]
  language           text,
  recorded_at        timestamptz,
  source_format      text,                                       -- 'webvtt' | 'srt' | 'whisper-json' | 'vtt+diarized'
  metadata           jsonb       not null default '{}'::jsonb,

  created_at         timestamptz not null default now(),
  updated_at         timestamptz not null default now()
);

create index if not exists transcript_metadata_tenant_idx
  on public.transcript_metadata (tenant_id, client_id);

create index if not exists transcript_metadata_study_idx
  on public.transcript_metadata (study_id)
  where study_id is not null;

create index if not exists transcript_metadata_recorded_at_idx
  on public.transcript_metadata (tenant_id, recorded_at desc nulls last);


-- ── 3. transcript_segments (many per document) ─────────────────────────────
-- Optionally maps each segment to the chunk that ended up containing it
-- (one chunk usually spans several adjacent segments).
create table if not exists public.transcript_segments (
  id              uuid        primary key default gen_random_uuid(),
  document_id     uuid        not null references public.documents(id) on delete cascade,
  chunk_id        uuid        references public.chunks(id) on delete set null,
  tenant_id       uuid        not null,
  client_id       uuid,

  segment_index   int         not null,
  speaker_label   text,                                       -- 'Speaker 1' or participants[i].id
  start_seconds   numeric(9,3) not null,
  end_seconds     numeric(9,3) not null,
  text            text        not null,
  confidence      float4,
  metadata        jsonb       not null default '{}'::jsonb,   -- words, voiceprint id, etc.

  created_at      timestamptz not null default now(),

  unique (document_id, segment_index),
  check (end_seconds >= start_seconds)
);

create index if not exists transcript_segments_doc_idx
  on public.transcript_segments (document_id, segment_index);

create index if not exists transcript_segments_doc_time_idx
  on public.transcript_segments (document_id, start_seconds);

create index if not exists transcript_segments_speaker_idx
  on public.transcript_segments (document_id, speaker_label)
  where speaker_label is not null;

create index if not exists transcript_segments_chunk_idx
  on public.transcript_segments (chunk_id)
  where chunk_id is not null;


-- ── 4. survey_responses (structured per-participant answers) ───────────────
-- Distinct from `survey_outputs` (which stores the generated *questions*).
-- A response captures what one participant answered for one survey.
create table if not exists public.survey_responses (
  id               uuid        primary key default gen_random_uuid(),
  tenant_id        uuid        not null,
  client_id        uuid,
  study_id         uuid        references public.studies(id) on delete set null,
  survey_id        uuid        not null,                       -- refs survey_outputs.id (no FK because survey_outputs auto-expires)

  participant_id   text,                                       -- external identifier; PII-safe alias
  responses        jsonb       not null default '{}'::jsonb,   -- {question_id: answer, ...}
  completion_pct   float4,                                     -- 0.0 - 1.0
  duration_seconds int,
  status           text        not null default 'submitted'
                   check (status in ('in_progress','submitted','flagged','discarded')),

  metadata         jsonb       not null default '{}'::jsonb,

  started_at       timestamptz,
  submitted_at     timestamptz not null default now(),
  created_at       timestamptz not null default now(),
  updated_at       timestamptz not null default now()
);

create index if not exists survey_responses_tenant_survey_idx
  on public.survey_responses (tenant_id, survey_id, submitted_at desc);

create index if not exists survey_responses_study_idx
  on public.survey_responses (study_id, submitted_at desc)
  where study_id is not null;

create index if not exists survey_responses_participant_idx
  on public.survey_responses (tenant_id, participant_id)
  where participant_id is not null;

create index if not exists survey_responses_status_idx
  on public.survey_responses (tenant_id, status);


-- ── 5. updated_at triggers ─────────────────────────────────────────────────
do $$
begin
  if exists (select 1 from pg_proc where proname = 'touch_updated_at') then
    if not exists (select 1 from pg_trigger where tgname = 'transcript_metadata_touch_updated_at') then
      execute $trg$
        create trigger transcript_metadata_touch_updated_at
          before update on public.transcript_metadata
          for each row execute function public.touch_updated_at();
      $trg$;
    end if;
    if not exists (select 1 from pg_trigger where tgname = 'survey_responses_touch_updated_at') then
      execute $trg$
        create trigger survey_responses_touch_updated_at
          before update on public.survey_responses
          for each row execute function public.touch_updated_at();
      $trg$;
    end if;
  end if;
end $$;


-- ── 6. Convenience view ────────────────────────────────────────────────────
-- Read-only view that joins documents + transcript_metadata for the three
-- transcript-shaped source types. Lets queries use `from public.transcripts`
-- as if it were a dedicated table without fragmenting the underlying storage.
create or replace view public.transcripts as
select
  d.id              as document_id,
  d.tenant_id,
  d.client_id,
  d.study_id,
  d.source_type,
  d.title,
  d.status,
  d.is_canonical,
  d.is_pinned,
  d.metadata        as document_metadata,
  d.created_at      as document_created_at,
  d.updated_at      as document_updated_at,
  t.recording_url,
  t.duration_seconds,
  t.participant_count,
  t.participants,
  t.language,
  t.recorded_at,
  t.source_format,
  t.metadata        as transcript_metadata
from public.documents d
left join public.transcript_metadata t on t.document_id = d.id
where d.source_type in ('WebVTT','VideoTranscript','ChatTranscript');
