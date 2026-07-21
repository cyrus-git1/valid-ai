-- 01_types.sql
do $$
begin
  if not exists (select 1 from pg_type where typname = 'node_status') then
    create type node_status as enum ('active', 'pending_linking', 'archived');
  end if;

  if not exists (select 1 from pg_type where typname = 'artifact_type') then
    create type artifact_type as enum (
      'WebPage',
      'PDF',
      'Image',
      'PowerPoint',
      'Docx',
      'VideoTranscript',
      'ChatTranscript',
      'ChatSnapshot',
      'Chunk'
    );
  end if;

  if not exists (select 1 from pg_type where typname = 'relation_type') then
    create type relation_type as enum (
      'has_chunk',
      'derived_from',
      'references',
      'related_to',
      'supports',
      'duplicate_of'
    );
  end if;
end $$;
