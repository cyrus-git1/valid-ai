-- 31_pii_vault.sql
-- Per-tenant encrypted PII storage.
--
-- Design:
--   - tenant_keys: stores a wrapped DEK per tenant. The wrapping is done by an
--     external KMS (AWS KMS / GCP KMS / Vault Transit). We store only the
--     ciphertext blob + key_id reference; the application calls the KMS to
--     unwrap on demand for encrypt/decrypt operations.
--   - pii_vault: alias → encrypted original. INSERT-able by app role,
--     SELECT-able only by reveal_role.
--
-- The actual crypto happens in application code, not Postgres. Postgres just
-- stores opaque ciphertext bytes alongside the alias and metadata.


-- ── tenant_keys (wrapped DEK per tenant) ────────────────────────────────────
create table if not exists public.tenant_keys (
  tenant_id        uuid primary key,
  kms_key_id       text not null,                  -- e.g. AWS KMS ARN or GCP key resource name
  wrapped_dek      bytea not null,                 -- ciphertext of the DEK, unwrap via KMS at runtime
  salt             bytea not null,                 -- per-tenant salt for any HMAC-based aliasing
  created_at       timestamptz not null default now(),
  rotated_at       timestamptz not null default now(),
  rotation_version int not null default 1
);

create index if not exists tenant_keys_kms_idx on public.tenant_keys (kms_key_id);


-- ── pii_vault ───────────────────────────────────────────────────────────────
-- Application code:
--   1. Receives raw PII (e.g. "Jane Doe").
--   2. Generates a stable alias (e.g. HMAC(salt, normalized) → "SUBJ_a3f1").
--   3. Encrypts the original with the unwrapped tenant DEK.
--   4. INSERTs (tenant_id, alias, encrypted_original, pii_type).
-- Re-encountering the same PII upserts on (tenant_id, alias) — the alias is
-- deterministic so the same input always maps to the same row.

create table if not exists public.pii_vault (
  tenant_id          uuid not null,
  alias              text not null,
  encrypted_original bytea not null,
  pii_type           text not null,                -- 'person'|'org'|'email'|'phone'|'address'|'id'|...
  rotation_version   int not null default 1,       -- wrap version of the DEK used
  first_seen_at      timestamptz not null default now(),
  last_seen_at       timestamptz not null default now(),
  seen_count         int not null default 1,
  primary key (tenant_id, alias)
);

create index if not exists pii_vault_type_idx on public.pii_vault (tenant_id, pii_type);
create index if not exists pii_vault_first_seen_idx on public.pii_vault (tenant_id, first_seen_at);


-- Upsert helper — called by ingest path when redaction extracts PII.
create or replace function public.upsert_pii_alias(
  p_tenant_id          uuid,
  p_alias              text,
  p_encrypted_original bytea,
  p_pii_type           text,
  p_rotation_version   int default 1
) returns void
language plpgsql
as $$
begin
  insert into public.pii_vault (
    tenant_id, alias, encrypted_original, pii_type, rotation_version,
    first_seen_at, last_seen_at, seen_count
  ) values (
    p_tenant_id, p_alias, p_encrypted_original, p_pii_type, p_rotation_version,
    now(), now(), 1
  ) on conflict (tenant_id, alias) do update set
    last_seen_at = now(),
    seen_count = public.pii_vault.seen_count + 1;
end;
$$;


-- Erase helper — used by /privacy/erase. Removes the vault entry; the caller
-- is responsible for redacting any chunks.pii_annotations that reference
-- this alias.
create or replace function public.erase_pii_alias(
  p_tenant_id uuid,
  p_alias     text
) returns int
language plpgsql
as $$
declare v int;
begin
  delete from public.pii_vault where tenant_id = p_tenant_id and alias = p_alias;
  get diagnostics v = row_count;
  return v;
end;
$$;


-- ── Roles (best-effort: skip silently if role plumbing differs in your env) ─
-- Supabase typically runs as `authenticator` with `service_role` and
-- `anon`/`authenticated` GRANTs. Adjust role names to your environment.
do $$ begin
  -- App role: INSERT/UPDATE only on pii_vault. No SELECT of encrypted_original.
  if exists (select 1 from pg_roles where rolname = 'service_role') then
    execute 'grant insert, update on public.pii_vault to service_role';
    execute 'grant insert, update on public.tenant_keys to service_role';
    -- Explicitly REVOKE select to prevent accidental reads
    execute 'revoke select on public.pii_vault from service_role';
    -- Allow select on tenant_keys (need it to fetch the wrapped DEK)
    execute 'grant select on public.tenant_keys to service_role';
  end if;

  -- Reveal role: gets to SELECT pii_vault. Used by the `/privacy/reveal`
  -- endpoint, which must also write an audit_log REVEAL row.
  if not exists (select 1 from pg_roles where rolname = 'pii_reveal_role') then
    execute 'create role pii_reveal_role';
  end if;
  execute 'grant select on public.pii_vault to pii_reveal_role';
  execute 'grant select on public.tenant_keys to pii_reveal_role';
exception when others then
  -- Role grant errors shouldn't block the migration in environments where
  -- role plumbing is managed externally (e.g. Supabase managed).
  raise notice 'Role grants partially skipped: %', sqlerrm;
end $$;
