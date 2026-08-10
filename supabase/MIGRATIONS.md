# Migrations — Supabase CLI workflow

Single directory:

- **`supabase/migrations/`** — the one source of truth, in CLI-timestamp format (`20240101004400_seen_count_scoring.sql`). **This is the only directory the Supabase CLI reads** for `db push`. The timestamp prefix sorts in order and encodes the ordinal (`004400` → migration 44); the `_name` suffix carries the readable label.

> Historical note: a second human-numbered mirror (`src/supabase/migrations/`)
> was maintained for readability but drifted out of sync, so it was removed in
> favour of this single directory. Author new migrations here directly.

The DB was originally built by pasting SQL (out of band), so the CLI's remote
migration-history table doesn't yet know what's applied. Do the one-time
adoption below, then it's a two-command loop forever.

---

## One-time adoption

Set the DB password once so the CLI doesn't prompt per command
(Project Settings → Database → Database password):

```powershell
$env:SUPABASE_DB_PASSWORD = "<db-password>"
```

**1. Tell the CLI which migrations are already on the DB (00–42).** This writes
their versions into the remote history table without re-running them:

```powershell
$applied = @(
  '20240101000000','20240101000100','20240101000200','20240101000300','20240101000400',
  '20240101000500','20240101000600','20240101000700','20240101000800','20240101000900',
  '20240101000930','20240101001000','20240101001100','20240101001200','20240101001300',
  '20240101001400','20240101001500','20240101001600','20240101001700','20240101001800',
  '20240101001900','20240101002000','20240101002100','20240101002200','20240101002300',
  '20240101002400','20240101002500','20240101002600','20240101002700','20240101002800',
  '20240101002900','20240101003000','20240101003100','20240101003200','20240101003300',
  '20240101003400','20240101003500','20240101003600','20240101003700','20240101003800',
  '20240101003900','20240101004000','20240101004100','20240101004200'
)
# one call if your CLI accepts multiple versions:
npx supabase migration repair --status applied $applied
# ...or, if it complains, loop:
foreach ($v in $applied) { npx supabase migration repair --status applied $v }
```

**2. Apply the pending ones (43–47):**

```powershell
npx supabase db push
```

`db push` sees 00–42 as already applied and runs only 43→47, in order.

**3. Verify:**

```powershell
npx supabase migration list        # every version should show applied both sides
```

---

## Going forward (every future change)

```powershell
npx supabase migration new my_change      # creates supabase/migrations/<ts>_my_change.sql
#   ... write SQL in that file ...
npx supabase db push                       # applies only the new pending migration
```

Notes:
- You already have the CLI via scoop, so `supabase …` works too; `npx supabase@latest …` just pins the newest.
- `db push` needs the DB password (from `$env:SUPABASE_DB_PASSWORD` or it prompts).
- Enum-add gotcha (migrations that add an `artifact_type`/`node_status` value and use it in the same file, e.g. 43/47) is safe under `db push` because it applies each migration file in its own transaction with the enum committed first — no `55P04`.
