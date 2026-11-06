# 2.9.95 (ocr-load-guard) Deploy Notes — `.env` CHANGE REQUIRED

> **This release must ship together with a change to production's `.env`.** The
> database connection pools in `/opt/mindrouter/.env` are larger than MariaDB
> allows, which is how the 2026-11-05 OCR flood became "Too many connections"
> (MariaDB error 1040) and 500s for every user. The code in this release removes
> the triggers; the `.env` change removes the amplifier. Do both.
>
> No database migration. `alembic upgrade head` is a no-op.

## Why the `.env` change

Each uvicorn worker has its **own** pool of up to `DATABASE_POOL_SIZE +
DATABASE_MAX_OVERFLOW` connections. Production as of 2026-11-06:

| Setting | Production now | Required |
|---|---|---|
| `UVICORN_WORKERS` | 8 | 8 (unchanged) |
| `DATABASE_POOL_SIZE` | 60 | **20** |
| `DATABASE_MAX_OVERFLOW` | 60 | **25** |
| App connections, worst case | 8 × 120 = **960** | 8 × 45 = **360** |

MariaDB allows 512 (`mariadb/custom.cnf`). The budget after the change:

| Client | Max connections |
|---|---|
| App (8 workers × 45) | 360 |
| `mcp` container (1 worker, compose defaults 30 + 20) | 50 |
| Admin / ad-hoc clients, headroom | ~100 |
| **Total** | **~410 of 512** |

The archive engine connects to the separate `mariadb-archive` container and does
not count here.

With the pools inside the limit, a worker that runs out of connections waits up to
10 s (`pool_timeout`) and fails only its own request. Over the limit, MariaDB
refuses the connection to whoever asks next: the dashboard, health checks, other
workers.

**Rule for the future:** keep `UVICORN_WORKERS × (DATABASE_POOL_SIZE +
DATABASE_MAX_OVERFLOW)` plus the `mcp` container's pool well under
`max_connections`. If you add workers, shrink the pools to match.

## Before the deploy: check that 45 connections per worker is enough

The cut is from 120 to 45 connections per worker. A worker that needs more than 45
at once now waits up to 10 s and then fails that request, where before it would have
opened more. Normal load should be far below 45, but check on a busy day first:

```bash
cd /opt/mindrouter
# Connections in use (not Sleep) right now, and the all-time peak since MariaDB
# started. The peak includes the 2026-11-05 outage (~465), so it overstates normal.
docker compose exec -T mariadb sh -c 'mariadb -N -u$MYSQL_USER -p$MYSQL_PASSWORD -e "
  SELECT COUNT(*) FROM information_schema.processlist WHERE command <> \"Sleep\";
  SHOW GLOBAL STATUS LIKE \"Max_used_connections\";"'
```

If busy-hour "in use" counts approach 8 × 45 = 360, use 25 + 30 instead (8 × 55 = 440 +
mcp 50 = 490, still under 512) and say so in the deploy log.

## Step-by-step (prod, `/opt/mindrouter` on mindrouter.uidaho.edu)

```bash
ssh mindrouter@mindrouter.uidaho.edu
cd /opt/mindrouter

# 1. Back up and edit .env (it is not in git). Change only these two lines.
cp -p .env ~/backups/env.before-2.9.95-$(date +%Y%m%d-%H%M%S)
sed -i -e 's/^DATABASE_POOL_SIZE=60$/DATABASE_POOL_SIZE=20/' \
       -e 's/^DATABASE_MAX_OVERFLOW=60$/DATABASE_MAX_OVERFLOW=25/' .env
grep -nE '^(UVICORN_WORKERS|DATABASE_POOL_SIZE|DATABASE_MAX_OVERFLOW)=' .env
# Expect: UVICORN_WORKERS=8, DATABASE_POOL_SIZE=20, DATABASE_MAX_OVERFLOW=25.
# If the old values were not exactly 60/60, sed changed nothing: edit by hand.

# 2. Fetch the code and build the new image WITHOUT restarting the app.
git pull
docker compose build app

# 3. Migrations (none in this release; this is a no-op, run it anyway).
docker compose run --rm --no-deps -T app alembic upgrade head

# 4. Recreate the app. Compose re-reads .env here, so the pool change and
#    the new code go live together. Then restart mcp: it needs an explicit
#    restart on code-only deploys.
docker compose up -d app
docker compose restart mcp
```

The app restart also clears the zombie `pdftoppm` processes left by the outage. Their
parent is the uvicorn master.

## Verify

```bash
# Version flipped
curl -s http://127.0.0.1:8000/status | python3 -c 'import sys,json; print(json.load(sys.stdin)["version"])'
# Expect 2.9.95

# Pools picked up the new size (inside the container)
docker compose exec -T app printenv DATABASE_POOL_SIZE DATABASE_MAX_OVERFLOW
# Expect 20 and 25

# Connection count well under 512 (idle pools hold at most 8 x 20 = 160 for
# the app, plus the mcp container's)
docker compose exec -T mariadb sh -c 'mariadb -N -u$MYSQL_USER -p$MYSQL_PASSWORD -e "SELECT COUNT(*) FROM information_schema.processlist"'

# Key-usage batching works: these should be absent, and api_keys.last_used_at
# should keep moving (within ~15 s) for an active key.
docker compose logs --since 5m app | grep -c key_usage_flush_failed
```

Then exercise OCR once, from a client that sends several documents at once. A burst
over the per-account limit should produce `ocr_waiting` in the logs and, past the
wait line or the 60 s wait, `ocr_refused` with a 429 + `Retry-After` to the client.
It should never produce `Too many connections`.

## Settings this release adds (Admin → OCR, stored in `app_config`)

| Key | Default | Meaning |
|---|---|---|
| `ocr.max_documents_per_user` | 2 | Documents one account may have in progress (0 = no limit) |
| `ocr.max_documents` | 8 | Documents in progress, whole service (0 = no limit) |
| `ocr.max_concurrent_conversions` | 2 | Page conversions at once, per uvicorn worker (8 workers → 16) |
| `ocr.queue_wait_seconds` | 60 | How long an over-limit document waits for a slot (0 = refuse at once; max 600) |
| `ocr.max_waiting_per_user` | 4 | Documents one account may have waiting (1–100) |
| `ocr.max_waiting` | 16 | Documents waiting, whole service (1–1000) |

Also new: each OCR document counts as one request against the account's per-minute
limit and is checked against its token quota before admission. Its chunks no longer
count against the per-minute limit.

No row needs creating. Absent keys use the defaults above.

## Rollback

```bash
cd /opt/mindrouter
cp -p ~/backups/env.before-2.9.95-<timestamp> .env
git checkout <previous tag>          # e.g. v2.9.94
docker compose build app && docker compose up -d app && docker compose restart mcp
```

The code rolls back cleanly: nothing in the schema changed. Rolling back the `.env`
is **optional and not recommended**. The smaller pools are correct for 2.9.94 too.
