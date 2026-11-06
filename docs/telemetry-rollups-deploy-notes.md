# 2.9.97 (telemetry-5m-rollups) Deploy Notes

> No `.env` change and no app-DB migration (`alembic upgrade head` is a no-op).
> The three new archive tables are created at startup. What changes on prod is
> **how much raw telemetry the app DB keeps**: read "Disk" before deploying.

## What changes on prod

| | Before (2.9.94) | After |
|---|---|---|
| Raw telemetry in the app DB | 30 days (hidden `TELEMETRY_RETENTION_DAYS` default, hourly registry loop) | `retention.telemetry.tier1_days` (prod: **90**, set 2026-10-30) |
| Long-term telemetry | none (the "archive" step only deleted) | 5-minute averages in the archive DB for `tier2_days` (prod: **730**) |
| Who deletes raw telemetry | registry loop (every worker) + retention cycle | retention cycle only (plus the Purge tab and node/backend deletion) |

## Disk

Raw retention goes from 30 to 90 days. The app DB's telemetry roughly triples
over the 60 days after the deploy: about **+22 GB** at ~11 GB/month (2026-10-30
estimate). The app DB lives on `/var`.

```bash
df -h /var /archivedb
```

If `/var` cannot take +22 GB, set Admin → Retention → Telemetry Tier 1 back to
30 **before** deploying. Averages are still kept for 730 days either way.

## First cycles after the deploy

- Each uvicorn worker runs `create_all` on the archive DB at startup.
  `archived_node_telemetry_5m`, `archived_gpu_device_telemetry_5m` and
  `archived_backend_telemetry_5m` appear on the first restart. The workers race
  to create them, which is handled: no `archive_db_init_failed` should appear.
- The first retention cycle (one `retention.cleanup_interval` after startup, 1 h by default) starts the backfill from the
  oldest live reading, 14 days per table per cycle. The ~30 days of live
  history take about three cycles. Each cycle reads ~14 days × 3 tables in
  6-hour windows on the app DB: a few minutes of steady reads, not a spike.
- Raw readings past Tier 1 are deleted only after their bucket is archived.

## Verify

```bash
cd /opt/mindrouter
docker compose logs --since 2h app | grep -E 'archive_db_init|retention_telemetry_(rollup|delete|backstop|watermark)' | tail -20
# Expect archive_db_initialized and retention_telemetry_rollup lines;
# no retention_telemetry_rollup_failed, _delete_failed or _backstop_delete.
```

Admin → Retention → Archive stats lists the three `*_5m` tables, with row
counts growing each cycle.

## If the archive breaks later

A table whose rollup fails keeps its raw readings (they have no averages yet)
and logs `retention_telemetry_rollup_failed` every cycle. After 14 days of
failures, readings more than 14 days past Tier 1 are deleted anyway, with
`retention_telemetry_backstop_delete` logged as an **error**. Those readings
are lost, never averaged. Watch for either event.

## Rollback

```bash
cd /opt/mindrouter
git checkout <previous tag>
docker compose build app && docker compose up -d app && docker compose restart mcp
```

The `*_5m` tables stay in the archive DB, unused by the old code. The old
registry loop resumes deleting raw telemetry at 30 days.
