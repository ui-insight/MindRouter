# Video queue robustness + transparency — Plan

Status: **implementing**
Trigger: a gateway redeploy mid-render orphaned a 90 s job (id 34) in `rendering`
forever. Root cause: re-adoption (`readopt_stale_video_jobs`) runs **once at
startup** and blindly requeues; at the restart the job's heartbeat was only ~12 s
old (< 120 s threshold) so it was skipped, and never reconsidered. The worker-side
`backend_job_id` ("for resume") was never used to check ground truth.

## Goals

1. **No permanently stuck jobs.** A mid-render restart must self-heal.
2. **Proactive, transparent queue management** that periodically checks **ground
   truth** with the worker rather than trusting DB state.
3. **Rich, auto-updating UI**: graphical queue, per-job elapsed time, ETA, and
   re-attach when the user navigates back.

## Part 1 — Reconciliation (no stuck jobs) — CRITICAL

A periodic **reconcile loop** (every `video_reconcile_interval_seconds`, default
20 s; also once at startup) that, for every RENDERING job whose heartbeat is
stale (runner that owned it is gone), checks the worker by `backend_job_id`:

| Worker ground truth | Action |
|---|---|
| `completed` | fetch + complete (recover the render — no rework) |
| `failed` | fail + refund |
| still rendering | **re-adopt** (claim + fresh heartbeat) and resume polling to completion |
| 404 / worker doesn't know it | render was lost → requeue (under retry cap) else fail + refund |
| no `backend_job_id` yet (died before submit) | requeue |
| elapsed > `video_job_max_wall_seconds` | fail + refund (belt-and-suspenders) |

This replaces the one-shot startup readopt. Because it runs continuously and
consults the worker, a redeploy mid-render is recovered (resumed if the worker
kept going, failed cleanly if not) — never stuck.

Implementation: `crud.list_stale_rendering_video_jobs`, `crud.get_backend_snapshot`;
`VideoRunner.reconcile()` / `reconcile_job()`; `run_forever` calls it on a timer;
new setting `video_reconcile_interval_seconds`. Fully unit-tested via the existing
injected fake-repo/fake-worker harness (test_video_runner.py pattern).

## Part 2 — Deploy safety

Reconciliation makes restarts safe. Operationally: **check for active renders
before deploying** (`SELECT ... status IN (queued,planning,rendering,assembling)`)
and prefer to wait/drain. Documented here; the robust guarantee is Part 1.

## Part 3 — Queue transparency API

- `GET /video/api/queue`: the shared render queue — the job in progress (with
  elapsed render seconds) + the waiting line (position each), plus this user's own
  jobs flagged, and an ETA per job.
- ETA = (jobs ahead) × avg recent render seconds + remaining on the in-flight job.
  `crud.get_recent_avg_render_seconds` from the last N completed jobs; per-job
  expected time scales by duration/quality.
- `video_poll` gains `elapsed_seconds`, `eta_seconds` alongside queue position.

## Part 4 — UI: graphical queue + auto-refresh

- A **queue panel** in the Video tab: horizontal bars, one row per active/queued
  job, newest job highlighted if it's yours; the rendering job shows a live
  elapsed timer + progress; queued jobs show position + ETA.
- **Auto-update**: poll `/video/api/queue` on an interval; refresh immediately on
  `visibilitychange`/`focus` (navigate-away-and-back); persist the active job id
  in `localStorage` so returning re-attaches and resumes polling.

## Verification

Unit tests for reconcile (completed/failed/still-rendering/lost/wall-timeout);
queue-position/ETA math. Deploy only after confirming no active renders.
