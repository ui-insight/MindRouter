# mindrouter video-worker

Standalone async video-generation service (LTX-2.3) that the MindRouter gateway
drives via `submit / poll / fetch / cancel`. Runs GPU-resident on an H200,
plain HTTP behind nginx — **no ComfyUI, no diffusers**. This is a separate
deploy artifact with its own venv; it does **not** import the gateway.

See `../docs/video-generation-plan.md` for the full design.

## Modes

- `VIDEO_WORKER_MODE=mock` (default) — no GPU, deterministic placeholder MP4.
  Used for dev, CI, and gateway integration before the GPU node is ready.
- `VIDEO_WORKER_MODE=ltx` — real LTX-2.3 on the H200 (wired during Phase 0).

## Run (mock)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 18300
```

## HTTP contract (what the gateway runner expects)

| Method | Path | Purpose |
|---|---|---|
| GET  | `/health` | liveness — answers <5s even mid-render (generation is off the event loop) |
| GET  | `/version` | version + mode + model |
| GET  | `/v1/models` | served model ids |
| GET  | `/v1/capabilities` | preset matrix (sizes, durations, fps, quality, pipelines) |
| POST | `/v1/videos` | submit a job → `202 {"id","status":"queued"}` |
| GET  | `/v1/videos/{id}` | poll → `{status, progress, step, total_steps, error}` |
| GET  | `/v1/videos/{id}/content` | fetch the MP4 (Range/206 capable) |
| DELETE | `/v1/videos/{id}` | cancel (queued: immediate; in-flight: cooperative) |

## Config (env)

| Var | Default | Notes |
|---|---|---|
| `VIDEO_WORKER_MODE` | `mock` | `mock` \| `ltx` |
| `VIDEO_WORKER_MODEL` | `lightricks/ltx-2.3-distilled` | served-model-name |
| `VIDEO_WORKER_OUTPUT_DIR` | `/tmp/mindrouter-video-worker` | artifact dir |
| `VIDEO_WORKER_CKPT_DIR` | — | LTX checkpoint dir (mode=ltx) |
| `VIDEO_WORKER_MOCK_STEP_DELAY` | `0` | mock: seconds/step (load tests) |

## Register with the gateway (after it's up)

Never a direct DB insert — through the admin API so the in-memory registry stays in sync:

```bash
curl -X POST https://mindrouter.uidaho.edu/api/admin/backends/register \
  -H "X-API-Key: $ADMIN_KEY" -H 'Content-Type: application/json' \
  -d '{"name":"aspen1-gpu2-ltx","url":"http://aspen1:18300","engine":"video",
       "max_concurrent":1,"node_id":1,"gpu_indices":[2],"gpu_memory_gb":141}'
```

## Tests

```bash
pytest tests/ -q     # mock mode, no GPU required
```
