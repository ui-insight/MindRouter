# Image-to-Mesh API (`/v1/meshes`)

Turn one picture of an object into a textured 3D model (GLB). Added in 2.9.102
(migration 091). The design is the same as video generation
([video-generation-plan.md](video-generation-plan.md)): a job is saved and
`202` returned at once, a gateway runner (`services/mesh_runner.py`) drives a
GPU worker (`mesh-worker/`) by submit/poll/fetch/cancel, and the result is
stored on the gateway for download.

There is no OpenAI equivalent, so the request and job shapes are MindRouter's
own, modelled on `/v1/videos`.

## Turning it on

The feature ships **off**: deploying the migration changes nothing for users.

1. Deploy a worker on a GPU node (`mesh-worker/README.md`): one Docker image
   per model family, pinned to one GPU, behind the node's nginx with TLS.
   Keep HTTP/2 off on that listener (fleet rule after CVE-2026-123007).
2. Register it through the admin API (never a direct DB insert), with
   `"engine": "mesh"` and `"max_concurrent": 1`. Discovery reads the worker's
   `/v1/models`; every model on a mesh backend gets modality `mesh_generation`.
3. The migration seeds the alias `default-mesh` → `tencent/hunyuan3d-2.1`
   (Hunyuan3D 2.1, the default model). To use another model, repoint the
   alias (Admin → Model Aliases) or pick one under **Admin → Meshes → Default
   model**.
4. **Accept Hunyuan3D's license** on Admin → Meshes (Licenses & terms). Until
   an admin does, Hunyuan3D takes no jobs (`503`) and the 3D tab does not
   offer it. Acceptance is logged (who, when) and can be withdrawn.
5. Switch on **Admin → Meshes → Enable /v1/meshes and the 3D tab**, then
   enable users one by one on the same page (`users.mesh_generation_enabled`).

Background removal reuses the images feature's matting server
(`img.matting_url`); **Admin → Meshes → Remove backgrounds** controls whether
mesh jobs use it, independently of the images feature's own switch.

## Admin → Meshes

One page (`/admin/meshes`) configures the whole service:

- **Service**: the `/v1/meshes` + 3D tab switch (`mesh.enabled`), background
  removal, the default model, allowed texture sizes.
- **Defaults & ranges** (service-wide): default target faces (40,000), min
  (5,000) and max (300,000); default texture size (2048).
- **Limits & cost**: in-flight jobs per user, token cost per job, storage cap,
  largest upload.
- **Models**: per registered model, its own caps (max faces, max texture
  size) and settings, stored in `mesh.model_settings`. Hunyuan3D starts
  capped at 100,000 faces. The settings each model family takes:

  | Model | Setting | Values | Worker default |
  |---|---|---|---|
  | Hunyuan3D 2.1 | `octree_resolution` | 256, 384, 512 | 256 |
  | | `steps` (shape diffusion) | 5–50 | 30 |
  | | `guidance_scale` | 1.0–10.0 | 5.0 |
  | TRELLIS.2 | `pipeline_type` | 512, 1024, 1024_cascade, 1536_cascade | 1024_cascade |
  | Pixal3D | `pipeline_type` | 1024_cascade, 1536_cascade | 1536_cascade |

  A job snapshots its model's settings at submit (`model_options` on the job
  and in the audit row) and sends them to the worker as `options`; the
  worker checks them against its own table (`ENGINE_OPTIONS` in
  `mesh-worker/config.py`, which a contract test keeps equal to the
  gateway's `backend/app/services/mesh_models.py`).
- **Licenses & terms**: each model's license and what it asks of the
  University as a hosted-service operator; Hunyuan3D's acceptance.
- **User access**: per-account enablement.

### Why Hunyuan3D is capped

A Hunyuan3D job is mostly CPU work around a short GPU burst. Profiled on an
H200 (portrait picture, 40,000 faces, 2048 px, production settings, 82 s):
mesh decimation (pymeshlab, one core) 41 s, texture inpainting 16 s, UV
unwrapping (xatlas, one core) 4.5 s, export and other CPU work about 5 s; all
the neural-network work (shape model, multiview paint model, upscaler) about
12 s. Its VRAM use is modest (about 25 GB peak), so nothing spills from the
GPU; the GPU simply waits for those CPU steps. Below 100,000 faces the time
hardly changes (72-86 s from 8,000 to 100,000 faces), but UV unwrapping
grows steeply with face count: a 1M-face, 4096 px job took about 11.5 minutes.

Since 2.9.103 the worker replaces the two slowest steps by default
(`MESH_WORKER_HY_FAST_POST=1`): fast-simplification decimates in about 0.6 s
instead of 41 s, and a GPU pull-push fill pads the texture atlas in about
0.02 s instead of OpenCV's Navier-Stokes inpainting (upstream's mesh-aware
vertex inpainting still runs first). The same 40,000-face, 2048 px jobs now
take about 24 s (portrait 72 → 24 s, capybara 44 → 24 s) with output that
looks the same. UV unwrapping is unchanged, so the 100,000-face cap stays.

## Endpoints

All endpoints take the usual `Authorization: Bearer <api key>` header.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v1/meshes/models` | Models and limits (texture sizes, face range, max image size) |
| `POST` | `/v1/meshes` | Create a job (multipart upload or JSON); `202` + the job |
| `GET` | `/v1/meshes` | Your jobs, newest first (`status_filter`, `limit`, `offset`) |
| `GET` | `/v1/meshes/{id}` | Poll one job (includes `queue_position` while queued) |
| `GET` | `/v1/meshes/{id}/content` | Download the GLB; `?variant=preview` for the PNG preview |
| `DELETE` | `/v1/meshes/{id}` | Cancel a running job (refunded), or delete a finished one and its files. `?cancel_only=true` never deletes: if the job already finished, it is returned unchanged |

### Create

Send the picture as a multipart file field `image`, or as JSON with `image`
holding base64 (a `data:` URL is accepted). The format is read from the bytes:
PNG, JPEG or WebP, 16–8192 px per side, up to `mesh.max_image_upload_mb`.

| Field | Default | Meaning |
|---|---|---|
| `model` | `mesh.default_model` (`default-mesh`) | A mesh model or alias |
| `texture` | `true` | Paint PBR textures; `false` = shape only (faster, cheaper) |
| `texture_size` | `mesh.default_texture_size` (2048) | One of `mesh.allowed_texture_sizes` |
| `target_faces` | `mesh.default_target_faces` (40,000) | Decimation target, within `mesh.min/max_target_faces` and the model's own cap (Hunyuan3D: 100,000) |
| `remove_background` | `auto` | `auto` (cut out unless the picture already has transparency), `on`, `off` |
| `seed` | random | 0 – 2147483647; echoed on the job so a result can be reproduced |

```bash
curl -s https://mindrouter.uidaho.edu/v1/meshes \
  -H "Authorization: Bearer $MINDROUTER_API_KEY" \
  -F image=@chair.jpg -F target_faces=50000
```

```python
import base64, os, time, requests

base = "https://mindrouter.uidaho.edu/v1/meshes"
headers = {"Authorization": f"Bearer {os.environ['MINDROUTER_API_KEY']}"}
with open("chair.jpg", "rb") as fh:
    job = requests.post(base, headers=headers, files={"image": fh}, data={"texture": "true"}).json()
while job["status"] in ("queued", "in_progress"):
    time.sleep(3)
    job = requests.get(f"{base}/{job['id']}", headers=headers).json()
if job["status"] == "completed":
    glb = requests.get(f"{base}/{job['id']}/content", headers=headers)
    open("chair.glb", "wb").write(glb.content)
```

### The job object

```json
{
  "id": "mesh-3f9c0e5b7a1d4c2e8b6f0a9d",
  "object": "mesh",
  "status": "in_progress",
  "stage": "texture",
  "progress": 62.0,
  "model": "microsoft/trellis.2-4b",
  "seed": 1872634, "texture": true, "texture_size": 2048, "target_faces": 40000,
  "remove_background": "auto",
  "content_url": null, "preview_url": null,
  "faces": null, "vertices": null, "size_bytes": null,
  "error": null,
  "usage": {"gpu_seconds": 0, "token_equivalent": 10000},
  "model_options": {"pipeline_type": "1536_cascade"},
  "license": {"name": "MIT License", "url": "https://github.com/microsoft/TRELLIS.2/blob/main/LICENSE"}
}
```

`status` is `queued`, `in_progress`, `completed`, `failed` or `cancelled`.
`stage` is the worker's step (`queued`, `preprocess`, `shape`, `texture`,
`export`, `done`). `content_url` and `preview_url` appear once the job is
completed (a missing preview never fails a job).

Every model returns a binary glTF (`.glb`) in glTF's convention: +Y is up and
the object's front faces +Z, so results from different models line up in the
same viewer. `faces` is at most `target_faces`.

### Errors

| Code | When |
|---|---|
| 400 | Bad picture or option (the body names the field) |
| 403 | 3D generation is not enabled for your account |
| 404 | Unknown model, or a job that is not yours |
| 409 | Content requested before the job finished |
| 429 | Too many jobs in progress (`mesh.max_concurrent_jobs_per_user`) or not enough token quota |
| 503 | The service is off, the model has no healthy server, or its license has not been accepted by an admin |
| 507 | Your 3D storage cap is reached; delete some models |

## Cost, storage and retention

- Each job reserves a flat token-equivalent at submit: `mesh.token_cost_textured`
  (10,000) or `mesh.token_cost_untextured` (5,000). A failed or cancelled job
  is refunded.
- The input picture, GLB and preview count toward `mesh.user_storage_cap_gb`
  (20 GB). Files live in `<MESH_STORAGE_PATH>/<user_id>/` under names derived
  from the job id; stored paths never reach a file operation.
- Finished jobs and their files are deleted after `retention.user_meshes_days`
  (90; Admin → Retention → 3D Meshes). Running jobs are never touched.

## Operations

Gateway settings (each also in `docker-compose.yml`): `MESH_STORAGE_PATH`
(`/data/mesh`, bind-mounted from `MESH_HOST_PATH`), `MESH_RUNNER_ENABLED`,
`MESH_RUNNER_POLL_INTERVAL_SECONDS`, `MESH_WORKER_TIMEOUT_SECONDS`,
`MESH_WORKER_FETCH_TIMEOUT_SECONDS`, `MESH_WORKER_API_KEY` (the shared secret
sent as `X-Worker-Key`; host `.env` only), `MESH_WORKER_FETCH_MAX_MB`,
`MESH_JOB_MAX_WALL_SECONDS`, `MESH_JOB_STALE_HEARTBEAT_SECONDS`,
`MESH_RECONCILE_INTERVAL_SECONDS`, `MESH_RUNNER_LEASE_TTL_SECONDS`,
`MESH_MAX_RETRIES`.

The runner is elected by a Redis lease (`mesh:runner:leader`), so one app
worker drives jobs. It re-checks the service switch and each model's license
before starting queued work: switching the service off, or withdrawing a
license, holds queued jobs (they start when allowed again). Jobs still queued
after `mesh.max_queue_hours` (default 24) are failed and refunded. Models of a
family MindRouter does not know (a worker served under a custom name that is
not recognizably Hunyuan3D, TRELLIS.2 or Pixal3D) are refused with `503`, so
no model can slip past its license gate or settings.

Each mesh worker runs one job at a time, so the runner keeps **one job per
healthy backend**: a model served by one worker takes one job at a time, and
registering a second worker for the same model (for example a second Hunyuan3D
container on the same H200, on its own port) lets it take two. Jobs for
different models always run side by side. Two Hunyuan3D workers on one H200
(fast post-processing on, 40k faces) finished 4 jobs in 70 s against ~96 s
for one worker: 1.36× the throughput, each job ~30-50% slower (24 s → 32-36 s).
Each worker needs its own port and backend registration. If the gateway restarts mid-job, the reconcile loop asks
the worker what happened (by `backend_job_id`) and finishes, resumes or
requeues the job; a worker that lost the job (404) gets it again, up to
`MESH_MAX_RETRIES`.

## Licensing

TRELLIS.2 and Pixal3D are MIT. Their upstream pipelines download RMBG-2.0
(non-commercial), which the worker replaces with BiRefNet (MIT), and Meta's
gated DINOv3, whose weights an administrator must stage after accepting its
license.

Hunyuan3D 2.1 is under the [Tencent Hunyuan 3D 2.1 Community License
Agreement](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/blob/main/LICENSE).
What it asks of the University as the operator of a hosted service:

- **Territory**: the license does not apply in the European Union, the United
  Kingdom or South Korea; the model and its outputs may not be used there.
- **Provider disclosure**: users must be told who provides the service (the
  University of Idaho) and that Tencent is not affiliated with, associated
  with, sponsoring or endorsing it. MindRouter shows this notice on the 3D tab,
  in `/v1/meshes/models` and on every Hunyuan3D job object (`license.notice`).
- **A copy of the license** must be available to users: the notice links to it.
- **Outputs** may not be used to improve any other AI model.
- **Acceptable Use Policy** (part of the license): among other things, no
  impersonation without consent, no high-stakes automated decisions, no
  military use, and machine-generated content placed in public must be
  identified as such.
- **Scale**: above 1 million monthly active users (at 2.1's release) a
  separate license from Tencent is required.

An admin accepts these terms on Admin → Meshes before Hunyuan3D takes jobs.
The MindRouter terms of use carry a Third-Party Model Licenses section
covering this.
