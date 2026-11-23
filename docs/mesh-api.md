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
3. Create the alias `default-mesh` (Admin → Model Aliases) pointing at the
   chosen model, or set **Admin → 3D → Default model** to a model name.
4. Switch on **Admin → 3D → 3D generation enabled**, then enable users one by
   one on the same page (`users.mesh_generation_enabled`).

Background removal reuses the images feature's matting server
(`img.matting_url`); **Admin → 3D → Remove backgrounds** controls whether mesh
jobs use it, independently of the images feature's own switch.

## Endpoints

All endpoints take the usual `Authorization: Bearer <api key>` header.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v1/meshes/models` | Models and limits (texture sizes, face range, max image size) |
| `POST` | `/v1/meshes` | Create a job (multipart upload or JSON); `202` + the job |
| `GET` | `/v1/meshes` | Your jobs, newest first (`status_filter`, `limit`, `offset`) |
| `GET` | `/v1/meshes/{id}` | Poll one job (includes `queue_position` while queued) |
| `GET` | `/v1/meshes/{id}/content` | Download the GLB; `?variant=preview` for the PNG preview |
| `DELETE` | `/v1/meshes/{id}` | Cancel a running job (refunded), or delete a finished one and its files |

### Create

Send the picture as a multipart file field `image`, or as JSON with `image`
holding base64 (a `data:` URL is accepted). The format is read from the bytes:
PNG, JPEG or WebP, 16–8192 px per side, up to `mesh.max_image_upload_mb`.

| Field | Default | Meaning |
|---|---|---|
| `model` | `mesh.default_model` (`default-mesh`) | A mesh model or alias |
| `texture` | `true` | Paint PBR textures; `false` = shape only (faster, cheaper) |
| `texture_size` | `mesh.default_texture_size` (2048) | One of `mesh.allowed_texture_sizes` |
| `target_faces` | `mesh.default_target_faces` (100,000) | Decimation target, within `mesh.min/max_target_faces` |
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
  "seed": 1872634, "texture": true, "texture_size": 2048, "target_faces": 100000,
  "remove_background": "auto",
  "content_url": null, "preview_url": null,
  "faces": null, "vertices": null, "size_bytes": null,
  "error": null,
  "usage": {"gpu_seconds": 0, "token_equivalent": 10000}
}
```

`status` is `queued`, `in_progress`, `completed`, `failed` or `cancelled`.
`stage` is the worker's step (`queued`, `preprocess`, `shape`, `texture`,
`export`, `done`). `content_url` and `preview_url` appear once the job is
completed (a missing preview never fails a job).

### Errors

| Code | When |
|---|---|
| 400 | Bad picture or option (the body names the field) |
| 403 | 3D generation is not enabled for your account |
| 404 | Unknown model, or a job that is not yours |
| 409 | Content requested before the job finished |
| 429 | Too many jobs in progress (`mesh.max_concurrent_jobs_per_user`) or not enough token quota |
| 503 | The service is off, or the model has no healthy server |
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
worker drives jobs. It runs one job per model at a time (each model has its own
single-GPU worker), so jobs for different models run side by side while a
model's own jobs wait their turn in the queue. If the gateway restarts mid-job, the reconcile loop asks
the worker what happened (by `backend_job_id`) and finishes, resumes or
requeues the job; a worker that lost the job (404) gets it again, up to
`MESH_MAX_RETRIES`.

## Licensing

TRELLIS.2 and Pixal3D are MIT. Their upstream pipelines download RMBG-2.0
(non-commercial), which the worker replaces with BiRefNet (MIT), and Meta's
gated DINOv3, whose weights an administrator must stage after accepting its
license. Hunyuan3D 2.1 is under the Tencent Hunyuan 3D 2.1 Community License:
not licensed in the European Union, the United Kingdom or South Korea, and its
outputs may not be used to improve other AI models. The MindRouter terms of use
carry a Third-Party Model Licenses section covering this.
