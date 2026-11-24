# mindrouter mesh-worker

Standalone async **image → 3D mesh (GLB)** service that the MindRouter gateway
drives via `submit / poll / fetch / cancel`, the 3D analogue of
`../video-worker/`. One GPU-resident model per process, plain HTTP behind
nginx TLS. It is a separate deploy artifact (own venv or container) and does
**not** import the gateway.

## Modes (`MESH_WORKER_MODE`)

| Mode | Model (served name) | License | VRAM (upstream README) | Image |
|---|---|---|---|---|
| `mock` (default) | `mindrouter/mesh-mock` | — | none | none: fastapi + Pillow + numpy |
| `trellis2` | `microsoft/trellis.2-4b` | MIT (RMBG-2.0 avoided, DINOv3 gated) | ≥ 24 GB | `docker/trellis2` |
| `pixal3d` | `tencentarc/pixal3d` | MIT (RMBG-2.0 avoided, DINOv3 gated) | ~18 GB (10-12 GB low-VRAM) | `docker/trellis2` |
| `hunyuan3d21` | `tencent/hunyuan3d-2.1` | Tencent Hunyuan 3D 2.1 Community License | 10 GB shape, 29 GB shape+paint | `docker/hunyuan3d21` |

- **mock** produces a valid, deterministic GLB from the seed. The mesh is a
  subdivided "pillowed" cube with up to 49,152 faces. When
  `texture: true`, the alpha-cropped input image is wrapped onto every side as
  an embedded PNG. The preview is a real render of that mesh. Use it for dev,
  CI and gateway load tests (`MESH_WORKER_MOCK_STAGE_DELAY`).
- **trellis2 / pixal3d**: one shared adapter, `engines/trellis_family.py`.
  Pixal3D vendors the TRELLIS.2 code as its own `pixal3d` package, adds
  camera-projected conditioning, and needs a camera FOV. That FOV comes from
  MoGe-2, or is fixed with `MESH_WORKER_PIXAL3D_FOV`.
- **hunyuan3d21**: runs shape (hy3dshape) then PBR paint (hy3dpaint)
  **in-process**. Upstream `api_server.py` is never used: it has no auth, blocks
  its event loop, and ignores seed, face count and texture.

The real-engine adapters were written from upstream source at pinned commits
(listed in each Dockerfile) and validated on an H200 (aspen1 GPU2) in the
Nov 2026 bake-off: TRELLIS.2, Pixal3D and Hunyuan3D 2.1 each completed 18/18
test images at 100k faces / 2048 px, plus 8k to 1M-face and 4096 px runs. The
fixes that bake-off needed are in the Dockerfiles and adapters; the checklist
below records what was and was not covered.

## Run (mock)

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 127.0.0.1 --port 18401
# slow stages for progress/cancel testing:
MESH_WORKER_MOCK_STAGE_DELAY=2 uvicorn app:app --host 127.0.0.1 --port 18401
```

## HTTP contract

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | open | `{"status":"ok","mode","model","busy","current_job","queue_depth"}`; answers <5 s mid-job (generation runs on its own executor thread). **503** `status:"stuck"` once a job's GPU thread outlives timeout + grace |
| GET | `/version` | open | `{"version","mode","model"}` |
| GET | `/v1/models` | open | `{"object":"list","data":[{"id":<served name>,"object":"model"}]}`, the same shape as video-worker, read by VLLMAdapter |
| GET | `/v1/capabilities` | open | `{"model","inputs":["image"],"formats":["glb"],"supports_texture","texture_sizes":[1024,2048,4096],"min_faces","max_faces","default_faces","options"}` plus `default_texture_size`, `max_image_bytes`, `max_queue`; `options` is the per-job settings schema with this worker's defaults |
| POST | `/v1/meshes` | key | body `{"model","image"(base64 PNG/JPEG/WEBP, RGBA ok),"seed","texture","texture_size","target_faces","options"}` → **202** `{"id","status":"queued"}`; **400** bad input; **413** body too large; **503** queue full (`Retry-After`) |
| GET | `/v1/meshes/{id}` | key | status object (below); **404** unknown id |
| GET | `/v1/meshes/{id}/content?variant=glb\|preview` | key | `model/gltf-binary` (default) or a 512×512 `image/png`; **409** not completed; **400** bad variant; Range/206 capable |
| DELETE | `/v1/meshes/{id}` | key | cancel: queued jobs cancel immediately; in-flight jobs stop at the next stage boundary. Returns **200** with the status object. Idempotent on finished jobs |

**Key** means the `X-Worker-Key` header when `MESH_WORKER_API_KEY` is set
(constant-time compare). An empty key means auth is off, for the rollout window
only. `/v1/models` and `/v1/capabilities` stay open like `/health`: the
gateway's registry discovers models with `VLLMAdapter`, which sends no key, and
they expose only the served model name and limits. Every `/v1/meshes` route
(the work and its outputs) requires the key.

Status object:

```json
{"id": "mjob-…", "status": "queued|in_progress|completed|failed|cancelled",
 "progress": 0-100, "stage": "queued|preprocess|shape|texture|export|done",
 "error": null | {"code", "message"}, "duration_ms": int|null,
 "outputs": null | {"glb": {"size_bytes", "sha256"}, "preview": {"size_bytes", "sha256"}},
 "faces": int|null, "vertices": int|null, "cancel_requested": bool}
```

- `outputs`, `faces` and `vertices` are non-null only when the job is `completed`.
- `progress` never decreases.
- `cancel_requested` is an additive field: true while a DELETE waits for a
  stage boundary.

Error codes:

- `timeout`
- `invalid_input`: e.g. no foreground found
- `invalid_output`: the engine wrote a structurally invalid GLB
- `generation_failed`
- `internal_error`

**Input rules:**

- `image` must decode as PNG, JPEG or WEBP.
  - At most 32 MiB decoded and 64 MP; each side at least 32 px.
  - A `data:` prefix is tolerated.
- An RGBA image whose alpha is not fully opaque is used **as the object mask**,
  so background removal is skipped.
- An RGBA image with no pixel above alpha 204 is rejected with 400 (the TRELLIS
  family would crash on it).
- `seed` is an integer in [0, 2³²−1]; a random seed is used if it is omitted.
- `texture` must be a boolean (default true).
- `texture_size` must be one of `texture_sizes` (default 2048).
- `target_faces` must be in [`min_faces`, `max_faces`].
- `options` (optional) holds per-job model settings, checked against
  `ENGINE_OPTIONS` in `config.py` (unknown names, wrong types and out-of-range
  values are 400); anything left out takes the worker's env default. Hunyuan3D:
  `octree_resolution` 256/384/512, `steps` 5-50, `guidance_scale` 1-10.
  TRELLIS.2: `pipeline_type` 512/1024/1024_cascade/1536_cascade. Pixal3D:
  `pipeline_type` 1024_cascade/1536_cascade. The mock engine takes none. The
  gateway sends the settings an admin chose on Admin → Meshes.
- `model` is not checked against the served name.

**Lifecycle:**

- **Scheduling:** one job runs on the GPU at a time. The FIFO queue is bounded
  (`MESH_WORKER_MAX_QUEUE`), and cancelling a queued job frees its slot at once.
- **Output check:** before a job is marked completed, every engine's output is
  validated (GLB header, chunks, accessors and buffer bounds; the preview must
  be a PNG). `faces` and `vertices` are counted from the GLB itself.
- **Timeout:** a job is marked failed when it hits the timeout. Its GPU thread
  is asked to cancel, and the next job waits until that thread stops, so the GPU
  stays serialized.
- **Expiry:** results expire after `MESH_WORKER_RESULT_TTL_SECONDS`; the files
  are deleted and the id becomes 404.
- **Restart:** state is in memory only. After a restart every id is 404 (the
  gateway requeues), and leftover `mjob-*` files are deleted at startup.

## Config (env)

| Var | Default | Notes |
|---|---|---|
| `MESH_WORKER_MODE` | `mock` | `mock` \| `trellis2` \| `pixal3d` \| `hunyuan3d21` (anything else refuses to start) |
| `MESH_WORKER_MODEL` | per mode (table above) | served model name = the gateway's model name |
| `MESH_WORKER_API_KEY` | — | `X-Worker-Key` secret (host env file only, never in the repo) |
| `MESH_WORKER_OUTPUT_DIR` | `/tmp/mindrouter-mesh-worker` | GLB + preview files (`/data/out` in the images) |
| `MESH_WORKER_MAX_QUEUE` | `8` | queued jobs beyond the running one → 503 |
| `MESH_WORKER_JOB_TIMEOUT_SECONDS` | `900` | per job, from start of execution |
| `MESH_WORKER_STUCK_GRACE_SECONDS` | `300` | after timeout + grace a still-running GPU thread makes `/health` 503 |
| `MESH_WORKER_RESULT_TTL_SECONDS` | `86400` | finished jobs and their files are deleted after this |
| `MESH_WORKER_SWEEP_INTERVAL_SECONDS` | ttl/4, max 60 | TTL sweeper period |
| `MESH_WORKER_MAX_IMAGE_BYTES` / `_MAX_IMAGE_PIXELS` | 32 MiB / 64 MP | input caps |
| `MESH_WORKER_MIN_FACES` / `_MAX_FACES` / `_DEFAULT_FACES` | 1000 / 1,000,000 / 200,000 (hunyuan default 40,000) | face budget |
| `MESH_WORKER_CKPT` | per mode HF id | local dir (TRELLIS family) or HF repo id resolved from the offline cache (Hunyuan: must stay the repo id) |
| `MESH_WORKER_BIREFNET` | `ZhengPeng7/BiRefNet` | TRELLIS family background removal for RGB inputs; any `RMBG` name is refused |
| `MESH_WORKER_DINOV3` | `facebook/dinov3-vitl16-pretrain-lvd1689m` | gated DINOv3 (admin-staged local dir) |
| `MESH_WORKER_PIPELINE_TYPE` | trellis2 `1024_cascade`, pixal3d `1536_cascade` | TRELLIS.2: `512` `1024` `1024_cascade` `1536_cascade`; Pixal3D: the two cascades |
| `MESH_WORKER_LOW_VRAM` | `0` | upstream low-VRAM mode (sub-models shuttle CPU↔GPU per stage) |
| `MESH_WORKER_PIXAL3D_FOV` | `0` | radians; 0 = estimate with MoGe-2 |
| `MESH_WORKER_MOGE` | `Ruicheng/moge-2-vitl` | Pixal3D camera estimator (MIT) |
| `MESH_WORKER_NAF_DIR` | `/opt/NAF` in the image | local clone of `valeoai/NAF` for `torch.hub` `source="local"` |
| `MESH_WORKER_HUNYUAN_REPO` | `/opt/Hunyuan3D-2.1` | checkout holding `hy3dshape/` + `hy3dpaint/` |
| `MESH_WORKER_HY_DINO` | `facebook/dinov2-giant` | paint image encoder (Apache-2.0) |
| `MESH_WORKER_REALESRGAN_CKPT` | `/weights/realesrgan/RealESRGAN_x4plus.pth` in the image | paint super-resolution (BSD-3) |
| `MESH_WORKER_HY_STEPS` / `_HY_OCTREE` | `30` / `256` | shape sampler defaults (upstream Gradio defaults; the pipeline's own are 50 / 384); a job's `options` override them |
| `MESH_WORKER_HY_GUIDANCE` | `5.0` | shape guidance default; a job's `options` override it |
| `MESH_WORKER_LOG_LEVEL` | `INFO` | the worker's own log level (model loading, license guard, jobs) |
| `MESH_WORKER_GLB_WEBP` | `0` | embed textures as WebP (`EXT_texture_webp`, ~5× smaller) instead of PNG |
| `MESH_WORKER_GLB_JPEG` | `1` | embed opaque PBR textures as JPEG (quality 92, 4:4:4) instead of PNG: PNG encoding was 5-10% of a TRELLIS-family job. TRELLIS.2 at 100k faces: ~2.5 s faster per job, GLB 12.9 → 5.5 MB, 42–48 dB PSNR against the PNG maps. Normal maps, and base colours of BLEND/MASK materials, stay PNG |
| `MESH_WORKER_HY_FAST_POST` | `1` | Hunyuan3D: decimate with fast-simplification and fill the texture atlas with a GPU pull-push, instead of upstream's single-core pymeshlab decimation and OpenCV Navier-Stokes inpainting (40k faces / 2048 px: 72 → 25 s per job, same-looking output). `0` = upstream |
| `MESH_WORKER_WARMUP` | `1` | TRELLIS family: one small job at startup |
| `MESH_WORKER_MOCK_STAGE_DELAY` | `0` | mock: seconds per stage |

## Licenses: read before deploying

- **TRELLIS.2** (code + `microsoft/TRELLIS.2-4B` weights) and **Pixal3D**
  (code + `TencentARC/Pixal3D` weights) are **MIT**. Both stacks pull in
  third-party models with their own licenses:
  - **`briaai/RMBG-2.0` is CC BY-NC 4.0 (non-commercial) and is never
    loaded.** Both `pipeline.json` files name it as the background remover.
    `load()` swaps the upstream `BiRefNet` factory for **`ZhengPeng7/BiRefNet`
    (MIT)**: same architecture, and it is already the wrapper's own default.
    The worker refuses to start if `MESH_WORKER_BIREFNET` names any RMBG
    weights, or if the pipeline builds a remover we did not vet (meaning
    `pipeline.json` changed upstream). Inputs that already carry alpha never
    reach a remover; the gateway normally sends RGBA after its own BiRefNet
    matting.
  - **DINOv3** (`facebook/dinov3-vitl16-pretrain-lvd1689m`) is **gated** under
    Meta's DINOv3 License. An admin who accepted that license must stage the
    weights; the worker never downloads them. Pixal3D's `inference.py` loads a
    third-party re-upload (`camenduru/dinov3-…`) instead; the adapter replaces
    that with `MESH_WORKER_DINOV3`.
  - Pixal3D extras: MoGe-2 `Ruicheng/moge-2-vitl` (MIT) and NAF
    `valeoai/NAF` (Apache-2.0). natten and utils3d are build-time
    dependencies.
- **Hunyuan3D 2.1** is under the **Tencent Hunyuan 3D 2.1 Community License**.
  Note that upstream `constants.py` mislabels it "non-commercial". Its terms
  include:
  - "**does not apply in the European Union, United Kingdom and South Korea**".
  - Outputs may **not** be used **to improve any other AI model** (§5(b)).
  - The use restrictions must be passed through in our terms of service, and
    users must be given a copy of the agreement (§3(a), §5(a)).
  - A separate license is required above 1M monthly active users (§4).
  - Tencent claims no rights in the outputs.
  - Its paint model includes Stable Diffusion components under CreativeML Open
    RAIL++-M (see `Notice.txt`).
  - Background removal uses rembg's `u2net` (rembg MIT, U-2-Net Apache-2.0).
  - Paint also loads `facebook/dinov2-giant` (Apache-2.0) and RealESRGAN x4plus
    (BSD-3).
  - **Get a licensing sign-off before exposing this mode to users.**

## Staging weights (offline: `HF_HUB_OFFLINE=1`, no runtime network)

Everything lives under a read-only tree mounted at `/weights` (uid 10001 must
be able to read it). Fetch the weights on a staging host with network, then
copy them to the node.

- **trellis2:**
  - `hf download microsoft/TRELLIS.2-4B --local-dir /weights/microsoft/TRELLIS.2-4B`
  - The sparse-structure decoder comes from a **second repo**
    (`pipeline.json`: `microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16`).
    Place `ss_dec_conv3d_16l8_fp16.{json,safetensors}` at
    `/weights/microsoft/TRELLIS.2-4B/microsoft/TRELLIS-image-large/ckpts/`,
    where the loader looks first. The other option is the HF cache under
    `/weights/hf`.
  - `ZhengPeng7/BiRefNet` → `/weights/ZhengPeng7/BiRefNet`. Take the full
    repo, including its `*.py`: the model loads with `trust_remote_code`.
  - DINOv3 (gated; accept the license first) →
    `/weights/facebook/dinov3-vitl16-pretrain-lvd1689m`.
- **pixal3d:**
  - `hf download TencentARC/Pixal3D --local-dir /weights/TencentARC/Pixal3D`.
    It ships its own sparse-structure decoder.
  - BiRefNet and DINOv3, as for trellis2.
  - `Ruicheng/moge-2-vitl` → `/weights/Ruicheng/moge-2-vitl`
    (`MESH_WORKER_MOGE`).
  - The NAF pretrained checkpoint goes into
    `/weights/torch/hub/checkpoints/` (`TORCH_HOME=/weights/torch`): run
    `torch.hub.load("valeoai/NAF", "naf", pretrained=True)` once on the
    staging host and copy the file.
- **hunyuan3d21:**
  - Stage an **HF cache** (`HF_HOME=/weights/hf`): paint resolves its weights
    with `snapshot_download(repo_id=…)`, which cannot take a local dir.
    - `HF_HOME=/weights/hf hf download tencent/Hunyuan3D-2.1 --include 'hunyuan3d-dit-v2-1/*' 'hunyuan3d-paintpbr-v2-1/*'`
    - `HF_HOME=/weights/hf hf download facebook/dinov2-giant`
  - `RealESRGAN_x4plus.pth` (github.com/xinntao/Real-ESRGAN v0.1.0) →
    `/weights/realesrgan/`.
  - `u2net.onnx` (rembg release v0.0.0) → `/weights/u2net/`.

## Deploy (GPU node)

1. Build **on the node**, never locally:
   `docker build -f docker/trellis2/Dockerfile -t mindrouter-mesh-trellis2:0.1.0 .`
   (context = `mesh-worker/`). Likewise `docker/hunyuan3d21/Dockerfile`.
   - Both images target CUDA 12.4, sm_90 (`TORCH_CUDA_ARCH_LIST=9.0`) and
     Python 3.10, and run as non-root uid 10001.
   - Torch pins: trellis2 uses 2.6.0 (TRELLIS.2 README); hunyuan3d21 uses 2.5.1
     (Hunyuan README).
   - `pip freeze` is saved at `/opt/mesh-worker/pip-freeze.txt` in each image.
2. `deploy/mesh-worker@.service` + `deploy/mesh-worker.env.example`: one
   container per GPU (`--gpus device=N`), published only on
   `127.0.0.1:<MESH_HOST_PORT>`.
3. `deploy/nginx-mesh-worker.conf`: TLS on a gateway-reachable port. Note
   `listen <port> ssl;` **without** `http2` (CVE-2026-123007 fleet rule),
   `client_max_body_size 64m`, `proxy_buffering off`.

## Register with the gateway (after it's up)

Never a direct DB insert. Register through the admin API so the in-memory
registry stays in sync:

```bash
curl -X POST https://mindrouter.uidaho.edu/api/admin/backends/register \
  -H "X-API-Key: $ADMIN_KEY" -H 'Content-Type: application/json' \
  -d '{"name":"aspen1-gpu3-trellis2","url":"https://aspen1.hpc.uidaho.edu:8006","engine":"mesh",
       "max_concurrent":1,"node_id":1,"gpu_indices":[3],"gpu_memory_gb":141}'
```

Discovery (`core/telemetry/registry.py`) reads the open `/v1/models` and
assigns every model on a `mesh` backend the `mesh_generation` modality, so it
shows up in `/v1/meshes/models` and the 3D tab, and never in the LLM catalogs.

## Phase 0 bake-off checklist

Run Nov 2026 on aspen1 GPU2 (H200, 141 GB): 18 ISO3D images per engine at
100k faces / 2048 px; a portrait and one object at 8k / 20k / 40k / 100k
faces (Hunyuan3D); one 1M-face / 4096 px run per engine. Keep further test
and profiling runs at or below 100k faces.

- [x] **Time per asset** (GPU-seconds, median of 18, 100k / 2048): TRELLIS.2
      30 s, Pixal3D 53 s, Hunyuan3D 68 s. Hunyuan3D at production settings:
      72-86 s from 8k to 100k faces (the face count barely matters there).
      1M / 4096: 127 s (TRELLIS.2, 1536 cascade), 124 s (Pixal3D), 689 s
      (Hunyuan3D, octree 512). Shape-only jobs were not timed.
- [x] **Peak VRAM** over the GPU's idle baseline at 100k: TRELLIS.2 34.8 GB,
      Pixal3D 55.1 GB, Hunyuan3D 24.5 GB; resident 16.8 / 19.9 / 14.1 GB.
      The three peaks together (~115 GB) fit one H200; they were not co-run.
- [x] **Failure rate**: no failed jobs. Every GLB loads in trimesh. Hunyuan3D
      gives one watertight part at exactly the target face count; the TRELLIS
      family gives open, multi-part meshes.
- [ ] **Blender / game-engine import**: not done. Outputs render correctly in
      three.js and model-viewer. Orientation: TRELLIS.2 and Hunyuan3D come out
      +Y up with the front at +Z; Pixal3D faced -Z and is now rotated to match
      (`Pixal3DEngine.ROTATION`, checked on the GPU with a face landmarker).
      `MESH_WORKER_GLB_WEBP=1` was not exercised.
- [ ] **Khronos glTF-Validator**: not run (trimesh validity only).
- [x] **Every upstream call** in `engines/` ran on the GPU; the fixes the
      bake-off needed are in the adapters and Dockerfiles. One open detail is
      noted in `hunyuan3d21.py`: the bake-atlas setting's own effect.
- [ ] **No egress**: images run with `HF_HUB_OFFLINE=1` and only local
      weights, but no run used `--network none` (the workers used host
      networking, bound to 127.0.0.1).
- [x] **RMBG is never loaded**: no RMBG weights exist on the node and the hub
      is offline. The "license guard" log line was not visible because the
      worker did not configure logging; it now logs at INFO
      (`MESH_WORKER_LOG_LEVEL`).
- [x] **The status endpoint stays responsive** during 1536-cascade / 4096 /
      1M-face jobs (polled every 0.5 s); `/health` was not timed on its own.
- [ ] **Cancel latency and the timeout path on the GPU**: covered only by the
      mock tests.
- [ ] **Seed determinism**: not checked.
- [x] **Hunyuan3D:**
  - [x] Where the time goes (py-spy; portrait, 40k / 2048; 82 s): mesh
        decimation (pymeshlab, one core) 41 s; texture inpainting 16 s; UV
        unwrapping (xatlas, one core) 4.5 s; export and other CPU work about
        5 s; all neural-network work about 12 s. VRAM is not the bottleneck
        (about 25 GB peak, nothing offloaded). At 1M faces UV unwrapping took
        about 8 minutes, and inpainting the 8192 px atlas of a 4096 px
        texture about 85 s.
  - [x] `use_remesh=False` above 40k paints cleanly (the 100k runs); targets
        at or below 40k come back at exactly the requested face count.
  - [x] 4096 px textures: 32 GB peak VRAM; the CPU inpaint of the 8192 px
        atlas is the cost.
  - [x] Licensing: the gateway refuses Hunyuan3D jobs until an admin accepts
        its license (Admin → Meshes) and shows users the notice it requires.

## Tests

```bash
pytest -q     # mock mode, no GPU, no torch; run from mesh-worker/
```
