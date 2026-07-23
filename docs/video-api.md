# MindRouter Video Generation API (`/v1/videos`)

**Last updated: 2026-07-23**

Base URL: `https://mindrouter.uidaho.edu`

## What this API is (and is not)

MindRouter is a **provider of single-shot generative primitives**. For video, that
means it renders **one clip per request** — a text prompt (optionally guided by one
or two keyframe images) becomes a single MP4 with a synchronized audio track.

MindRouter does **not** do multi-shot assembly, stitching, timeline editing, or
cross-shot coherence. Those belong to the **separate studio/storyboarding
application** that consumes this API — it stores its own storyboards, submits one
`POST /v1/videos` per shot, and stitches the returned MP4s (ffmpeg) on its own side.
Keep that boundary in mind: **MindRouter renders clips; your app assembles them.**

> **Changed in this release.** Image-guided (keyframe) video is now available over
> the API. Earlier docs described image conditioning as a "future" addition — that
> is no longer true. You upload a reference image with `POST /v1/videos/assets`, then
> pass the returned asset id as `start_image_asset_id` and/or `end_image_asset_id` on
> `POST /v1/videos`. See [Image-guided video](#image-guided-video-keyframes).

## Authentication

Every request needs an API key, sent as **either** header:

```
X-API-Key: <key>
```
```
Authorization: Bearer <key>
```

The caller's account must have `video_generation_enabled`, and the service must be
enabled server-side (`vid.enabled`). Otherwise you get `403` (account) or `503`
(service disabled).

## The async model: submit → poll → fetch

A render outlives an HTTP request, so the API is asynchronous:

1. **Submit** — `POST /v1/videos` validates and enqueues a job, then returns **`202
   Accepted`** immediately with a **job object** in status `queued`. It never returns
   the video inline.
2. **Poll** — `GET /v1/videos/{id}` until `status` is `completed` or `failed`
   (or `cancelled`). Poll every 2–5s with backoff.
3. **Fetch** — once `completed`, download the MP4 from `GET /v1/videos/{id}/content`.

**One clip renders at a time.** The GPU serializes renders into a single queue, so a
job can sit in `queued` behind other users' work before it moves to `in_progress`.
Design your client for a queue and show honest progress/ETA rather than blocking.

---

## Endpoints

### `POST /v1/videos` — create a text-to-video job

`Content-Type: application/json`. Only the fields below are honored. Unknown or
mistyped fields are surfaced according to the server's `field_validation` setting
(logged by default; some fields carry a hint telling you the supported alternative —
e.g. sending `duration` tells you to use `seconds`, sending `width`/`height` tells
you to use `size`).

| field | type | required | default | notes |
|---|---|---|---|---|
| `prompt` | string | **yes** | — | The scene description. Must be non-empty. |
| `model` | string | no | `vid.default_model` (`lightricks/ltx-2.3-distilled`) | Served video model id. Must exist; aliases are resolved. |
| `size` | string | no | `vid.default_size` (`1280x704`) | `"WIDTHxHEIGHT"` from the fixed preset menu (see [Sizes & durations](#sizes-durations-fps)). Off-menu → `400`. |
| `seconds` | int / string | no | `vid.default_seconds` (`5`) | Whole number of seconds, `4`–`30` inclusive (`vid.min_seconds`..`vid.max_total_seconds`). Non-integer or out-of-range → `400`. |
| `fps` | int | no | `24` | Accepted but **not validated**: any value is stored and echoed back, but only `24` is actually rendered. Off-`24` values are neither honored nor rejected (no `400`). |
| `quality` | string | no | `standard` | One of `draft` \| `standard` \| `final`. Controls denoising steps/guidance and the token cost multiplier. Anything else → `400`. |
| `seed` | int | no | random | Optional determinism seed. |
| `negative_prompt` | string | no | — | Optional. |
| `callback_url` | string | no | — | Optional webhook fired on completion. |
| `start_image_asset_id` | int | no | — | Asset id (from `POST /v1/videos/assets`) used as the **first-frame** keyframe. Must be an asset you own → else `400`. |
| `end_image_asset_id` | int | no | — | Asset id used as the **last-frame** keyframe. Must be an asset you own → else `400`. |

Fields other OpenAI-video clients send (`user`, `response_format`, `metadata`) are
accepted and harmlessly ignored.

**Response:** `202 Accepted` with the [job object](#the-job-object), `status:
"queued"`.

**Errors:**

| status | when |
|---|---|
| `400` | missing/empty `prompt`; off-menu `size`; non-integer or out-of-range `seconds`; bad `quality`; invalid `start_image_asset_id`/`end_image_asset_id` (not found or not owned by you); invalid JSON body |
| `401` | missing/invalid/inactive/expired API key |
| `403` | your account does not have video generation enabled |
| `404` | `model` does not exist |
| `429` | over the [per-user concurrency cap](#limits-quotas--costs), or insufficient token quota for this render |
| `503` | video generation is disabled service-wide |
| `507` | you are at/over your [storage cap](#limits-quotas--costs) |

### `POST /v1/videos/assets` — upload a keyframe image

`Content-Type: multipart/form-data`. **This is how you supply keyframes over the
API** — the create endpoint takes asset *ids*, not raw image bytes, so you upload the
image here first and pass back the id it returns.

| form field | type | required | notes |
|---|---|---|---|
| `image` | file | **yes** | An image file. Content-Type must start with `image/`. PNG, JPEG, and WebP are recognized (others stored as `.png`). Max size `vid.max_image_upload_mb` (default **10 MB**). |

**Response:** `201 Created`

```json
{
  "id": 12345,
  "object": "video.asset",
  "kind": "reference",
  "content_type": "image/png",
  "size_bytes": 482013
}
```

Use `id` as `start_image_asset_id` / `end_image_asset_id` on `POST /v1/videos`.
Reference images count toward your storage cap.

**Errors:** `400` (missing/non-image file, or over the size limit), `401`
(missing/invalid/inactive/expired API key), `403` (account not enabled), `503`
(service disabled), `507` (upload would exceed your storage cap).

### `GET /v1/videos/{id}` — poll one job

**Response:** the [job object](#the-job-object). Returns `404` for an id you do not
own (no existence leak — other users' jobs are indistinguishable from missing ones).

### `GET /v1/videos/{id}/content` — fetch the MP4

Streams the rendered file from disk. Sets `Accept-Ranges: bytes` and returns `206
Partial Content` for `Range` requests, so a browser `<video>` element can scrub.

| status | when |
|---|---|
| `200` / `206` | success (full body / range) |
| `401` | missing/invalid/inactive/expired API key |
| `409` | job exists but the video is not ready yet (not `completed`) |
| `404` | job not found / not owned, or the file is missing on disk |

### `GET /v1/videos` — list your jobs

Query params: `status_filter` (optional; e.g. `queued`, `in_progress`, `completed`,
`failed`, `cancelled`), `limit` (default `50`), `offset` (default `0`).

**Response:**

```json
{ "object": "list", "data": [ { …job… } ], "total": 42 }
```

(List entries carry the job fields but omit the per-project `model`/`size`/`fps`/
`quality` echo — poll `GET /v1/videos/{id}` for those.)

### `DELETE /v1/videos/{id}` — cancel or delete

Behavior depends on the job's state:

- **Terminal job** (`completed`, `failed`, or `cancelled`) — **deletes** the job's
  rows and artifact files and **reclaims the per-user storage** they occupied.
  Returns `{"id": "...", "object": "video", "deleted": true}`. This is how you free
  space against the 50 GB storage cap over the API.
- **Running or queued job** — requests cancellation (propagated to the worker;
  a queued job's reserved quota is refunded). Returns the (updated) job object.

`404` if not found/owned.

### `GET /v1/videos/models` — capability discovery

Render your UI controls from this so the preset matrix has one source of truth.

**Response:**

```json
{
  "object": "list",
  "data": [
    {
      "id": "lightricks/ltx-2.3-distilled",
      "supported_sizes": ["1280x704", "704x1280", "1024x576", "768x448"],
      "min_seconds": 4,
      "max_seconds": 30,
      "supported_fps": [24],
      "supported_qualities": ["draft", "standard", "final"],
      "supports_text_to_video": true,
      "supports_image_to_video": false,
      "supports_keyframes": false,
      "max_shots": 1,
      "license_notice": "AI-generated"
    }
  ]
}
```

> **Note on the capability flags.** `supports_image_to_video` and
> `supports_keyframes` report the model runner's advertised text-to-video mode and
> currently read `false`. The **create endpoint nonetheless accepts**
> `start_image_asset_id` / `end_image_asset_id` today (see below) — the flags are the
> runner's self-description, not a gate on the request fields. Treat the fields as the
> authoritative way to do keyframe conditioning over the API.

---

## Image-guided video (keyframes)

To condition a clip on reference images:

1. `POST /v1/videos/assets` with the image → get an asset `id`.
2. `POST /v1/videos` with `start_image_asset_id` (first frame), `end_image_asset_id`
   (last frame), or both — alongside the usual `prompt`/`size`/`seconds`.

Both asset ids must reference images **you** uploaded; anything else returns `400`.
You can supply just a start frame, just an end frame, or both to bracket the motion.

---

## The job object

```json
{
  "id": "vid-<24 hex>",
  "object": "video",
  "status": "queued | in_progress | completed | failed | cancelled",
  "progress": 0,
  "created_at": 1753305600,
  "started_at": null,
  "completed_at": null,
  "expires_at": null,
  "content_url": null,
  "error": null,
  "usage": {
    "duration_seconds": 5.0,
    "gpu_seconds": 0,
    "token_equivalent": 10000
  },

  "model": "lightricks/ltx-2.3-distilled",
  "size": "1280x704",
  "fps": 24,
  "quality": "standard"
}
```

| field | meaning |
|---|---|
| `id` | Job id, `vid-` + 24 hex chars. |
| `object` | Always `"video"`. |
| `status` | External status (below). OpenAI-video-shaped — `in_progress`, not `rendering` — so stock SDK polling loops work. |
| `progress` | 0–100, one decimal. |
| `created_at` / `started_at` / `completed_at` / `expires_at` | Unix seconds, or `null` if not yet reached. |
| `content_url` | `"/v1/videos/{id}/content"` once the output exists; `null` until then. |
| `error` | `{ "code", "message" }` on failure, else `null`. |
| `usage.duration_seconds` | Clip length rendered. |
| `usage.gpu_seconds` | GPU time consumed (populated by the runner). |
| `usage.token_equivalent` | Tokens charged for this render (reserved up front; refunded on fail/cancel). |
| `model` / `size` / `fps` / `quality` | Echoed on single-job responses (`POST`, `GET /{id}`); omitted in list rows. |

### Statuses

| external `status` | meaning |
|---|---|
| `queued` | Accepted, waiting for the single render slot. |
| `in_progress` | Actively planning, rendering, or assembling. (Internal `planning`/`rendering`/`assembling` all map here.) |
| `completed` | Done; fetch `content_url`. |
| `failed` | Render failed; see `error`. Quota refunded. |
| `cancelled` | Cancelled via `DELETE`. Quota refunded. |

---

## Sizes, durations, fps

- **Sizes** are a fixed preset menu (torch.compile shape set), default
  `1280x704`, `704x1280`, `1024x576`, `768x448`. Off-menu sizes are rejected.
- **Duration** (`seconds`) is a **continuous whole-second value** from `4` to `30`
  inclusive (defaults may be tuned server-side; always read `GET /v1/videos/models`
  for the live `min_seconds`/`max_seconds`). Fractional seconds are rejected. Frame
  count is derived as `24 × seconds + 1`.
- **fps** is `24`.
- Every clip includes a **synchronized audio track** (the model generates audio
  natively).

Always read `GET /v1/videos/models` rather than hardcoding these — the preset matrix
is admin-tunable.

## Limits, quotas & costs

- **Serialized queue.** One clip renders at a time across the whole service.
- **Per-user concurrency cap.** By default **1** in-flight job per user
  (`vid.max_concurrent_jobs_per_user`). Submitting while at the cap returns `429`.
- **Per-user storage cap.** Default **50 GB** (`vid.user_storage_cap_gb`), counting
  rendered clips **and** uploaded reference images. Being at/over the cap returns
  **`507 Insufficient Storage`** on both `POST /v1/videos` and `POST
  /v1/videos/assets`. The check gates on *current* usage, so a single in-flight clip
  can push slightly over; the next submit is then blocked until you delete videos to
  free space. Use `DELETE /v1/videos/{id}` to reclaim space.
- **Token cost.** Each render reserves tokens up front from your group budget:
  `token_equivalent = seconds × per_second_rate × quality_multiplier ×
  resolution_multiplier` (defaults: `per_second_rate` 2000; quality `draft` 0.5 /
  `standard` 1.0 / `final` 2.0; resolution multiplier 1.0 unless configured).
  Insufficient budget → `429`. The reservation is **refunded** if the job fails or is
  cancelled. The charged amount is echoed as `usage.token_equivalent`.

---

## Worked example: keyframe-guided clip, end to end

Upload a keyframe → create an image-guided clip → poll → fetch.

### curl

```bash
BASE=https://mindrouter.uidaho.edu
KEY=your_api_key_here

# 1) Upload a reference image → get an asset id.
ASSET_ID=$(curl -s -X POST "$BASE/v1/videos/assets" \
  -H "X-API-Key: $KEY" \
  -F "image=@./first_frame.png" \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
echo "asset id: $ASSET_ID"

# 2) Create an image-guided text-to-video job (202 + job object).
JOB_ID=$(curl -s -X POST "$BASE/v1/videos" \
  -H "X-API-Key: $KEY" \
  -H "Content-Type: application/json" \
  -d "{
        \"prompt\": \"a paper boat drifting down a rain-slick gutter at dusk\",
        \"size\": \"1280x704\",
        \"seconds\": 5,
        \"quality\": \"standard\",
        \"start_image_asset_id\": $ASSET_ID
      }" \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')
echo "job id: $JOB_ID"

# 3) Poll until completed.
while true; do
  STATUS=$(curl -s "$BASE/v1/videos/$JOB_ID" -H "X-API-Key: $KEY" \
    | python3 -c 'import sys,json; print(json.load(sys.stdin)["status"])')
  echo "status: $STATUS"
  [ "$STATUS" = "completed" ] && break
  [ "$STATUS" = "failed" ] && { echo "render failed"; exit 1; }
  [ "$STATUS" = "cancelled" ] && { echo "cancelled"; exit 1; }
  sleep 3
done

# 4) Fetch the MP4.
curl -s "$BASE/v1/videos/$JOB_ID/content" -H "X-API-Key: $KEY" -o out.mp4
echo "saved out.mp4"
```

### Python

```python
import time
import requests

BASE = "https://mindrouter.uidaho.edu"
KEY = "your_api_key_here"
HEADERS = {"X-API-Key": KEY}  # or {"Authorization": f"Bearer {KEY}"}

# 1) Upload a reference image → asset id.
with open("first_frame.png", "rb") as fh:
    r = requests.post(
        f"{BASE}/v1/videos/assets",
        headers=HEADERS,
        files={"image": ("first_frame.png", fh, "image/png")},
    )
r.raise_for_status()
asset_id = r.json()["id"]
print("asset id:", asset_id)

# 2) Create an image-guided text-to-video job (returns 202 + job object).
r = requests.post(
    f"{BASE}/v1/videos",
    headers={**HEADERS, "Content-Type": "application/json"},
    json={
        "prompt": "a paper boat drifting down a rain-slick gutter at dusk",
        "size": "1280x704",
        "seconds": 5,
        "quality": "standard",
        "start_image_asset_id": asset_id,   # first-frame keyframe
        # "end_image_asset_id": other_asset_id,  # optional last-frame keyframe
    },
)
r.raise_for_status()
job = r.json()
job_id = job["id"]
print("job id:", job_id, "status:", job["status"])

# 3) Poll until terminal.
while True:
    job = requests.get(f"{BASE}/v1/videos/{job_id}", headers=HEADERS).json()
    print("status:", job["status"], "progress:", job.get("progress"))
    if job["status"] == "completed":
        break
    if job["status"] in ("failed", "cancelled"):
        raise SystemExit(f"job {job['status']}: {job.get('error')}")
    time.sleep(3)

# 4) Fetch the MP4 (streamed; supports Range).
with requests.get(f"{BASE}/v1/videos/{job_id}/content", headers=HEADERS, stream=True) as resp:
    resp.raise_for_status()
    with open("out.mp4", "wb") as out:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            out.write(chunk)
print("saved out.mp4")
```

For a **text-only** clip, omit the asset upload and the `start_image_asset_id` /
`end_image_asset_id` fields — everything else is identical.

---

## Notes for the consuming (studio) application

- **Multi-shot is your job.** Store the storyboard/timeline yourself, submit each
  shot as an independent `POST /v1/videos`, and stitch the returned MP4s (ffmpeg) on
  your side. MindRouter renders clips; it does not assemble them.
- **Coherence across shots.** Generate consistent stills (e.g. via the image API),
  upload them as assets, and pass them as `start_image_asset_id` / `end_image_asset_id`
  keyframes to anchor each clip.
- **Design for a queue.** With one render slot and a default per-user concurrency of
  1, expect `queued` waits. Poll with backoff and surface honest progress + ETA.
- **Discover, don't hardcode.** Read `GET /v1/videos/models` for the live size menu,
  duration range, fps, and qualities.
