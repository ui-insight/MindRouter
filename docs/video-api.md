# MindRouter Video Generation API (`/v1/videos`)

**Stability contract.** MindRouter is a **single-clip text-to-video provider**.
This API is the integration surface for downstream applications (e.g. a separate
multi-shot "video studio" app that stores its own storyboards and calls this API
once per clip, assembling the results itself). Treat these shapes as a stable
contract; changes will be additive.

Scope: **text-to-video, one clip per request.** Multi-shot assembly, storyboards,
coherence workflows, and timeline editing are **out of scope for MindRouter** —
they belong to the consuming application.

## Auth

All endpoints require an API key: `Authorization: Bearer <key>` (or
`X-API-Key: <key>`). The caller's user must have `video_generation_enabled`, and
the service must be enabled (`vid.enabled`).

## Async model

A render outlives an HTTP request, so the API is **submit → poll → fetch**:
`POST /v1/videos` returns `202` immediately with a job in `queued`; poll
`GET /v1/videos/{id}` until `completed` or `failed`; then fetch the MP4 from
`/v1/videos/{id}/content`. Never block on a single request.

Poll interval: 2–5s with backoff. A 5s 720p clip currently takes ~30–43s.

## Endpoints

### `POST /v1/videos` — submit
Body (only these fields are honored; unknown/typo'd fields are surfaced per the
`field_validation` setting):

| field | type | notes |
|---|---|---|
| `model` | string | served model id; defaults to `vid.default_model` |
| `prompt` | string | **required** |
| `size` | string | `"WIDTHxHEIGHT"` from the preset list (below); default `vid.default_size` |
| `seconds` | string | duration from the preset list; default `vid.default_seconds` |
| `fps` | int | default 24 |
| `quality` | string | `draft` \| `standard` \| `final` |
| `seed` | int | optional |
| `negative_prompt` | string | optional |
| `callback_url` | string | optional webhook (HMAC-signed) on completion |

→ `202` with the **job object** (below), `status: "queued"`.
Errors: `503` service disabled, `403` user not enabled, `400` missing prompt /
off-menu size or duration / bad quality, `404` model not found, `429` over the
per-user concurrency cap or insufficient quota.

### `GET /v1/videos/{id}` — poll
→ the job object. `404` for an id the caller does not own (no existence leak).

### `GET /v1/videos/{id}/content` — fetch the MP4
→ `video/mp4`, `Accept-Ranges: bytes`, `206` on Range (browser `<video>`
scrubbing works). `409` if not yet completed, `404` if not found/owned.

### `GET /v1/videos` — list the caller's jobs
Query: `status`, `limit` (≤50), `offset`. → `{object:"list", data:[job…], total}`.

### `DELETE /v1/videos/{id}` — cancel / delete
Cancels a running job (propagated to the worker) or removes a terminal one.
Reserved quota is refunded on cancel/failure.

### `GET /v1/videos/models` — capability discovery
→ per-model `{supported_sizes, supported_durations, supported_fps,
supported_qualities, supports_text_to_video, supports_image_to_video (false in
v1), max_shots (1), license_notice}`. **Render your UI controls from this** so
the preset matrix has one source of truth.

## The job object

```json
{
  "id": "vid-<24hex>",
  "object": "video",
  "status": "queued | in_progress | completed | failed | cancelled",
  "progress": 0-100,
  "created_at": 0, "started_at": 0, "completed_at": 0, "expires_at": 0,
  "model": "lightricks/ltx-2.3-distilled",
  "size": "1280x704", "seconds": "5", "fps": 24, "quality": "standard",
  "content_url": "/v1/videos/<id>/content | null (until completed)",
  "error": { "code": "...", "message": "..." } ,
  "usage": { "duration_seconds": 5.0, "gpu_seconds": 0, "token_equivalent": 10000 }
}
```

`status` is OpenAI-video-shaped (`in_progress`, not `rendering`) so stock SDK
polling loops work.

## Preset matrix (measured, LTX-2.3 on H200)

Sizes must be divisible by 64 (two-stage pipeline). Current presets:
`1280x704`, `704x1280`, `1024x576`, `768x448`. Durations (seconds): `4, 5, 8, 10`.
fps: 24. Off-menu values are rejected — always read `GET /v1/videos/models`.

Every clip includes a **synchronized audio track** (LTX generates audio natively).

## Notes for the consuming (studio) application

- **Multi-shot = your job.** Store the storyboard/timeline yourself; submit each
  shot as an independent `POST /v1/videos`; stitch the returned MP4s (ffmpeg) on
  your side. MindRouter renders clips; it does not assemble them.
- **Reference frames** for coherence: use MindRouter's image API (`/v1/images`)
  to generate consistent stills, then feed them to your own pipeline. (Image
  conditioning on the video endpoint — `supports_image_to_video` — is a future
  MindRouter addition, currently `false`.)
- **Concurrency:** one in-flight job per user by default (a single GPU
  serializes renders); design for a queue, show honest progress + ETA.
