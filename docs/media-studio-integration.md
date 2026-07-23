# Building a Storyboarding / Ad-Mockup Studio on MindRouter

**Last updated: 2026-07-23**

This is the end-to-end integration guide for building a **creative studio app** —
storyboarding, ad mockups, short promotional sequences — on top of MindRouter's
generative-media APIs. It is the keystone document; the per-API references
([images](images-api.md), [video](video-api.md), [voice](voice-api.md)) give the
exhaustive field-by-field contract, while this guide shows how to **compose** them
into a full pipeline.

---

## The two-layer architecture (read this first)

MindRouter is a **provider of single-shot generative primitives**:

- one image (text-to-image, or image-to-image edit)
- one clip (a single continuous video shot, ≤ 30 s)
- one utterance (a single text-to-speech render)

MindRouter does **not** do multi-shot assembly, stitching, timeline editing,
crossfades, or cross-shot coherence. **That is your app's job.** Your studio app
is the "second layer": it holds the storyboard/timeline state, calls MindRouter
once per primitive, and does its own **ffmpeg** assembly (concat, crossfade, audio
mux) to produce the final deliverable.

```
┌─────────────────────────────────────────────────────────────┐
│  YOUR STUDIO APP  (storyboard state, timeline, ffmpeg)       │
│                                                              │
│   shot list → per-shot prompts → continuity keyframes →      │
│   concat/crossfade → mux narration → final 30–60s deliverable│
└───────────────┬──────────────────────────────────────────────┘
                │  single-shot API calls (one primitive each)
                ▼
┌─────────────────────────────────────────────────────────────┐
│  MINDROUTER  (https://mindrouter.uidaho.edu)                 │
│   /v1/images/generations   one still (txt2img)               │
│   /v1/images/edits         one still (img2img / ref-edit)    │
│   /v1/videos/assets        upload a still → asset id         │
│   /v1/videos               one clip (async job)              │
│   /v1/audio/speech         one narration render (TTS)        │
└─────────────────────────────────────────────────────────────┘
```

If you find yourself wanting MindRouter to "join these two clips" or "keep the
character consistent across shots" — stop. That logic lives in your app. This
guide shows you the patterns to do it.

---

## Auth

Every request needs an API key, sent either way:

```
X-API-Key: <key>
# or
Authorization: Bearer <key>
```

Your account must additionally have the relevant capability flags enabled by an
administrator:

- `image_generation_enabled` — for `/v1/images/*`
- `video_generation_enabled` — for `/v1/videos/*`

and the corresponding service must be enabled server-side (`img.enabled`,
`vid.enabled`, `voice.tts_enabled`, `voice.stt_enabled`). If a capability is off
you'll get `403` (account) or `503`/`404` (service).

Base URL for all examples: `https://mindrouter.uidaho.edu`.

---

## The creative pipeline as a recipe

The full studio pipeline is four stages. Each maps to one MindRouter primitive
(or a small loop over one), plus assembly you do yourself.

| Stage | What you do | MindRouter primitive | Assembly (your app) |
|---|---|---|---|
| 1. Frames | Generate & refine still keyframes per shot | `POST /v1/images/generations`, `POST /v1/images/edits` | pick/version frames |
| 2. Clips | Animate each still into a clip | `POST /v1/videos/assets` → `POST /v1/videos` → poll → `/content` | download MP4s |
| 3. Sequence | Stitch clips into a 30–60 s sequence | *(none — your ffmpeg)* | concat / crossfade |
| 4. Narration | Add voiceover, mux into the sequence | `POST /v1/audio/speech` | ffmpeg audio mux |

---

## Stage 1 — Generate and refine still frames

Stills are your **storyboard**. Get each shot's look right as a still *before*
spending render time on video.

### 1a. Text-to-image (`POST /v1/images/generations`)

JSON body. Fields (source: `_prepare_image_canonical` / `image_generations` in
`backend/app/api/v1_openai.py`):

| field | type | required | default | notes |
|---|---|---|---|---|
| `prompt` | string | **yes** | — | the shot description |
| `model` | string | no | `black-forest-labs/FLUX.2-dev` (`img.default_model`) | served model id |
| `n` | int | no | 1 | capped at `img.max_n` (4) |
| `size` | string | no | `1024x1024` | must be in `img.allowed_sizes`: `512x512,768x768,1024x1024,1024x768,768x1024`; also capped by `img.max_width`/`img.max_height` (1024×1024) |
| `quality` | string | no | `standard` | |
| `style` | string | no | — | |
| `response_format` | string | no | `url` | `url` or `b64_json` |
| `num_inference_steps` | int | no | 20 (`img.default_steps`) | capped at `img.max_steps` (50) |
| `guidance_scale` | float | no | 3.5 (`img.default_guidance_scale`) | |
| `seed` | int | no | — | **set this** for reproducible shots |
| `user` | string | no | — | end-user tag |

The response is OpenAI-compatible: `{ "created": …, "data": [ { "url": … } ] }`
(or `b64_json` per `response_format`).

**Errors:** `400` missing prompt / off-menu size / dimensions over max / content
policy violation; `403` account not enabled; `404` model not found; `503` image
generation disabled.

**Content policy:** prompts are screened by an LLM-as-judge (`img.policy`). A
denied prompt returns `400` with `type: "content_policy_violation"`.

### 1b. Image-to-image / reference edit (`POST /v1/images/edits`)

This is your **continuity tool**: feed a prior frame back in and restyle/adjust it
so shot N+1 shares the look of shot N. `multipart/form-data` (source: `image_edits`
in `v1_openai.py`):

| field | type | required | default | notes |
|---|---|---|---|---|
| `image` | file(s) | **yes** | — | 1–4 reference images (`_MAX_EDIT_IMAGES`); each must be `image/*`, ≤ `img.max_image_upload_mb` (10 MB) |
| `prompt` | string | **yes** | — | edit instruction |
| `model` | string | no | default model | |
| `n` | int | no | 1 | |
| `size` | string | no | default | same allow-list as above |
| `response_format` | string | no | `url` | |
| `strength` | float | no | — | **accepted but ignored** for FLUX.2 Klein edits (structure-preserving) |
| `num_inference_steps` | int | no | **8** (`img.edit_default_steps`) | edits look better with fewer steps |
| `guidance_scale` | float | no | 3.5 | |
| `seed` | int | no | — | |
| `user` | string | no | — | |

The policy judge is **edit-aware** here: deictic references ("this man", "her
jacket") are not auto-failed as *ambiguous*, since they refer to the supplied
image rather than a described scene. The **full content policy still applies** to
the described transformation and its likely resulting image — the edit exemption
only removes the ambiguity penalty, it does not relax the policy itself.

> See [images-api.md](images-api.md) for the complete field/response contract.

---

## Stage 2 — Turn a still into a clip

Video is **asynchronous**: submit → poll → fetch. A submit never returns the
video. The path is three calls.

### 2a. Upload the still as an asset (`POST /v1/videos/assets`)

This is what makes **image-guided (keyframe) video** possible over the API: the
create endpoint takes *asset ids*, not raw image bytes. `multipart/form-data`
(source: `create_video_asset` in `backend/app/api/video_api.py`):

| field | type | required | notes |
|---|---|---|---|
| `image` | file | **yes** | must be `image/*`, ≤ `vid.max_image_upload_mb` (10 MB); counts toward your storage cap |

→ `201`:

```json
{ "id": 1234, "object": "video.asset", "kind": "reference",
  "content_type": "image/png", "size_bytes": 512345 }
```

Keep that `id`. **Errors:** `400` not an image / too large; `401`
missing/invalid/inactive/expired API key; `403` account not enabled; `503` video
disabled; `507` over your storage cap.

### 2b. Create the clip (`POST /v1/videos`)

JSON body (source: `submit_video_job` in `video_api.py`). Fields the create path
reads:

| field | type | required | default | notes |
|---|---|---|---|---|
| `prompt` | string | **yes** | — | motion / action description for the shot |
| `model` | string | no | `lightricks/ltx-2.3-distilled` (`vid.default_model`) | |
| `size` | string | no | `1280x704` | must be in `vid.allowed_sizes`: `1280x704,704x1280,1024x576,768x448` (fixed preset menu) |
| `seconds` | string/int | no | 5 | **whole seconds**, `vid.min_seconds` (4) ≤ n ≤ `vid.max_total_seconds` (**30**) |
| `fps` | int | no | 24 | accepted but **not validated** — any value is stored/echoed, only 24 is actually rendered; off-24 values are neither honored nor rejected |
| `quality` | string | no | `standard` | `draft` \| `standard` \| `final` |
| `seed` | int | no | — | |
| `start_image_asset_id` | int | no | — | asset id from 2a — **first frame** conditioning |
| `end_image_asset_id` | int | no | — | asset id from 2a — **last frame** conditioning |
| `callback_url` | string | no | — | optional completion webhook |

Referenced assets must be owned by the caller (else `400 Invalid image asset id`).

→ `202` with the **job object**:

```json
{
  "id": "vid-ab12…", "object": "video", "status": "queued",
  "progress": 0.0, "created_at": 1753000000,
  "started_at": null, "completed_at": null, "expires_at": null,
  "content_url": null, "error": null,
  "usage": { "duration_seconds": null, "gpu_seconds": null, "token_equivalent": 12345 },
  "model": "lightricks/ltx-2.3-distilled", "size": "1280x704", "fps": 24, "quality": "standard"
}
```

**Errors:** `400` missing prompt / off-menu size / non-whole or out-of-range
seconds / bad quality / bad asset id; `401` missing/invalid/inactive/expired API
key; `403` account not enabled; `404` model not found; `429` over per-user
concurrency cap **or** insufficient token quota; `503` video disabled; `507` over
storage cap.

### 2c. Poll and fetch

Poll `GET /v1/videos/{id}` until `status` is terminal. External statuses (source:
`_EXTERNAL_STATUS`): `queued`, `in_progress`, `completed`, `failed`, `cancelled`.

When `completed`, fetch the MP4 from `GET /v1/videos/{id}/content` (also available
as the `content_url` on the job). It streams with `Accept-Ranges`/`206` so browser
`<video>` scrubbing works. `409` if you fetch before it's ready.

Other endpoints:
- `GET /v1/videos` — list your jobs (`status_filter`, `limit` ≤ 50, `offset`).
- `DELETE /v1/videos/{id}` — for a **terminal** job (`completed`/`failed`/
  `cancelled`) it deletes the job's rows + artifact files and **reclaims the
  per-user storage** (this is how you free space against the 50 GB cap over the
  API), returning `{"id": ..., "object": "video", "deleted": true}`; for a
  **running/queued** job it requests cancellation (refunding a queued job's
  reserved quota) and returns the job object.
- `GET /v1/videos/models` — capability discovery (sizes, second range, qualities).

> See [video-api.md](video-api.md) for the full job lifecycle and error table.

---

## Stage 3 — Assemble multiple clips (YOUR app, not MindRouter)

MindRouter renders **one clip at a time and never joins them.** To produce a
coherent 30–60 s sequence you concatenate your downloaded MP4s yourself with
ffmpeg. This is a hard architectural boundary, not a missing feature.

### Continuity: the gallery → keyframe pattern

The single most important trick for shot-to-shot coherence: **carry the last frame
of clip N as the start keyframe of clip N+1.**

1. Render clip N (with `start_image_asset_id` = your storyboard still for shot N).
2. After download, extract its final frame with ffmpeg:
   ```bash
   ffmpeg -sseof -0.1 -i clipN.mp4 -frames:v 1 -q:v 2 lastframe_N.png
   ```
3. Upload that frame via `POST /v1/videos/assets` → get an asset id.
4. Use that asset id as `start_image_asset_id` for clip N+1.

Now clip N+1 begins exactly where clip N ended, so a hard cut between them looks
continuous. You can also pin an `end_image_asset_id` (e.g. your storyboard still
for the *next* shot) to steer clip N toward its destination — giving you a clean
handoff on both sides.

### Concatenation

Same codec/size/fps clips — use the concat demuxer (no re-encode):

```bash
printf "file 'clip1.mp4'\nfile 'clip2.mp4'\nfile 'clip3.mp4'\n" > list.txt
ffmpeg -f concat -safe 0 -i list.txt -c copy sequence.mp4
```

### Crossfades (softer transitions)

```bash
ffmpeg -i clip1.mp4 -i clip2.mp4 -filter_complex \
  "[0][1]xfade=transition=fade:duration=0.5:offset=4.5" -c:v libx264 sequence.mp4
```

(`offset` = clip1 length − crossfade duration. Chain `xfade` for 3+ clips.)

---

## Stage 4 — Add narration

### Text-to-speech (`POST /v1/audio/speech`)

JSON body (source: `TTSRequest` / `tts_speech` in
`backend/app/api/voice_api.py`):

| field | type | required | default | notes |
|---|---|---|---|---|
| `input` | string | **yes** | — | the narration text |
| `model` | string | no | `kokoro` | |
| `voice` | string | no | `af_heart` | |
| `response_format` | string | no | `mp3` | content type follows this |
| `speed` | float | no | 1.0 | clamped 0.25–4.0 |

Returns **streamed audio bytes** (`audio/mpeg` for mp3). `400` if `input` is
empty; `404` if TTS is not enabled.

### Mux narration into the sequence

```bash
ffmpeg -i sequence.mp4 -i narration.mp3 \
  -c:v copy -c:a aac -shortest final.mp4
```

To duck under or replace existing clip audio, add the appropriate `-map` / audio
filters. That mixing is your app's job — MindRouter only hands you the raw
narration render.

> (Optional) `POST /v1/audio/transcriptions` (STT) turns an audio file back into
> text/`srt`/`vtt` — handy for auto-generating captions from your narration.
> See [voice-api.md](voice-api.md).

---

## Async orchestration at scale

Video is the bottleneck — plan your orchestration around it.

**The queue is serialized.** MindRouter renders **one clip at a time** across the
whole service. Submitting more jobs does not make them render in parallel; they
wait in `queued`.

**Per-user concurrency cap.** `vid.max_concurrent_jobs_per_user` (default **1**):
you may have only one *active* job at a time. A second submit while one is in
flight returns `429`. So a 3-shot ad is **submit → wait → submit → wait → submit**,
not three parallel submits. Design your orchestrator as a per-user serial queue.

**Per-user storage cap.** `vid.user_storage_cap_gb` (default **50 GB**), counting
both rendered clips *and* uploaded reference assets. At/over the cap, new
submits/uploads return `507`. Download finished clips to your own storage and
`DELETE` them from MindRouter to reclaim space.

**Quota.** Each video reserves token-equivalent quota up front (refunded on
fail/cancel); insufficient quota returns `429` on submit. Image and TTS calls
also draw on the same token quota. Over quota → `429 Token quota exceeded`.
Per-key RPM limits also apply to image/voice calls.

**Poll/backoff.** Poll `GET /v1/videos/{id}` every **2–5 s with backoff** (a 5 s
720p clip currently renders in ~30–43 s). Do not hammer at sub-second intervals.
Prefer a `callback_url` webhook if you can receive one, and poll only as a
fallback.

**Recommended orchestrator loop (per user):**

```
for shot in storyboard:                # serialize — cap is 1 active job
    asset = upload(still_for(shot))     # POST /v1/videos/assets
    job   = submit(prompt, start_image_asset_id=asset.id)  # POST /v1/videos → 202
    while job.status in (queued, in_progress):
        sleep(backoff())                # 2–5s, capped
        job = poll(job.id)
    if job.status == failed: handle/retry
    else: download(job.id + "/content"); delete_if_low_on_space(job.id)
```

**Limits recap:** clip ≤ **30 s**, fps **24**, sizes from `GET /v1/videos/models`;
final 30–60 s sequences come from stitching several clips in your app.

---

## Worked walkthrough: a 3-shot ad mockup

Goal: a ~15 s three-shot ad — (1) product on a desk, (2) product in a hand, (3)
product with a tagline — with narration. Exact calls, in order. `$KEY` is your API
key; `$BASE` is `https://mindrouter.uidaho.edu`.

### Shot 1 — storyboard still (txt2img)

```bash
curl -s $BASE/v1/images/generations \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{
    "prompt": "a sleek matte-black water bottle on a sunlit oak desk, product photography, soft shadows",
    "size": "1024x768",
    "seed": 42,
    "response_format": "b64_json"
  }'
```

> Note: image sizes and video sizes are **different allow-lists**. Generate the
> still at an image size (e.g. `1024x768`), then request the video at a video size
> (e.g. `1280x704`); the img2video conditioning handles the reframe.

### Shots 2 & 3 — keep continuity with edits (img2img)

Feed shot 1's still back in so the product matches:

```bash
curl -s $BASE/v1/images/edits \
  -H "Authorization: Bearer $KEY" \
  -F "image=@shot1.png" \
  -F 'prompt=the same matte-black water bottle now held in a person\'s hand, same lighting' \
  -F "size=1024x768" -F "seed=43"
```

Repeat for shot 3 (add the tagline scene). You now have `shot1.png`, `shot2.png`,
`shot3.png` — a visually consistent 3-frame storyboard.

### Animate each still into a clip (serial, because the cap is 1)

For each shot: upload the still, submit the clip, poll, download.

```bash
# 1) upload shot1 as a keyframe asset
ASSET=$(curl -s $BASE/v1/videos/assets \
  -H "Authorization: Bearer $KEY" \
  -F "image=@shot1.png" | jq -r .id)

# 2) submit a 5s clip conditioned on that first frame
JOB=$(curl -s $BASE/v1/videos \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d "{\"prompt\":\"slow push-in on the water bottle, gentle rim light\",
       \"size\":\"1280x704\",\"seconds\":\"5\",\"quality\":\"standard\",
       \"start_image_asset_id\":$ASSET}" | jq -r .id)

# 3) poll until terminal
while :; do
  S=$(curl -s $BASE/v1/videos/$JOB -H "Authorization: Bearer $KEY" | jq -r .status)
  [ "$S" = completed ] && break
  [ "$S" = failed ]    && { echo "render failed"; exit 1; }
  sleep 3
done

# 4) download the MP4
curl -s $BASE/v1/videos/$JOB/content -H "Authorization: Bearer $KEY" -o clip1.mp4
```

For **shot 2's clip**, first extract shot 1's last frame and use it as the start
keyframe (the continuity pattern):

```bash
ffmpeg -sseof -0.1 -i clip1.mp4 -frames:v 1 -q:v 2 last1.png
ASSET2=$(curl -s $BASE/v1/videos/assets -H "Authorization: Bearer $KEY" \
  -F "image=@last1.png" | jq -r .id)
# submit clip2 with start_image_asset_id=$ASSET2 … (same poll/download loop)
```

Repeat for clip 3. Result: `clip1.mp4`, `clip2.mp4`, `clip3.mp4`, each starting
where the previous ended.

### Narration (TTS)

```bash
curl -s $BASE/v1/audio/speech \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"input":"Hydration, redefined. Meet the bottle built for your day.",
       "voice":"af_heart","response_format":"mp3"}' -o narration.mp3
```

### Assemble (YOUR ffmpeg — MindRouter does none of this)

```bash
printf "file 'clip1.mp4'\nfile 'clip2.mp4'\nfile 'clip3.mp4'\n" > list.txt
ffmpeg -f concat -safe 0 -i list.txt -c copy sequence.mp4
ffmpeg -i sequence.mp4 -i narration.mp3 -c:v copy -c:a aac -shortest ad_mockup.mp4
```

`ad_mockup.mp4` is your ~15 s deliverable — three coherent shots plus voiceover,
assembled entirely in your app.

---

## Python: end-to-end orchestrator sketch

```python
import time, requests

BASE = "https://mindrouter.uidaho.edu"
H = {"Authorization": f"Bearer {KEY}"}

def txt2img(prompt, size="1024x768", seed=None):
    r = requests.post(f"{BASE}/v1/images/generations",
                      headers={**H, "Content-Type": "application/json"},
                      json={"prompt": prompt, "size": size, "seed": seed,
                            "response_format": "b64_json"})
    r.raise_for_status()
    return r.json()["data"][0]["b64_json"]

def upload_asset(path):
    with open(path, "rb") as f:
        r = requests.post(f"{BASE}/v1/videos/assets", headers=H,
                          files={"image": f})
    r.raise_for_status()
    return r.json()["id"]

def render_clip(prompt, start_asset=None, seconds="5", size="1280x704"):
    body = {"prompt": prompt, "seconds": seconds, "size": size,
            "quality": "standard"}
    if start_asset:
        body["start_image_asset_id"] = start_asset
    r = requests.post(f"{BASE}/v1/videos",
                      headers={**H, "Content-Type": "application/json"}, json=body)
    r.raise_for_status()                      # 202
    job_id = r.json()["id"]

    delay = 2.0
    while True:                               # serial: cap is 1 active job
        time.sleep(delay)
        delay = min(delay * 1.5, 5.0)         # backoff
        job = requests.get(f"{BASE}/v1/videos/{job_id}", headers=H).json()
        if job["status"] == "completed":
            break
        if job["status"] in ("failed", "cancelled"):
            raise RuntimeError(job.get("error"))
    mp4 = requests.get(f"{BASE}/v1/videos/{job_id}/content", headers=H).content
    return job_id, mp4

def tts(text, voice="af_heart"):
    r = requests.post(f"{BASE}/v1/audio/speech",
                      headers={**H, "Content-Type": "application/json"},
                      json={"input": text, "voice": voice,
                            "response_format": "mp3"})
    r.raise_for_status()
    return r.content   # mp3 bytes

# Then: stitch the returned mp4s + narration with ffmpeg (subprocess) — your job.
```

---

## Cross-references

- [images-api.md](images-api.md) — `/v1/images/generations` & `/v1/images/edits`
  full field/response/error contract.
- [video-api.md](video-api.md) — `/v1/videos*` job lifecycle, statuses, caps, and
  capability discovery.
- [voice-api.md](voice-api.md) — `/v1/audio/speech` (TTS) & `/v1/audio/transcriptions`
  (STT).
