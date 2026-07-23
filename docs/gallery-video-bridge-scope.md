# Gallery → Video keyframe bridge — Scope

Status: **BUILT (Release 2.8.18) — ships with the pending gateway deploy; no FLUX restart needed for this piece**
Owner: Luke Sheneman (RCDS)
Related: `docs/img2img-scope.md`, `docs/video-generation-plan.md`, `docs/video-api.md`

## Built (2026-07-23)

- `crud.get_user_image(image_id, user_id)` — ownership-checked gallery lookup.
- `dashboard/video.py`: `_store_reference_asset` helper (shared with the upload
  endpoint) + `POST /video/api/assets/from-gallery {image_id}` — resolves the
  UserImage, `ArtifactStorage.retrieve`s its bytes, **copies** them into
  `/data/video/<uid>/refs/<sha>.<ext>`, mints a `VideoAsset(kind=REFERENCE)`,
  returns `{asset_id}`. Downstream submit/runner/worker unchanged.
- `templates/user/video.html`: "Choose from gallery" button on each Start/End
  slot → scrollable picker modal (data from `/images/history`, paginated) →
  import → thumbnail + clear; gallery selection wins over the file input in
  `generate()`. Validated: py_compile, Jinja parse, `node --check` JS.

**QUOTA — NOW ENFORCED (Release 2.8.19):** `crud.get_user_video_storage_bytes`
sums a user's `VideoAsset.size_bytes` (outputs + reference images). The cap
(`vid.user_storage_cap_gb`, default 50, already seeded in migration 067) is gated
at: (a) **submit** (`submit_video_job`) — soft gate on *current* usage ≥ cap →
`507`, since output size isn't known until rendered (one in-flight clip may push
slightly over, then the next submit is blocked); (b) **upload + gallery-import**
(`_storage_cap_error`) — usage + incoming bytes > cap → `507`. Video page shows
`used / cap GB`. cap_gb=0 disables the check.

## Problem

A user can generate a still in the **Images** tab (FLUX, incl. the new img2img
reference-edit) but cannot use that still as an LTX **start/end keyframe** in the
**Video** tab. Video conditioning only accepts images uploaded through the video
tab. To animate a gallery image today you must download the PNG and re-upload it.
This is the video-side gap ("gap #1") flagged during the img2img work — the
keystone for shot-to-shot storyboard continuity.

## Why the two don't connect (confirmed in code)

| | Gallery image (Images tab) | Video conditioning frame |
|---|---|---|
| DB row | `UserImage` (`models.py:921`) | `VideoAsset` kind=`REFERENCE` (`models.py:172`) |
| Storage | `ArtifactStorage` → `/data/artifacts` | `<video_storage_path>/<uid>/refs/<sha>.<ext>` |
| Created by | image-gen path | `POST /video/api/assets` (upload only) |
| Consumed by | `/images/serve/{id}` | video submit `start_image_asset_id` / `end_image_asset_id` |

Video submit resolves conditioning strictly via `crud.get_video_asset`
(`video_api.py:_resolve_ref`, line ~331) — a `VideoAsset` id. There is no path from
a `UserImage` id into that flow.

## Core mechanism — materialize a VideoAsset from a UserImage

The bridge is a single **asset-materialization** step: copy the gallery image's
bytes into the video asset domain and mint a `VideoAsset(kind=REFERENCE)`. The
**entire downstream is unchanged** — submit → `create_video_shot(first/last_frame_
asset_id)` → runner `_ref_path` (reads `VideoAsset.storage_path`) → worker image
conditioning (the red→blue path already validated). We touch nothing in the runner
or worker; we only produce an asset id the system already knows how to consume.

```
UserImage(id) ──ownership check (user_id)──▶ ArtifactStorage.retrieve(storage_path)
      └─▶ write bytes to <video_storage_path>/<uid>/refs/<sha>.<ext>  (dedup by sha)
            └─▶ crud.create_video_asset(kind=REFERENCE, storage_path, content_type, sha256, size)
                  └─▶ returns asset_id  ──▶ existing submit uses it as start/end frame
```

### Why COPY, not reference-in-place (key design point)

Referencing the `UserImage` file directly from a `VideoAsset` would be less code
but is **wrong**: `delete_video_job` deletes a job's `VideoAsset` rows *and their
files*. If the video asset pointed at the gallery original, deleting the video
would delete the user's gallery image. Copying keeps the two lifecycles cleanly
separated and makes video-asset cleanup self-contained. Images are ~1–2 MB, so the
copy is cheap; dedup by sha avoids duplicate refs.

## Layers to change

| # | Layer | File | Change |
|---|-------|------|--------|
| 1 | CRUD helper | `db/crud.py` | `get_user_image(db, image_id, user_id)` (ownership-checked; currently inline selects in images.py). Small. |
| 2 | **Import endpoint** | `dashboard/video.py` | `POST /video/api/assets/from-gallery` `{image_id}` → resolve UserImage (ownership) → `ArtifactStorage.retrieve` → write to refs (dedup by sha) → `create_video_asset(kind=REFERENCE)` → `{asset_id}`. Mirrors `video_upload_asset`; returns the **same shape** so the submit flow is byte-identical. |
| 3 | Video UI | `templates/user/video.html` | Next to each Start/End image file input, add a **"Choose from gallery"** button → gallery picker modal (data from `/images/history`) → on select, call from-gallery import → set returned `asset_id` as start/end + show thumbnail. |
| 4 | Provenance (optional) | `models.py` `VideoAsset`, migration | `source_user_image_id` nullable FK for audit ("keyframe from gallery image X"). Deferred unless wanted (needs a migration). |
| 5 | Reverse deep-link (optional) | `templates/user/images.html` | "Use in video" on a gallery image → open Video tab with it preselected as the start frame. Phase 2 nicety. |

## Design decisions

1. **Scope: dashboard-only.** The `/video/api/assets` upload endpoint is already
   dashboard-scoped (session auth); the public `/v1/videos` API is for the separate
   studio app, which supplies its own images. Keep the bridge dashboard-only for
   symmetry. (If the studio app later wants it, expose an equivalent under the API.)
2. **Endpoint vs inline.** Recommend a **dedicated import endpoint** returning an
   `asset_id` (not folding a `start_image_gallery_id` into submit). Keeps submit
   unchanged, is reusable, and mirrors the existing upload endpoint exactly.
3. **Dedup.** Reuse the sha-named `refs/<sha>.<ext>` scheme; importing the same
   gallery image twice reuses the file (and may reuse the asset row — cheap to
   just create a new row pointing at the same file, matching upload behavior).
4. **Format.** Gallery saves PNG (`content_type image/png`); ext map png/jpg/webp
   as in upload. LTX conditioning decodes it fine.

## Non-goals / notes

- **Video storage quota (50 GB/user):** the copy counts toward it; note that this
  quota is a decision but is **not yet enforced** in crud (no `used_bytes` check
  found). Out of scope here; flag if enforcement lands.
- **Multi-keyframe at arbitrary timesteps:** still v1-unsupported (start/end only).
- **No GPU / infra / model changes.** Pure gateway + UI.

## Effort & risk

- **Effort:** ~½–1 day. Steps 1–3 are the feature; 4–5 optional.
- **Risk: low.** The video pipeline (runner/worker/submit) is untouched — the bridge
  only produces a `VideoAsset` id, already a first-class input. The one correctness
  subtlety (copy vs reference, for delete safety) is settled above.

## Payoff

Closes the loop: **prompt → FLUX still (txt2img or img2img reference-edit) → LTX
keyframe → animated clip**, entirely inside MindRouter, no download/re-upload. This
is the concrete substrate the separate storyboard/studio app builds on for
shot-to-shot continuity — see `docs/video-generation-plan.md`.
