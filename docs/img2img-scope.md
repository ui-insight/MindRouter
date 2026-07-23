# Image-to-Image (input-image conditioning) — Scope

Status: **Phase A COMPLETE (passed) — ready for Phase B**
Owner: Luke Sheneman (RCDS)
Related: `docs/video-generation-plan.md`, `docs/video-api.md` (FLUX→LTX storyboard bridge)

## Phase A results (2026-07-23) — GATE PASSED

Standalone probe (`test_edit.py`) on webbyg2 **GPU0** (empty 81GB PCIe H100 — the
live FLUX instances on GPU2 were untouched). Note: `CUDA_DEVICE_ORDER=PCI_BUS_ID`
is REQUIRED to target the free card; without it CUDA "fastest-first" ordering
picks a busy 95GB NVL and OOMs.

- ✅ **Reference-edit works.** `Flux2KleinPipeline.__call__` (diffusers 0.37.1)
  takes `image=` as its first param; the working call shape is **`image=[pil]`**
  (a list of PIL images). Structure-preserving: red apple → blue apple, identical
  shape/stem/lenticels/wood-grain/background.
- ✅ **Semantic = reference-edit, not strength-denoise.** No `strength` param;
  `guidance_scale` is ignored (step-wise distilled model). Schema still carries
  optional `strength` for forward-compat but Klein won't use it.
- ✅ **img2img costs NO extra VRAM vs txt2img.** Load 34.7GB; peak **37.31GB for
  BOTH** txt2img and reference-edit. **Implication: edits can run on the existing
  GPU2 instances — no GPU reclaim / no new deployment needed for Phase B.**
  (Caveat: measured with ONE 1024² reference; multiple/larger references would add
  memory — cap input count/resolution in the API.)
- ⏱ Timing @ 20 steps, 1024²: txt2img **7.3s** (3.3 it/s); reference-edit **13.6s**
  (1.6 it/s — ~2× slower/step from processing reference tokens, still well within
  the ≥300s proxy timeout).

Artifacts: `webbyg2:/data/flux2/test_{base,edit}.png`.

## Problem

MindRouter's image-generation path is **text-to-image only**. There is no way to
condition generation on an input image (img2img / reference edit). Confirmed at
three layers:

- `CanonicalImageRequest` (`backend/app/core/canonical_schemas.py:420`) has no
  `image` / `init_image` / `mask` / `strength` field — only `prompt` + knobs.
- Public API exposes only `POST /v1/images/generations` (`v1_openai.py:683`);
  the OpenAI img2img endpoints `/images/edits` and `/images/variations` are not
  implemented.
- The dashboard proxy (`images.py:images_api_generate`) builds the canonical
  request from `prompt` only.

## Pivotal finding: the FLUX backend is a custom server we own

Backend 45 = `diffusion` at `https://webbyg2.hpc.uidaho.edu:8002` (nginx LB over
`flux2-klein-gpu2{a,b}` on 18100/18101). The server is **NOT** openedai-images-flux —
it is a custom 228-line FastAPI app at `webbyg2:/data/flux2/serve_klein.py`
(`venv` at `/data/flux2/venv`, diffusers `Flux2KleinPipeline`, model
`black-forest-labs/FLUX.2-klein-9B`).

Routes today: `/health`, `/v1/models`, `/v1/images/generations`.
Generate call: `pipe(prompt=…, width, height, num_inference_steps, guidance_scale)`
— **no image input**. Adding img2img therefore requires backend work too, but the
server is ours to modify.

## Current txt2img chain (what img2img must extend)

```
images.html ──▶ POST /images/api/generate (dashboard proxy)   ┐
   or  API  ──▶ POST /v1/images/generations (v1_openai.py)    ┘
        └▶ CanonicalImageRequest (prompt-only)
             └▶ InferenceService.image_generation            (inference.py:608)
                  └▶ _proxy_image_request                    (inference.py:1762)
                       └▶ DiffusionOutTranslator.translate_image_request (diffusion_out.py:38)
                            └▶ serve_klein.py /v1/images/generations
                                 └▶ Flux2KleinPipeline(prompt=…)
        └▶ b64 back ▶ ArtifactStorage.store + UserImage row  (images.py:320)
```

## Layers to change

| # | Layer | File | Change |
|---|-------|------|--------|
| 1 | **Backend server** ⚠️ | `webbyg2:/data/flux2/serve_klein.py` | Accept input image(s); decode b64→PIL; `pipe(prompt=…, image=[…])`; new route. **Gating work.** |
| 2 | Canonical schema | `canonical_schemas.py:420` | Add `image: Optional[List[str]]` (b64), `strength: Optional[float]`, optional `mask`. |
| 3 | Out-translator | `diffusion_out.py:38` | Pass `image`/`strength` into backend payload when present. |
| 4 | Inference service | `inference.py:1762` | `_proxy_image_request` targets the edits route when an image is present; timeout already ≥300s. |
| 5 | Public API | `v1_openai.py:683` | Add `/v1/images/edits` (OpenAI-compatible multipart: `image`, `mask`, `prompt`); new multipart handling. |
| 6 | Dashboard proxy + UI | `images.py:182`, `templates/user/images.html` | Accept an uploaded/selected input image; strength slider; pass as b64. |
| — | Provenance (optional) | `models.py` `UserImage`, `crud` | Persist input image + link for audit/gallery ("edited from X"). |

## Gating verification (Phase A — do first)

FLUX.2 Klein is a **reference-editing** model. In diffusers, `Flux2KleinPipeline.__call__`
takes an `image=` list of reference images (multi-image edit) — a *different*
semantic from classic img2img `strength`-based denoise. Before any plumbing:

- Confirm `Flux2KleinPipeline.__call__` accepts `image=` (and whether `strength`
  applies) on the installed diffusers version in `/data/flux2/venv`.
- Run a standalone edit on webbyg2 (`test_edit.py`, mirroring the LTX `test_cond.py`
  validation): one reference image + prompt → confirm output + measure **VRAM delta**
  (two Klein instances already share GPU2; reference-image activations add memory
  and could OOM).

This step settles the semantic (reference-edit vs strength img2img) and whether
GPU2 has headroom. Everything downstream keys off it.

## Design decisions (resolve after Phase A)

1. **Endpoint shape** — Recommend **both**: add OpenAI-compatible `/v1/images/edits`
   (multipart) for API parity, and let the dashboard send JSON b64 through the
   existing proxy. Simpler non-standard alternative: accept `image[]` inside
   `/images/generations`. Prefer the standard `/edits` route.
2. **Semantic** — RESOLVED (Phase A): reference-edit (Klein native, `image=[pil]`).
   No `strength`. Schema keeps optional `strength` for forward-compat only.

## Effort & phasing

- **Phase A — Backend proof (½–1 day):** standalone edit script on webbyg2;
  confirm pipeline API + VRAM. **Gate.**
- **Phase B — Backend route + MindRouter plumbing (1 day):** `/v1/images/edits`
  on `serve_klein.py`; schema + translator + service + API.
- **Phase C — Dashboard + UI (½–1 day):** upload/pick input image, strength slider,
  wire proxy.

≈ **2–3 focused days**; most risk in Phase A.

## Payoff — unlocks the storyboard bridge

Once the image path accepts an input image, two earlier gaps collapse into one
capability: FLUX can condition on a prior shot's frame (shot-to-shot continuity),
and a gallery image can feed forward as an LTX start/end keyframe. img2img on FLUX
is the keystone for the storyboard-coherence workflow (the separate studio app),
while MindRouter only needs to expose the clip + conditioning primitives cleanly.
