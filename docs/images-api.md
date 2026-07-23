# MindRouter Image Generation API

_Last updated: 2026-07-23_

Base URL: `https://mindrouter.uidaho.edu`

This is the public reference for MindRouter's **image** generation primitives:
text-to-image (`POST /v1/images/generations`) and image-to-image /
reference-edit (`POST /v1/images/edits`). Both endpoints are
OpenAI-compatible (they mirror the OpenAI Images API request/response shapes,
with a few FLUX-specific extensions).

## Architecture note: MindRouter is a provider of single-shot primitives

MindRouter produces **one thing per call** — here, one batch of still images
from one prompt. It does **not** do multi-shot assembly, storyboard layout,
collage/montage stitching, timeline editing, or cross-image character
consistency. That orchestration is the job of the *separate* studio /
storyboarding app that consumes these APIs and does its own composition. Treat
each `/v1/images/*` call as a stateless render of a single prompt; keep any
multi-image narrative logic in your app.

---

## Authentication

Every request requires an API key, sent as **either** header:

```
X-API-Key: <your-key>
```
or
```
Authorization: Bearer <your-key>
```

Additional account gating (both must be true or the request is rejected):

- The image subsystem must be enabled globally (admin key `img.enabled`,
  default `true`). If disabled: **503 Service Unavailable** —
  `"Image generation is currently disabled"`.
- Your account must have image generation enabled
  (`user.image_generation_enabled`). If not: **403 Forbidden** —
  `"Image generation is not enabled for your account. Contact an
  administrator."`

---

## Endpoint 1 — `POST /v1/images/generations` (text-to-image)

Generate one or more images from a text prompt. Request body is **JSON**
(`Content-Type: application/json`).

### Request fields

| Field | Type | Required | Default | Notes |
|-------|------|----------|---------|-------|
| `prompt` | string | **yes** | — | The text description. Empty/missing → 400. |
| `model` | string | no | `img.default_model` (`black-forest-labs/FLUX.2-dev`) | Must resolve to a known image model or → 404 `model_not_found`. Aliases are resolved. |
| `n` | integer | no | `1` | Number of images. Clamped to `img.max_n` (default `4`) — values above the cap are silently reduced, not rejected. |
| `size` | string | no | `img.default_size` (`1024x1024`) | `WIDTHxHEIGHT`. Must be in `img.allowed_sizes` (see guardrails) or → 400. Also bounded by `img.max_width` / `img.max_height`. |
| `quality` | string | no | `"standard"` | `"standard"` or `"hd"`. Passed through to the backend. |
| `style` | string | no | `null` | Optional, e.g. `"vivid"` / `"natural"`. Passed through only if set. |
| `response_format` | string | no | `"url"` | `"url"` or `"b64_json"` — see [Response format](#response-format). |
| `num_inference_steps` | integer | no | `img.default_steps` (`20`) | Diffusion steps. Clamped to `img.max_steps` (default `50`). |
| `guidance_scale` | float | no | `img.default_guidance_scale` (`3.5`) | Classifier-free guidance strength. |
| `seed` | integer | no | `null` | Fixed seed for reproducibility. Omit for random. |
| `user` | string | no | `null` | Opaque end-user identifier for your own auditing. |

> Defaults shown in parentheses are the shipped fallback values. An
> administrator can change any `img.*` key, so treat them as defaults, not
> guarantees. Discover the currently-allowed sizes from a 400 error's message
> or from your admin.

### Example — curl

```bash
curl https://mindrouter.uidaho.edu/v1/images/generations \
  -H "X-API-Key: $MINDROUTER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "black-forest-labs/FLUX.2-dev",
    "prompt": "a lighthouse on a rocky coast at golden hour, cinematic",
    "n": 1,
    "size": "1024x1024",
    "num_inference_steps": 20,
    "guidance_scale": 3.5,
    "response_format": "b64_json"
  }'
```

### Example — Python

```python
import base64
import requests

BASE = "https://mindrouter.uidaho.edu"
API_KEY = "..."  # your MindRouter API key

resp = requests.post(
    f"{BASE}/v1/images/generations",
    headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
    json={
        "model": "black-forest-labs/FLUX.2-dev",
        "prompt": "a lighthouse on a rocky coast at golden hour, cinematic",
        "size": "1024x1024",
        "num_inference_steps": 20,
        "response_format": "b64_json",
    },
    timeout=180,
)
resp.raise_for_status()
data = resp.json()

# b64_json response: decode and save
b64 = data["data"][0]["b64_json"]
with open("out.png", "wb") as f:
    f.write(base64.b64decode(b64))
```

### Example — JavaScript (fetch)

```javascript
const BASE = "https://mindrouter.uidaho.edu";
const res = await fetch(`${BASE}/v1/images/generations`, {
  method: "POST",
  headers: {
    "X-API-Key": process.env.MINDROUTER_API_KEY,
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    model: "black-forest-labs/FLUX.2-dev",
    prompt: "a lighthouse on a rocky coast at golden hour, cinematic",
    size: "1024x1024",
    response_format: "b64_json",
  }),
});
if (!res.ok) throw new Error(`image generation failed: ${res.status}`);
const data = await res.json();
const b64 = data.data[0].b64_json; // decode as needed
```

---

## Endpoint 2 — `POST /v1/images/edits` (image-to-image / reference-edit)

Condition generation on **one or more reference images**. This is the endpoint
to use for "put glasses on this person", "restyle this scene", "change the
background", etc. Request is **`multipart/form-data`** (because it carries file
uploads).

### Request fields (form fields)

| Field | Type | Required | Default | Notes |
|-------|------|----------|---------|-------|
| `image` | file (repeated) | **yes** | — | 1 to **4** image files. Each part must be named `image`; repeat the part to send several. See the [reference-image cap](#reference-image-cap). |
| `prompt` | string | **yes** | — | The edit instruction. Empty/missing → 400. |
| `model` | string | no | `img.default_model` | Same resolution/validation as txt2img. |
| `n` | integer | no | `1` | Clamped to `img.max_n` (default `4`). |
| `size` | string | no | `img.default_size` | Same allowed-sizes / max-dimension guardrails as txt2img. |
| `response_format` | string | no | `"url"` | `"url"` or `"b64_json"`. |
| `num_inference_steps` | integer | no | **`img.edit_default_steps` (`8`)** | Edits default to **8 steps**, not 20 — see [8-step edit default](#8-step-edit-default). Still clamped to `img.max_steps`. |
| `guidance_scale` | float | no | `img.default_guidance_scale` (`3.5`) | — |
| `seed` | integer | no | `null` | — |
| `strength` | float | no | `null` | Accepted for forward-compat but **ignored by FLUX.2 Klein** — see [strength note](#strength-is-ignored-on-flux2-klein). |
| `user` | string | no | `null` | Sent as form field named `user`. Opaque end-user id. |

### Reference-image cap

At most **4** reference images per request (server constant `_MAX_EDIT_IMAGES`,
mirroring the diffusion backend's `MAX_REF_IMAGES` VRAM protection). Sending
more → **400** `"at most 4 reference image(s) allowed (got N)"`.

Each file must have an `image/*` content type (→ 400 `"each 'image' must be an
image file"`) and must be under `img.max_image_upload_mb` (default **10 MB**)
or → 400 `"reference image exceeds 10MB"`.

### 8-step edit default

FLUX.2 Klein reference-edits are structure-preserving and look better with
**fewer** diffusion steps. When you do **not** pass `num_inference_steps`, edits
default to `img.edit_default_steps` (**8**), whereas plain text-to-image
defaults to `img.default_steps` (**20**). Pass an explicit
`num_inference_steps` to override.

### `strength` is ignored on FLUX.2 Klein

The `strength` field (the usual img2img denoising-strength knob) is accepted and
carried through the canonical request for forward-compatibility, but **FLUX.2
Klein — a distilled, structure-preserving edit model — ignores it**. Do not rely
on `strength` to control how much the reference is preserved on that model;
control the result through the prompt and step count instead.

### Example — curl (multipart)

```bash
curl https://mindrouter.uidaho.edu/v1/images/edits \
  -H "X-API-Key: $MINDROUTER_API_KEY" \
  -F "image=@portrait.png;type=image/png" \
  -F "prompt=add round wire-frame glasses to this man, keep everything else" \
  -F "size=1024x1024" \
  -F "response_format=b64_json"
```

Two reference images (repeat the `image` part):

```bash
curl https://mindrouter.uidaho.edu/v1/images/edits \
  -H "X-API-Key: $MINDROUTER_API_KEY" \
  -F "image=@subject.png;type=image/png" \
  -F "image=@style_ref.png;type=image/png" \
  -F "prompt=render the subject in the painterly style of the second image" \
  -F "response_format=b64_json"
```

### Example — Python

```python
import base64
import requests

BASE = "https://mindrouter.uidaho.edu"
API_KEY = "..."

with open("portrait.png", "rb") as fh:
    files = [("image", ("portrait.png", fh, "image/png"))]
    data = {
        "prompt": "add round wire-frame glasses to this man",
        "size": "1024x1024",
        "response_format": "b64_json",
        # num_inference_steps omitted -> defaults to 8 for edits
    }
    resp = requests.post(
        f"{BASE}/v1/images/edits",
        headers={"X-API-Key": API_KEY},  # do NOT set Content-Type; requests sets the multipart boundary
        files=files,
        data=data,
        timeout=180,
    )

resp.raise_for_status()
out = resp.json()
with open("edited.png", "wb") as f:
    f.write(base64.b64decode(out["data"][0]["b64_json"]))
```

### Example — JavaScript (fetch + FormData)

```javascript
const BASE = "https://mindrouter.uidaho.edu";
const form = new FormData();
form.append("image", fileBlob, "portrait.png"); // fileBlob: a File/Blob of type image/*
form.append("prompt", "add round wire-frame glasses to this man");
form.append("size", "1024x1024");
form.append("response_format", "b64_json");

const res = await fetch(`${BASE}/v1/images/edits`, {
  method: "POST",
  headers: { "X-API-Key": process.env.MINDROUTER_API_KEY }, // let the browser set the multipart Content-Type
  body: form,
});
if (!res.ok) throw new Error(`edit failed: ${res.status}`);
const data = await res.json();
const b64 = data.data[0].b64_json;
```

---

## Response format

Both endpoints return the same JSON shape (HTTP 200). It is OpenAI-compatible,
plus two MindRouter metadata fields.

```json
{
  "created": 1753280000,
  "data": [
    {
      "url": "https://.../generated.png",
      "b64_json": null,
      "revised_prompt": null
    }
  ],
  "backend_id": 45,
  "processing_time_ms": 4213
}
```

| Field | Type | Notes |
|-------|------|-------|
| `created` | integer | Unix timestamp. |
| `data` | array | One entry per generated image (`n` entries). |
| `data[].url` | string \| null | Populated when `response_format` is `"url"`. |
| `data[].b64_json` | string \| null | Base64 PNG, populated when `response_format` is `"b64_json"`. |
| `data[].revised_prompt` | string \| null | Present only if the backend rewrote the prompt. |
| `backend_id` | integer | MindRouter metadata — which backend served it. |
| `processing_time_ms` | integer | MindRouter metadata — server-side render time. |

**`response_format`:**
- `"url"` (default for the public API) — `url` is set, `b64_json` is null.
- `"b64_json"` — `b64_json` holds a base64-encoded PNG; `url` is null. Use this
  when you want the bytes inline without a second fetch.

---

## Content-policy judge (LLM-as-judge)

If an administrator has configured a policy (`img.policy` non-empty), **every**
prompt on both endpoints is screened by an LLM judge **before** any image is
generated.

- **Models:** primary judge `img.judge_model`, optional fallback
  `img.judge_model_secondary`. If the primary errors, the secondary is tried.
- **Fail-closed:** if no judge model is reachable / both error, the request is
  **denied** (not silently allowed).
- **Injection-resistant:** the prompt is passed to the judge inside delimited
  `<PROMPT>` tags and the judge is instructed to ignore any instructions inside
  it. You cannot "talk past" the policy from within the prompt.
- **Ambiguity → FAIL:** the judge is told to err on the side of denial for
  ambiguous prompts (except under the edit exemption below).

### Edit-aware exemption

The judge is told **by the system** (based on whether reference images were
attached — a signal that cannot be spoofed from prompt text) when a request is
an **edit**. For edits, **deictic references** like "this man", "the person",
"the image", or "their shirt" refer to the reference image the judge cannot see
and are **expected** — the judge must **not** FAIL such prompts merely as
"ambiguous" or for "not providing a description". It still applies the full
policy to the described transformation and its likely resulting image. This is
why `/v1/images/edits` accepts natural edit phrasing that plain text-to-image
prompts would need to spell out.

### Denial response

A blocked prompt returns **400** with:

```json
{
  "detail": {
    "error": {
      "message": "Your image request was denied by content policy: <reason>",
      "type": "content_policy_violation",
      "code": "policy_violation"
    }
  }
}
```

Denied requests are recorded (status `FAILED`, `error_code=policy_violation`)
for auditing.

---

## Guardrails & admin config keys (`img.*`)

All of these are admin-tunable config keys. Shipped defaults in parentheses.

| Key | Default | Effect |
|-----|---------|--------|
| `img.enabled` | `true` | Global on/off. `false` → 503 on every request. |
| `img.default_model` | `black-forest-labs/FLUX.2-dev` | Model used when `model` is omitted. |
| `img.default_size` | `1024x1024` | Size used when `size` is omitted. |
| `img.allowed_sizes` | `512x512,768x768,1024x1024,1024x768,768x1024` | Comma list; `size` not in this list → 400. |
| `img.max_width` | `1024` | Hard width cap; exceeded → 400. |
| `img.max_height` | `1024` | Hard height cap; exceeded → 400. |
| `img.default_steps` | `20` | Default `num_inference_steps` for **text-to-image**. |
| `img.edit_default_steps` | `8` | Default `num_inference_steps` for **edits**. |
| `img.max_steps` | `50` | Steps above this are silently clamped down. |
| `img.default_guidance_scale` | `3.5` | Default `guidance_scale`. |
| `img.max_n` | `4` | `n` above this is silently clamped down. |
| `img.max_image_upload_mb` | `10` | Per-reference-image size cap (edits only); exceeded → 400. |
| `img.policy` | `""` | Policy text; empty disables the judge. |
| `img.judge_model` | `""` | Primary judge model. |
| `img.judge_model_secondary` | `""` | Fallback judge model. |

**Clamp vs. reject:** `n` and `num_inference_steps` above their caps are
**silently clamped** (the request still succeeds with the capped value). `size`
outside `img.allowed_sizes` or beyond `img.max_width`/`img.max_height`, and
oversized/too-many reference images, are **rejected with 400**.

---

## Error reference

| HTTP | When |
|------|------|
| 400 | Invalid JSON body; missing `prompt`; disallowed `size`; dimensions over max; > 4 reference images; non-image upload; reference image over size cap; **content-policy denial** (`content_policy_violation`). |
| 401 | Missing / invalid API key. |
| 403 | Image generation not enabled for your account. |
| 404 | `model` does not resolve to a known image model (`model_not_found`). |
| 503 | Image subsystem globally disabled (`img.enabled=false`). |

> Note: `/v1/images/generations` returns most 4xx errors in the standard
> `{"detail": ...}` FastAPI envelope; policy denials nest an OpenAI-style
> `{"error": {...}}` inside `detail`. Read `detail` first, then `detail.error`
> if present.
