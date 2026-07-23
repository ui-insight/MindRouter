# FLUX.2 Klein image server (`serve_klein.py`)

Custom OpenAI-compatible image server backing MindRouter **backend 45**
(`diffusion` @ `webbyg2.hpc.uidaho.edu:8002`, nginx LB over `flux2-klein-gpu2{a,b}`
on 18100/18101). Deployed on webbyg2 at `/data/flux2/serve_klein.py`
(venv `/data/flux2/venv`, model `black-forest-labs/FLUX.2-klein-9B`). This copy is
the version-controlled source of truth — keep it in sync with the node.

## Endpoints

- `GET /health`, `GET /v1/models`
- `POST /v1/images/generations` — txt2img (OpenAI-compatible, JSON in / b64 or url out)
- `POST /v1/images/edits` — **img2img / reference-edit** (see below)

## `/v1/images/edits` (img2img)

JSON body (backend↔backend contract; MindRouter translates the public OpenAI
multipart form into this):

```json
{
  "prompt": "the apple is now bright green",
  "image": ["<base64 PNG/JPEG, data-URI ok>"],
  "size": "1024x1024",
  "num_inference_steps": 20,
  "seed": 7,
  "response_format": "b64_json"
}
```

- `image`: 1..`MAX_REF_IMAGES` (=4) base64 reference images. Reference-edit is
  structure-preserving and prompt-conditioned (validated Phase A: red→green apple,
  same shape/stem/wood-grain/background).
- `strength`: accepted for forward-compat but **ignored** — Klein is a step-wise
  distilled model (no denoise strength; `guidance_scale` is likewise ignored).
- Guards: `400` on no image, >4 images, or undecodable base64.
- VRAM: reference-edit peak == txt2img peak (~37GB), so edits run on the existing
  GPU2 instances — no extra GPU. Timing @20 steps 1024²: ~13.6s (vs ~7.3s txt2img).

## Deploy / restart

Overwriting `/data/flux2/serve_klein.py` is inert until the services restart
(running process holds the old code in memory). A backup is kept at
`serve_klein.py.pre-img2img.bak` on the node. To activate the edits route:

```
sudo systemctl restart flux2-klein-gpu2a flux2-klein-gpu2b
```

This briefly interrupts image generation on backend 45 (two instances, ~10s model
reload each). Requires `CUDA_DEVICE_ORDER=PCI_BUS_ID` in the unit (already set) so
CUDA targets the correct physical GPU2 rather than a busy NVL card.

See `docs/img2img-scope.md` for the full plan and Phase A results.
