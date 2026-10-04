# clef_service

A System One HTTP server for [Cloudflare's Clef](https://huggingface.co/Cloudflare/clef)
decision model, so MindRouter can use Clef as a decisions upstream.

Clef is Qwen3.8-27B post-trained for decisions, with a small "joint schema
head" (Apache-2.0). It reads the state and every question in **one** forward
pass and returns a probability for every allowed option. Cloudflare publishes
the weights and a Python function (`systemone()` in the model repo's
`joint_schema_model.py`) but no server; this package is the server.

## What it adds

- `POST /v1/systemone` in TypeSafe's System One wire format (the same body
  MindRouter's `/v1/systemone` takes), answered by Clef.
- A bearer key (`CLEF_API_KEY`). Without one the service refuses to start
  unless `CLEF_ALLOW_NO_AUTH=1`.
- One inference thread with a queue in front, so concurrent callers are served
  in turn. **Optional dynamic batching** (`CLEF_MAX_BATCH` > 1): requests
  arriving within `CLEF_BATCH_WAIT_MS` of each other are answered by one
  forward pass (up to `CLEF_MAX_BATCH_TOKENS` padded tokens). It is off by
  default because it did not help mixed-length traffic (see Performance). Do
  not run two instances on one GPU: they share the same compute and hold the
  55 GB of weights twice.
- A bounded queue: beyond `CLEF_MAX_QUEUE` waiting requests the answer is
  `503` with `Retry-After`, not an ever-growing backlog.
- A caller that disconnects or times out while waiting is dropped from the
  queue; the GPU is not spent on an answer nobody will read.
- `GET /health`: `{"status": "ok"}` for anyone (`"loading"` while the weights
  load); model, device, queue depth and batch statistics with the key.
- Truncation is reported. Clef cuts the state to fit `CLEF_MAX_LENGTH` tokens;
  `usage.truncated` and `usage.state_tokens_dropped` say when and by how much.
- Nothing about a request is logged, including error text.

**Images.** A request may carry `images`, in the format Cloudflare publishes
for Clef on Workers AI: up to 4 items, each a base64 data URL
(`data:image/png;base64,...`) or `{"content_type": "image/png", "base64": "..."}`;
PNG, JPEG or WebP; 4 MiB and 16 megapixels each, 8 MiB in total. They are shown
to the model before the state. Remote URLs are not fetched. `videos` are
refused with 422.

## Settings (environment)

| Variable | Default | Meaning |
|---|---|---|
| `CLEF_API_KEY` | (required) | Bearer key callers must send |
| `CLEF_MODEL` | `Cloudflare/clef` | HF repo id or a local snapshot directory |
| `CLEF_SERVED_NAME` | `clef` | Name reported in replies |
| `CLEF_DEVICE` | `cuda` | Torch device (pick the card with `CUDA_VISIBLE_DEVICES`) |
| `CLEF_HOST` / `CLEF_PORT` | `127.0.0.1` / `18004` | Bind address; put nginx TLS in front |
| `CLEF_MAX_LENGTH` | `16384` | Tokens per request; the state is cut to fit |
| `CLEF_MAX_BATCH` | `1` | Requests per forward pass. Keep 1 for mixed traffic (see Performance) |
| `CLEF_BATCH_WAIT_MS` | `5` | How long the first request waits for others |
| `CLEF_MAX_BATCH_TOKENS` | `65536` | Padded tokens per forward pass (batch size × longest request) |
| `CLEF_MAX_QUEUE` | `64` | Waiting requests before 503 |
| `CLEF_MAX_BODY_BYTES` | `13631488` | Request body limit, 13 MiB (413 above it) |
| `CLEF_CUDNN_ATTENTION` | off | `1` re-enables PyTorch's cuDNN attention backend (see Performance) |

## Install and run

```bash
python3.11 -m venv /path/to/.venv-clef
/path/to/.venv-clef/bin/pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128
/path/to/.venv-clef/bin/pip install -r clef_service/requirements.txt
HF_HOME=/path/to/models /path/to/.venv-clef/bin/hf download Cloudflare/clef   # 55 GB

CLEF_API_KEY=... CUDA_VISIBLE_DEVICES=1 HF_HOME=/path/to/models \
    /path/to/.venv-clef/bin/python -m clef_service
```

`deploy/clef-service.service` is a systemd unit template.

## Performance (measured on one H200, 2026-10-04)

Three things decide how fast this is. Together they are worth about 6x on
real traffic and do not change accuracy (0.726 vs 0.725 on 2,000 benchmark
decisions).

1. **Install the fast path.** Without `flash-linear-attention` and
   `causal-conv1d`, the backbone falls back to a plain-torch implementation and
   every request takes about 2.5x longer (a 313-token request: 194 ms vs 76 ms).
   `causal-conv1d` compiles against CUDA:

   ```bash
   pip install flash-linear-attention wheel setuptools ninja
   CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH \
       CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install --no-build-isolation --no-deps causal-conv1d
   ```

   `/health` (with the key) reports `"fast_path": true` when both are in use,
   and the service logs `clef_slow_path` at start when they are not. Point
   `TRITON_CACHE_DIR` at local disk; kernels are compiled on first use.

2. **Leave the cuDNN attention backend off** (the default here). PyTorch's
   cuDNN attention re-plans for every new input length, which added about
   1.2 s to the first request at each length. Real requests almost all differ
   in length, so that was 1.2 s on nearly every request.

3. **One request per forward pass** (`CLEF_MAX_BATCH=1`, the default). The
   model's cost is close to linear in tokens, so a batch of requests padded to
   the longest costs as much as running them one after another, and padded
   batches of mixed lengths sometimes stalled for about 10 s. Batching only
   paid off for many short requests of the same length (about 2x).

**Check it with a real request, not just `/health`.** A model server can be
"healthy" and fail every inference (we hit exactly that with another decision
model: its container reported healthy and returned 500 to every request).

```bash
curl -s http://127.0.0.1:18004/v1/systemone -H "Authorization: Bearer $CLEF_API_KEY" \
  -H 'Content-Type: application/json' -d '{
    "model": "clef", "state": "Checkout has been failing for every customer for the last hour.",
    "questions": {"urgent": {"type": "noul", "instructions": "Is this urgent?"}}}'
```

## Using it from MindRouter

Admin → Settings → Decisions API → upstream servers:

```json
{"clef": {"url": "https://<host>:<tls-port>", "api_key": "<CLEF_API_KEY>", "model": "clef", "images": true}}
```

`"images": true` tells MindRouter this upstream can see; without it a request
with images for this model is refused rather than answered from the text alone.

Callers then send `"model": "clef"` to MindRouter's `/v1/systemone`. See
`docs/decisions-api.md`.

## The model's code

The service imports `joint_schema_model.py` from the downloaded model
snapshot (that file is part of Cloudflare's release, not of this repo). It
uses `load_release_model`, `encode_record`, `collate_records` and
`systemone_answer` from it.
