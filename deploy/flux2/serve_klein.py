############################################################
#
# MindRouter FLUX.2 Klein Image Generation API Server
#
# OpenAI-compatible /v1/images/generations + /v1/images/edits
# endpoints backed by FLUX.2 Klein 9B via HuggingFace diffusers.
#
# GPU-resident, BF16, no CPU offload.
#
# Luke Sheneman / RCDS / University of Idaho
#
############################################################

import argparse
import asyncio
import base64
import io
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field

# Cap reference images per edit to protect VRAM. Phase A showed a single 1024²
# reference costs the same peak as txt2img (~37GB); more/larger references add
# memory, and GPU2 only has ~7.6GB headroom over the two live instances.
MAX_REF_IMAGES = 4

# ---------------------------------------------------------------------------
# Request / Response schemas (OpenAI-compatible)
# ---------------------------------------------------------------------------

class ImageGenerationRequest(BaseModel):
    model: str = "black-forest-labs/FLUX.2-klein-9B"
    prompt: str
    n: int = 1
    size: str = "1024x1024"
    quality: str = "standard"
    style: Optional[str] = None
    response_format: str = "b64_json"  # "url" or "b64_json"
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None
    user: Optional[str] = None


class ImageEditRequest(BaseModel):
    """Reference-edit (img2img). Same knobs as generation plus one or more
    base64-encoded reference images. FLUX.2 Klein edits are structure-preserving
    and conditioned on the prompt; `strength` is accepted for forward-compat but
    Klein (a step-wise distilled model) does not use it."""
    model: str = "black-forest-labs/FLUX.2-klein-9B"
    prompt: str
    image: List[str] = Field(default_factory=list)  # base64 PNG/JPEG (data-URI ok)
    n: int = 1
    size: str = "1024x1024"
    quality: str = "standard"
    strength: Optional[float] = None  # accepted, ignored by Klein
    response_format: str = "b64_json"  # "url" or "b64_json"
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    seed: Optional[int] = None
    user: Optional[str] = None


class ImageData(BaseModel):
    b64_json: Optional[str] = None
    url: Optional[str] = None
    revised_prompt: Optional[str] = None


class ImageGenerationResponse(BaseModel):
    created: int
    data: List[ImageData]


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
pipe = None
model_id = None
device_str = None
gen_lock = threading.Lock()  # Serialize GPU access


def parse_size(size: str):
    """Parse 'WxH' string into (width, height)."""
    parts = size.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid size format: {size}")
    return int(parts[0]), int(parts[1])


def _decode_b64_image(s: str) -> Image.Image:
    """Decode a base64 image (raw or data-URI) into an RGB PIL image."""
    if s.strip().startswith("data:") and "," in s:
        s = s.split(",", 1)[1]
    raw = base64.b64decode(s)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _pack_image(image: Image.Image, response_format: str, prompt: str) -> ImageData:
    """Serialize a rendered PIL image to the requested response shape."""
    if response_format == "b64_json":
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return ImageData(b64_json=b64, revised_prompt=prompt)
    os.makedirs("/data/flux2/output", exist_ok=True)
    fname = f"{uuid.uuid4().hex}.png"
    image.save(f"/data/flux2/output/{fname}", format="PNG")
    return ImageData(url=f"/images/{fname}", revised_prompt=prompt)


def load_pipeline(model: str, device: str):
    """Load the FLUX.2 Klein pipeline, fully on GPU."""
    from diffusers import Flux2KleinPipeline

    print(f"Loading {model} on {device}...")
    t0 = time.time()

    pipeline = Flux2KleinPipeline.from_pretrained(
        model,
        torch_dtype=torch.bfloat16,
    ).to(device)

    elapsed = time.time() - t0
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"Pipeline loaded in {elapsed:.1f}s, VRAM: {vram:.1f} GB")
    return pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global pipe, model_id, device_str
    pipe = load_pipeline(model_id, device_str)
    yield
    del pipe
    torch.cuda.empty_cache()


app = FastAPI(title="FLUX.2 Klein Image Generation API", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    vram = torch.cuda.memory_allocated() / 1e9
    return {"status": "ok", "vram_gb": round(vram, 1), "model": "FLUX.2-klein-9B"}


@app.get("/v1/models")
async def list_models():
    served_name = app.state.served_model_name
    return {
        "object": "list",
        "data": [
            {
                "id": served_name,
                "object": "model",
                "created": 0,
                "owned_by": "black-forest-labs",
            }
        ],
    }


def _generate_sync(req, width, height, num_steps, guidance, generator):
    """Run generation under the GPU lock (called from thread pool)."""
    with gen_lock:
        with torch.inference_mode():
            result = pipe(
                prompt=req.prompt,
                width=width,
                height=height,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                generator=generator,
            )
        return result.images[0]


def _edit_sync(prompt, ref_images, width, height, num_steps, guidance, generator):
    """Run a reference-edit under the GPU lock. `image=[pil,...]` is the
    reference-edit call shape validated in Phase A (diffusers 0.37.1)."""
    with gen_lock:
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                image=ref_images,
                width=width,
                height=height,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                generator=generator,
            )
        return result.images[0]


@app.post("/v1/images/generations")
async def generate_images(req: ImageGenerationRequest):
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    width, height = parse_size(req.size)

    # Klein defaults: fewer steps needed than Dev
    num_steps = req.num_inference_steps or (28 if req.quality == "hd" else 20)
    guidance = req.guidance_scale if req.guidance_scale is not None else 4.0

    # Seed handling
    generator = None
    if req.seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(req.seed)

    images_data: List[ImageData] = []

    for i in range(req.n):
        t0 = time.time()

        # Run in thread pool so we don't block the event loop
        loop = asyncio.get_event_loop()
        image = await loop.run_in_executor(
            None, _generate_sync, req, width, height, num_steps, guidance, generator
        )

        elapsed = time.time() - t0
        print(f"Generated image {i+1}/{req.n} in {elapsed:.1f}s ({width}x{height}, {num_steps} steps)")

        images_data.append(_pack_image(image, req.response_format, req.prompt))

    return ImageGenerationResponse(
        created=int(time.time()),
        data=images_data,
    )


@app.post("/v1/images/edits")
async def edit_images(req: ImageEditRequest):
    """Reference-edit (img2img): render `prompt` conditioned on reference image(s)."""
    if pipe is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not req.image:
        raise HTTPException(status_code=400, detail="at least one reference image is required")
    if len(req.image) > MAX_REF_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"at most {MAX_REF_IMAGES} reference image(s) allowed (got {len(req.image)})",
        )
    try:
        ref_images = [_decode_b64_image(s) for s in req.image]
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid reference image: {e}")

    width, height = parse_size(req.size)
    num_steps = req.num_inference_steps or (28 if req.quality == "hd" else 20)
    guidance = req.guidance_scale if req.guidance_scale is not None else 4.0

    generator = None
    if req.seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(req.seed)

    images_data: List[ImageData] = []

    for i in range(req.n):
        t0 = time.time()

        loop = asyncio.get_event_loop()
        image = await loop.run_in_executor(
            None, _edit_sync, req.prompt, ref_images, width, height, num_steps, guidance, generator
        )

        elapsed = time.time() - t0
        print(
            f"Edited image {i+1}/{req.n} in {elapsed:.1f}s "
            f"({len(ref_images)} ref, {width}x{height}, {num_steps} steps)"
        )

        images_data.append(_pack_image(image, req.response_format, req.prompt))

    return ImageGenerationResponse(
        created=int(time.time()),
        data=images_data,
    )


from fastapi.staticfiles import StaticFiles


def main():
    global model_id, device_str

    parser = argparse.ArgumentParser(description="FLUX.2 Klein OpenAI-compatible API server")
    parser.add_argument("--model", default="/data/flux2/models/FLUX.2-klein-9B",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--served-model-name", default=None,
                        help="Model name exposed via /v1/models (default: --model value)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18100)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    model_id = args.model
    device_str = args.device
    app.state.served_model_name = args.served_model_name or args.model

    os.makedirs("/data/flux2/output", exist_ok=True)
    app.mount("/images", StaticFiles(directory="/data/flux2/output"), name="images")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
