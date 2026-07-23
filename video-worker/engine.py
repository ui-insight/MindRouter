############################################################
#
# mindrouter video-worker - LTX-2.3 async video generation service
#
# engine.py: Pluggable generation engine.
#
#   MockEngine — no GPU, deterministic placeholder MP4 (dev + CI).
#   LTXEngine  — real ltx_pipelines on the H200 (torch/ltx imported lazily so
#                this module loads without them).
#
# Generation is a blocking call run OFF the event loop by the JobManager, so
# GET /health stays under 5s while a render is in flight.
#
# Luke Sheneman — University of Idaho RCDS — sheneman@uidaho.edu
#
############################################################

"""Video generation engines (mock + LTX-2.3)."""

import time
from typing import Any, Callable, Dict, Protocol

from config import WorkerConfig, frames_for

# A minimal but structurally-valid MP4 (ftyp + empty moov). Enough for
# content-type, Range, and non-empty-file assertions without a codec.
_PLACEHOLDER_MP4 = (
    b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
    b"\x00\x00\x00\x08free"
    + b"\x00" * 512
)


class Cancelled(Exception):
    """Raised by an engine when cooperative cancellation is requested."""


ProgressCb = Callable[[int, int], None]      # (step, total_steps)
ShouldCancel = Callable[[], bool]


class VideoEngine(Protocol):
    def capabilities(self) -> Dict[str, Any]: ...

    def model_ids(self) -> list: ...

    def generate(
        self, spec: Dict[str, Any], dest_path: str,
        progress_cb: ProgressCb, should_cancel: ShouldCancel,
    ) -> Dict[str, Any]: ...


class MockEngine:
    """Deterministic placeholder engine. Simulates the two-stage step schedule,
    honors cancellation, and writes a valid-enough MP4."""

    def __init__(self, config: WorkerConfig):
        self.config = config

    def capabilities(self) -> Dict[str, Any]:
        return self.config.capabilities()

    def model_ids(self) -> list:
        return [self.config.model_id]

    def generate(self, spec, dest_path, progress_cb, should_cancel) -> Dict[str, Any]:
        # Mirror the distilled 8+4 two-stage schedule = 12 steps.
        total_steps = 12
        for step in range(1, total_steps + 1):
            if should_cancel():
                raise Cancelled()
            if self.config.mock_step_delay:
                time.sleep(self.config.mock_step_delay)
            progress_cb(step, total_steps)
        with open(dest_path, "wb") as fh:
            fh.write(_PLACEHOLDER_MP4)
        return {"duration_ms": int(float(spec["seconds"]) * 1000)}


class LTXEngine:
    """Real LTX-2.3 engine (mode=ltx). torch + ltx_pipelines are imported lazily
    in load() so this file imports on a machine without them.

    Phase-0 validated recipe (aspen1 GPU2, H200; see
    docs/video-generation-plan.md and the phase0 memory):
      - DistilledPipeline (two-stage 8+3 distilled), model resident via one
        construction; generation runs under torch.inference_mode() — WITHOUT it
        autograd retains the graph and OOMs at ~139GB.
      - quantization="fp8-cast": ~24GB peak (vs bf16 ~44GB and right at the
        141GB edge). fp8-cast needs NO custom ltx-kernels build.
      - Measured: ~35s per 5s 720p clip (121f), stable 24GB, zero leak.
      - Attention is torch SDPA (cuDNN on Hopper) — LTX ships no FA3 path.
      - LTX generates synchronized audio natively.
    Requires (installed in the worker's uv venv on the GPU node): torch cu130,
    torchvision, ltx-core, ltx-pipelines. Checkpoints under VIDEO_WORKER_CKPT_DIR.
    """

    def __init__(self, config: WorkerConfig):
        self.config = config
        self._pipeline = None
        self._encode_video = None
        self._tiling = None
        self._get_chunks = None

    def _paths(self):
        import os
        d = self.config.checkpoint_dir
        return {
            "dit": os.path.join(d, "ltx-2.3", "ltx-2.3-22b-distilled-1.1.safetensors"),
            "upsampler": os.path.join(d, "ltx-2.3", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
            "gemma": os.path.join(d, "gemma-3-12b"),
        }

    def load(self) -> None:  # pragma: no cover - requires GPU + ltx_pipelines
        import logging
        from ltx_pipelines.distilled import DistilledPipeline
        from ltx_pipelines.utils.media_io import encode_video
        from ltx_pipelines.utils.quantization_factory import QuantizationKind
        from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number

        logging.getLogger(__name__).info("Loading LTX-2.3 distilled pipeline (fp8-cast)…")
        p = self._paths()
        policy = QuantizationKind("fp8-cast").to_policy(checkpoint_path=p["dit"])
        self._pipeline = DistilledPipeline(
            distilled_checkpoint_path=p["dit"],
            gemma_root=p["gemma"],
            spatial_upsampler_path=p["upsampler"],
            loras=(),
            quantization=policy,
        )
        self._encode_video = encode_video
        self._tiling = TilingConfig.default()
        self._get_chunks = get_video_chunks_number

    def capabilities(self) -> Dict[str, Any]:
        return self.config.capabilities()

    def model_ids(self) -> list:
        return [self.config.model_id]

    def _build_images(self, spec, num_frames) -> tuple:  # pragma: no cover
        """Decode optional start/end conditioning images (base64) to temp files
        and place them at frame 0 / the last frame. Returns (images, tmp_paths)."""
        import base64
        import os
        import uuid as _uuid

        from ltx_pipelines.utils.args import ImageConditioningInput

        strength = float(spec.get("image_strength") or 1.0)
        images, tmp = [], []
        for key, frame_idx in (("start_image", 0), ("end_image", num_frames - 1)):
            b64 = spec.get(key)
            if not b64:
                continue
            path = os.path.join(self.config.output_dir, f"cond-{_uuid.uuid4().hex[:12]}.png")
            with open(path, "wb") as fh:
                fh.write(base64.b64decode(b64))
            tmp.append(path)
            images.append(ImageConditioningInput(path, frame_idx, strength))
        return images, tmp

    def generate(self, spec, dest_path, progress_cb, should_cancel) -> Dict[str, Any]:  # pragma: no cover
        import os
        import time
        import torch

        if self._pipeline is None:
            self.load()
        if should_cancel():
            raise Cancelled()

        width, height = (int(x) for x in spec["size"].split("x"))
        num_frames = spec.get("num_frames") or frames_for(spec["seconds"])
        fps = float(spec.get("fps") or self.config.default_fps)
        seed = int(spec["seed"]) if spec.get("seed") is not None else 42
        images, tmp_paths = self._build_images(spec, num_frames)

        # LTX drives its own internal tqdm denoise loops; we surface coarse
        # phase progress (per-step callbacks would require pipeline hooks).
        progress_cb(1, 3)
        t0 = time.time()
        try:
            with torch.inference_mode():
                video, audio = self._pipeline(
                    prompt=spec["prompt"], seed=seed, height=height, width=width,
                    num_frames=num_frames, frame_rate=fps, images=images, tiling_config=self._tiling,
                )
                progress_cb(2, 3)
                self._encode_video(
                    video=video, fps=fps, audio=audio, output_path=dest_path,
                    video_chunks_number=self._get_chunks(num_frames, self._tiling),
                )
        finally:
            for p in tmp_paths:
                try:
                    os.remove(p)
                except OSError:
                    pass
        progress_cb(3, 3)
        return {"duration_ms": int(num_frames / fps * 1000), "render_ms": int((time.time() - t0) * 1000)}


def build_engine(config: WorkerConfig) -> VideoEngine:
    if config.mode == "ltx":
        return LTXEngine(config)
    return MockEngine(config)
