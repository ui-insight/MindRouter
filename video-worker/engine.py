############################################################
#
# mindrouter video-worker - async video generation service
#
# engine.py: Pluggable generation engine.
#
#   MockEngine — no GPU, deterministic placeholder MP4 (dev + CI).
#   LTXEngine  — real ltx_pipelines on the H200 (torch/ltx_pipelines imported
#                lazily so this module loads without them). Serves LTX-2.3
#                (one checkpoint) or LTX-2.5 (split checkpoints), weights
#                resident between renders.
#
# Generation is a blocking call run OFF the event loop by the JobManager, so
# GET /health stays under 5s while a render is in flight.
#
# Luke Sheneman — University of Idaho RCDS — sheneman@uidaho.edu
#
############################################################

"""Video generation engines (mock + real GPU-backed video model)."""

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
    """Real video-model engine (mode=ltx). torch + ltx_pipelines are imported
    lazily in load() so this file imports on a machine without them.

    Two checkpoint layouts, picked by ``config.ltx_layout``:
      - "monolith" (LTX-2.3): one fat checkpoint + a separate Gemma 3 folder,
        driven by the ltx_pipelines release of July 2026 (its own venv).
      - "split" (LTX-2.5): one file per component (transformer, Gemma 4 text
        encoder, video VAE, audio VAE), driven by ltx_pipelines >= 1.2 (its
        own venv; the July release cannot load 2.5).

    Recipe (aspen1 GPU2, H200; see docs/video-generation-plan.md):
      - DistilledPipeline (two-stage 8+4 distilled), quantization fp8-cast.
      - Generation runs under torch.inference_mode() — WITHOUT it autograd
        retains the graph and OOMs at ~139GB.
      - Weights stay resident (``config.resident``): the pipeline is built with
        a weight registry, so the first render loads every component onto the
        GPU and later renders reuse them. Measured on 2.3: ~31 s -> ~11 s per
        5 s 720p clip, output bit-identical, ~55 GB held between renders
        (vs ~24 GB peak / 1.7 GB idle when reloading per render).
      - ``config.warmup`` renders one small clip at startup so the first user
        job does not pay the load.
      - The model generates synchronized audio natively.
    """

    # LTX-2.5 split layout, relative to the checkpoint dir (the Hugging Face
    # repo's own folder layout, kept by `hf download --local-dir`).
    SPLIT_FILES = {
        "transformer": "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors",
        "text_encoder": "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors",
        "video_vae": "vae/ltx-2.5-video-vae-bf16.safetensors",
        "audio_vae": "vae/ltx-2.5-audio-vae-bf16.safetensors",
        "upsampler": "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    }

    def __init__(self, config: WorkerConfig):
        self.config = config
        self._pipeline = None
        self._encode_video = None
        self._tiling = None
        self._get_chunks = None

    def _paths(self):
        import os
        d = self.config.checkpoint_dir
        if self.config.ltx_layout == "split":
            paths = {k: os.path.join(d, v) for k, v in self.SPLIT_FILES.items()}
            if self.config.video_vae_file:
                paths["video_vae"] = os.path.join(d, self.config.video_vae_file)
            return paths
        return {
            "dit": os.path.join(d, "ltx-2.3", "ltx-2.3-22b-distilled-1.1.safetensors"),
            "upsampler": os.path.join(d, "ltx-2.3", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"),
            "gemma": os.path.join(d, "gemma-3-12b"),
        }

    def load(self) -> None:  # pragma: no cover - requires GPU + ltx_pipelines
        if self.config.ltx_layout not in ("monolith", "split"):
            raise ValueError(f"VIDEO_WORKER_LTX_LAYOUT must be 'monolith' or 'split', not {self.config.ltx_layout!r}")
        import logging
        from ltx_pipelines.distilled import DistilledPipeline
        from ltx_pipelines.utils.media_io import encode_video
        from ltx_pipelines.utils.quantization_factory import QuantizationKind
        from ltx_core.model.video_vae import get_video_chunks_number

        log = logging.getLogger(__name__)
        p = self._paths()
        if self.config.ltx_layout == "split":
            from ltx_core.loader import ModelRegistry
            from ltx_pipelines.utils.model_paths import ModelPaths

            log.info("Loading the two-stage distilled video pipeline, split layout (fp8-cast, resident=%s)…",
                     self.config.resident)
            policy = QuantizationKind("fp8-cast").to_policy(checkpoint_path=p["transformer"])
            self._pipeline = DistilledPipeline(
                model_paths=ModelPaths.from_split(
                    transformer_path=p["transformer"],
                    text_encoder_path=p["text_encoder"],
                    video_vae_path=p["video_vae"],
                    audio_vae_path=p["audio_vae"],
                ),
                spatial_upsampler_path=p["upsampler"],
                loras=(),
                quantization=policy,
                registry=ModelRegistry() if self.config.resident else None,
            )
            self._tiling = None  # the pipeline sizes decode tiles itself (AUTO_TILING)
        else:
            from ltx_core.loader import StateDictRegistry
            from ltx_core.model.video_vae import TilingConfig

            log.info("Loading the two-stage distilled video pipeline (fp8-cast, resident=%s)…",
                     self.config.resident)
            policy = QuantizationKind("fp8-cast").to_policy(checkpoint_path=p["dit"])
            self._pipeline = DistilledPipeline(
                distilled_checkpoint_path=p["dit"],
                gemma_root=p["gemma"],
                spatial_upsampler_path=p["upsampler"],
                loras=(),
                quantization=policy,
                registry=StateDictRegistry() if self.config.resident else None,
            )
            self._tiling = TilingConfig.default()
        self._encode_video = encode_video
        self._get_chunks = get_video_chunks_number

        if self.config.resident and self.config.warmup:
            try:
                self._warm_up(log)
            except Exception:
                # Not fatal: e.g. out of memory because the other model on this
                # GPU is mid-render. Serve anyway; the first job pays the load.
                log.warning("Warm-up render failed; the first job will load the weights.", exc_info=True)
                import torch

                torch.cuda.empty_cache()

    def _warm_up(self, log) -> None:  # pragma: no cover - requires GPU
        """Render one small clip so every component is loaded and cached before
        the first real job. Uses the smallest preset; output is discarded."""
        import os
        import time
        import uuid as _uuid

        os.makedirs(self.config.output_dir, exist_ok=True)
        dest = os.path.join(self.config.output_dir, f"warmup-{_uuid.uuid4().hex[:8]}.mp4")
        started = time.time()
        try:
            self._render(prompt="A calm lake at dawn, gentle water sounds.", seed=1,
                         width=768, height=448, num_frames=frames_for(4), fps=24.0,
                         images=[], dest_path=dest)
            log.info("Warm-up render done in %.1fs; weights are resident.", time.time() - started)
        finally:
            self._release_working_memory()
            try:
                os.remove(dest)
            except OSError:
                pass

    def _render(self, *, prompt, seed, width, height, num_frames, fps, images, dest_path,
                on_generated=None) -> None:  # pragma: no cover
        """Generate and encode one clip; ``on_generated`` runs between the two."""
        import torch

        on_generated = on_generated or (lambda: None)

        with torch.inference_mode():
            if self.config.ltx_layout == "split":
                result = self._pipeline(
                    prompt=prompt, seed=seed, height=height, width=width,
                    num_frames=num_frames, frame_rate=fps, images=images,
                )
                on_generated()
                self._encode_video(
                    video=result.video, fps=fps, audio=result.audio, output_path=dest_path,
                    video_chunks_number=self._get_chunks(result.num_frames, result.tiling_config),
                )
            else:
                video, audio = self._pipeline(
                    prompt=prompt, seed=seed, height=height, width=width,
                    num_frames=num_frames, frame_rate=fps, images=images, tiling_config=self._tiling,
                )
                on_generated()
                self._encode_video(
                    video=video, fps=fps, audio=audio, output_path=dest_path,
                    video_chunks_number=self._get_chunks(num_frames, self._tiling),
                )

    @staticmethod
    def _release_working_memory() -> None:  # pragma: no cover - requires GPU
        """Hand a render's working memory back to the GPU. PyTorch otherwise
        keeps it reserved for this process, and with two models resident on
        one GPU that reserve squeezes the other model's renders: measured on
        aspen1, a 30 s 2.3 render peaked within 0.7 GB of the 141 GB card
        while 2.5 sat on ~10 GB it no longer used. Resident weights stay."""
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()

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

        if self.config.ltx_layout == "split":
            from ltx_pipelines.utils.types import ImageConditioningInput
        else:
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

        if self._pipeline is None:
            self.load()
        if should_cancel():
            raise Cancelled()

        width, height = (int(x) for x in spec["size"].split("x"))
        num_frames = spec.get("num_frames") or frames_for(spec["seconds"])
        fps = float(spec.get("fps") or self.config.default_fps)
        seed = int(spec["seed"]) if spec.get("seed") is not None else 42
        images, tmp_paths = self._build_images(spec, num_frames)

        # The pipeline drives its own internal tqdm denoise loops; we surface coarse
        # phase progress (per-step callbacks would require pipeline hooks).
        progress_cb(1, 3)
        t0 = time.time()
        try:
            self._render(prompt=spec["prompt"], seed=seed, width=width, height=height,
                         num_frames=num_frames, fps=fps, images=images, dest_path=dest_path,
                         on_generated=lambda: progress_cb(2, 3))
        finally:
            self._release_working_memory()
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
