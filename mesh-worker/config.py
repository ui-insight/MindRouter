############################################################
#
# mindrouter mesh-worker - async image-to-3D mesh service
#
# config.py: Worker configuration, per-engine defaults and the legal
#     request ranges (texture sizes, face budget).
#
# Standalone deploy artifact (own venv / container, GPU-resident). NOT
# imported by the gateway.
#
# Luke Sheneman — University of Idaho RCDS — sheneman@uidaho.edu
#
############################################################

"""Mesh worker configuration."""

import os
from dataclasses import dataclass, field
from typing import Dict, List

MODES = ("mock", "trellis2", "pixal3d", "hunyuan3d21")

# Served model name per engine (the gateway's models.name). Lowercase, like
# every other served name in the fleet. Override with MESH_WORKER_MODEL.
DEFAULT_MODEL: Dict[str, str] = {
    "mock": "mindrouter/mesh-mock",
    "trellis2": "microsoft/trellis.2-4b",
    "pixal3d": "tencentarc/pixal3d",
    "hunyuan3d21": "tencent/hunyuan3d-2.1",
}

# Checkpoint (local dir or Hugging Face repo id) per engine. In production
# this is ALWAYS a local dir mounted read-only with HF_HUB_OFFLINE=1.
DEFAULT_CKPT: Dict[str, str] = {
    "mock": "",
    "trellis2": "microsoft/TRELLIS.2-4B",
    "pixal3d": "TencentARC/Pixal3D",
    "hunyuan3d21": "tencent/Hunyuan3D-2.1",
}

# Face budget per engine: (min, max, default). TRELLIS.2's own demo offers a
# 100k-1M decimation slider (default 500k); we allow lower targets because game
# engines want them, and default lower because a 200k-face GLB is plenty for a
# preview and much smaller to move. Hunyuan3D 2.1 decimates with its own
# FaceReducer (upstream default 40k faces).
DEFAULT_FACES: Dict[str, tuple] = {
    "mock": (1_000, 1_000_000, 200_000),
    "trellis2": (1_000, 1_000_000, 200_000),
    "pixal3d": (1_000, 1_000_000, 200_000),
    "hunyuan3d21": (1_000, 1_000_000, 40_000),
}

# Hunyuan3D 2.1's paint pipeline bakes at its own texture_size (upstream
# default 4096); TRELLIS.2's o_voxel bake takes any size (demo slider
# 1024-4096). Same menu for every engine keeps the gateway simple.
TEXTURE_SIZES: List[int] = [1024, 2048, 4096]
DEFAULT_TEXTURE_SIZE = 2048

# Seeds are fed to torch.manual_seed and numpy; keep them in uint32 range.
MAX_SEED = 2**32 - 1


def _env(name: str, default: str = "") -> str:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip()


def _env_flag(name: str, default: bool) -> bool:
    raw = _env(name)
    if not raw:
        return default
    return raw.lower() not in ("0", "false", "no", "off")


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    return float(raw) if raw else default


def _mode() -> str:
    return _env("MESH_WORKER_MODE", "mock").lower()


@dataclass
class WorkerConfig:
    # mock (no GPU, deterministic placeholder GLB — dev + CI) | trellis2 |
    # pixal3d | hunyuan3d21. Unknown modes are refused at startup.
    mode: str = field(default_factory=_mode)
    # Served model name ("" = the engine default from DEFAULT_MODEL).
    model_id: str = field(default_factory=lambda: _env("MESH_WORKER_MODEL"))
    # Where finished GLB + preview files live until the result TTL expires.
    output_dir: str = field(
        default_factory=lambda: _env("MESH_WORKER_OUTPUT_DIR", "/tmp/mindrouter-mesh-worker")
    )
    # Shared secret for the X-Worker-Key header on every /v1/* route. Empty =
    # auth DISABLED (rollout window only). Same value on the gateway.
    api_key: str = field(default_factory=lambda: _env("MESH_WORKER_API_KEY"))

    # -- queue / lifecycle --------------------------------------------------
    max_queue: int = field(default_factory=lambda: _env_int("MESH_WORKER_MAX_QUEUE", 8))
    job_timeout_seconds: float = field(
        default_factory=lambda: _env_float("MESH_WORKER_JOB_TIMEOUT_SECONDS", 900.0)
    )
    # A timed-out job is reported failed at once, but the GPU thread cannot be
    # killed: it stops at the next stage boundary. If it is still running this
    # long after the timeout, /health answers 503 so the gateway stops routing.
    stuck_grace_seconds: float = field(
        default_factory=lambda: _env_float("MESH_WORKER_STUCK_GRACE_SECONDS", 300.0)
    )
    result_ttl_seconds: float = field(
        default_factory=lambda: _env_float("MESH_WORKER_RESULT_TTL_SECONDS", 86400.0)
    )
    # How often the TTL sweeper runs (0 = derive from the TTL, max 60 s).
    sweep_interval_seconds: float = field(
        default_factory=lambda: _env_float("MESH_WORKER_SWEEP_INTERVAL_SECONDS", 0.0)
    )

    # -- input limits -------------------------------------------------------
    max_image_bytes: int = field(
        default_factory=lambda: _env_int("MESH_WORKER_MAX_IMAGE_BYTES", 32 * 1024 * 1024)
    )
    max_image_pixels: int = field(
        default_factory=lambda: _env_int("MESH_WORKER_MAX_IMAGE_PIXELS", 64 * 1024 * 1024)
    )
    min_image_side: int = 32

    # -- face / texture menu (0 = engine default) ---------------------------
    min_faces: int = field(default_factory=lambda: _env_int("MESH_WORKER_MIN_FACES", 0))
    max_faces: int = field(default_factory=lambda: _env_int("MESH_WORKER_MAX_FACES", 0))
    default_faces: int = field(default_factory=lambda: _env_int("MESH_WORKER_DEFAULT_FACES", 0))
    texture_sizes: List[int] = field(default_factory=lambda: list(TEXTURE_SIZES))
    default_texture_size: int = DEFAULT_TEXTURE_SIZE

    # -- real engines -------------------------------------------------------
    # Local checkpoint dir or HF repo id ("" = DEFAULT_CKPT for the mode).
    checkpoint: str = field(default_factory=lambda: _env("MESH_WORKER_CKPT"))
    # Background removal for RGB inputs on the TRELLIS family. MUST stay an
    # MIT BiRefNet: briaai/RMBG-2.0 (the upstream default) is CC BY-NC and is
    # refused at load time.
    birefnet: str = field(default_factory=lambda: _env("MESH_WORKER_BIREFNET", "ZhengPeng7/BiRefNet"))
    # Gated DINOv3 image encoder (local dir staged by an admin who accepted
    # the DINOv3 license, or the official HF id resolved from the HF cache).
    dinov3: str = field(
        default_factory=lambda: _env("MESH_WORKER_DINOV3", "facebook/dinov3-vitl16-pretrain-lvd1689m")
    )
    # TRELLIS.2: 512 | 1024 | 1024_cascade | 1536_cascade; Pixal3D:
    # 1024_cascade | 1536_cascade. "" = upstream default for the mode.
    pipeline_type: str = field(default_factory=lambda: _env("MESH_WORKER_PIPELINE_TYPE"))
    # Keep every sub-model on the GPU (H200: plenty of room). 1 = upstream
    # low-VRAM mode (per-stage CPU<->GPU shuffling, slower).
    low_vram: bool = field(default_factory=lambda: _env_flag("MESH_WORKER_LOW_VRAM", False))
    # Pixal3D camera: manual horizontal FOV in radians; 0 = estimate with MoGe-2.
    pixal3d_fov: float = field(default_factory=lambda: _env_float("MESH_WORKER_PIXAL3D_FOV", 0.0))
    moge: str = field(default_factory=lambda: _env("MESH_WORKER_MOGE", "Ruicheng/moge-2-vitl"))
    # Pixal3D NAF upsampler: local clone of github.com/valeoai/NAF (torch.hub
    # source="local"); "" = the torch.hub cache under $TORCH_HOME.
    naf_dir: str = field(default_factory=lambda: _env("MESH_WORKER_NAF_DIR"))
    # Hunyuan3D 2.1 paint: RealESRGAN x4 weights (local file; "" = the
    # repo's hy3dpaint/ckpt/RealESRGAN_x4plus.pth).
    realesrgan_ckpt: str = field(default_factory=lambda: _env("MESH_WORKER_REALESRGAN_CKPT"))
    # Hunyuan3D 2.1 paint: DINOv2 image encoder (Apache-2.0, not gated).
    hunyuan_dino: str = field(default_factory=lambda: _env("MESH_WORKER_HY_DINO", "facebook/dinov2-giant"))
    # Hunyuan3D 2.1 checkout (hy3dshape/ + hy3dpaint/ live under it).
    hunyuan_repo: str = field(default_factory=lambda: _env("MESH_WORKER_HUNYUAN_REPO", "/opt/Hunyuan3D-2.1"))
    # Hunyuan3D 2.1 shape sampler (upstream Gradio defaults: 30 steps, octree 256;
    # the pipeline's own call defaults are 50 / 384 — slower, finer).
    hunyuan_steps: int = field(default_factory=lambda: _env_int("MESH_WORKER_HY_STEPS", 30))
    hunyuan_octree: int = field(default_factory=lambda: _env_int("MESH_WORKER_HY_OCTREE", 256))
    # Embed textures as WebP (EXT_texture_webp, ~5x smaller) instead of PNG.
    # Off by default: not every DCC tool / game engine imports WebP.
    glb_webp: bool = field(default_factory=lambda: _env_flag("MESH_WORKER_GLB_WEBP", False))
    # Run one small job at startup so the first user job does not pay the
    # CUDA/autotune warm-up.
    warmup: bool = field(default_factory=lambda: _env_flag("MESH_WORKER_WARMUP", True))

    # -- mock ---------------------------------------------------------------
    # Seconds of simulated work per stage (load-test progress/cancel).
    mock_stage_delay: float = field(
        default_factory=lambda: _env_float("MESH_WORKER_MOCK_STAGE_DELAY", 0.0)
    )

    def __post_init__(self) -> None:
        self.mode = (self.mode or "mock").lower()
        if self.mode not in MODES:
            raise ValueError(f"MESH_WORKER_MODE must be one of {', '.join(MODES)}, not {self.mode!r}")
        if not self.model_id:
            self.model_id = DEFAULT_MODEL[self.mode]
        if not self.checkpoint:
            self.checkpoint = DEFAULT_CKPT[self.mode]
        lo, hi, default = DEFAULT_FACES[self.mode]
        self.min_faces = self.min_faces or lo
        self.max_faces = self.max_faces or hi
        self.default_faces = self.default_faces or default
        if not (0 < self.min_faces <= self.default_faces <= self.max_faces):
            raise ValueError(
                f"face budget must satisfy 0 < min ({self.min_faces}) <= default "
                f"({self.default_faces}) <= max ({self.max_faces})"
            )
        if self.default_texture_size not in self.texture_sizes:
            raise ValueError(f"default texture size {self.default_texture_size} not in {self.texture_sizes}")
        if self.max_queue < 1:
            raise ValueError("MESH_WORKER_MAX_QUEUE must be >= 1")

    @property
    def sweep_interval(self) -> float:
        if self.sweep_interval_seconds > 0:
            return self.sweep_interval_seconds
        return max(0.05, min(60.0, self.result_ttl_seconds / 4))

    @property
    def max_body_bytes(self) -> int:
        # base64 is 4/3 of the raw bytes, plus slack for the other JSON fields.
        return self.max_image_bytes * 4 // 3 + 64 * 1024
