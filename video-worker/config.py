############################################################
#
# mindrouter video-worker - LTX-2.3 async video generation service
#
# config.py: Worker configuration + the legal preset matrix.
#
# Standalone deploy artifact (own venv, GPU-resident). NOT imported by the
# gateway. See docs/video-generation-plan.md.
#
# Luke Sheneman — University of Idaho RCDS — sheneman@uidaho.edu
#
############################################################

"""Video worker configuration and preset matrix."""

import os
from dataclasses import dataclass, field
from typing import Dict, List

# The legal preset matrix. torch.compile specializes to exact shapes, so these
# are warmed at boot and anything off-menu is REJECTED (never silently
# recompiled inside a user's job). Kept in one place — the gateway mirrors it
# via vid.allowed_sizes / vid.allowed_durations, and the UI renders from
# GET /v1/capabilities so there is one source of truth.
#
# LTX legal constraints (measured Phase 0): the TWO-STAGE distilled pipeline
# requires width/height divisible by 64 (stage 1 renders at half-res, which must
# then be divisible by 32); frame count = 8k+1. 960x544 was invalid (544/64 is
# not integer) and is replaced by 1024x576. All four presets below are ÷64.
SUPPORTED_SIZES: List[str] = ["1280x704", "704x1280", "1024x576", "768x448"]

# duration seconds -> frame count at 24 fps. frames = 24*seconds + 1, which is
# always 8k+1 for whole seconds (LTX's required format).
DURATION_FRAMES: Dict[str, int] = {
    "4": 97, "5": 121, "8": 193, "10": 241, "12": 289, "15": 361, "20": 481,
}

QUALITY_TIERS: List[str] = ["draft", "standard", "final"]


@dataclass
class WorkerConfig:
    # "mock" (no GPU, deterministic placeholder output — dev + CI) or "ltx".
    mode: str = field(default_factory=lambda: os.environ.get("VIDEO_WORKER_MODE", "mock"))
    model_id: str = field(
        default_factory=lambda: os.environ.get("VIDEO_WORKER_MODEL", "lightricks/ltx-2.3-distilled")
    )
    # Where rendered artifacts land on the worker node.
    output_dir: str = field(
        default_factory=lambda: os.environ.get("VIDEO_WORKER_OUTPUT_DIR", "/tmp/mindrouter-video-worker")
    )
    # LTX checkpoint dir + fp8 flag (only used in mode=ltx).
    checkpoint_dir: str = field(default_factory=lambda: os.environ.get("VIDEO_WORKER_CKPT_DIR", ""))
    default_fps: int = 24
    # Mock-only: seconds of simulated work per step (keeps /health-under-load
    # tests meaningful; 0 in CI for speed).
    mock_step_delay: float = field(
        default_factory=lambda: float(os.environ.get("VIDEO_WORKER_MOCK_STEP_DELAY", "0"))
    )

    def capabilities(self) -> dict:
        return {
            "model_id": self.model_id,
            "supported_sizes": SUPPORTED_SIZES,
            "supported_durations": list(DURATION_FRAMES.keys()),
            "supported_fps": [self.default_fps],
            "quality_tiers": QUALITY_TIERS,
            "pipelines": ["t2v"],  # v1 is text-to-video only
            "max_frames": max(DURATION_FRAMES.values()),
        }


def frames_for(seconds) -> int:
    """Frame count for a legal duration, or raise ValueError.

    Tolerant of numeric forms: "10", "10.0", 10, and 10.0 all resolve to the
    "10" preset key (durations are whole seconds), so a float that round-tripped
    through the DB (10.0) doesn't spuriously miss the matrix.
    """
    key = str(seconds).strip()
    if key not in DURATION_FRAMES:
        try:
            f = float(key)
            if f.is_integer():
                key = str(int(f))
        except (ValueError, TypeError):
            pass
    if key not in DURATION_FRAMES:
        raise ValueError(f"duration '{seconds}' not in preset matrix {list(DURATION_FRAMES)}")
    return DURATION_FRAMES[key]
