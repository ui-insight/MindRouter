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

    The GPU-resident serving decisions (see docs/video-generation-plan.md):
      - fp8-scaled-mm checkpoints, no offload (141GB / 80GB holds it)
      - assert FlashAttention-3 actually dispatched on sm_90 (log if it fell
        back to SDPA — the public reference runs silently lost ~2x)
      - torch.compile warmed over the preset matrix at load()
    """

    def __init__(self, config: WorkerConfig):
        self.config = config
        self._pipeline = None

    def load(self) -> None:  # pragma: no cover - requires GPU + ltx_pipelines
        import torch  # noqa: F401
        from ltx_pipelines import distilled  # type: ignore

        # Build/load the distilled fp8 pipeline resident on the GPU, assert FA3,
        # then warm torch.compile over every (size, frames) preset. Left as the
        # single GPU-node integration point; Phase 0 measures and tunes it.
        self._pipeline = distilled  # placeholder binding for the real pipeline
        raise NotImplementedError("LTXEngine.load is wired during Phase 0 on the GPU node")

    def capabilities(self) -> Dict[str, Any]:
        return self.config.capabilities()

    def model_ids(self) -> list:
        return [self.config.model_id]

    def generate(self, spec, dest_path, progress_cb, should_cancel) -> Dict[str, Any]:  # pragma: no cover
        if self._pipeline is None:
            self.load()
        width, height = (int(x) for x in spec["size"].split("x"))
        num_frames = frames_for(spec["seconds"])
        # The real pipeline call reports per-step progress via a callback and
        # checks should_cancel between denoising steps; it writes the MP4 to
        # dest_path and returns wall/gpu timing. Implemented on the GPU node.
        raise NotImplementedError("LTXEngine.generate is wired during Phase 0 on the GPU node")


def build_engine(config: WorkerConfig) -> VideoEngine:
    if config.mode == "ltx":
        return LTXEngine(config)
    return MockEngine(config)
