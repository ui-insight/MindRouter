############################################################
#
# mindrouter mesh-worker - async image-to-3D mesh service
#
# engines/base.py: The engine contract shared by mock + real engines.
#
#   An engine's generate() is a BLOCKING call that the JobManager runs on its
#   single GPU executor thread (never on the event loop). It reports stages
#   through a StageTracker, which is also the cooperative-cancel point: every
#   stage boundary checks the job's cancel flag and raises Cancelled.
#
# Luke Sheneman — University of Idaho RCDS — sheneman@uidaho.edu
#
############################################################

"""Engine protocol, stage tracking and engine errors."""

from typing import Any, Callable, Dict, Protocol

from PIL import Image

STAGES = ("queued", "preprocess", "shape", "texture", "export", "done")

ReportFn = Callable[[str, float], None]    # (stage, progress 0-100)
ShouldCancel = Callable[[], bool]


class Cancelled(Exception):
    """Raised at a stage boundary when cancellation was requested."""


class EngineInputError(ValueError):
    """The input was valid HTTP but unusable by the model (e.g. background
    removal found no foreground). Reported as error code 'invalid_input'."""


class StageTracker:
    """Stage/progress reporting + the cooperative cancel check."""

    def __init__(self, report: ReportFn, should_cancel: ShouldCancel):
        self._report = report
        self._should_cancel = should_cancel

    def check(self) -> None:
        if self._should_cancel():
            raise Cancelled()

    def enter(self, stage: str, progress: float) -> None:
        """Boundary: cancel check first, then publish the new stage."""
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage!r}")
        self.check()
        self._report(stage, progress)

    def progress(self, stage: str, progress: float) -> None:
        """In-stage progress update (no cancel check)."""
        self._report(stage, progress)


class MeshEngine(Protocol):
    def load(self) -> None: ...

    def capabilities(self) -> Dict[str, Any]: ...

    def generate(
        self,
        spec: Any,                 # inputs.MeshSpec
        image: Image.Image,        # RGBA (real alpha) or RGB
        glb_path: str,
        preview_path: str,
        tracker: StageTracker,
    ) -> Dict[str, Any]: ...
