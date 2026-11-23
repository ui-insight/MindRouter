############################################################
#
# mindrouter mesh-worker - async image-to-3D mesh service
#
# engines/__init__.py: build_engine(config) picks the engine for
#     MESH_WORKER_MODE. The real-engine modules import NO heavy dependency at
#     module level (torch, trellis2, pixal3d, hy3dshape, ...): those load in
#     engine.load(), so this package imports and the tests run on mock deps.
#
#   NOTE: never add a top-level module or package named `utils` to the
#   mesh-worker: Hunyuan3D 2.1's paint pipeline imports its own top-level
#   `utils.*` package from sys.path.
#
# Luke Sheneman — University of Idaho RCDS — sheneman@uidaho.edu
#
############################################################

"""Engine selection."""

from config import WorkerConfig
from engines.base import Cancelled, EngineInputError, MeshEngine, StageTracker

__all__ = ["build_engine", "Cancelled", "EngineInputError", "MeshEngine", "StageTracker"]


def build_engine(config: WorkerConfig) -> MeshEngine:
    if config.mode == "mock":
        from engines.mock import MockEngine
        return MockEngine(config)
    if config.mode == "trellis2":
        from engines.trellis_family import Trellis2Engine
        return Trellis2Engine(config)
    if config.mode == "pixal3d":
        from engines.trellis_family import Pixal3DEngine
        return Pixal3DEngine(config)
    if config.mode == "hunyuan3d21":
        from engines.hunyuan3d21 import Hunyuan3D21Engine
        return Hunyuan3D21Engine(config)
    raise ValueError(f"unknown MESH_WORKER_MODE {config.mode!r}")
