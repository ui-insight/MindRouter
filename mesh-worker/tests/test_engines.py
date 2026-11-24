"""Real-engine adapters WITHOUT a GPU: they must import on mock deps, refuse
bad configuration before any heavy import, and their load-time patches (the
RMBG-2.0 license guard, the DINOv3 path override) are exercised against FAKE
upstream modules. The actual model calls are validated in the GPU bake-off.
"""

import math
import os
import sys
import types

import numpy as np
import pytest

from config import DEFAULT_CKPT, DEFAULT_MODEL, MODES, WorkerConfig
from engines import build_engine
from engines.base import Cancelled, StageTracker
from engines.common import check_rembg_allowed, gltf_axes, stage_hooks, trimesh_preview_png
from engines.hunyuan3d21 import Hunyuan3D21Engine, paint_plan
from engines.trellis_family import Pixal3DEngine, Trellis2Engine, pixal3d_camera_from_fov

MESH_WORKER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ── selection / config ────────────────────────────────────────────────────

@pytest.mark.parametrize("mode,cls", [("trellis2", Trellis2Engine), ("pixal3d", Pixal3DEngine),
                                      ("hunyuan3d21", Hunyuan3D21Engine)])
def test_real_engines_construct_without_gpu_deps(mode, cls):
    cfg = WorkerConfig(mode=mode)
    engine = build_engine(cfg)
    assert isinstance(engine, cls)
    assert cfg.model_id == DEFAULT_MODEL[mode] and cfg.checkpoint == DEFAULT_CKPT[mode]
    # Construction imports nothing heavy (that happens in load(), on the GPU node).
    for heavy in ("trellis2", "pixal3d", "hy3dshape", "textureGenPipeline", "o_voxel"):
        assert heavy not in sys.modules


def test_unknown_mode_refused(monkeypatch):
    monkeypatch.setenv("MESH_WORKER_MODE", "comfyui")
    with pytest.raises(ValueError, match="MESH_WORKER_MODE"):
        WorkerConfig()


def test_env_parsing(monkeypatch):
    monkeypatch.setenv("MESH_WORKER_MODE", "TRELLIS2")
    monkeypatch.setenv("MESH_WORKER_MAX_QUEUE", "3")
    monkeypatch.setenv("MESH_WORKER_JOB_TIMEOUT_SECONDS", "120")
    monkeypatch.setenv("MESH_WORKER_RESULT_TTL_SECONDS", "3600")
    monkeypatch.setenv("MESH_WORKER_LOW_VRAM", "yes")
    monkeypatch.setenv("MESH_WORKER_MODEL", "")            # blank = default
    cfg = WorkerConfig()
    assert (cfg.mode, cfg.max_queue, cfg.job_timeout_seconds, cfg.result_ttl_seconds) == ("trellis2", 3, 120.0, 3600.0)
    assert cfg.low_vram is True and cfg.model_id == "microsoft/trellis.2-4b"
    assert cfg.sweep_interval == 60.0
    assert set(DEFAULT_MODEL) == set(MODES)


def test_bad_face_budget_refused():
    with pytest.raises(ValueError, match="face budget"):
        WorkerConfig(mode="mock", min_faces=5000, default_faces=1000)


@pytest.mark.parametrize("mode,bad", [("trellis2", "2048_cascade"), ("pixal3d", "512")])
def test_bad_pipeline_type_refused(mode, bad):
    with pytest.raises(ValueError, match="PIPELINE_TYPE"):
        build_engine(WorkerConfig(mode=mode, pipeline_type=bad))


# ── RMBG-2.0 license guard ────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["briaai/RMBG-2.0", "/weights/briaai--RMBG-1.4", "rmbg"])
def test_rmbg_weights_refused(name):
    with pytest.raises(ValueError, match="CC BY-NC"):
        check_rembg_allowed(name)
    for mode in ("trellis2", "pixal3d"):
        with pytest.raises(ValueError, match="CC BY-NC"):
            build_engine(WorkerConfig(mode=mode, birefnet=name))


def _fake_stack(monkeypatch, package, rembg_name="BiRefNet"):
    """Install a fake upstream package whose from_pretrained behaves like the
    real one: it builds the remover named in pipeline.json with
    model_name='briaai/RMBG-2.0' and the DINOv3 extractor by HF id."""
    built = {"rembg": [], "dino": []}

    class BiRefNet:
        def __init__(self, model_name="ZhengPeng7/BiRefNet"):
            built["rembg"].append(model_name)

    class OtherRemover:
        def __init__(self, model_name=None):
            built["rembg"].append(f"other:{model_name}")

    class DinoV3FeatureExtractor:
        def __init__(self, model_name, image_size=512):
            built["dino"].append(model_name)

    rembg = types.ModuleType(f"{package}.pipelines.rembg")
    rembg.BiRefNet = BiRefNet
    rembg.OtherRemover = OtherRemover
    ife = types.ModuleType(f"{package}.modules.image_feature_extractor")
    ife.DinoV3FeatureExtractor = DinoV3FeatureExtractor

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, path, config_file="pipeline.json"):
            p = cls()
            p.path = path
            p.image_cond_model = getattr(ife, "DinoV3FeatureExtractor")(
                model_name="facebook/dinov3-vitl16-pretrain-lvd1689m")
            p.rembg_model = getattr(rembg, rembg_name)(model_name="briaai/RMBG-2.0")
            return p

    pipelines = types.ModuleType(f"{package}.pipelines")
    pipelines.rembg = rembg
    pipelines.Trellis2ImageTo3DPipeline = FakePipeline
    pipelines.Pixal3DImageTo3DPipeline = FakePipeline
    modules = types.ModuleType(f"{package}.modules")
    modules.image_feature_extractor = ife
    root = types.ModuleType(package)
    root.pipelines, root.modules = pipelines, modules
    for name, mod in ((package, root), (f"{package}.pipelines", pipelines),
                      (f"{package}.pipelines.rembg", rembg), (f"{package}.modules", modules),
                      (f"{package}.modules.image_feature_extractor", ife)):
        monkeypatch.setitem(sys.modules, name, mod)
    return built, rembg, ife


@pytest.mark.parametrize("mode,package,cls", [("trellis2", "trellis2", Trellis2Engine),
                                              ("pixal3d", "pixal3d", Pixal3DEngine)])
def test_rmbg_is_never_built_birefnet_is(monkeypatch, mode, package, cls):
    built, rembg, ife = _fake_stack(monkeypatch, package)
    original_birefnet, original_dino = rembg.BiRefNet, ife.DinoV3FeatureExtractor
    cfg = WorkerConfig(mode=mode, checkpoint="/weights/ckpt", birefnet="/weights/BiRefNet",
                       dinov3="/weights/dinov3-vitl16")
    pipeline = cls(cfg)._build_pipeline()
    assert pipeline.path == "/weights/ckpt"
    assert built["rembg"] == ["/weights/BiRefNet"]
    assert not any("rmbg" in name.lower() for name in built["rembg"])
    if mode == "trellis2":
        assert built["dino"] == ["/weights/dinov3-vitl16"]       # gated weights from the staged dir
    # Patches are scoped to the load.
    assert rembg.BiRefNet is original_birefnet and ife.DinoV3FeatureExtractor is original_dino


def test_load_refuses_a_remover_it_did_not_vet(monkeypatch):
    built, *_ = _fake_stack(monkeypatch, "trellis2", rembg_name="OtherRemover")
    with pytest.raises(RuntimeError, match="review its license"):
        Trellis2Engine(WorkerConfig(mode="trellis2"))._build_pipeline()


# ── stage hooks / helpers ─────────────────────────────────────────────────

def test_stage_hooks_report_cancel_and_clean_up():
    events, cancel = [], {"flag": False}

    class Pipe:
        def sample_sparse_structure(self, x):
            return x + 1

        def sample_tex_slat(self, x):
            return x * 2

    p = Pipe()
    tracker = StageTracker(lambda s, pr: events.append((s, pr)), lambda: cancel["flag"])
    hooks = {"sample_sparse_structure": ("shape", 15.0), "sample_tex_slat": ("texture", 60.0),
             "not_there": ("export", 90.0)}
    with stage_hooks(p, tracker, hooks):
        assert p.sample_sparse_structure(1) == 2
        cancel["flag"] = True
        with pytest.raises(Cancelled):
            p.sample_tex_slat(3)
    assert events == [("shape", 15.0)]
    assert "sample_sparse_structure" not in vars(p) and p.sample_tex_slat(3) == 6


def test_gltf_axes_matches_o_voxel_swap():
    v = np.array([[1.0, 2.0, 3.0]])
    assert gltf_axes(v).tolist() == [[1.0, 3.0, -2.0]]


def test_trimesh_preview_with_duck_typed_meshes():
    from engines.mock import _box
    from PIL import Image

    pos, uv, faces = _box(4, 0.2)
    tex = Image.new("RGB", (8, 8), (0, 200, 0))
    textured = types.SimpleNamespace(
        vertices=pos, faces=faces,
        visual=types.SimpleNamespace(kind="texture", uv=uv,
                                     material=types.SimpleNamespace(baseColorTexture=tex)))
    vertex = types.SimpleNamespace(
        vertices=pos, faces=faces,
        visual=types.SimpleNamespace(kind="vertex", vertex_colors=np.full((len(pos), 4), 200, np.uint8)))
    plain = types.SimpleNamespace(vertices=pos, faces=faces)
    for mesh in (textured, vertex, plain):
        assert trimesh_preview_png(mesh)[:8] == b"\x89PNG\r\n\x1a\n"


def test_pixal3d_camera_matches_upstream_formula():
    # inference.py distance_from_fov(), transliterated with its matrices.
    fov, res = 0.2, 512
    rot = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    gp = np.array([-1.0, 0, 0]) @ rot.T / 1.0 / 2
    f_pixels = (16.0 / math.tan(fov / 2)) * res / 32.0
    expected = f_pixels * gp[0] / (0 - res / 2.0) - gp[1]
    cam = pixal3d_camera_from_fov(fov)
    assert cam["camera_angle_x"] == fov and cam["mesh_scale"] == 1.0
    assert cam["distance"] == pytest.approx(expected)


def test_pixal3d_output_faces_plus_z_like_the_other_engines():
    up = np.array(Pixal3DEngine.UPSTREAM_ROTATION, dtype=float)
    rot = np.array(Pixal3DEngine.ROTATION, dtype=float)
    ry180 = np.diag([-1.0, 1.0, -1.0, 1.0])
    # Our rotation is upstream's plus a half turn about +Y ...
    assert np.allclose(rot, ry180 @ up)
    # ... a proper rotation (no mirroring) ...
    assert np.isclose(np.linalg.det(rot[:3, :3]), 1.0)
    # ... that turns what upstream leaves facing -Z to +Z, keeping +Y up.
    front_upstream = up[:3, :3] @ np.linalg.inv(up[:3, :3]) @ np.array([0.0, 0.0, -1.0])
    assert np.allclose(ry180[:3, :3] @ front_upstream, [0.0, 0.0, 1.0])
    assert np.allclose(rot[:3, :3] @ np.linalg.inv(up[:3, :3]) @ np.array([0.0, 1.0, 0.0]), [0.0, 1.0, 0.0])


def test_hunyuan_paint_plan_honours_face_and_texture_budget():
    assert paint_plan(40_000, 2048) == {"use_remesh": 1, "bake_atlas": 4096}   # upstream default path
    assert paint_plan(200_000, 2048)["use_remesh"] == 0     # remesh would cut to 40k
    assert paint_plan(10_000, 1024)["bake_atlas"] == 2048
    assert paint_plan(10_000, 4096)["bake_atlas"] == 8192


def test_hunyuan_sys_path_and_no_utils_collision(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    Hunyuan3D21Engine(WorkerConfig(mode="hunyuan3d21", hunyuan_repo="/opt/hy"))._sys_path()
    # Upstream precedence: hy3dpaint, hy3dshape, repo root.
    assert sys.path[:3] == ["/opt/hy/hy3dpaint", "/opt/hy/hy3dshape", "/opt/hy"]
    # hy3dpaint imports a top-level `utils` package: we must never shadow it.
    assert not os.path.exists(os.path.join(MESH_WORKER_DIR, "utils.py"))
    assert not os.path.exists(os.path.join(MESH_WORKER_DIR, "utils"))


# ── per-job options ───────────────────────────────────────────────────────

from config import ENGINE_OPTIONS, option_defaults, options_schema  # noqa: E402
from inputs import RequestError, validate_options  # noqa: E402


def test_option_defaults_come_from_the_worker_env(monkeypatch):
    monkeypatch.setenv("MESH_WORKER_HY_OCTREE", "384")
    monkeypatch.setenv("MESH_WORKER_HY_STEPS", "20")
    cfg = WorkerConfig(mode="hunyuan3d21")
    assert option_defaults(cfg) == {"octree_resolution": 384, "steps": 20, "guidance_scale": 5.0}
    assert option_defaults(WorkerConfig(mode="trellis2")) == {"pipeline_type": "1024_cascade"}
    assert option_defaults(WorkerConfig(mode="pixal3d")) == {"pipeline_type": "1536_cascade"}
    assert option_defaults(WorkerConfig(mode="mock")) == {}
    schema = options_schema(cfg)
    assert schema["octree_resolution"]["default"] == 384 and schema["steps"]["max"] == 50


def test_illegal_env_default_refused_at_startup(monkeypatch):
    monkeypatch.setenv("MESH_WORKER_HY_OCTREE", "300")
    with pytest.raises(ValueError, match="octree_resolution"):
        WorkerConfig(mode="hunyuan3d21")


def test_engine_pipeline_menus_match_the_option_table():
    assert Trellis2Engine.pipeline_types == tuple(ENGINE_OPTIONS["trellis2"]["pipeline_type"]["choices"])
    assert Pixal3DEngine.pipeline_types == tuple(ENGINE_OPTIONS["pixal3d"]["pipeline_type"]["choices"])


@pytest.mark.parametrize("raw,match", [
    ([1], "must be an object"),
    ({"bogus": 1}, "unknown option"),
    ({"octree_resolution": 300}, "one of"),
    ({"octree_resolution": True}, "one of"),
    ({"steps": 4}, r"\[5, 50\]"),
    ({"steps": 20.5}, "integer"),
    ({"guidance_scale": 11}, r"\[1.0, 10.0\]"),
    ({"guidance_scale": "7"}, "number"),
])
def test_bad_options_refused(raw, match):
    with pytest.raises(RequestError, match=match):
        validate_options(raw, WorkerConfig(mode="hunyuan3d21"))


def test_options_merged_over_defaults():
    cfg = WorkerConfig(mode="hunyuan3d21")
    assert validate_options({"steps": 12, "guidance_scale": 7}, cfg) == {
        "octree_resolution": 256, "steps": 12, "guidance_scale": 7.0}
    assert validate_options(None, cfg) == option_defaults(cfg)
    with pytest.raises(RequestError, match="allowed: none"):
        validate_options({"steps": 10}, WorkerConfig(mode="mock"))
    assert validate_options({"pipeline_type": "1536_cascade"}, WorkerConfig(mode="trellis2")) == {
        "pipeline_type": "1536_cascade"}
