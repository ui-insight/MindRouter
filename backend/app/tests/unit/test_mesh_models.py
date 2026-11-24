"""mesh_models: the gateway's table of image-to-mesh model families.

Pure module (no DB, no registry), loaded by path. The contract tests read
mesh-worker/config.py as text (the gateway never imports the worker), so the
per-job settings the admin page offers can never drift from what the worker
accepts.
"""

import ast
import importlib.util
import os

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_spec = importlib.util.spec_from_file_location(
    "mesh_models_under_test", os.path.join(ROOT, "backend", "app", "services", "mesh_models.py"))
mm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mm)


def _worker_assign(name):
    tree = ast.parse(open(os.path.join(ROOT, "mesh-worker", "config.py")).read())
    for node in tree.body:
        target = getattr(node, "target", None) or (node.targets[0] if getattr(node, "targets", None) else None)
        if getattr(target, "id", None) == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in mesh-worker/config.py")


def test_settings_table_equals_the_workers():
    worker = _worker_assign("ENGINE_OPTIONS")
    assert worker.pop("mock") == {}
    assert mm.FAMILY_OPTIONS == worker


def test_defaults_and_model_ids_match_the_worker():
    assert mm.FAMILY_MODEL_IDS == {k: v for k, v in _worker_assign("DEFAULT_MODEL").items() if k != "mock"}
    for family, pt in _worker_assign("DEFAULT_PIPELINE_TYPE").items():
        assert mm.FAMILY_OPTION_DEFAULTS[family] == {"pipeline_type": pt}
    cfg = open(os.path.join(ROOT, "mesh-worker", "config.py")).read()
    hy = mm.FAMILY_OPTION_DEFAULTS["hunyuan3d21"]
    assert f'"MESH_WORKER_HY_STEPS", {hy["steps"]})' in cfg
    assert f'"MESH_WORKER_HY_OCTREE", {hy["octree_resolution"]})' in cfg
    assert f'"MESH_WORKER_HY_GUIDANCE", {hy["guidance_scale"]})' in cfg
    for family, opts in mm.FAMILY_OPTION_DEFAULTS.items():
        assert mm.clean_options(family, opts) == opts          # every default is a legal setting


@pytest.mark.parametrize("name,family", [
    ("tencent/hunyuan3d-2.1", "hunyuan3d21"), ("Tencent/Hunyuan3D-2.1-custom", "hunyuan3d21"),
    ("microsoft/trellis.2-4b", "trellis2"), ("tencentarc/pixal3d", "pixal3d"),
    ("mindrouter/mesh-mock", None), (None, None),
])
def test_family_for_model(name, family):
    assert mm.family_for_model(name) == family


def test_only_hunyuan_needs_license_acceptance():
    assert [f for f, lic in mm.LICENSES.items() if lic["requires_acceptance"]] == ["hunyuan3d21"]
    hy = mm.LICENSES["hunyuan3d21"]
    text = " ".join(hy["obligations"]) + hy["user_notice"]
    for must in ("European Union", "United Kingdom", "South Korea", "not affiliated", "improve",
                 "Acceptable Use", "1 million"):
        assert must in text
    assert mm.license_for_model("mindrouter/mesh-mock") is None


def test_clean_options_casts_form_strings_and_refuses_bad_values():
    assert mm.clean_options("hunyuan3d21", {"octree_resolution": "384", "steps": " 20 ", "guidance_scale": "7.5"}) == {
        "octree_resolution": 384, "steps": 20, "guidance_scale": 7.5}
    assert mm.clean_options("hunyuan3d21", {"steps": ""}) == {}                     # blank = worker default
    assert mm.clean_options("trellis2", {"pipeline_type": "1536_cascade"}) == {"pipeline_type": "1536_cascade"}
    for family, raw, match in [
        ("hunyuan3d21", {"octree_resolution": "300"}, "256, 384, 512"),
        ("hunyuan3d21", {"steps": "51"}, "5 to 50"),
        ("hunyuan3d21", {"guidance_scale": "x"}, "1.0 to 10.0"),
        ("pixal3d", {"pipeline_type": "512"}, "1024_cascade"),
        ("hunyuan3d21", {"pipeline_type": "512"}, "unknown"),
        (None, {"steps": "10"}, "unknown"),
    ]:
        with pytest.raises(ValueError, match=match):
            mm.clean_options(family, raw)


def test_model_limits_narrow_and_clamp():
    base = {"texture_sizes": [1024, 2048, 4096], "default_texture_size": 4096, "min_faces": 5000,
            "max_faces": 300000, "default_faces": 100000, "max_image_mb": 10}
    assert mm.model_limits(base, None) == base
    out = mm.model_limits(base, {"max_faces": 40000, "max_texture_size": 2048})
    assert out["max_faces"] == 40000 and out["default_faces"] == 40000
    assert out["texture_sizes"] == [1024, 2048] and out["default_texture_size"] == 2048
    assert mm.model_limits(base, {"max_faces": 10_000_000})["max_faces"] == 300000   # never above the service cap
    assert mm.model_limits(base, {"max_faces": 10})["max_faces"] == 5000             # never below the floor
    assert mm.model_limits(base, {"max_texture_size": 512})["texture_sizes"] == [1024]
