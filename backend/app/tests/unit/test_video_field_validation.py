"""Tests for the video request-field validation dialect (v1 text-to-video).

field_validation is spec-loaded (its lazy settings import only fires when
mode=None; tests pass mode explicitly). canonical_schemas is spec-loaded too
(pydantic-only) for the drift guard — neither pulls the db chain.
"""

import importlib.util
import os

import pytest
from fastapi import HTTPException

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel), submodule_search_locations=[])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fv = _load("fv_video", "backend/app/core/translators/field_validation.py")
V = fv.validate_request_fields
VIDEO_ACCEPTED = fv.VIDEO_ACCEPTED
VIDEO_IGNORED = fv.VIDEO_IGNORED
VIDEO_HINTS = fv.VIDEO_DIALECT_HINTS


def _v(data, mode):
    return V(data, dialect="video", accepted=VIDEO_ACCEPTED, ignored=VIDEO_IGNORED,
             hints=VIDEO_HINTS, mode=mode)


def test_enforce_accepts_all_declared_video_fields():
    # A full valid v1 body must not raise in enforce mode.
    _v({
        "model": "lightricks/ltx-2.3-distilled", "prompt": "a fox",
        "size": "1280x704", "seconds": "5", "fps": 24, "quality": "draft",
        "seed": 7, "negative_prompt": "blurry", "callback_url": "https://x/y",
    }, mode="enforce")


def test_enforce_rejects_duration_typo_with_hint():
    with pytest.raises(HTTPException) as ei:
        _v({"prompt": "x", "duration": 5}, mode="enforce")
    err = ei.value.detail["error"]
    assert ei.value.status_code == 400
    assert err["param"] == "duration"
    assert "seconds" in err["message"]  # points at the supported alternative


def test_enforce_rejects_width_height_pointing_to_size():
    for field in ("width", "height", "aspect_ratio"):
        with pytest.raises(HTTPException) as ei:
            _v({"prompt": "x", field: 720}, mode="enforce")
        assert "size" in ei.value.detail["error"]["message"]


def test_enforce_rejects_v1_unsupported_conditioning():
    # image-to-video / keyframe fields are surfaced, not silently dropped
    for field in ("image", "input_reference", "first_frame", "storyboard"):
        with pytest.raises(HTTPException) as ei:
            _v({"prompt": "x", field: "y"}, mode="enforce")
        assert "v1" in ei.value.detail["error"]["message"]


def test_enforce_rejects_unknown_field_generically():
    with pytest.raises(HTTPException) as ei:
        _v({"prompt": "x", "frobnicate": 1}, mode="enforce")
    assert "Unsupported request field" in ei.value.detail["error"]["message"]


def test_ignored_fields_do_not_raise():
    _v({"prompt": "x", "user": "u", "response_format": {}, "metadata": {}}, mode="enforce")


def test_log_and_off_never_raise():
    body = {"prompt": "x", "duration": 5, "frobnicate": 1}
    _v(body, mode="log")   # dark-launch: observes, rejects nothing
    _v(body, mode="off")   # disabled


def test_accepted_fields_are_real_canonical_fields():
    # Drift guard: every VIDEO_ACCEPTED key must map to a real request field
    # (or be a routing-only extra the canonical model doesn't carry).
    cs = _load("cs_video", "backend/app/core/canonical_schemas.py")
    fields = set(cs.CanonicalVideoRequest.model_fields)
    routing_only = {"callback_url"}  # handled by the job, not the canonical request
    for name in VIDEO_ACCEPTED - routing_only:
        assert name in fields, f"VIDEO_ACCEPTED '{name}' is not a CanonicalVideoRequest field"
