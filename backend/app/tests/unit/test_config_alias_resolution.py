############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_config_alias_resolution.py: internal paths that route a
#     configured model name resolve aliases first
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Configured model names may be aliases; routing needs the real name.

The API endpoints resolve aliases before routing.  Internal paths that take a
model name from app_config did not: prod's OCR model is the alias
`default-ocr`, and every scanned PDF uploaded in the dashboard chat failed
OCR (192 "Model 'default-ocr' not found" in a week) -- that one is covered in
test_ocr_pipeline_json_and_runaway.py.  The same gap, latent because the
admin dropdowns offer real names, sat in the dashboard's image generation
(`img.default_model`) and the image-policy judge (`img.judge_model`, which
fails closed and would have blocked every image).  The chat also told the
user nothing when OCR failed.
"""

import ast
from pathlib import Path
from unittest.mock import patch

import pytest

import backend.app.services.image_policy as ip

_APP = Path(__file__).resolve().parents[2]


class _Registry:
    def __init__(self):
        self.looked_up = []

    def resolve_alias(self, name):
        return ("Qwen/Qwen3.8-27B", name) if name == "default-judge" else (name, None)

    async def get_backends_with_model(self, name, modality=None):
        self.looked_up.append(name)
        return []


@pytest.mark.parametrize("configured, routed", [
    ("default-judge", "Qwen/Qwen3.8-27B"),
    ("Qwen/Qwen3.8-27B", "Qwen/Qwen3.8-27B"),
])
async def test_the_image_judge_is_looked_up_by_its_real_name(configured, routed):
    registry = _Registry()
    with patch.object(ip, "get_registry", lambda: registry):
        with pytest.raises(RuntimeError, match=f"judge model '{routed}'"):
            await ip._call_judge("a cat", "policy", configured)
    assert registry.looked_up == [routed]


def test_dashboard_image_generation_resolves_the_model_after_defaults():
    src = (_APP / "dashboard" / "images.py").read_text()
    tree = ast.parse(src)
    fns = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and "body['model'] = default_model" in ast.unparse(n)
    ]
    assert len(fns) == 1
    body = ast.unparse(fns[0])
    default = body.index("body['model'] = default_model")
    resolved = body.index("body['model'], _ = get_registry().resolve_alias(body['model'])")
    # After the default is applied (the default itself may be an alias) and
    # before the request reaches InferenceService.
    assert default < resolved < body.index("InferenceService(")


def test_chat_tells_the_user_when_ocr_failed():
    src = (_APP / "dashboard" / "chat.py").read_text()
    handler = src[src.index('logger.warning(\n                            "pdf_ocr_fallback_failed"'):]
    assert "ocr_failed = True" in handler[: handler.index("else:")]
    note = src[src.index("elif ocr_failed:"):]
    assert 'resp["processing_note"]' in note[:300]
    assert "OCR is unavailable right now" in note[:300]
