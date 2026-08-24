############################################################
# test_image_quota.py: flat per-image quota charge
############################################################
"""Diffusion backends report no `usage`, so image requests used to be billed
for the LENGTH OF THE PROMPT TEXT — measured in production, a 24-character
prompt cost 6 tokens while occupying an entire max_concurrent=1 diffusion
worker for ~7s of exclusive GPU time. These pin the flat charge that replaced
it."""
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

_APP_DIR = Path(__file__).resolve().parents[2]
INFERENCE_SRC = (_APP_DIR / "services" / "inference.py").read_text()
ROUTES_SRC = (_APP_DIR / "dashboard" / "routes.py").read_text()
IMAGES_HTML = (_APP_DIR / "dashboard" / "templates" / "admin" / "images_config.html").read_text()
MIGRATIONS = _APP_DIR / "db" / "migrations" / "versions"


class TestImageTokenCost:
    """`_image_token_cost` does `from backend.app.db import crud`, i.e. it
    resolves the PACKAGE ATTRIBUTE — so the real function is patched here
    rather than sys.modules, which would not be seen."""

    def _svc(self):
        from backend.app.services.inference import InferenceService

        svc = InferenceService.__new__(InferenceService)
        svc.db = MagicMock()
        return svc

    def _patch(self, monkeypatch, value=None, raises=False):
        from backend.app.db import crud as real_crud

        async def get_config_json(db, key, default=None):
            if raises:
                raise RuntimeError("db down")
            return default if value is None else value

        monkeypatch.setattr(real_crud, "get_config_json", get_config_json)

    @pytest.mark.asyncio
    async def test_default_is_one_average_chat_exchange(self, monkeypatch):
        from backend.app.services.inference import InferenceService

        assert InferenceService.DEFAULT_IMAGE_TOKEN_COST == 10000
        self._patch(monkeypatch)
        assert await self._svc()._image_token_cost() == 10000

    @pytest.mark.asyncio
    async def test_configured_value_is_used(self, monkeypatch):
        self._patch(monkeypatch, value=2500)
        assert await self._svc()._image_token_cost() == 2500

    @pytest.mark.asyncio
    async def test_zero_disables_the_charge(self, monkeypatch):
        self._patch(monkeypatch, value=0)
        assert await self._svc()._image_token_cost() == 0

    @pytest.mark.asyncio
    async def test_negative_is_clamped_not_credited(self, monkeypatch):
        """A negative cost would REFUND quota on every image."""
        self._patch(monkeypatch, value=-5000)
        assert await self._svc()._image_token_cost() == 0

    @pytest.mark.asyncio
    async def test_garbage_falls_back_to_default(self, monkeypatch):
        self._patch(monkeypatch, value="lots")
        assert await self._svc()._image_token_cost() == 10000

    @pytest.mark.asyncio
    async def test_config_outage_charges_default_not_zero(self, monkeypatch):
        """A config read failure must not silently make images free."""
        self._patch(monkeypatch, raises=True)
        assert await self._svc()._image_token_cost() == 10000


class TestCompletionCharging:
    """The charge is applied in _complete_request, so it flows through the
    existing quota / api-key-usage / cluster-token accounting unchanged."""

    def test_applied_only_for_image_modality(self):
        body = INFERENCE_SRC[INFERENCE_SRC.index("# Estimate if not provided"):]
        body = body[:body.index("# Release backend capacity FIRST")]
        assert "if modality == Modality.IMAGE_GENERATION:" in body
        assert "_image_token_cost()" in body
        assert "if per_image > 0:" in body, "0 must leave the old estimate alone"

    def test_charges_per_image_returned(self):
        body = INFERENCE_SRC[INFERENCE_SRC.index("if modality == Modality.IMAGE_GENERATION:"):]
        body = body[:body.index("total_tokens = prompt_tokens + completion_tokens")]
        assert 'len(response.get("data") or [])' in body, "n>1 must cost n times"
        assert "per_image * images" in body
        assert "completion_tokens = 0" in body

    def test_marked_estimated(self):
        """It is a synthetic charge, not a measured count — the audit row
        must say so."""
        body = INFERENCE_SRC[INFERENCE_SRC.index("if modality == Modality.IMAGE_GENERATION:"):]
        body = body[:body.index("total_tokens = prompt_tokens + completion_tokens")]
        assert "tokens_estimated = True" in body


class TestAdminSurface:
    def test_config_key_read_and_written(self):
        assert '"img.quota_tokens_per_image"' in ROUTES_SRC
        assert 'set_config(db, "img.quota_tokens_per_image", quota_per_image)' in ROUTES_SRC

    def test_validated_before_any_write(self):
        """A bad value must not land half a save."""
        i = ROUTES_SRC.index("raw_cost = (form.get(\"quota_tokens_per_image\")")
        j = ROUTES_SRC.index('set_config(db, "img.quota_tokens_per_image"')
        assert i < j
        seg = ROUTES_SRC[i:j]
        assert "must+be+a+whole+number" in seg
        assert "0 <= quota_per_image <= 10_000_000" in seg

    def test_control_on_the_images_page(self):
        assert 'name="quota_tokens_per_image"' in IMAGES_HTML
        assert 'min="0"' in IMAGES_HTML
        assert "0" in IMAGES_HTML and "free" in IMAGES_HTML.lower()

    def test_migration_085_seeds_the_default(self):
        mig = (MIGRATIONS / "20260824_000000_085_image_quota_tokens.py").read_text()
        assert 'revision = "085"' in mig and 'down_revision = "084"' in mig
        assert "_DEFAULT = 10000" in mig
        assert "INSERT IGNORE" in mig, "must not clobber an operator's value"
