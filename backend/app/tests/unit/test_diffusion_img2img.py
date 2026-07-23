############################################################
#
# mindrouter - unit tests for img2img (reference-edit) plumbing
#
# Covers the DiffusionOutTranslator payload passthrough for image/strength and
# the canonical schema fields. Route selection (generations vs edits) lives in
# InferenceService._proxy_image_request and is asserted here at the translator
# boundary (payload contains `image` iff the canonical carries reference images).
#
############################################################

"""Unit tests for image-to-image (reference-edit) request translation."""

from backend.app.core.canonical_schemas import CanonicalImageRequest
from backend.app.core.translators.diffusion_out import DiffusionOutTranslator


def test_txt2img_payload_has_no_image_or_strength():
    """A plain generation request must not carry image/strength keys."""
    req = CanonicalImageRequest(model="flux", prompt="a cat", size="1024x1024")
    payload = DiffusionOutTranslator.translate_image_request(req)
    assert "image" not in payload
    assert "strength" not in payload
    assert payload["prompt"] == "a cat"


def test_img2img_payload_passes_reference_images():
    """When `image` is set, the backend payload carries the base64 list."""
    req = CanonicalImageRequest(
        model="flux",
        prompt="make it blue",
        size="1024x1024",
        image=["QUJD", "REVG"],
        strength=0.6,
    )
    payload = DiffusionOutTranslator.translate_image_request(req)
    assert payload["image"] == ["QUJD", "REVG"]
    assert payload["strength"] == 0.6
    assert payload["prompt"] == "make it blue"


def test_img2img_strength_optional():
    """`image` without `strength` still passes the images; no strength key."""
    req = CanonicalImageRequest(model="flux", prompt="x", image=["QUJD"])
    payload = DiffusionOutTranslator.translate_image_request(req)
    assert payload["image"] == ["QUJD"]
    assert "strength" not in payload


def test_empty_image_list_is_falsy_no_key():
    """An empty image list must not emit an `image` key (stays txt2img)."""
    req = CanonicalImageRequest(model="flux", prompt="x", image=[])
    payload = DiffusionOutTranslator.translate_image_request(req)
    assert "image" not in payload


def test_canonical_defaults_none():
    """Defaults keep the txt2img shape (image/strength unset)."""
    req = CanonicalImageRequest(model="flux", prompt="x")
    assert req.image is None
    assert req.strength is None
