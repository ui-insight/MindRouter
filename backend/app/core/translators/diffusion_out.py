############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# diffusion_out.py: Canonical schema to diffusion backend translator
#
# Targets OpenAI-compatible image generation APIs such as
# openedai-images-flux and vLLM-Omni.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Canonical schema to diffusion backend (OpenAI images API) translator."""

import time
from typing import Any, Dict, List

from backend.app.core.canonical_schemas import (
    CanonicalImageData,
    CanonicalImageRequest,
    CanonicalImageResponse,
)


class DiffusionOutTranslator:
    """Translates canonical image requests to/from diffusion backend format.

    The target backend is expected to implement the OpenAI
    ``/v1/images/generations`` contract (e.g. openedai-images-flux).
    """

    @staticmethod
    def translate_image_request(canonical: CanonicalImageRequest) -> Dict[str, Any]:
        """Convert a canonical image request into the backend payload."""
        payload: Dict[str, Any] = {
            "model": canonical.model,
            "prompt": canonical.prompt,
            "n": canonical.n,
            "size": canonical.size,
        }

        if canonical.quality:
            payload["quality"] = canonical.quality
        if canonical.style:
            payload["style"] = canonical.style
        if canonical.response_format:
            payload["response_format"] = canonical.response_format

        # FLUX-specific extended parameters
        if canonical.num_inference_steps is not None:
            payload["num_inference_steps"] = canonical.num_inference_steps
        if canonical.guidance_scale is not None:
            payload["guidance_scale"] = canonical.guidance_scale
        if canonical.seed is not None:
            payload["seed"] = canonical.seed

        return payload

    @staticmethod
    def translate_image_response(
        data: Dict[str, Any],
    ) -> CanonicalImageResponse:
        """Convert backend response to canonical image response."""
        images: List[CanonicalImageData] = []
        for item in data.get("data", []):
            images.append(
                CanonicalImageData(
                    url=item.get("url"),
                    b64_json=item.get("b64_json"),
                    revised_prompt=item.get("revised_prompt"),
                )
            )

        return CanonicalImageResponse(
            created=data.get("created", int(time.time())),
            data=images,
        )
