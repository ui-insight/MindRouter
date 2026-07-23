############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# field_validation.py: Surface request fields that MindRouter
#   would otherwise silently drop (vLLM-dialect payloads sent
#   to the OpenAI dialect), instead of ignoring them.
#
############################################################

"""Request-field validation for the input translators.

Some clients treat MindRouter as a raw vLLM endpoint and send vLLM-specific
fields (e.g. ``structured_outputs``, ``guided_json``). MindRouter builds its
canonical request from a known set of fields and silently discards the rest —
so those requests behave wrong with no error. This module makes that explicit.

Three modes (``field_validation`` setting):
* ``off``     — no checking (legacy behavior)
* ``log``     — log unknown / dialect fields, reject nothing (dark-launch)
* ``enforce`` — 400 on any field that is neither accepted nor deliberately
                ignored; known dialect fields get a helpful message

Deploy at ``log`` first to learn what real clients send, finalize the
ignore-list, then flip to ``enforce`` — so strict rejection never blindsides a
working integration.
"""

from typing import Any, Dict, Optional, Set

import structlog
from fastapi import HTTPException

logger = structlog.get_logger(__name__)


# Fields the OpenAI chat translator actually consumes.
CHAT_ACCEPTED: Set[str] = {
    "model", "messages", "temperature", "top_p", "max_tokens",
    "max_completion_tokens", "stream", "stream_options", "stop",
    "presence_penalty", "frequency_penalty", "seed", "top_k",
    "repetition_penalty", "min_p", "tools", "tool_choice", "response_format",
    "n", "user", "reasoning_effort",
    # thinking control (resolved together)
    "think", "thinking", "chat_template_kwargs",
}

# Standard OpenAI chat fields MindRouter does not act on but that are safe to
# accept and drop. Deliberate, documented — not a default catch-all.
CHAT_IGNORED: Set[str] = {
    "logit_bias", "logprobs", "top_logprobs", "parallel_tool_calls",
    "metadata", "store", "service_tier", "modalities", "audio", "prediction",
    "web_search_options", "function_call", "functions",  # legacy function-calling
    "stream_cls", "extra_headers", "extra_body", "extra_query", "timeout",
}

# Known vLLM-dialect fields that MindRouter silently drops today. Each maps to a
# hint pointing at the supported cross-backend alternative.
CHAT_DIALECT_HINTS: Dict[str, str] = {
    "structured_outputs": "Use 'response_format' with type 'json_schema', which MindRouter enforces across backends.",
    "guided_json": "Use 'response_format' with type 'json_schema'.",
    "guided_regex": "Regex-guided decoding is not exposed; use 'response_format' with 'json_schema'.",
    "guided_choice": "Use 'response_format' with 'json_schema', or constrain via the prompt.",
    "guided_grammar": "Grammar-guided decoding is not exposed; use 'response_format'.",
    "guided_decoding_backend": "Guided-decoding backends are not configurable per request.",
    "use_beam_search": "Beam search is not supported.",
    "best_of": "'best_of' is not supported.",
    "stop_token_ids": "Use 'stop' with string sequences.",
    "prompt_logprobs": "Prompt logprobs are not exposed.",
    "mm_processor_kwargs": "Multimodal processor kwargs are configured per backend, not per request.",
    "add_generation_prompt": "Chat templating is managed server-side.",
    "continue_final_message": "'continue_final_message' is not supported.",
    "echo": "'echo' is not supported on chat completions.",
    "top_a": "'top_a' is not supported.",
    "typical_p": "'typical_p' is not supported.",
    "length_penalty": "'length_penalty' is not supported.",
    "early_stopping": "'early_stopping' is not supported.",
}


# --- Video generation dialect (v1 text-to-video) --------------------------
# Fields the video create path (submit_video_job) actually consumes.
VIDEO_ACCEPTED: Set[str] = {
    "model", "prompt", "size", "seconds", "fps", "quality", "seed",
    "negative_prompt", "callback_url",
}

# Accepted-and-dropped (OpenAI-video clients send these harmlessly).
VIDEO_IGNORED: Set[str] = {"user", "response_format", "metadata"}

# Common video-request fields MindRouter does not honor in v1, each mapped to
# the supported alternative — so a typo or a wrong-API field is surfaced, not
# silently dropped (the v2.8.2 field-validation contract).
VIDEO_DIALECT_HINTS: Dict[str, str] = {
    "duration": "Use 'seconds' (a value from the preset duration list).",
    "length": "Use 'seconds' (a value from the preset duration list).",
    "width": "Use 'size' as 'WIDTHxHEIGHT' from the preset size list.",
    "height": "Use 'size' as 'WIDTHxHEIGHT' from the preset size list.",
    "aspect_ratio": "Use 'size' as 'WIDTHxHEIGHT' from the preset size list.",
    "resolution": "Use 'size' as 'WIDTHxHEIGHT' from the preset size list.",
    "num_frames": "Frame count is derived from 'seconds'; pick a preset duration.",
    "frames": "Frame count is derived from 'seconds'; pick a preset duration.",
    "steps": "Denoising steps are fixed by the 'quality' tier.",
    "num_inference_steps": "Denoising steps are fixed by the 'quality' tier.",
    "guidance_scale": "Guidance is fixed by the 'quality' tier.",
    "cfg_scale": "Guidance is fixed by the 'quality' tier.",
    "n": "Only one clip is generated per request in v1.",
    "image": "Image conditioning is not available in v1 (text-to-video only).",
    "input_reference": "Image-to-video is not available in v1 (text-to-video only).",
    "first_frame": "Keyframe conditioning is not available in v1 (text-to-video only).",
    "last_frame": "Keyframe conditioning is not available in v1 (text-to-video only).",
    "storyboard": "Multi-shot storyboards are not available in v1 (single clip only).",
    "audio": "Audio generation is not available in v1.",
    "generate_audio": "Audio generation is not available in v1.",
}


def _error(field: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail={"error": {
            "type": "invalid_request_error",
            "param": field,
            "message": message,
        }},
    )


def validate_request_fields(
    data: Dict[str, Any],
    *,
    dialect: str = "chat",
    accepted: Optional[Set[str]] = None,
    ignored: Optional[Set[str]] = None,
    hints: Optional[Dict[str, str]] = None,
    mode: Optional[str] = None,
) -> None:
    """Log or reject request fields MindRouter would otherwise drop.

    ``mode`` defaults to the ``field_validation`` setting. Raises an
    OpenAI-style 400 in ``enforce`` mode; logs in ``log`` mode; no-op in ``off``.
    """
    if mode is None:
        from backend.app.settings import get_settings
        mode = getattr(get_settings(), "field_validation", "log")
    if mode == "off":
        return

    accepted = accepted if accepted is not None else CHAT_ACCEPTED
    ignored = ignored if ignored is not None else CHAT_IGNORED
    hints = hints if hints is not None else CHAT_DIALECT_HINTS
    known = accepted | ignored

    for field in data:
        if field in known:
            continue
        hint = hints.get(field)
        if mode == "enforce":
            if hint:
                raise _error(field, f"Unsupported field '{field}' (vLLM dialect). {hint}")
            raise _error(field, f"Unsupported request field '{field}'.")
        # log mode: observe, reject nothing
        logger.info(
            "request_unknown_field",
            field=field,
            dialect=dialect,
            kind="dialect" if hint else "unknown",
        )
