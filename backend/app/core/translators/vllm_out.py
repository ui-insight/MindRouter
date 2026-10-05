############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# vllm_out.py: Canonical schema to vLLM/OpenAI API format translator
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Canonical schema to vLLM (OpenAI-compatible) API format translator."""

import re
import time
from typing import Any, AsyncIterator, Dict, List, Optional

import orjson
import structlog

from backend.app.core.text_stream import Utf8StreamDecoder
from backend.app.core.canonical_schemas import (
    CanonicalChatRequest,
    CanonicalChatResponse,
    CanonicalChoice,
    CanonicalCompletionRequest,
    CanonicalEmbeddingRequest,
    CanonicalEmbeddingResponse,
    CanonicalFunctionCall,
    CanonicalMessage,
    CanonicalRerankRequest,
    CanonicalRerankResponse,
    CanonicalRerankResult,
    CanonicalScoreData,
    CanonicalScoreRequest,
    CanonicalScoreResponse,
    CanonicalToolCall,
    ImageBase64Content,
    ImageUrlContent,
    MessageRole,
    ResponseFormatType,
    TextContent,
    UsageInfo,
)
from backend.app.core.url_guard import image_url_reason

logger = structlog.get_logger(__name__)


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

# Gemma 4 outputs thinking as "thought\n...<actual content>" when the
# reasoning parser doesn't extract it.  The "thought\n" prefix comes
# from the <|channel|>thought delimiter being partially consumed.
_GEMMA4_THOUGHT_RE = re.compile(r"^thought\n(.*?)(?=\n(?:[A-Z]|\d|$))", re.DOTALL)


def reasoning_promotion_applies(model: str) -> bool:
    """True when the disabled-thinking reasoning→content promotion may fire.

    The promotion works around a vLLM/Qwen3.5 bug: with enable_thinking
    false, the model sometimes emits its ANSWER in reasoning_content and
    leaves content empty. That premise only holds for qwen-family models,
    whose chat template actually honors the disable switch. Models whose
    reasoning cannot be switched off (e.g. Muse Glimmer's harmony-style
    channels) emit GENUINE reasoning in that field — promoting it would
    present the model's internal monologue as the answer.
    """
    return "qwen" in (model or "").lower()


def _extract_think_tags(content: str) -> tuple:
    """Extract <think>...</think> from content, returning (reasoning, cleaned_content).

    Returns (None, original_content) if no tags found.
    """
    match = _THINK_RE.search(content)
    if not match:
        return None, content
    reasoning = match.group(1).strip() or None
    cleaned = (content[: match.start()] + content[match.end() :]).strip()
    return reasoning, cleaned or None


def _extract_gemma4_thought(content: str) -> tuple:
    """Extract Gemma 4 'thought\\n...' prefix from content.

    Gemma 4 prefixes thinking with 'thought\\n' followed by reasoning
    lines, then the actual answer.  We split on the last reasoning line
    by finding where the thinking ends (lines starting with non-whitespace
    after a paragraph break).

    Returns (reasoning, cleaned_content) or (None, original_content).
    """
    if not content.startswith("thought\n"):
        return None, content

    # Remove the "thought\n" prefix
    rest = content[len("thought\n"):]

    # Find the boundary between reasoning and answer.
    # Reasoning lines are typically indented or analytical.  The answer
    # is usually preceded by an empty line or starts with a direct statement.
    lines = rest.split("\n")
    split_idx = len(lines)  # default: everything is reasoning

    for i, line in enumerate(lines):
        stripped = line.strip()
        # Empty line followed by non-empty = start of answer
        if stripped == "" and i + 1 < len(lines) and lines[i + 1].strip():
            # Check if the next line looks like answer (not reasoning)
            next_line = lines[i + 1].strip()
            if not next_line.startswith(("*", "-", "The user", "I need", "Let me", "So ", "This ")):
                split_idx = i
                break

    reasoning = "\n".join(lines[:split_idx]).strip()
    answer = "\n".join(lines[split_idx:]).strip()

    return reasoning or None, answer or None


class VLLMOutTranslator:
    """Translate canonical requests to vLLM (OpenAI-compatible) API format."""

    @staticmethod
    def translate_chat_request(canonical: CanonicalChatRequest) -> Dict[str, Any]:
        """Translate canonical chat request to vLLM/OpenAI format.

        vLLM is fully OpenAI-compatible, so this produces standard OpenAI format.

        Args:
            canonical: Canonical chat request

        Returns:
            OpenAI-compatible API request body
        """
        messages = []
        for msg in canonical.messages:
            messages.append(VLLMOutTranslator._translate_message(msg))

        # Ensure system messages come first and are merged into one —
        # some chat templates (e.g. Qwen3.5) reject requests where system
        # messages appear after non-system messages or where there are
        # multiple system messages.
        system = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]
        if len(system) > 1:
            # System content may legally be a list of text parts (OpenAI
            # spec) — flatten before the string join.
            def _sys_text(content: Any) -> str:
                if isinstance(content, list):
                    return " ".join(
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict)
                    )
                return content or ""

            merged = "\n\n".join(
                _sys_text(m.get("content")) for m in system if m.get("content")
            )
            messages = [{"role": "system", "content": merged}] + non_system
        else:
            messages = system + non_system

        payload: Dict[str, Any] = {
            "model": canonical.model,
            "messages": messages,
            "stream": canonical.stream,
        }
        # Request usage on the final streaming chunk so quota accounting uses
        # real token counts instead of a char-length estimate. MindRouter
        # consumes this usage chunk internally; it is not forwarded to clients.
        if canonical.stream:
            payload["stream_options"] = {"include_usage": True}

        # Add optional parameters
        if canonical.temperature is not None:
            payload["temperature"] = canonical.temperature
        if canonical.top_p is not None:
            payload["top_p"] = canonical.top_p
        if canonical.max_tokens is not None:
            payload["max_tokens"] = canonical.max_tokens
        if canonical.stop is not None:
            payload["stop"] = canonical.stop
        if canonical.presence_penalty is not None:
            payload["presence_penalty"] = canonical.presence_penalty
        if canonical.frequency_penalty is not None:
            payload["frequency_penalty"] = canonical.frequency_penalty
        if canonical.seed is not None:
            payload["seed"] = canonical.seed
        if canonical.top_k is not None:
            payload["top_k"] = canonical.top_k
        if canonical.repeat_penalty is not None:
            payload["repetition_penalty"] = canonical.repeat_penalty  # vLLM name
        if canonical.min_p is not None:
            payload["min_p"] = canonical.min_p
        # Reasoning/thinking support for vLLM.  By the time a request reaches
        # here the inference policy has resolved think/reasoning_effort to
        # the target family's own switch and level name (core/reasoning.py);
        # this translator just spells them the way vLLM reads them.  Client
        # chat_template_kwargs we do not model (Qwen's preserve_thinking, …)
        # pass through beside the switch.
        template_kwargs: Dict[str, Any] = dict(canonical.chat_template_kwargs or {})
        if canonical.reasoning_effort is not None:
            payload["reasoning_effort"] = canonical.reasoning_effort
        if canonical.think is not None:
            if isinstance(canonical.think, str):
                # Legacy string effort that bypassed the canonical validator
                payload["reasoning_effort"] = canonical.think
            else:
                # Qwen-style boolean → vLLM chat_template_kwargs
                template_kwargs["enable_thinking"] = canonical.think
        if template_kwargs:
            payload["chat_template_kwargs"] = template_kwargs
        if canonical.n != 1:
            payload["n"] = canonical.n
        if canonical.user:
            payload["user"] = canonical.user

        # Handle tool calling. `tool_choice` is only emitted alongside `tools`:
        # vLLM rejects the field when no tools are set (400 "When using
        # `tool_choice`, `tools` must be set"), whereas OpenAI tolerates the
        # pair — and Codex sends `tools: []` + `tool_choice: "auto"` on every
        # context compaction, which wedged agent sessions on 2026-09-28 (the
        # failed compaction is retried before each later turn). This is the
        # one place every inbound dialect converges, so dropping it here
        # covers chat, Responses and Anthropic alike. An empty tools list is
        # likewise never forwarded (vLLM rejects `tools: []` too).
        if canonical.tools:
            payload["tools"] = [t.model_dump() for t in canonical.tools]
            if canonical.tool_choice is not None:
                payload["tool_choice"] = canonical.tool_choice

        # Handle structured output
        if canonical.response_format:
            payload["response_format"] = VLLMOutTranslator._translate_response_format(
                canonical
            )

        return payload

    @staticmethod
    def translate_completion_request(
        canonical: CanonicalCompletionRequest,
    ) -> Dict[str, Any]:
        """Translate canonical completion request to vLLM/OpenAI format.

        Args:
            canonical: Canonical completion request

        Returns:
            OpenAI-compatible API request body
        """
        payload: Dict[str, Any] = {
            "model": canonical.model,
            "prompt": canonical.prompt,
            "stream": canonical.stream,
        }

        if canonical.temperature is not None:
            payload["temperature"] = canonical.temperature
        if canonical.top_p is not None:
            payload["top_p"] = canonical.top_p
        if canonical.max_tokens is not None:
            payload["max_tokens"] = canonical.max_tokens
        if canonical.stop is not None:
            payload["stop"] = canonical.stop
        if canonical.presence_penalty is not None:
            payload["presence_penalty"] = canonical.presence_penalty
        if canonical.frequency_penalty is not None:
            payload["frequency_penalty"] = canonical.frequency_penalty
        if canonical.seed is not None:
            payload["seed"] = canonical.seed
        if canonical.top_k is not None:
            payload["top_k"] = canonical.top_k
        if canonical.repeat_penalty is not None:
            payload["repetition_penalty"] = canonical.repeat_penalty  # vLLM name
        if canonical.min_p is not None:
            payload["min_p"] = canonical.min_p
        # Note: backend_options is Ollama-only, intentionally ignored
        if canonical.suffix is not None:
            payload["suffix"] = canonical.suffix
        if canonical.echo:
            payload["echo"] = canonical.echo
        if canonical.n != 1:
            payload["n"] = canonical.n

        return payload

    @staticmethod
    def translate_embedding_request(
        canonical: CanonicalEmbeddingRequest,
    ) -> Dict[str, Any]:
        """Translate canonical embedding request to vLLM/OpenAI format.

        Args:
            canonical: Canonical embedding request

        Returns:
            OpenAI-compatible API request body
        """
        payload: Dict[str, Any] = {
            "model": canonical.model,
            "input": canonical.input,
        }

        if canonical.encoding_format != "float":
            payload["encoding_format"] = canonical.encoding_format
        if canonical.dimensions is not None:
            payload["dimensions"] = canonical.dimensions

        return payload

    @staticmethod
    def translate_chat_response(
        openai_response: Dict[str, Any],
        request_id: Optional[str] = None,
        thinking_enabled: bool = True,
    ) -> CanonicalChatResponse:
        """Translate vLLM/OpenAI chat response to canonical format.

        Args:
            openai_response: Raw OpenAI-format response
            request_id: Override request ID

        Returns:
            CanonicalChatResponse
        """
        choices = []
        for choice_data in openai_response.get("choices", []):
            message_data = choice_data.get("message", {})

            # Parse tool_calls if present
            tool_calls = None
            if "tool_calls" in message_data and message_data["tool_calls"]:
                tool_calls = [
                    CanonicalToolCall(
                        id=tc["id"],
                        type=tc.get("type", "function"),
                        function=CanonicalFunctionCall(
                            name=tc["function"]["name"],
                            arguments=tc["function"]["arguments"],
                        ),
                    )
                    for tc in message_data["tool_calls"]
                ]

            content = message_data.get("content")
            reasoning = message_data.get("reasoning_content") or message_data.get("reasoning")

            # Fallback: extract <think> tags from content when vLLM
            # doesn't separate reasoning (e.g. Qwen3-32B)
            if not reasoning and content and "<think>" in content:
                reasoning, content = _extract_think_tags(content)

            # Fallback: extract Gemma 4 "thought\n..." prefix
            if not reasoning and content and content.startswith("thought\n"):
                reasoning, content = _extract_gemma4_thought(content)

            # vLLM/Qwen3.5 bug: when thinking is disabled the model
            # may put all output into reasoning_content with content
            # empty.  Promote reasoning to content in that case.
            # Only when thinking was explicitly disabled (think=False) AND
            # the model is one whose template honors the disable switch —
            # for always-reasoning models (Muse Glimmer's channel format)
            # the reasoning field is GENUINE reasoning, and promoting it
            # presents the internal monologue as the answer.
            if (
                not thinking_enabled and not content and reasoning and not tool_calls
                and reasoning_promotion_applies(openai_response.get("model", ""))
            ):
                content = reasoning
                reasoning = None

            message = CanonicalMessage(
                role=MessageRole(message_data.get("role", "assistant")),
                content=content,
                reasoning=reasoning,
                tool_calls=tool_calls,
            )
            choices.append(
                CanonicalChoice(
                    index=choice_data.get("index", 0),
                    message=message,
                    finish_reason=choice_data.get("finish_reason"),
                )
            )

        usage_data = openai_response.get("usage", {})
        usage = UsageInfo(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        return CanonicalChatResponse(
            id=request_id or openai_response.get("id", ""),
            created=openai_response.get("created", int(time.time())),
            model=openai_response.get("model", ""),
            choices=choices,
            usage=usage,
        )

    @staticmethod
    async def translate_chat_stream(
        openai_stream: AsyncIterator[bytes],
        request_id: str,
        model: str,
        thinking_enabled: bool = True,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Translate vLLM/OpenAI streaming response to canonical stream chunks.

        OpenAI streams Server-Sent Events:
        data: {"id":"...","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hi"}}]}
        data: [DONE]

        Args:
            openai_stream: Async iterator of response bytes
            request_id: Request ID
            model: Model name

        Yields:
            Plain dicts shaped exactly like
            ``CanonicalStreamChunk.model_dump(exclude_none=True, by_alias=True)``
            (None keys omitted, ``reasoning`` emitted as ``reasoning_content``)
            — the hot path skips per-chunk Pydantic model construction.
        """
        buffer = ""
        # State for extracting <think> tags from streaming content when
        # vLLM doesn't provide reasoning_content (e.g. Qwen3-32B).  The
        # per-chunk state machine is gated: until "<think>" is actually
        # seen, each chunk only pays for a substring probe, with a
        # partial-prefix hold-back across chunk boundaries so a split
        # tag can't slip past.
        think_seen = False
        in_think = False
        think_tag_buf = ""  # buffer for partial tag matching

        # A character split across two network chunks must not fail the stream.
        decoder = Utf8StreamDecoder()

        async for chunk_bytes in openai_stream:
            buffer += decoder.feed(chunk_bytes)

            # Process complete SSE messages
            while "\n\n" in buffer or "\r\n\r\n" in buffer:
                # Handle both Unix and Windows line endings
                if "\r\n\r\n" in buffer:
                    message, buffer = buffer.split("\r\n\r\n", 1)
                else:
                    message, buffer = buffer.split("\n\n", 1)

                for line in message.split("\n"):
                    line = line.strip()

                    if not line or line.startswith(":"):
                        continue

                    if line.startswith("data:"):
                        data_str = line[5:].strip()

                        if data_str == "[DONE]":
                            return

                        try:
                            data = orjson.loads(data_str)
                        except orjson.JSONDecodeError:
                            continue

                        choices = []
                        for choice_data in data.get("choices", []):
                            delta_data = choice_data.get("delta", {})

                            # Parse tool_calls deltas (omit None keys to
                            # match the old exclude_none serialization)
                            tc_deltas = None
                            if delta_data.get("tool_calls"):
                                tc_deltas = []
                                for tcd in delta_data["tool_calls"]:
                                    tc: Dict[str, Any] = {"index": tcd.get("index", 0)}
                                    if tcd.get("id") is not None:
                                        tc["id"] = tcd["id"]
                                    if tcd.get("type") is not None:
                                        tc["type"] = tcd["type"]
                                    if tcd.get("function") is not None:
                                        tc["function"] = tcd["function"]
                                    tc_deltas.append(tc)

                            content_delta = delta_data.get("content")
                            reasoning_delta = delta_data.get("reasoning_content") or delta_data.get("reasoning")

                            # Same Qwen 3.5 workaround as non-streaming:
                            # if content is empty but reasoning has data,
                            # promote reasoning to content. Only when
                            # thinking was explicitly disabled AND the
                            # model honors the disable switch — an
                            # always-reasoning model's reasoning deltas
                            # are genuine and must stay reasoning_content.
                            if (
                                not thinking_enabled and not content_delta
                                and reasoning_delta and not tc_deltas
                                and reasoning_promotion_applies(model)
                            ):
                                content_delta = reasoning_delta
                                reasoning_delta = None

                            # Fallback: extract <think> tags from content
                            # stream when vLLM doesn't provide
                            # reasoning_content (e.g. Qwen3-32B)
                            if content_delta and not reasoning_delta:
                                if not think_seen:
                                    # Cheap gate: probe held-back tail +
                                    # chunk for the opening tag before
                                    # engaging the state machine.
                                    probe = think_tag_buf + content_delta
                                    if "<think>" in probe:
                                        think_seen = True
                                        # Run the machine over the held
                                        # tail + this chunk
                                        content_delta = probe
                                        think_tag_buf = ""
                                    elif "<" not in probe:
                                        # Fast path: no possible tag start
                                        content_delta = probe
                                        think_tag_buf = ""
                                    else:
                                        # Hold back a trailing partial
                                        # <think> prefix (same rule as the
                                        # machine) so a tag split across
                                        # chunks can't slip past the gate
                                        partial = ""
                                        for i in range(1, min(len("<think>"), len(probe) + 1)):
                                            if probe.endswith("<think>"[:i]):
                                                partial = probe[-i:]
                                                break
                                        think_tag_buf = partial
                                        emitted = probe[: len(probe) - len(partial)]
                                        content_delta = emitted if emitted else None

                                if think_seen:
                                    think_tag_buf += content_delta
                                    content_delta = None
                                    emit_content = ""
                                    emit_reasoning = ""

                                    while think_tag_buf:
                                        if in_think:
                                            end_idx = think_tag_buf.find("</think>")
                                            if end_idx != -1:
                                                emit_reasoning += think_tag_buf[:end_idx]
                                                think_tag_buf = think_tag_buf[end_idx + 8:]
                                                in_think = False
                                            else:
                                                # Check for partial </think> at end
                                                partial = ""
                                                for i in range(1, min(len("</think>"), len(think_tag_buf) + 1)):
                                                    if think_tag_buf.endswith("</think>"[:i]):
                                                        partial = think_tag_buf[-i:]
                                                        break
                                                emit_reasoning += think_tag_buf[:len(think_tag_buf) - len(partial)]
                                                think_tag_buf = partial
                                                break
                                        else:
                                            start_idx = think_tag_buf.find("<think>")
                                            if start_idx != -1:
                                                emit_content += think_tag_buf[:start_idx]
                                                think_tag_buf = think_tag_buf[start_idx + 7:]
                                                in_think = True
                                            else:
                                                # Check for partial <think> at end
                                                partial = ""
                                                for i in range(1, min(len("<think>"), len(think_tag_buf) + 1)):
                                                    if think_tag_buf.endswith("<think>"[:i]):
                                                        partial = think_tag_buf[-i:]
                                                        break
                                                emit_content += think_tag_buf[:len(think_tag_buf) - len(partial)]
                                                think_tag_buf = partial
                                                break

                                    content_delta = emit_content if emit_content else None
                                    reasoning_delta = emit_reasoning if emit_reasoning else None

                            delta: Dict[str, Any] = {}
                            if delta_data.get("role") is not None:
                                delta["role"] = delta_data["role"]
                            if content_delta is not None:
                                delta["content"] = content_delta
                            if reasoning_delta is not None:
                                delta["reasoning_content"] = reasoning_delta
                            if tc_deltas is not None:
                                delta["tool_calls"] = tc_deltas

                            choice: Dict[str, Any] = {
                                "index": choice_data.get("index", 0),
                                "delta": delta,
                            }
                            if choice_data.get("finish_reason") is not None:
                                choice["finish_reason"] = choice_data["finish_reason"]
                            choices.append(choice)

                        chunk: Dict[str, Any] = {
                            "id": data.get("id") or request_id or f"chatcmpl-{int(time.time())}",
                            "object": "chat.completion.chunk",
                            "created": data.get("created") or int(time.time()),
                            "model": data.get("model") or model,
                            "choices": choices,
                        }

                        # Check for usage in final chunks
                        if data.get("usage"):
                            usage_data = data["usage"]
                            chunk["usage"] = {
                                "prompt_tokens": usage_data.get("prompt_tokens", 0),
                                "completion_tokens": usage_data.get(
                                    "completion_tokens", 0
                                ),
                                "total_tokens": usage_data.get("total_tokens", 0),
                                "is_estimated": False,
                            }

                        yield chunk

    @staticmethod
    def translate_rerank_request(
        canonical: CanonicalRerankRequest,
    ) -> Dict[str, Any]:
        """Translate canonical rerank request to vLLM format.

        Args:
            canonical: Canonical rerank request

        Returns:
            vLLM-compatible API request body
        """
        payload: Dict[str, Any] = {
            "model": canonical.model,
            "query": canonical.query,
            "documents": canonical.documents,
        }

        if canonical.top_n is not None:
            payload["top_n"] = canonical.top_n

        return payload

    @staticmethod
    def translate_rerank_response(
        data: Dict[str, Any],
    ) -> CanonicalRerankResponse:
        """Translate vLLM rerank response to canonical format.

        Args:
            data: Raw vLLM rerank response

        Returns:
            CanonicalRerankResponse
        """
        usage_data = data.get("usage", {})
        usage = UsageInfo(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        results = []
        for r in data.get("results", []):
            results.append(CanonicalRerankResult(
                index=r["index"],
                relevance_score=r["relevance_score"],
                document=r.get("document"),
            ))

        return CanonicalRerankResponse(
            id=data.get("id", ""),
            model=data.get("model", ""),
            results=results,
            usage=usage,
        )

    @staticmethod
    def translate_score_request(
        canonical: CanonicalScoreRequest,
    ) -> Dict[str, Any]:
        """Translate canonical score request to vLLM format.

        Args:
            canonical: Canonical score request

        Returns:
            vLLM-compatible API request body
        """
        return {
            "model": canonical.model,
            "text_1": canonical.text_1,
            "text_2": canonical.text_2,
        }

    @staticmethod
    def translate_score_response(
        data: Dict[str, Any],
    ) -> CanonicalScoreResponse:
        """Translate vLLM score response to canonical format.

        Args:
            data: Raw vLLM score response

        Returns:
            CanonicalScoreResponse
        """
        usage_data = data.get("usage", {})
        usage = UsageInfo(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        score_data = []
        for d in data.get("data", []):
            score_data.append(CanonicalScoreData(
                index=d["index"],
                score=d["score"],
            ))

        return CanonicalScoreResponse(
            id=data.get("id", ""),
            object=data.get("object", "list"),
            model=data.get("model", ""),
            data=score_data,
            usage=usage,
        )

    @staticmethod
    def translate_embedding_response(
        openai_response: Dict[str, Any],
    ) -> CanonicalEmbeddingResponse:
        """Translate vLLM/OpenAI embedding response to canonical format.

        Args:
            openai_response: Raw OpenAI-format response

        Returns:
            CanonicalEmbeddingResponse
        """
        usage_data = openai_response.get("usage", {})
        usage = UsageInfo(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        return CanonicalEmbeddingResponse(
            data=openai_response.get("data", []),
            model=openai_response.get("model", ""),
            usage=usage,
        )

    @staticmethod
    def _translate_message(msg: CanonicalMessage) -> Dict[str, Any]:
        """Translate canonical message to OpenAI format."""
        result: Dict[str, Any] = {
            "role": msg.role.value,
        }

        # Handle multimodal content
        if isinstance(msg.content, list):
            content_blocks = []

            for block in msg.content:
                if isinstance(block, TextContent):
                    content_blocks.append({"type": "text", "text": block.text})
                elif isinstance(block, ImageUrlContent):
                    # SSRF guard: only relay http(s) URLs that resolve to
                    # public addresses. data: URIs pass through unchanged.
                    # Unsafe URLs are dropped, not fatal to the request.
                    relay_url = ""
                    if isinstance(block.image_url, dict):
                        relay_url = block.image_url.get("url", "")
                    ok, reason = image_url_reason(relay_url)
                    if not ok:
                        logger.warning(
                            "dropping unsafe image_url before backend relay",
                            reason=reason,
                        )
                    else:
                        content_blocks.append(
                            {
                                "type": "image_url",
                                "image_url": block.image_url,
                            }
                        )
                elif isinstance(block, ImageBase64Content):
                    # Convert to data URL format
                    data_url = f"data:{block.media_type};base64,{block.data}"
                    content_blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        }
                    )
                elif isinstance(block, dict):
                    # Already in dict format
                    content_blocks.append(block)

            result["content"] = content_blocks
        else:
            # content can be None for tool-call-only assistant messages
            result["content"] = msg.content

        if msg.name:
            result["name"] = msg.name

        # Add tool_calls for assistant messages
        if msg.tool_calls:
            result["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]

        # Add tool_call_id for tool messages
        if msg.tool_call_id:
            result["tool_call_id"] = msg.tool_call_id

        return result

    @staticmethod
    def _translate_response_format(canonical: CanonicalChatRequest) -> Dict[str, Any]:
        """Translate response format to OpenAI format."""
        if not canonical.response_format:
            return {"type": "text"}

        if canonical.response_format.type == ResponseFormatType.JSON_OBJECT:
            # vLLM nightly rejects bare {"type": "json_object"} — it requires an
            # actual schema.  Promote to a permissive json_schema so the model is
            # constrained to produce valid JSON without a specific structure.
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "json_response",
                    "schema": {"type": "object"},
                },
            }

        if canonical.response_format.type == ResponseFormatType.JSON_SCHEMA:
            result: Dict[str, Any] = {"type": "json_schema"}
            if canonical.response_format.json_schema:
                js = dict(canonical.response_format.json_schema)
                # vLLM/OpenAI requires a "name" field in json_schema
                if "name" not in js:
                    js["name"] = "response"
                result["json_schema"] = js
            return result

        return {"type": "text"}
