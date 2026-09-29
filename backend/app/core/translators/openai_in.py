############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# openai_in.py: OpenAI API format to canonical schema translator
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""OpenAI API format to canonical schema translator."""

from typing import Any, Dict, List, Optional, Union

import structlog

from backend.app.core.canonical_schemas import (
    CanonicalChatRequest,
    CanonicalCompletionRequest,
    CanonicalEmbeddingRequest,
    CanonicalFunctionCall,
    CanonicalMessage,
    CanonicalRerankRequest,
    CanonicalScoreRequest,
    CanonicalToolCall,
    CanonicalToolDefinition,
    ContentBlock,
    ImageBase64Content,
    ImageUrlContent,
    MessageRole,
    ResponseFormat,
    ResponseFormatType,
    TextContent,
)
from backend.app.core.url_guard import image_url_reason

logger = structlog.get_logger(__name__)


class OpenAIInTranslator:
    """Translate OpenAI API requests to canonical format."""

    @staticmethod
    def translate_chat_request(data: Dict[str, Any]) -> CanonicalChatRequest:
        """Translate OpenAI chat completion request to canonical format.

        Args:
            data: Raw request body from OpenAI-compatible endpoint

        Returns:
            CanonicalChatRequest
        """
        from backend.app.core.translators.field_validation import validate_request_fields
        validate_request_fields(data, dialect="chat")

        messages = []
        for msg in data.get("messages", []):
            messages.append(OpenAIInTranslator._translate_message(msg))

        response_format = None
        if "response_format" in data:
            response_format = OpenAIInTranslator._translate_response_format(
                data["response_format"]
            )

        # Handle tools
        tools = None
        if "tools" in data:
            tools = [
                CanonicalToolDefinition(
                    type=t.get("type", "function"),
                    function=t["function"],
                )
                for t in data["tools"]
            ]

        return CanonicalChatRequest(
            model=data["model"],
            messages=messages,
            temperature=data.get("temperature"),
            top_p=data.get("top_p"),
            max_tokens=data.get("max_completion_tokens") or data.get("max_tokens"),
            stream=data.get("stream", False),
            stop=data.get("stop"),
            presence_penalty=data.get("presence_penalty"),
            frequency_penalty=data.get("frequency_penalty"),
            seed=data.get("seed"),
            top_k=data.get("top_k"),
            repeat_penalty=data.get("repetition_penalty"),  # OpenAI/vLLM name
            min_p=data.get("min_p"),
            tools=tools,
            tool_choice=data.get("tool_choice"),
            response_format=response_format,
            n=data.get("n", 1),
            user=data.get("user"),
            think=OpenAIInTranslator._resolve_think(data),
            reasoning_effort=OpenAIInTranslator._resolve_effort(data),
            chat_template_kwargs=OpenAIInTranslator._passthrough_template_kwargs(data),
            include_usage=bool((data.get("stream_options") or {}).get("include_usage", False)),
        )

    @staticmethod
    def translate_completion_request(data: Dict[str, Any]) -> CanonicalCompletionRequest:
        """Translate OpenAI completion request to canonical format.

        Args:
            data: Raw request body from OpenAI-compatible endpoint

        Returns:
            CanonicalCompletionRequest
        """
        return CanonicalCompletionRequest(
            model=data["model"],
            prompt=data["prompt"],
            temperature=data.get("temperature"),
            top_p=data.get("top_p"),
            max_tokens=data.get("max_completion_tokens") or data.get("max_tokens"),
            stream=data.get("stream", False),
            stop=data.get("stop"),
            presence_penalty=data.get("presence_penalty"),
            frequency_penalty=data.get("frequency_penalty"),
            seed=data.get("seed"),
            top_k=data.get("top_k"),
            repeat_penalty=data.get("repetition_penalty"),  # OpenAI/vLLM name
            min_p=data.get("min_p"),
            suffix=data.get("suffix"),
            echo=data.get("echo", False),
            n=data.get("n", 1),
            best_of=data.get("best_of", 1),
        )

    @staticmethod
    def translate_embedding_request(data: Dict[str, Any]) -> CanonicalEmbeddingRequest:
        """Translate OpenAI embedding request to canonical format.

        Args:
            data: Raw request body from OpenAI-compatible endpoint

        Returns:
            CanonicalEmbeddingRequest
        """
        return CanonicalEmbeddingRequest(
            model=data["model"],
            input=data["input"],
            encoding_format=data.get("encoding_format", "float"),
            dimensions=data.get("dimensions"),
        )

    @staticmethod
    def translate_rerank_request(data: Dict[str, Any]) -> CanonicalRerankRequest:
        """Translate OpenAI rerank request to canonical format.

        Args:
            data: Raw request body from rerank endpoint

        Returns:
            CanonicalRerankRequest
        """
        return CanonicalRerankRequest(
            model=data["model"],
            query=data["query"],
            documents=data["documents"],
            top_n=data.get("top_n"),
            return_documents=data.get("return_documents", True),
        )

    @staticmethod
    def translate_score_request(data: Dict[str, Any]) -> CanonicalScoreRequest:
        """Translate OpenAI score request to canonical format.

        Args:
            data: Raw request body from score endpoint

        Returns:
            CanonicalScoreRequest
        """
        return CanonicalScoreRequest(
            model=data["model"],
            text_1=data["text_1"],
            text_2=data["text_2"],
        )

    @staticmethod
    def _translate_message(msg: Dict[str, Any]) -> CanonicalMessage:
        """Translate a single message to canonical format."""
        role = MessageRole(msg["role"])
        content = msg.get("content")

        # Extract tool_calls from assistant messages
        tool_calls = None
        if "tool_calls" in msg and msg["tool_calls"]:
            tool_calls = [
                CanonicalToolCall(
                    id=tc["id"],
                    type=tc.get("type", "function"),
                    function=CanonicalFunctionCall(
                        name=tc["function"].get("name") or "",
                        arguments=tc["function"].get("arguments") or "{}",
                    ),
                )
                for tc in msg["tool_calls"]
            ]

        # Extract tool_call_id from tool messages
        tool_call_id = msg.get("tool_call_id")

        # Handle multimodal content
        if isinstance(content, list):
            content_blocks: List[ContentBlock] = []
            for item in content:
                if isinstance(item, str):
                    content_blocks.append(TextContent(text=item))
                elif isinstance(item, dict):
                    block = OpenAIInTranslator._translate_content_block(item)
                    if block:
                        content_blocks.append(block)
            return CanonicalMessage(
                role=role,
                content=content_blocks,
                name=msg.get("name"),
                tool_calls=tool_calls,
                tool_call_id=tool_call_id,
            )

        # Simple text content (content can be None for tool-call-only messages)
        return CanonicalMessage(
            role=role,
            content=content,
            name=msg.get("name"),
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
        )

    @staticmethod
    def _translate_content_block(item: Dict[str, Any]) -> Optional[ContentBlock]:
        """Translate a content block to canonical format."""
        item_type = item.get("type")

        if item_type == "text":
            return TextContent(text=item.get("text", ""))

        elif item_type == "image_url":
            image_url = item.get("image_url", {})
            url = image_url.get("url", "")

            # Check if it's a base64 data URL
            if url.startswith("data:"):
                # Parse data URL: data:image/png;base64,<data>
                try:
                    header, data = url.split(",", 1)
                    media_type = header.split(":")[1].split(";")[0]
                    return ImageBase64Content(data=data, media_type=media_type)
                except (ValueError, IndexError):
                    pass

            # SSRF guard: user-supplied http(s) image URLs are forwarded to
            # backends inside the GPU/HPC network. Drop (skip) any URL that
            # resolves to a private/internal/metadata address instead of
            # failing the whole request. data: URIs are handled above.
            ok, reason = image_url_reason(url)
            if not ok:
                logger.warning(
                    "dropping unsafe image_url from vision request",
                    reason=reason,
                )
                return None

            return ImageUrlContent(
                image_url={
                    "url": url,
                    "detail": image_url.get("detail", "auto"),
                }
            )

        return None

    @staticmethod
    def _translate_response_format(
        response_format: Dict[str, Any]
    ) -> ResponseFormat:
        """Translate response format specification."""
        format_type = response_format.get("type", "text")

        if format_type == "json_object":
            return ResponseFormat(type=ResponseFormatType.JSON_OBJECT)

        elif format_type == "json_schema":
            return ResponseFormat(
                type=ResponseFormatType.JSON_SCHEMA,
                json_schema=response_format.get("json_schema"),
            )

        return ResponseFormat(type=ResponseFormatType.TEXT)

    @staticmethod
    def _resolve_think(data: Dict[str, Any]) -> Optional[Union[bool, str]]:
        """Resolve thinking/think from multiple possible input formats.

        Accepts:
        - think: true/false/"low"/"medium"/"high" (Ollama/MindRouter native)
        - thinking: {"type": "enabled"/"disabled"} (OpenAI/Anthropic style)
        - chat_template_kwargs: {"enable_thinking": true/false} (vLLM native)

        A level given without a switch (``reasoning_effort`` alone) is an
        opt-in too; that is resolved later from the canonical pair, so this
        function only answers "did the client say on or off explicitly".
        """
        # Direct think field takes priority
        if "think" in data:
            return data["think"]

        # OpenAI/Anthropic-style thinking object
        thinking = data.get("thinking")
        if isinstance(thinking, dict):
            thinking_type = thinking.get("type")
            if thinking_type in ("enabled", "adaptive"):
                return True
            elif thinking_type == "disabled":
                return False

        # vLLM-native chat_template_kwargs
        ctk = data.get("chat_template_kwargs")
        if isinstance(ctk, dict) and "enable_thinking" in ctk:
            return ctk["enable_thinking"]

        return None

    @staticmethod
    def _resolve_effort(data: Dict[str, Any]) -> Optional[str]:
        """Resolve the reasoning level from every spelling clients use.

        Priority: top-level ``reasoning_effort`` (OpenAI chat), then
        ``reasoning.effort`` (Responses-style object some SDKs send on chat),
        then ``chat_template_kwargs.reasoning_effort`` (vLLM native, the form
        the Qwen3.8 model card documents), then an Anthropic-style
        ``thinking.budget_tokens`` bucketed into a level.  The value is a
        gateway level; validation and per-family translation happen in the
        canonical model and the inference policy.
        """
        effort = data.get("reasoning_effort")
        if effort is not None:
            return effort
        reasoning = data.get("reasoning")
        if isinstance(reasoning, dict) and reasoning.get("effort") is not None:
            return reasoning["effort"]
        ctk = data.get("chat_template_kwargs")
        if isinstance(ctk, dict) and ctk.get("reasoning_effort") is not None:
            return ctk["reasoning_effort"]
        thinking = data.get("thinking")
        if isinstance(thinking, dict) and thinking.get("budget_tokens") is not None:
            from backend.app.core.reasoning import budget_to_level

            return budget_to_level(thinking["budget_tokens"])
        return None

    # Keys of chat_template_kwargs the gateway owns and rebuilds itself.
    _OWNED_TEMPLATE_KWARGS = frozenset({"enable_thinking", "reasoning_effort"})

    @staticmethod
    def _passthrough_template_kwargs(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Keep the chat_template_kwargs keys we do not model (e.g. Qwen's
        ``preserve_thinking``) so they reach a vLLM backend unchanged.

        Only scalar values pass; the keys the gateway derives itself
        (``enable_thinking``, ``reasoning_effort``) are dropped here and
        re-emitted from the canonical fields by the outbound translator.
        """
        ctk = data.get("chat_template_kwargs")
        if not isinstance(ctk, dict):
            return None
        kept = {
            k: v
            for k, v in ctk.items()
            if k not in OpenAIInTranslator._OWNED_TEMPLATE_KWARGS
            and isinstance(v, (bool, int, float, str))
        }
        return kept or None
