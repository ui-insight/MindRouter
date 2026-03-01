############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# ollama_out.py: Canonical schema to Ollama API format translator
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Canonical schema to Ollama API format translator."""

import json
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from backend.app.core.canonical_schemas import (
    CanonicalChatRequest,
    CanonicalChatResponse,
    CanonicalChoice,
    CanonicalCompletionRequest,
    CanonicalEmbeddingRequest,
    CanonicalEmbeddingResponse,
    CanonicalFunctionCall,
    CanonicalMessage,
    CanonicalStreamChunk,
    CanonicalStreamChoice,
    CanonicalStreamDelta,
    CanonicalToolCall,
    ImageBase64Content,
    ImageUrlContent,
    MessageRole,
    ResponseFormatType,
    TextContent,
    UsageInfo,
)


class OllamaOutTranslator:
    """Translate canonical requests to Ollama API format."""

    @staticmethod
    def translate_chat_request(canonical: CanonicalChatRequest) -> Dict[str, Any]:
        """Translate canonical chat request to Ollama /api/chat format.

        Args:
            canonical: Canonical chat request

        Returns:
            Ollama API request body
        """
        messages = []
        for msg in canonical.messages:
            messages.append(OllamaOutTranslator._translate_message(msg))

        payload: Dict[str, Any] = {
            "model": canonical.model,
            "messages": messages,
            "stream": canonical.stream,
        }

        # Build options dict
        options = OllamaOutTranslator._build_options(canonical)
        if options:
            payload["options"] = options

        # Handle structured output
        if canonical.response_format:
            format_spec = OllamaOutTranslator._translate_response_format(canonical)
            if format_spec:
                payload["format"] = format_spec

        # Handle tool calling
        if canonical.tools:
            payload["tools"] = [t.model_dump() for t in canonical.tools]

        # Thinking mode goes at top level, NOT inside options
        if canonical.think is not None:
            payload["think"] = canonical.think

        return payload

    @staticmethod
    def translate_generate_request(
        canonical: CanonicalCompletionRequest,
    ) -> Dict[str, Any]:
        """Translate canonical completion request to Ollama /api/generate format.

        Args:
            canonical: Canonical completion request

        Returns:
            Ollama API request body
        """
        prompt = canonical.prompt if isinstance(canonical.prompt, str) else canonical.prompt[0]

        payload: Dict[str, Any] = {
            "model": canonical.model,
            "prompt": prompt,
            "stream": canonical.stream,
        }

        # Build options
        options = OllamaOutTranslator._build_options_from_completion(canonical)
        if options:
            payload["options"] = options

        # Thinking mode goes at top level, NOT inside options
        if canonical.think is not None:
            payload["think"] = canonical.think

        return payload

    @staticmethod
    def translate_embedding_request(
        canonical: CanonicalEmbeddingRequest,
    ) -> Dict[str, Any]:
        """Translate canonical embedding request to Ollama format.

        Args:
            canonical: Canonical embedding request

        Returns:
            Ollama API request body
        """
        # Ollama uses "prompt" for single text, but we may have a list
        input_text = canonical.input
        if isinstance(input_text, list):
            input_text = input_text[0] if input_text else ""

        return {
            "model": canonical.model,
            "prompt": input_text,
        }

    @staticmethod
    def translate_chat_response(
        ollama_response: Dict[str, Any],
        request_id: str,
        model: str,
    ) -> CanonicalChatResponse:
        """Translate Ollama chat response to canonical format.

        Ollama response format:
        {
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "..."},
            "done": true,
            "total_duration": 123456789,
            "load_duration": 1234567,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 12345678,
            "eval_count": 50,
            "eval_duration": 123456789
        }

        Args:
            ollama_response: Raw Ollama response
            request_id: Request ID for the response
            model: Model name

        Returns:
            CanonicalChatResponse
        """
        message = ollama_response.get("message", {})

        # Parse tool_calls from Ollama response (dict arguments → JSON string)
        tool_calls = None
        if "tool_calls" in message and message["tool_calls"]:
            tool_calls = []
            for i, tc in enumerate(message["tool_calls"]):
                func = tc.get("function", {})
                args = func.get("arguments", {})
                if isinstance(args, dict):
                    args = json.dumps(args)
                tool_calls.append(
                    CanonicalToolCall(
                        id=tc.get("id", f"call_{i}"),
                        type="function",
                        function=CanonicalFunctionCall(
                            name=func.get("name", ""),
                            arguments=args,
                        ),
                    )
                )

        finish_reason = "stop" if ollama_response.get("done") else None
        if tool_calls:
            finish_reason = "tool_calls"

        choice = CanonicalChoice(
            index=0,
            message=CanonicalMessage(
                role=MessageRole(message.get("role", "assistant")),
                content=message.get("content") or None,
                reasoning=message.get("thinking") or None,
                tool_calls=tool_calls,
            ),
            finish_reason=finish_reason,
        )

        # Extract usage info
        usage = UsageInfo(
            prompt_tokens=ollama_response.get("prompt_eval_count", 0),
            completion_tokens=ollama_response.get("eval_count", 0),
            total_tokens=(
                ollama_response.get("prompt_eval_count", 0)
                + ollama_response.get("eval_count", 0)
            ),
        )

        return CanonicalChatResponse(
            id=request_id,
            created=int(time.time()),
            model=model,
            choices=[choice],
            usage=usage,
        )

    @staticmethod
    async def translate_chat_stream(
        ollama_stream: AsyncIterator[bytes],
        request_id: str,
        model: str,
    ) -> AsyncIterator[CanonicalStreamChunk]:
        """Translate Ollama streaming response to canonical stream chunks.

        Ollama streams JSON objects line by line:
        {"model":"llama3.2","message":{"role":"assistant","content":"Hello"},"done":false}
        {"model":"llama3.2","message":{"role":"assistant","content":" there"},"done":false}
        {"model":"llama3.2","message":{"role":"assistant","content":"!"},"done":true,...}

        Args:
            ollama_stream: Async iterator of response bytes
            request_id: Request ID
            model: Model name

        Yields:
            CanonicalStreamChunk objects
        """
        buffer = ""

        async for chunk_bytes in ollama_stream:
            buffer += chunk_bytes.decode("utf-8")

            # Process complete lines
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()

                if not line:
                    continue

                try:
                    data = json.loads(line)

                    message = data.get("message", {})
                    content = message.get("content", "")
                    is_done = data.get("done", False)

                    # Parse tool_calls from streaming chunk
                    tc_deltas = None
                    if "tool_calls" in message and message["tool_calls"]:
                        from backend.app.core.canonical_schemas import CanonicalStreamToolCallDelta
                        tc_deltas = []
                        for i, tc in enumerate(message["tool_calls"]):
                            func = tc.get("function", {})
                            args = func.get("arguments", {})
                            if isinstance(args, dict):
                                args = json.dumps(args)
                            tc_deltas.append(
                                CanonicalStreamToolCallDelta(
                                    index=i,
                                    id=tc.get("id", f"call_{i}"),
                                    type="function",
                                    function={
                                        "name": func.get("name", ""),
                                        "arguments": args,
                                    },
                                )
                            )

                    finish_reason = None
                    if is_done:
                        finish_reason = "tool_calls" if tc_deltas else "stop"

                    thinking = message.get("thinking", "")

                    delta = CanonicalStreamDelta(
                        role=MessageRole.ASSISTANT if content or thinking or tc_deltas else None,
                        content=content if content else None,
                        reasoning=thinking if thinking else None,
                        tool_calls=tc_deltas,
                    )

                    choice = CanonicalStreamChoice(
                        index=0,
                        delta=delta,
                        finish_reason=finish_reason,
                    )

                    # Include usage in final chunk
                    usage = None
                    if is_done:
                        usage = UsageInfo(
                            prompt_tokens=data.get("prompt_eval_count", 0),
                            completion_tokens=data.get("eval_count", 0),
                            total_tokens=(
                                data.get("prompt_eval_count", 0)
                                + data.get("eval_count", 0)
                            ),
                        )

                    yield CanonicalStreamChunk(
                        id=request_id or f"chatcmpl-{int(time.time())}",
                        created=int(time.time()),
                        model=model,
                        choices=[choice],
                        usage=usage,
                    )

                except json.JSONDecodeError:
                    continue

    @staticmethod
    def translate_embedding_response(
        ollama_response: Dict[str, Any],
        model: str,
    ) -> CanonicalEmbeddingResponse:
        """Translate Ollama embedding response to canonical format.

        Args:
            ollama_response: Raw Ollama response
            model: Model name

        Returns:
            CanonicalEmbeddingResponse
        """
        embedding = ollama_response.get("embedding", [])

        return CanonicalEmbeddingResponse(
            data=[
                {
                    "object": "embedding",
                    "embedding": embedding,
                    "index": 0,
                }
            ],
            model=model,
            usage=UsageInfo(
                prompt_tokens=ollama_response.get("prompt_eval_count", 0),
                total_tokens=ollama_response.get("prompt_eval_count", 0),
            ),
        )

    @staticmethod
    def _translate_message(msg: CanonicalMessage) -> Dict[str, Any]:
        """Translate canonical message to Ollama format."""
        result: Dict[str, Any] = {
            "role": msg.role.value,
        }

        # Handle multimodal content
        if isinstance(msg.content, list):
            text_parts = []
            images = []

            for block in msg.content:
                if isinstance(block, TextContent):
                    text_parts.append(block.text)
                elif isinstance(block, ImageBase64Content):
                    images.append(block.data)
                elif isinstance(block, ImageUrlContent):
                    # Ollama doesn't support URLs directly, would need to fetch
                    # For now, skip or log warning
                    pass
                elif isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "image_base64":
                        images.append(block.get("data", ""))

            result["content"] = " ".join(text_parts)
            if images:
                result["images"] = images
        else:
            result["content"] = msg.content if msg.content is not None else ""

        # Add tool_calls with arguments converted from JSON string to dict
        if msg.tool_calls:
            result["tool_calls"] = []
            for tc in msg.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                result["tool_calls"].append({
                    "function": {
                        "name": tc.function.name,
                        "arguments": args,
                    },
                })

        return result

    @staticmethod
    def _build_options(canonical: CanonicalChatRequest) -> Dict[str, Any]:
        """Build Ollama options dict from canonical request."""
        options: Dict[str, Any] = {}

        if canonical.temperature is not None:
            options["temperature"] = canonical.temperature
        if canonical.top_p is not None:
            options["top_p"] = canonical.top_p
        if canonical.max_tokens is not None:
            options["num_predict"] = canonical.max_tokens
        if canonical.stop is not None:
            options["stop"] = canonical.stop
        if canonical.presence_penalty is not None:
            options["presence_penalty"] = canonical.presence_penalty
        if canonical.frequency_penalty is not None:
            options["frequency_penalty"] = canonical.frequency_penalty
        if canonical.seed is not None:
            options["seed"] = canonical.seed
        if canonical.top_k is not None:
            options["top_k"] = canonical.top_k
        if canonical.repeat_penalty is not None:
            options["repeat_penalty"] = canonical.repeat_penalty
        if canonical.min_p is not None:
            options["min_p"] = canonical.min_p

        # Merge opaque backend options (mirostat, tfs_z, num_ctx, etc.)
        if canonical.backend_options:
            options.update(canonical.backend_options)

        return options

    @staticmethod
    def _build_options_from_completion(
        canonical: CanonicalCompletionRequest,
    ) -> Dict[str, Any]:
        """Build Ollama options dict from canonical completion request."""
        options: Dict[str, Any] = {}

        if canonical.temperature is not None:
            options["temperature"] = canonical.temperature
        if canonical.top_p is not None:
            options["top_p"] = canonical.top_p
        if canonical.max_tokens is not None:
            options["num_predict"] = canonical.max_tokens
        if canonical.stop is not None:
            options["stop"] = canonical.stop
        if canonical.presence_penalty is not None:
            options["presence_penalty"] = canonical.presence_penalty
        if canonical.frequency_penalty is not None:
            options["frequency_penalty"] = canonical.frequency_penalty
        if canonical.seed is not None:
            options["seed"] = canonical.seed
        if canonical.top_k is not None:
            options["top_k"] = canonical.top_k
        if canonical.repeat_penalty is not None:
            options["repeat_penalty"] = canonical.repeat_penalty
        if canonical.min_p is not None:
            options["min_p"] = canonical.min_p

        # Merge opaque backend options
        if canonical.backend_options:
            options.update(canonical.backend_options)

        return options

    @staticmethod
    def _translate_response_format(canonical: CanonicalChatRequest) -> Optional[Any]:
        """Translate response format to Ollama format."""
        if not canonical.response_format:
            return None

        if canonical.response_format.type == ResponseFormatType.JSON_OBJECT:
            return "json"

        if canonical.response_format.type == ResponseFormatType.JSON_SCHEMA:
            schema = canonical.response_format.json_schema
            if schema:
                # Ollama expects the schema directly
                return schema.get("schema", schema)

        return None
