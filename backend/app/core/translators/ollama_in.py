############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# ollama_in.py: Ollama API format to canonical schema translator
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Ollama API format to canonical schema translator."""

import base64
import json
from typing import Any, Dict, List, Optional

from backend.app.core.canonical_schemas import (
    CanonicalChatRequest,
    CanonicalCompletionRequest,
    CanonicalEmbeddingRequest,
    CanonicalFunctionCall,
    CanonicalMessage,
    CanonicalToolCall,
    CanonicalToolDefinition,
    ContentBlock,
    ImageBase64Content,
    MessageRole,
    ResponseFormat,
    ResponseFormatType,
    TextContent,
)


class OllamaInTranslator:
    """Translate Ollama API requests to canonical format."""

    @staticmethod
    def translate_chat_request(data: Dict[str, Any]) -> CanonicalChatRequest:
        """Translate Ollama /api/chat request to canonical format.

        Ollama chat format:
        {
            "model": "llama3.2",
            "messages": [
                {"role": "user", "content": "...", "images": ["base64..."]}
            ],
            "format": "json" or {...schema...},
            "options": {"temperature": 0.7, ...},
            "stream": true
        }

        Args:
            data: Raw request body from Ollama /api/chat endpoint

        Returns:
            CanonicalChatRequest
        """
        messages = []
        for msg in data.get("messages", []):
            messages.append(OllamaInTranslator._translate_message(msg))

        # Handle Ollama options
        options = data.get("options", {})

        # Handle Ollama format field
        response_format = OllamaInTranslator._translate_format(data.get("format"))

        # Known option keys that map to canonical fields
        _KNOWN_OPTION_KEYS = {
            "temperature", "top_p", "num_predict", "stop",
            "presence_penalty", "frequency_penalty", "seed",
            "top_k", "repeat_penalty", "min_p",
        }

        # Collect unknown option keys into backend_options
        backend_options = {
            k: v for k, v in options.items() if k not in _KNOWN_OPTION_KEYS
        } or None

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
            temperature=options.get("temperature"),
            top_p=options.get("top_p"),
            max_tokens=options.get("num_predict"),  # Ollama uses num_predict
            stream=data.get("stream", True),  # Ollama defaults to streaming
            stop=options.get("stop"),
            presence_penalty=options.get("presence_penalty"),
            frequency_penalty=options.get("frequency_penalty"),
            seed=options.get("seed"),
            top_k=options.get("top_k"),
            repeat_penalty=options.get("repeat_penalty"),
            min_p=options.get("min_p"),
            think=data.get("think"),
            backend_options=backend_options,
            tools=tools,
            response_format=response_format,
        )

    @staticmethod
    def translate_generate_request(data: Dict[str, Any]) -> CanonicalCompletionRequest:
        """Translate Ollama /api/generate request to canonical format.

        Ollama generate format:
        {
            "model": "llama3.2",
            "prompt": "...",
            "system": "...",
            "images": ["base64..."],
            "format": "json",
            "options": {...},
            "stream": true
        }

        Args:
            data: Raw request body from Ollama /api/generate endpoint

        Returns:
            CanonicalCompletionRequest
        """
        options = data.get("options", {})

        # Combine system prompt with main prompt if present
        prompt = data.get("prompt", "")
        system = data.get("system")
        if system:
            prompt = f"{system}\n\n{prompt}"

        # Handle Ollama format field (same as chat path)
        response_format = OllamaInTranslator._translate_format(data.get("format"))

        # Known option keys that map to canonical fields
        _KNOWN_OPTION_KEYS = {
            "temperature", "top_p", "num_predict", "stop",
            "presence_penalty", "frequency_penalty", "seed",
            "top_k", "repeat_penalty", "min_p",
        }

        # Collect unknown option keys into backend_options
        backend_options = {
            k: v for k, v in options.items() if k not in _KNOWN_OPTION_KEYS
        } or None

        return CanonicalCompletionRequest(
            model=data["model"],
            prompt=prompt,
            temperature=options.get("temperature"),
            top_p=options.get("top_p"),
            max_tokens=options.get("num_predict"),
            stream=data.get("stream", True),
            stop=options.get("stop"),
            presence_penalty=options.get("presence_penalty"),
            frequency_penalty=options.get("frequency_penalty"),
            seed=options.get("seed"),
            top_k=options.get("top_k"),
            repeat_penalty=options.get("repeat_penalty"),
            min_p=options.get("min_p"),
            backend_options=backend_options,
            response_format=response_format,
            think=data.get("think"),
        )

    @staticmethod
    def translate_embedding_request(data: Dict[str, Any]) -> CanonicalEmbeddingRequest:
        """Translate Ollama embedding request to canonical format.

        Ollama uses /api/embeddings with:
        {
            "model": "nomic-embed-text",
            "prompt": "..." or "input": "..."
        }

        Args:
            data: Raw request body

        Returns:
            CanonicalEmbeddingRequest
        """
        # Ollama uses "prompt" while OpenAI uses "input"
        input_text = data.get("input") or data.get("prompt", "")

        return CanonicalEmbeddingRequest(
            model=data["model"],
            input=input_text,
        )

    @staticmethod
    def _translate_message(msg: Dict[str, Any]) -> CanonicalMessage:
        """Translate a single Ollama message to canonical format."""
        role_str = msg.get("role", "user")
        role = MessageRole(role_str)

        content = msg.get("content", "")
        images = msg.get("images", [])

        # Extract tool_calls from assistant messages
        # Ollama uses dict arguments; canonical uses JSON strings
        tool_calls = None
        if "tool_calls" in msg and msg["tool_calls"]:
            tool_calls = []
            for i, tc in enumerate(msg["tool_calls"]):
                func = tc.get("function", {})
                args = func.get("arguments", {})
                # Convert dict arguments to JSON string
                if isinstance(args, dict):
                    args = json.dumps(args)
                tool_calls.append(
                    CanonicalToolCall(
                        id=tc.get("id", f"call_{i}"),
                        type=tc.get("type", "function"),
                        function=CanonicalFunctionCall(
                            name=func.get("name", ""),
                            arguments=args,
                        ),
                    )
                )

        # If there are images, create multimodal content
        if images:
            content_blocks: List[ContentBlock] = []

            # Add text content if present
            if content:
                content_blocks.append(TextContent(text=content))

            # Add image content
            for img_data in images:
                # Ollama uses raw base64, need to determine media type
                # Default to jpeg if we can't detect
                media_type = OllamaInTranslator._detect_image_type(img_data)
                content_blocks.append(
                    ImageBase64Content(data=img_data, media_type=media_type)
                )

            return CanonicalMessage(role=role, content=content_blocks, tool_calls=tool_calls)

        # Simple text message
        return CanonicalMessage(role=role, content=content, tool_calls=tool_calls)

    @staticmethod
    def _translate_format(
        format_spec: Optional[Any]
    ) -> Optional[ResponseFormat]:
        """Translate Ollama format specification to canonical response format.

        Ollama supports:
        - "json" - simple JSON mode
        - {...} - JSON schema object
        """
        if format_spec is None:
            return None

        if isinstance(format_spec, str):
            if format_spec.lower() == "json":
                return ResponseFormat(type=ResponseFormatType.JSON_OBJECT)
            return None

        if isinstance(format_spec, dict):
            # It's a JSON schema
            return ResponseFormat(
                type=ResponseFormatType.JSON_SCHEMA,
                json_schema={"schema": format_spec},
            )

        return None

    @staticmethod
    def _detect_image_type(base64_data: str) -> str:
        """Detect image type from base64 data.

        Args:
            base64_data: Base64 encoded image data

        Returns:
            MIME type string
        """
        try:
            # Decode first few bytes to check magic numbers
            decoded = base64.b64decode(base64_data[:32])

            # PNG: 89 50 4E 47
            if decoded[:4] == b'\x89PNG':
                return "image/png"
            # JPEG: FF D8 FF
            if decoded[:3] == b'\xff\xd8\xff':
                return "image/jpeg"
            # GIF: 47 49 46 38
            if decoded[:4] == b'GIF8':
                return "image/gif"
            # WebP: 52 49 46 46 ... 57 45 42 50
            if decoded[:4] == b'RIFF' and decoded[8:12] == b'WEBP':
                return "image/webp"

        except Exception:
            pass

        # Default to JPEG
        return "image/jpeg"
