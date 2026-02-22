############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# anthropic_in.py: Anthropic Messages API format to canonical schema translator
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Anthropic Messages API format to canonical schema translator."""

import json
from typing import Any, Dict, List, Optional

from backend.app.core.canonical_schemas import (
    CanonicalChatRequest,
    CanonicalFunctionCall,
    CanonicalMessage,
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


class AnthropicInTranslator:
    """Translate Anthropic Messages API requests to canonical format."""

    @staticmethod
    def translate_messages_request(data: Dict[str, Any]) -> CanonicalChatRequest:
        """Translate Anthropic Messages API request to canonical format.

        Args:
            data: Raw request body from Anthropic-compatible endpoint

        Returns:
            CanonicalChatRequest
        """
        messages: List[CanonicalMessage] = []

        # Handle system prompt (top-level in Anthropic API)
        system = data.get("system")
        if system is not None:
            system_msg = AnthropicInTranslator._translate_system(system)
            messages.append(system_msg)

        # Translate conversation messages
        # A single Anthropic message can expand to multiple canonical messages
        # (e.g. a user message with mixed text + tool_results)
        for msg in data.get("messages", []):
            translated = AnthropicInTranslator._translate_message(msg)
            if isinstance(translated, list):
                messages.extend(translated)
            else:
                messages.append(translated)

        # Handle tools
        tools = None
        if "tools" in data:
            tools = [
                CanonicalToolDefinition(
                    type="function",
                    function={
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", {}),
                    },
                )
                for t in data["tools"]
            ]

        # Handle tool_choice
        tool_choice = None
        if "tool_choice" in data:
            tc = data["tool_choice"]
            if isinstance(tc, dict):
                tc_type = tc.get("type")
                if tc_type == "auto":
                    tool_choice = "auto"
                elif tc_type == "any":
                    tool_choice = "required"
                elif tc_type == "tool":
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tc["name"]},
                    }
            elif isinstance(tc, str):
                # Already a string like "auto"
                tool_choice = tc

        # Handle structured output via output_config
        response_format = None
        output_config = data.get("output_config")
        if output_config:
            fmt = output_config.get("format")
            if fmt and fmt.get("type") == "json_schema":
                response_format = ResponseFormat(
                    type=ResponseFormatType.JSON_SCHEMA,
                    json_schema={
                        "name": fmt.get("name", "response"),
                        "schema": fmt.get("schema"),
                    },
                )

        # Handle thinking mode
        think = None
        thinking = data.get("thinking")
        if thinking:
            thinking_type = thinking.get("type")
            if thinking_type in ("enabled", "adaptive"):
                think = True
            elif thinking_type == "disabled":
                think = False

        # Map stop_sequences to stop
        stop = data.get("stop_sequences")

        # Map metadata.user_id to user
        user = None
        metadata = data.get("metadata")
        if metadata:
            user = metadata.get("user_id")

        return CanonicalChatRequest(
            model=data["model"],
            messages=messages,
            max_tokens=data.get("max_tokens"),
            temperature=data.get("temperature"),
            top_p=data.get("top_p"),
            top_k=data.get("top_k"),
            stream=data.get("stream", False),
            stop=stop,
            think=think,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            user=user,
        )

    @staticmethod
    def format_response(response: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Convert OpenAI-format canonical response dict to Anthropic Messages response.

        Args:
            response: OpenAI-format response dict from inference service
            model: Model name to include in response

        Returns:
            Anthropic Messages API formatted response dict
        """
        # Extract content from OpenAI format
        choices = response.get("choices", [])
        content_blocks = []
        finish_reason = "end_turn"
        if choices:
            choice = choices[0]
            message = choice.get("message", {})
            content_text = message.get("content") or ""
            finish_reason = AnthropicInTranslator._map_finish_reason(
                choice.get("finish_reason", "stop")
            )

            # Add text block if there's text content
            if content_text:
                content_blocks.append({"type": "text", "text": content_text})

            # Add tool_use blocks if present
            tool_calls = message.get("tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    func = tc.get("function", {})
                    args = func.get("arguments", "")
                    # Parse JSON string arguments to dict for Anthropic format
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            args = {}
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": func.get("name", ""),
                        "input": args,
                    })

        if not content_blocks:
            content_blocks.append({"type": "text", "text": ""})

        # Map usage fields
        usage = response.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        return {
            "id": response.get("id", ""),
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content_blocks,
            "stop_reason": finish_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        }

    @staticmethod
    def format_stream_event(event_type: str, data: Dict[str, Any]) -> str:
        """Format a single Anthropic SSE event.

        Args:
            event_type: The event type (e.g. message_start, content_block_delta)
            data: The event data payload

        Returns:
            Formatted SSE string
        """
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    @staticmethod
    def _translate_system(system: Any) -> CanonicalMessage:
        """Translate Anthropic system prompt to canonical system message.

        Anthropic system can be a string or an array of content blocks.
        """
        if isinstance(system, str):
            return CanonicalMessage(role=MessageRole.SYSTEM, content=system)

        # Array of content blocks (e.g. [{"type":"text","text":"..."}])
        if isinstance(system, list):
            text_parts = []
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            return CanonicalMessage(
                role=MessageRole.SYSTEM,
                content=" ".join(text_parts) if text_parts else "",
            )

        return CanonicalMessage(role=MessageRole.SYSTEM, content=str(system))

    @staticmethod
    def _translate_message(msg: Dict[str, Any]):
        """Translate a single Anthropic message to canonical format.

        Returns a single CanonicalMessage or a list of CanonicalMessages
        (when a user message contains tool_result blocks that need expansion).
        """
        role = MessageRole(msg["role"])
        content = msg.get("content")

        # Simple string content
        if isinstance(content, str):
            return CanonicalMessage(role=role, content=content)

        # Array of content blocks
        if isinstance(content, list):
            # Separate tool_use, tool_result, and regular content blocks
            content_blocks: List[ContentBlock] = []
            tool_calls: List[CanonicalToolCall] = []
            tool_results: List[Dict[str, Any]] = []

            for item in content:
                if isinstance(item, str):
                    content_blocks.append(TextContent(text=item))
                elif isinstance(item, dict):
                    item_type = item.get("type")

                    if item_type == "tool_use":
                        # Assistant's tool call → CanonicalToolCall
                        args = item.get("input", {})
                        tool_calls.append(
                            CanonicalToolCall(
                                id=item.get("id", ""),
                                type="function",
                                function=CanonicalFunctionCall(
                                    name=item.get("name", ""),
                                    arguments=json.dumps(args) if isinstance(args, dict) else str(args),
                                ),
                            )
                        )
                    elif item_type == "tool_result":
                        # User's tool result → will expand to separate TOOL messages
                        tool_results.append(item)
                    else:
                        block = AnthropicInTranslator._translate_content_block(item)
                        if block:
                            content_blocks.append(block)

            # Assistant message with tool_use blocks
            if role == MessageRole.ASSISTANT and tool_calls:
                text_content = None
                if content_blocks:
                    text_content = content_blocks
                elif not content_blocks:
                    # No text content, content can be None
                    text_content = None
                return CanonicalMessage(
                    role=role,
                    content=text_content,
                    tool_calls=tool_calls,
                )

            # User message with tool_result blocks → expand
            if tool_results:
                result_messages: list = []
                # If there's text content too, emit that first as a user message
                if content_blocks:
                    result_messages.append(
                        CanonicalMessage(role=MessageRole.USER, content=content_blocks)
                    )
                # Each tool_result becomes a separate TOOL role message
                for tr in tool_results:
                    tr_content = tr.get("content", "")
                    # tool_result content can be string or array of blocks
                    if isinstance(tr_content, list):
                        text_parts = []
                        for block in tr_content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text_parts.append(block.get("text", ""))
                            elif isinstance(block, str):
                                text_parts.append(block)
                        tr_content = " ".join(text_parts) if text_parts else ""
                    result_messages.append(
                        CanonicalMessage(
                            role=MessageRole.TOOL,
                            content=tr_content if tr_content else "",
                            tool_call_id=tr.get("tool_use_id", ""),
                        )
                    )
                return result_messages

            if content_blocks:
                return CanonicalMessage(role=role, content=content_blocks)
            return CanonicalMessage(role=role, content="")

        return CanonicalMessage(role=role, content=content or "")

    @staticmethod
    def _translate_content_block(item: Dict[str, Any]) -> Optional[ContentBlock]:
        """Translate an Anthropic content block to canonical format."""
        item_type = item.get("type")

        if item_type == "text":
            return TextContent(text=item.get("text", ""))

        elif item_type == "image":
            source = item.get("source", {})
            source_type = source.get("type")

            if source_type == "base64":
                return ImageBase64Content(
                    data=source.get("data", ""),
                    media_type=source.get("media_type", "image/png"),
                )
            elif source_type == "url":
                return ImageUrlContent(
                    image_url={
                        "url": source.get("url", ""),
                        "detail": "auto",
                    }
                )

        # tool_use and tool_result are handled in _translate_message directly
        return None

    @staticmethod
    def _map_finish_reason(reason: str) -> str:
        """Map OpenAI finish_reason to Anthropic stop_reason."""
        mapping = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
        }
        return mapping.get(reason, "end_turn")
