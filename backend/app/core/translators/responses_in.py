############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# responses_in.py: OpenAI Responses API format to canonical
# schema translator, plus Response-object formatting.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""OpenAI Responses API (/v1/responses) to canonical schema translator.

The Responses API differs from chat-completions in ways this module
absorbs:
- ``input`` is a string or a typed item array (messages, function_call,
  function_call_output, reasoning, ...); items may arrive with or
  without ``type`` (the SDK's EasyInputMessage form omits it) and with
  or without server ids (Codex strips them when store=false).
- Function tools are FLAT (``{"type":"function","name":...}``), not
  nested under a ``function`` key.
- ``instructions`` is a top-level system prompt; ``developer`` is a
  valid message role (mapped to system).
- Clients send fields we must tolerate but not forward (``store``,
  ``include``, ``prompt_cache_key``, Codex's ``client_metadata``, ...).

Non-function tools (``web_search``, ``custom``, ``mcp``) are stripped,
never rejected: the Codex agent sends a ``web_search`` tool by default
and would break out-of-the-box otherwise.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

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

# reasoning.effort values outside the vLLM/Ollama-supported set are
# clamped to the nearest supported effort.
_EFFORT_CLAMP = {"minimal": "low", "xhigh": "high"}


def _gen_id(prefix: str) -> str:
    """Generate a Responses-style id, e.g. resp_<32hex>, msg_..., fc_..., rs_...."""
    return f"{prefix}_{uuid.uuid4().hex}"


@dataclass
class ResponsesRequestContext:
    """Request parameters echoed back in Response snapshots.

    Snapshot echo uses the API's documented defaults for omitted fields
    (store=true, parallel_tool_calls=true, ...) because strict SDK
    clients type them as non-null.
    """

    model: str = ""
    instructions: Optional[str] = None
    tools: List[Dict[str, Any]] = field(default_factory=list)
    tool_choice: Any = "auto"
    temperature: float = 1.0
    top_p: float = 1.0
    max_output_tokens: Optional[int] = None
    parallel_tool_calls: bool = True
    reasoning: Optional[Dict[str, Any]] = None
    text: Optional[Dict[str, Any]] = None
    store: bool = True
    previous_response_id: Optional[str] = None
    truncation: str = "disabled"
    metadata: Dict[str, Any] = field(default_factory=dict)
    prompt_cache_key: Optional[str] = None
    background: bool = False
    stream: bool = False
    response_id: str = ""
    created_at: int = 0
    # Stamped by the route (used by the store service)
    user_id: Optional[int] = None
    api_key_id: Optional[int] = None

    @classmethod
    def from_body(cls, body: Dict[str, Any]) -> "ResponsesRequestContext":
        def _or(value, default):
            return default if value is None else value

        return cls(
            model=str(body.get("model", "")),
            instructions=body.get("instructions"),
            tools=body.get("tools") or [],
            tool_choice=_or(body.get("tool_choice"), "auto"),
            temperature=_or(body.get("temperature"), 1.0),
            top_p=_or(body.get("top_p"), 1.0),
            max_output_tokens=body.get("max_output_tokens"),
            parallel_tool_calls=_or(body.get("parallel_tool_calls"), True),
            reasoning=body.get("reasoning"),
            text=body.get("text"),
            store=_or(body.get("store"), True),
            previous_response_id=body.get("previous_response_id"),
            truncation=_or(body.get("truncation"), "disabled"),
            metadata=body.get("metadata") or {},
            prompt_cache_key=body.get("prompt_cache_key"),
            background=_or(body.get("background"), False),
            stream=_or(body.get("stream"), False),
            response_id=_gen_id("resp"),
            created_at=int(time.time()),
        )

    def stripped_tool_types(self) -> List[str]:
        """Tool types that will be stripped (everything non-function)."""
        return [
            str(t.get("type"))
            for t in self.tools
            if isinstance(t, dict) and t.get("type") != "function"
        ]

    def to_stored_parameters(self) -> Dict[str, Any]:
        """Echo-parameter dict persisted with a stored response, from
        which the Response object can be rebuilt (GET /v1/responses/{id})."""
        return {
            "model": self.model,
            "tools": self.tools,
            "tool_choice": self.tool_choice,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_output_tokens": self.max_output_tokens,
            "parallel_tool_calls": self.parallel_tool_calls,
            "reasoning": self.reasoning,
            "text": self.text,
            "truncation": self.truncation,
            "metadata": self.metadata,
            "prompt_cache_key": self.prompt_cache_key,
            "created_at": self.created_at,
        }

    @classmethod
    def from_stored(cls, stored: Any) -> "ResponsesRequestContext":
        """Rebuild a context from a StoredResponse row for snapshot
        reconstruction."""
        params = stored.parameters or {}
        ctx = cls(
            model=params.get("model") or stored.model,
            instructions=stored.instructions,
            tools=params.get("tools") or [],
            tool_choice=params.get("tool_choice", "auto"),
            temperature=params.get("temperature", 1.0),
            top_p=params.get("top_p", 1.0),
            max_output_tokens=params.get("max_output_tokens"),
            parallel_tool_calls=params.get("parallel_tool_calls", True),
            reasoning=params.get("reasoning"),
            text=params.get("text"),
            store=True,
            previous_response_id=stored.previous_response_id,
            truncation=params.get("truncation", "disabled"),
            metadata=params.get("metadata") or {},
            prompt_cache_key=params.get("prompt_cache_key"),
            response_id=stored.response_id,
            created_at=params.get("created_at") or int(stored.created_at.timestamp()),
            user_id=stored.user_id,
            api_key_id=stored.api_key_id,
        )
        return ctx


class ResponsesInTranslator:
    """Translate OpenAI Responses API requests to canonical format."""

    @staticmethod
    def translate_responses_request(data: Dict[str, Any]) -> CanonicalChatRequest:
        """Translate a POST /v1/responses body to a CanonicalChatRequest.

        Raises ValueError for request shapes we cannot serve (file
        inputs, item_reference without a store); unknown fields and
        unknown item types are tolerated.
        """
        messages: List[CanonicalMessage] = []

        instructions = data.get("instructions")
        if instructions:
            messages.append(
                CanonicalMessage(role=MessageRole.SYSTEM, content=instructions)
            )

        messages.extend(
            ResponsesInTranslator._translate_input(data.get("input"))
        )

        tools = ResponsesInTranslator._translate_tools(data.get("tools"))
        tool_choice = ResponsesInTranslator._translate_tool_choice(
            data.get("tool_choice")
        )
        response_format = ResponsesInTranslator._translate_text_format(
            data.get("text")
        )
        think = ResponsesInTranslator._resolve_reasoning(data.get("reasoning"))

        user = data.get("user") or data.get("safety_identifier")

        return CanonicalChatRequest(
            model=data["model"],
            messages=messages,
            temperature=data.get("temperature"),
            top_p=data.get("top_p"),
            max_tokens=data.get("max_output_tokens"),
            stream=data.get("stream") or False,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            think=think,
            user=user,
        )

    # ------------------------------------------------------------------
    # Input items
    # ------------------------------------------------------------------

    @staticmethod
    def _translate_input(input_value: Any) -> List[CanonicalMessage]:
        """Translate the polymorphic ``input`` field into canonical messages."""
        if input_value is None:
            return []

        if isinstance(input_value, str):
            return [CanonicalMessage(role=MessageRole.USER, content=input_value)]

        if not isinstance(input_value, list):
            raise ValueError("'input' must be a string or an array of items")

        messages: List[CanonicalMessage] = []
        for item in input_value:
            if not isinstance(item, dict):
                raise ValueError("input items must be objects")
            for msg in ResponsesInTranslator._translate_item(item):
                # Merge a function_call into the preceding assistant
                # message (content + tool_calls in one message is the
                # chat-completions shape for a single model turn).
                if (
                    msg.role == MessageRole.ASSISTANT
                    and msg.tool_calls
                    and msg.content is None
                    and messages
                    and messages[-1].role == MessageRole.ASSISTANT
                    and messages[-1].tool_call_id is None
                ):
                    prev = messages[-1]
                    prev.tool_calls = (prev.tool_calls or []) + msg.tool_calls
                    continue
                messages.append(msg)
        return messages

    @staticmethod
    def _translate_item(item: Dict[str, Any]) -> List[CanonicalMessage]:
        """Translate one input item; may yield zero, one, or two messages.

        Server-stamped keys (``id``, ``status``) are ignored on every
        item type — Codex-on-Azure and SDK replays resend them.
        """
        item_type = item.get("type")

        # EasyInputMessage: ``type`` is optional when role/content present.
        if item_type is None and "role" in item:
            item_type = "message"

        if item_type == "message":
            return ResponsesInTranslator._translate_message_item(item)

        if item_type == "function_call":
            call_id = item.get("call_id") or item.get("id") or _gen_id("call")
            return [
                CanonicalMessage(
                    role=MessageRole.ASSISTANT,
                    content=None,
                    tool_calls=[
                        CanonicalToolCall(
                            id=call_id,
                            type="function",
                            function=CanonicalFunctionCall(
                                name=item.get("name") or "",
                                arguments=item.get("arguments") or "{}",
                            ),
                        )
                    ],
                )
            ]

        if item_type == "function_call_output":
            return ResponsesInTranslator._translate_function_call_output(item)

        if item_type == "reasoning":
            # Opaque provider state (encrypted_content) / raw CoT: chat
            # backends cannot replay it. Dropped by design.
            return []

        if item_type == "item_reference":
            raise ValueError(
                "item_reference input items require server-side storage, "
                "which this endpoint does not provide"
            )

        # Unknown item types are skipped for forward compatibility.
        return []

    @staticmethod
    def _translate_message_item(item: Dict[str, Any]) -> List[CanonicalMessage]:
        role_str = item.get("role", "user")
        if role_str == "developer":
            role_str = "system"
        role = MessageRole(role_str)

        content = item.get("content")
        if content is None or isinstance(content, str):
            return [CanonicalMessage(role=role, content=content)]

        if not isinstance(content, list):
            return [CanonicalMessage(role=role, content=str(content))]

        # Assistant history (OutputMessage form) carries output_text /
        # refusal parts; flatten to plain text.
        if role == MessageRole.ASSISTANT:
            texts = [
                part.get("text") or part.get("refusal") or ""
                for part in content
                if isinstance(part, dict)
            ]
            return [CanonicalMessage(role=role, content="".join(texts))]

        blocks: List[ContentBlock] = []
        for part in content:
            if isinstance(part, str):
                blocks.append(TextContent(text=part))
            elif isinstance(part, dict):
                block = ResponsesInTranslator._translate_content_part(part)
                if block is not None:
                    blocks.append(block)
        return [CanonicalMessage(role=role, content=blocks)]

    @staticmethod
    def _translate_content_part(part: Dict[str, Any]) -> Optional[ContentBlock]:
        part_type = part.get("type")

        if part_type in ("input_text", "output_text", "text"):
            return TextContent(text=part.get("text", ""))

        if part_type == "input_image":
            if part.get("file_id"):
                raise ValueError("input_image with file_id is not supported")
            url = part.get("image_url") or ""
            # The Responses form is a bare string; tolerate the nested
            # chat-completions form too.
            if isinstance(url, dict):
                url = url.get("url", "")
            if url.startswith("file:"):
                # Never accept client-supplied file references.
                raise ValueError("file: URLs are not supported in input_image")
            if url.startswith("data:"):
                try:
                    header, b64data = url.split(",", 1)
                    media_type = header.split(":")[1].split(";")[0]
                    return ImageBase64Content(data=b64data, media_type=media_type)
                except (ValueError, IndexError):
                    pass
            return ImageUrlContent(
                image_url={"url": url, "detail": part.get("detail") or "auto"}
            )

        if part_type == "input_file":
            raise ValueError("input_file content is not supported")

        return None

    @staticmethod
    def _translate_function_call_output(
        item: Dict[str, Any]
    ) -> List[CanonicalMessage]:
        call_id = item.get("call_id") or ""
        output = item.get("output")

        if output is None or isinstance(output, str):
            return [
                CanonicalMessage(
                    role=MessageRole.TOOL,
                    content=output or "",
                    tool_call_id=call_id,
                )
            ]

        # Array-of-parts output: text parts become the tool result;
        # image parts are re-attached as a follow-up user message so
        # vision models can still see them (canonical tool messages
        # only carry text).
        texts: List[str] = []
        image_blocks: List[ContentBlock] = []
        for part in output:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type in ("input_text", "output_text", "text"):
                texts.append(part.get("text", ""))
            elif part_type == "input_image":
                block = ResponsesInTranslator._translate_content_part(part)
                if block is not None:
                    image_blocks.append(block)

        messages = [
            CanonicalMessage(
                role=MessageRole.TOOL,
                content="".join(texts),
                tool_call_id=call_id,
            )
        ]
        if image_blocks:
            messages.append(
                CanonicalMessage(role=MessageRole.USER, content=image_blocks)
            )
        return messages

    # ------------------------------------------------------------------
    # Tools / formats / reasoning
    # ------------------------------------------------------------------

    @staticmethod
    def _translate_tools(
        tools: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[CanonicalToolDefinition]]:
        """Re-nest flat Responses function tools; strip everything else."""
        if not tools:
            return None
        translated = []
        for tool in tools:
            if not isinstance(tool, dict) or tool.get("type") != "function":
                continue  # web_search / custom / mcp / ... — stripped
            # description is always a string: vLLM's gpt-oss tool parser
            # rejects tools whose description is missing/None.
            function: Dict[str, Any] = {
                "name": tool.get("name") or "",
                "description": tool.get("description") or "",
                "parameters": tool.get("parameters") or {},
            }
            translated.append(
                CanonicalToolDefinition(type="function", function=function)
            )
        return translated or None

    @staticmethod
    def _translate_tool_choice(tool_choice: Any) -> Optional[Any]:
        if tool_choice is None:
            return None
        if isinstance(tool_choice, str):
            return tool_choice
        if isinstance(tool_choice, dict):
            if tool_choice.get("type") == "function" and tool_choice.get("name"):
                return {
                    "type": "function",
                    "function": {"name": tool_choice["name"]},
                }
            # allowed_tools / hosted-tool choices have no chat equivalent.
            return "auto"
        return None

    @staticmethod
    def _translate_text_format(text: Optional[Dict[str, Any]]) -> Optional[ResponseFormat]:
        if not isinstance(text, dict):
            return None
        fmt = text.get("format")
        if not isinstance(fmt, dict):
            return None
        fmt_type = fmt.get("type", "text")

        if fmt_type == "json_object":
            return ResponseFormat(type=ResponseFormatType.JSON_OBJECT)

        if fmt_type == "json_schema":
            json_schema: Dict[str, Any] = {
                "name": fmt.get("name", "response"),
                "schema": fmt.get("schema"),
            }
            if fmt.get("strict") is not None:
                json_schema["strict"] = fmt["strict"]
            return ResponseFormat(
                type=ResponseFormatType.JSON_SCHEMA, json_schema=json_schema
            )

        return ResponseFormat(type=ResponseFormatType.TEXT)

    @staticmethod
    def _resolve_reasoning(
        reasoning: Optional[Dict[str, Any]]
    ) -> Optional[Union[bool, str]]:
        """Map reasoning.effort onto canonical ``think``.

        String think is the proven cross-engine path: vllm_out converts
        it to reasoning_effort for gpt-oss, ollama_out forwards the
        string; bool False disables thinking on qwen-style models.
        """
        if not isinstance(reasoning, dict):
            return None
        effort = reasoning.get("effort")
        if effort is None:
            return None
        if effort == "none":
            return False
        return _EFFORT_CLAMP.get(effort, effort)

    # ------------------------------------------------------------------
    # Response formatting (non-streaming) and snapshots
    # ------------------------------------------------------------------

    @staticmethod
    def format_response(
        chat_response: Dict[str, Any], ctx: ResponsesRequestContext
    ) -> Dict[str, Any]:
        """Reshape an OpenAI chat.completion dict into a Response object."""
        choices = chat_response.get("choices") or [{}]
        message = choices[0].get("message") or {}
        finish_reason = choices[0].get("finish_reason")

        output = ResponsesInTranslator.build_output_items(message)
        status, incomplete_details = ResponsesInTranslator.map_finish_reason(
            finish_reason
        )
        usage = ResponsesInTranslator.map_usage(chat_response.get("usage"))

        return ResponsesInTranslator.build_snapshot(
            ctx,
            status=status,
            output=output,
            usage=usage,
            incomplete_details=incomplete_details,
            completed_at=int(time.time()),
        )

    @staticmethod
    def build_output_items(message: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build the Response ``output`` array from a chat message dict."""
        output: List[Dict[str, Any]] = []

        reasoning_text = message.get("reasoning_content")
        if reasoning_text:
            output.append(
                {
                    "id": _gen_id("rs"),
                    "type": "reasoning",
                    "summary": [],
                    "content": [
                        {"type": "reasoning_text", "text": reasoning_text}
                    ],
                    "status": "completed",
                }
            )

        content = message.get("content")
        if content:
            output.append(
                {
                    "id": _gen_id("msg"),
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": content,
                            "annotations": [],
                            "logprobs": [],
                        }
                    ],
                }
            )

        for tc in message.get("tool_calls") or []:
            function = tc.get("function") or {}
            output.append(
                {
                    "id": _gen_id("fc"),
                    "type": "function_call",
                    "status": "completed",
                    "call_id": tc.get("id") or _gen_id("call"),
                    "name": function.get("name") or "",
                    "arguments": function.get("arguments") or "{}",
                }
            )

        return output

    @staticmethod
    def map_usage(usage: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Map chat-completions usage keys onto the Responses usage shape."""
        usage = usage or {}
        input_tokens = usage.get("prompt_tokens", 0) or 0
        output_tokens = usage.get("completion_tokens", 0) or 0
        return {
            "input_tokens": input_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": output_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": usage.get("total_tokens") or (input_tokens + output_tokens),
        }

    @staticmethod
    def map_finish_reason(
        finish_reason: Optional[str],
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Map a chat finish_reason to (status, incomplete_details)."""
        if finish_reason == "length":
            return "incomplete", {"reason": "max_output_tokens"}
        return "completed", None

    @staticmethod
    def build_snapshot(
        ctx: ResponsesRequestContext,
        status: str,
        output: List[Dict[str, Any]],
        usage: Optional[Dict[str, Any]],
        error: Optional[Dict[str, Any]] = None,
        incomplete_details: Optional[Dict[str, Any]] = None,
        completed_at: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Build a full Response object snapshot (used by both the
        non-streaming response body and every streaming lifecycle event)."""
        return {
            "id": ctx.response_id,
            "object": "response",
            "created_at": ctx.created_at,
            "status": status,
            "background": False,
            "completed_at": completed_at,
            "error": error,
            "incomplete_details": incomplete_details,
            "instructions": ctx.instructions,
            "max_output_tokens": ctx.max_output_tokens,
            "max_tool_calls": None,
            "model": ctx.model,
            "output": output,
            "parallel_tool_calls": ctx.parallel_tool_calls,
            "previous_response_id": ctx.previous_response_id,
            "prompt_cache_key": ctx.prompt_cache_key,
            "reasoning": ctx.reasoning or {"effort": None, "summary": None},
            "safety_identifier": None,
            "service_tier": "default",
            "store": ctx.store,
            "temperature": ctx.temperature,
            "text": ctx.text or {"format": {"type": "text"}},
            "tool_choice": ctx.tool_choice,
            "tools": ctx.tools,
            "top_logprobs": 0,
            "top_p": ctx.top_p,
            "truncation": ctx.truncation,
            "usage": usage,
            "user": None,
            "metadata": ctx.metadata,
        }
