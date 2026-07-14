############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_responses_in.py: Unit tests for the OpenAI Responses API
# inbound translator and Response-object formatting.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Covers:
- Request basics: string input, parameter mapping, unknown-field tolerance
- Input item polymorphism: typed items, typeless EasyInputMessage items,
  items carrying server ids/status, developer role, assistant history
- Content parts: input_text, input_image (data:/https/file:), input_file
- Function-call round trip via call_id; consecutive-call merging
- Tool translation: flat->nested, non-function tools stripped
- text.format -> ResponseFormat; reasoning.effort -> canonical think
- format_response / build_snapshot: output items, usage, status mapping
- Round trip through VLLMOutTranslator
"""

import pytest

from backend.app.core.canonical_schemas import (
    CanonicalChatRequest,
    ImageBase64Content,
    ImageUrlContent,
    MessageRole,
    ResponseFormatType,
    TextContent,
)
from backend.app.core.translators.responses_in import (
    ResponsesInTranslator,
    ResponsesRequestContext,
)
from backend.app.core.translators.vllm_out import VLLMOutTranslator

SAMPLE_TOOLS_RESPONSES = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
        "strict": False,
    },
    {"type": "web_search", "external_web_access": False},
]


class TestResponsesInBasics:
    def test_string_input(self):
        result = ResponsesInTranslator.translate_responses_request(
            {"model": "m", "input": "hello"}
        )
        assert isinstance(result, CanonicalChatRequest)
        assert len(result.messages) == 1
        assert result.messages[0].role == MessageRole.USER
        assert result.messages[0].content == "hello"
        assert result.stream is False

    def test_instructions_prepended_as_system(self):
        result = ResponsesInTranslator.translate_responses_request(
            {"model": "m", "instructions": "You are terse.", "input": "hi"}
        )
        assert result.messages[0].role == MessageRole.SYSTEM
        assert result.messages[0].content == "You are terse."
        assert result.messages[1].role == MessageRole.USER

    def test_parameter_mapping(self):
        result = ResponsesInTranslator.translate_responses_request(
            {
                "model": "m",
                "input": "hi",
                "max_output_tokens": 128,
                "temperature": 0.5,
                "top_p": 0.9,
                "stream": True,
                "user": "u1",
            }
        )
        assert result.max_tokens == 128
        assert result.temperature == 0.5
        assert result.top_p == 0.9
        assert result.stream is True
        assert result.user == "u1"

    def test_safety_identifier_maps_to_user(self):
        result = ResponsesInTranslator.translate_responses_request(
            {"model": "m", "input": "hi", "safety_identifier": "sid"}
        )
        assert result.user == "sid"

    def test_unknown_fields_tolerated(self):
        # Codex sends client_metadata, prompt_cache_key, include, store...
        result = ResponsesInTranslator.translate_responses_request(
            {
                "model": "m",
                "input": "hi",
                "store": False,
                "include": ["reasoning.encrypted_content"],
                "prompt_cache_key": "thread-123",
                "client_metadata": {"session_id": "s"},
                "service_tier": None,
                "parallel_tool_calls": False,
                "totally_unknown_field": {"x": 1},
            }
        )
        assert len(result.messages) == 1

    def test_missing_model_raises(self):
        with pytest.raises(KeyError):
            ResponsesInTranslator.translate_responses_request({"input": "hi"})

    def test_truncation_auto_sets_flag(self):
        result = ResponsesInTranslator.translate_responses_request(
            {"model": "m", "input": "hi", "truncation": "auto"}
        )
        assert result.auto_truncate is True
        result = ResponsesInTranslator.translate_responses_request(
            {"model": "m", "input": "hi", "truncation": "disabled"}
        )
        assert result.auto_truncate is False
        result = ResponsesInTranslator.translate_responses_request(
            {"model": "m", "input": "hi"}
        )
        assert result.auto_truncate is False


class TestResponsesInputItems:
    def test_typed_message_item(self):
        result = ResponsesInTranslator.translate_responses_request(
            {
                "model": "m",
                "input": [
                    {"type": "message", "role": "user", "content": "hello"}
                ],
            }
        )
        assert result.messages[0].role == MessageRole.USER
        assert result.messages[0].content == "hello"

    def test_typeless_easy_input_message(self):
        # The SDK's most common form omits "type" entirely.
        result = ResponsesInTranslator.translate_responses_request(
            {"model": "m", "input": [{"role": "user", "content": "hi"}]}
        )
        assert len(result.messages) == 1
        assert result.messages[0].content == "hi"

    def test_typeless_message_with_parts(self):
        result = ResponsesInTranslator.translate_responses_request(
            {
                "model": "m",
                "input": [
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": "hi"}],
                    }
                ],
            }
        )
        blocks = result.messages[0].content
        assert isinstance(blocks[0], TextContent)
        assert blocks[0].text == "hi"

    def test_developer_role_maps_to_system(self):
        result = ResponsesInTranslator.translate_responses_request(
            {
                "model": "m",
                "input": [{"role": "developer", "content": "be brief"}],
            }
        )
        assert result.messages[0].role == MessageRole.SYSTEM

    def test_system_and_developer_parts_flatten_to_string(self):
        # Codex sends developer/user_instructions messages as input_text
        # part arrays; system content must flatten to a plain string or
        # the outbound system-merge join breaks (prod regression).
        result = ResponsesInTranslator.translate_responses_request(
            {
                "model": "m",
                "instructions": "You are Codex.",
                "input": [
                    {
                        "type": "message",
                        "role": "developer",
                        "content": [
                            {"type": "input_text", "text": "<user_instructions>"},
                            {"type": "input_text", "text": "be terse"},
                        ],
                    },
                    {"role": "user", "content": "hi"},
                ],
            }
        )
        sys_msgs = [m for m in result.messages if m.role == MessageRole.SYSTEM]
        assert len(sys_msgs) == 2
        assert all(isinstance(m.content, str) for m in sys_msgs)
        assert sys_msgs[1].content == "<user_instructions>be terse"
        # And the multi-system merge in vllm_out must not raise
        payload = VLLMOutTranslator.translate_chat_request(result)
        assert payload["messages"][0]["role"] == "system"
        assert "be terse" in payload["messages"][0]["content"]

    def test_items_with_server_ids_and_status_ignored(self):
        # Codex-on-Azure / SDK replays resend msg_/fc_/rs_ ids + status.
        result = ResponsesInTranslator.translate_responses_request(
            {
                "model": "m",
                "input": [
                    {
                        "type": "message",
                        "id": "msg_abc",
                        "status": "completed",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "prev answer"}
                        ],
                    },
                    {
                        "type": "function_call",
                        "id": "fc_abc",
                        "status": "completed",
                        "call_id": "call_1",
                        "name": "f",
                        "arguments": "{}",
                    },
                ],
            }
        )
        assert result.messages[0].role == MessageRole.ASSISTANT
        assert result.messages[0].content == "prev answer"
        assert result.messages[0].tool_calls[0].id == "call_1"

    def test_assistant_output_message_flattened(self):
        result = ResponsesInTranslator.translate_responses_request(
            {
                "model": "m",
                "input": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "part one. "},
                            {"type": "refusal", "refusal": "no."},
                        ],
                    }
                ],
            }
        )
        assert result.messages[0].content == "part one. no."

    def test_function_call_round_trip_via_call_id(self):
        result = ResponsesInTranslator.translate_responses_request(
            {
                "model": "m",
                "input": [
                    {"role": "user", "content": "weather?"},
                    {
                        "type": "function_call",
                        "call_id": "call_w1",
                        "name": "get_weather",
                        "arguments": '{"location": "Moscow, ID"}',
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_w1",
                        "output": "sunny, 25C",
                    },
                ],
            }
        )
        assistant = result.messages[1]
        assert assistant.role == MessageRole.ASSISTANT
        assert assistant.tool_calls[0].id == "call_w1"
        assert assistant.tool_calls[0].function.name == "get_weather"
        tool = result.messages[2]
        assert tool.role == MessageRole.TOOL
        assert tool.tool_call_id == "call_w1"
        assert tool.content == "sunny, 25C"

    def test_consecutive_function_calls_merge(self):
        result = ResponsesInTranslator.translate_responses_request(
            {
                "model": "m",
                "input": [
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "a",
                        "arguments": "{}",
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_2",
                        "name": "b",
                        "arguments": "{}",
                    },
                ],
            }
        )
        assert len(result.messages) == 1
        assert len(result.messages[0].tool_calls) == 2
        assert result.messages[0].tool_calls[1].id == "call_2"

    def test_function_call_merges_into_assistant_message(self):
        # A model turn = message item then function_call item(s); the
        # chat-completions shape is one assistant message with both.
        result = ResponsesInTranslator.translate_responses_request(
            {
                "model": "m",
                "input": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "checking"}],
                    },
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "f",
                        "arguments": "{}",
                    },
                ],
            }
        )
        assert len(result.messages) == 1
        assert result.messages[0].content == "checking"
        assert result.messages[0].tool_calls[0].id == "call_1"

    def test_function_call_output_with_image_part(self):
        result = ResponsesInTranslator.translate_responses_request(
            {
                "model": "m",
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call_v",
                        "output": [
                            {"type": "input_text", "text": "screenshot:"},
                            {
                                "type": "input_image",
                                "image_url": "data:image/png;base64,QUJD",
                            },
                        ],
                    }
                ],
            }
        )
        assert result.messages[0].role == MessageRole.TOOL
        assert result.messages[0].content == "screenshot:"
        follow_up = result.messages[1]
        assert follow_up.role == MessageRole.USER
        assert isinstance(follow_up.content[0], ImageBase64Content)
        assert follow_up.content[0].data == "QUJD"

    def test_reasoning_items_dropped(self):
        result = ResponsesInTranslator.translate_responses_request(
            {
                "model": "m",
                "input": [
                    {
                        "type": "reasoning",
                        "summary": [],
                        "encrypted_content": None,
                    },
                    {"role": "user", "content": "hi"},
                ],
            }
        )
        assert len(result.messages) == 1
        assert result.messages[0].role == MessageRole.USER

    def test_unknown_item_type_skipped(self):
        result = ResponsesInTranslator.translate_responses_request(
            {
                "model": "m",
                "input": [
                    {"type": "web_search_call", "status": "completed"},
                    {"role": "user", "content": "hi"},
                ],
            }
        )
        assert len(result.messages) == 1

    def test_item_reference_raises(self):
        with pytest.raises(ValueError):
            ResponsesInTranslator.translate_responses_request(
                {"model": "m", "input": [{"type": "item_reference", "id": "msg_1"}]}
            )


class TestResponsesContentParts:
    def test_input_image_data_uri(self):
        result = ResponsesInTranslator.translate_responses_request(
            {
                "model": "m",
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": "data:image/jpeg;base64,QUJD",
                            }
                        ],
                    }
                ],
            }
        )
        block = result.messages[0].content[0]
        assert isinstance(block, ImageBase64Content)
        assert block.media_type == "image/jpeg"

    def test_input_image_http_url(self):
        result = ResponsesInTranslator.translate_responses_request(
            {
                "model": "m",
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": "https://example.com/x.png",
                                "detail": "high",
                            }
                        ],
                    }
                ],
            }
        )
        block = result.messages[0].content[0]
        assert isinstance(block, ImageUrlContent)
        assert block.image_url["url"] == "https://example.com/x.png"
        assert block.image_url["detail"] == "high"

    def test_input_image_file_url_rejected(self):
        with pytest.raises(ValueError):
            ResponsesInTranslator.translate_responses_request(
                {
                    "model": "m",
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_image",
                                    "image_url": "file:///etc/passwd",
                                }
                            ],
                        }
                    ],
                }
            )

    def test_input_image_file_id_rejected(self):
        with pytest.raises(ValueError):
            ResponsesInTranslator.translate_responses_request(
                {
                    "model": "m",
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_image", "file_id": "file-abc"}
                            ],
                        }
                    ],
                }
            )

    def test_input_file_rejected(self):
        with pytest.raises(ValueError):
            ResponsesInTranslator.translate_responses_request(
                {
                    "model": "m",
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_file", "filename": "a.pdf"}
                            ],
                        }
                    ],
                }
            )


class TestResponsesTools:
    def test_flat_function_tool_renested(self):
        result = ResponsesInTranslator.translate_responses_request(
            {"model": "m", "input": "hi", "tools": SAMPLE_TOOLS_RESPONSES}
        )
        assert len(result.tools) == 1  # web_search stripped
        tool = result.tools[0]
        assert tool.type == "function"
        assert tool.function["name"] == "get_weather"
        assert tool.function["parameters"]["required"] == ["location"]
        assert "description" in tool.function

    def test_all_tools_stripped_yields_none(self):
        result = ResponsesInTranslator.translate_responses_request(
            {
                "model": "m",
                "input": "hi",
                "tools": [{"type": "web_search"}, {"type": "custom", "name": "apply_patch"}],
            }
        )
        assert result.tools is None

    def test_tool_choice_strings_pass_through(self):
        for choice in ("auto", "none", "required"):
            result = ResponsesInTranslator.translate_responses_request(
                {"model": "m", "input": "hi", "tool_choice": choice}
            )
            assert result.tool_choice == choice

    def test_tool_choice_function_renested(self):
        result = ResponsesInTranslator.translate_responses_request(
            {
                "model": "m",
                "input": "hi",
                "tool_choice": {"type": "function", "name": "get_weather"},
            }
        )
        assert result.tool_choice == {
            "type": "function",
            "function": {"name": "get_weather"},
        }

    def test_tool_choice_allowed_tools_downgrades_to_auto(self):
        result = ResponsesInTranslator.translate_responses_request(
            {
                "model": "m",
                "input": "hi",
                "tool_choice": {"type": "allowed_tools", "mode": "auto", "tools": []},
            }
        )
        assert result.tool_choice == "auto"


class TestResponsesTextFormat:
    def test_text_format(self):
        result = ResponsesInTranslator.translate_responses_request(
            {"model": "m", "input": "hi", "text": {"format": {"type": "text"}}}
        )
        assert result.response_format.type == ResponseFormatType.TEXT

    def test_json_object_format(self):
        result = ResponsesInTranslator.translate_responses_request(
            {"model": "m", "input": "hi", "text": {"format": {"type": "json_object"}}}
        )
        assert result.response_format.type == ResponseFormatType.JSON_OBJECT

    def test_json_schema_format(self):
        result = ResponsesInTranslator.translate_responses_request(
            {
                "model": "m",
                "input": "hi",
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "answer",
                        "schema": {"type": "object"},
                        "strict": True,
                    }
                },
            }
        )
        rf = result.response_format
        assert rf.type == ResponseFormatType.JSON_SCHEMA
        assert rf.json_schema["name"] == "answer"
        assert rf.json_schema["schema"] == {"type": "object"}
        assert rf.json_schema["strict"] is True

    def test_absent_text_yields_no_format(self):
        result = ResponsesInTranslator.translate_responses_request(
            {"model": "m", "input": "hi"}
        )
        assert result.response_format is None


class TestResponsesReasoning:
    def test_effort_none_disables_thinking(self):
        result = ResponsesInTranslator.translate_responses_request(
            {"model": "m", "input": "hi", "reasoning": {"effort": "none"}}
        )
        assert result.think is False

    def test_effort_string_becomes_think(self):
        result = ResponsesInTranslator.translate_responses_request(
            {"model": "m", "input": "hi", "reasoning": {"effort": "high"}}
        )
        assert result.think == "high"
        assert result.reasoning_effort is None

    def test_effort_clamping(self):
        for effort, expected in (("minimal", "low"), ("xhigh", "high")):
            result = ResponsesInTranslator.translate_responses_request(
                {"model": "m", "input": "hi", "reasoning": {"effort": effort}}
            )
            assert result.think == expected

    def test_absent_reasoning_leaves_backend_default(self):
        result = ResponsesInTranslator.translate_responses_request(
            {"model": "m", "input": "hi", "reasoning": None}
        )
        assert result.think is None


class TestResponsesFormatResponse:
    def _ctx(self, **overrides):
        body = {"model": "test-model", "input": "hi"}
        body.update(overrides)
        return ResponsesRequestContext.from_body(body)

    def test_message_output(self):
        ctx = self._ctx()
        result = ResponsesInTranslator.format_response(
            {
                "id": "internal-uuid",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hello!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
            ctx,
        )
        assert result["object"] == "response"
        assert result["id"].startswith("resp_")
        assert result["status"] == "completed"
        assert result["model"] == "test-model"
        msg = result["output"][0]
        assert msg["type"] == "message"
        assert msg["id"].startswith("msg_")
        part = msg["content"][0]
        assert part["type"] == "output_text"
        assert part["text"] == "hello!"
        assert part["annotations"] == []
        assert part["logprobs"] == []
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 5
        assert result["usage"]["total_tokens"] == 15
        assert result["usage"]["input_tokens_details"] == {"cached_tokens": 0}

    def test_reasoning_and_tool_call_output(self):
        ctx = self._ctx()
        result = ResponsesInTranslator.format_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "reasoning_content": "thinking...",
                            "tool_calls": [
                                {
                                    "id": "call_9",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"location": "x"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            },
            ctx,
        )
        types = [item["type"] for item in result["output"]]
        assert types == ["reasoning", "function_call"]
        rs = result["output"][0]
        assert rs["id"].startswith("rs_")
        assert rs["summary"] == []
        assert rs["content"][0] == {"type": "reasoning_text", "text": "thinking..."}
        fc = result["output"][1]
        assert fc["id"].startswith("fc_")
        assert fc["call_id"] == "call_9"
        assert fc["arguments"] == '{"location": "x"}'
        assert result["status"] == "completed"

    def test_length_finish_reason_incomplete(self):
        ctx = self._ctx()
        result = ResponsesInTranslator.format_response(
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "partial"},
                        "finish_reason": "length",
                    }
                ],
                "usage": {},
            },
            ctx,
        )
        assert result["status"] == "incomplete"
        assert result["incomplete_details"] == {"reason": "max_output_tokens"}

    def test_snapshot_spec_defaults(self):
        ctx = self._ctx()
        snap = ResponsesInTranslator.build_snapshot(
            ctx, status="in_progress", output=[], usage=None
        )
        assert snap["parallel_tool_calls"] is True
        assert snap["store"] is True
        assert snap["truncation"] == "disabled"
        assert snap["temperature"] == 1.0
        assert snap["top_p"] == 1.0
        assert snap["text"] == {"format": {"type": "text"}}
        assert snap["tool_choice"] == "auto"
        assert snap["tools"] == []
        assert snap["reasoning"] == {"effort": None, "summary": None}
        assert snap["metadata"] == {}
        # Spec-required snapshot fields
        for field in (
            "id", "object", "created_at", "error", "incomplete_details",
            "instructions", "model", "output",
        ):
            assert field in snap

    def test_snapshot_echoes_request_values(self):
        ctx = self._ctx(
            store=False,
            temperature=0.2,
            truncation="auto",
            parallel_tool_calls=False,
            metadata={"k": "v"},
        )
        snap = ResponsesInTranslator.build_snapshot(
            ctx, status="completed", output=[], usage=None
        )
        assert snap["store"] is False
        assert snap["temperature"] == 0.2
        assert snap["truncation"] == "auto"
        assert snap["parallel_tool_calls"] is False
        assert snap["metadata"] == {"k": "v"}

    def test_stripped_tool_types(self):
        ctx = self._ctx(tools=SAMPLE_TOOLS_RESPONSES)
        assert ctx.stripped_tool_types() == ["web_search"]


class TestResponsesRoundTrip:
    def test_through_vllm_out(self):
        canonical = ResponsesInTranslator.translate_responses_request(
            {
                "model": "qwen/qwen3.5-122b",
                "instructions": "You are a coding agent.",
                "input": [
                    {"role": "user", "content": "list files"},
                    {
                        "type": "function_call",
                        "call_id": "call_ls",
                        "name": "shell_command",
                        "arguments": '{"command": "ls"}',
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_ls",
                        "output": "a.txt",
                    },
                ],
                "tools": SAMPLE_TOOLS_RESPONSES,
                "stream": True,
            }
        )
        payload = VLLMOutTranslator.translate_chat_request(canonical)
        assert payload["model"] == "qwen/qwen3.5-122b"
        roles = [m["role"] for m in payload["messages"]]
        assert roles == ["system", "user", "assistant", "tool"]
        assert payload["messages"][2]["tool_calls"][0]["id"] == "call_ls"
        assert payload["messages"][3]["tool_call_id"] == "call_ls"
        assert payload["tools"][0]["function"]["name"] == "get_weather"
        assert payload["stream"] is True
