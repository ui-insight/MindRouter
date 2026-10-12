############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_kimi_tool_names.py: Kimi K3 tool results carry a name.
#     K3's template refuses a tool message it cannot name
#     ("tool messages need a resolvable tool name") whenever
#     the result does not line up by position with the last
#     assistant message's calls. Codex replays history item by
#     item, so that happens; the vLLM translator now names
#     every K3 tool result from its tool_call_id.
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Kimi K3 tool results are named after the call they answer."""

import pytest

from backend.app.core.translators.responses_in import ResponsesInTranslator
from backend.app.core.translators.vllm_out import (
    UNKNOWN_TOOL_NAME,
    VLLMOutTranslator,
    name_tool_results,
)

KIMI = "moonshotai/kimi-k3"
TOOLS = [{"type": "function", "name": "shell", "parameters": {"type": "object", "properties": {}}},
         {"type": "function", "name": "read_file", "parameters": {"type": "object", "properties": {}}}]


def _call(call_id, name="shell"):
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": "{}"}


def _output(call_id, text="ok"):
    return {"type": "function_call_output", "call_id": call_id, "output": text}


def _assistant_text(text):
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}


USER = {"type": "message", "role": "user", "content": "list files"}


def _payload(items, model=KIMI):
    canonical = ResponsesInTranslator.translate_responses_request(
        {"model": model, "input": items, "tools": TOOLS, "stream": True})
    return VLLMOutTranslator.translate_chat_request(canonical)


def _tool_messages(payload):
    return [(m["tool_call_id"], m.get("name")) for m in payload["messages"] if m["role"] == "tool"]


class TestCodexHistoriesKimiRefused:
    """The three shapes that got HTTP 400 from K3 on aspen300 (probed 2026-10-12)."""

    def test_assistant_text_between_a_call_and_its_result(self):
        payload = _payload([USER, _call("call_1"), _assistant_text("\n"), _output("call_1")])
        assert _tool_messages(payload) == [("call_1", "shell")]

    def test_one_turns_calls_split_over_two_assistant_messages(self):
        items = [USER, _call("call_1", "shell"), _assistant_text("and also"), _call("call_2", "read_file"),
                 _output("call_1"), _output("call_2")]
        assert _tool_messages(_payload(items)) == [("call_1", "shell"), ("call_2", "read_file")]

    def test_a_result_whose_call_is_gone(self):
        assert _tool_messages(_payload([USER, _output("call_9")])) == [("call_9", UNKNOWN_TOOL_NAME)]


class TestNaming:
    def test_parallel_results_out_of_order_get_their_own_names(self):
        items = [USER, _call("a", "shell"), _call("b", "read_file"), _output("b"), _output("a")]
        assert _tool_messages(_payload(items)) == [("b", "read_file"), ("a", "shell")]

    def test_a_reused_id_takes_the_latest_call_before_it(self):
        items = [USER, _call("x", "shell"), _output("x"), _call("x", "read_file"), _output("x")]
        assert _tool_messages(_payload(items)) == [("x", "shell"), ("x", "read_file")]

    def test_a_name_the_client_sent_is_kept(self):
        messages = [{"role": "assistant", "content": None,
                     "tool_calls": [{"id": "c", "type": "function", "function": {"name": "shell", "arguments": "{}"}}]},
                    {"role": "tool", "tool_call_id": "c", "name": "custom", "content": "ok"}]
        name_tool_results(messages)
        assert messages[1]["name"] == "custom"

    def test_odd_shapes_do_not_raise(self):
        messages = [{"role": "assistant", "content": None, "tool_calls": [{"id": "c"}, "junk", {"function": None}]},
                    {"role": "tool", "content": "ok"},
                    {"role": "tool", "tool_call_id": None, "content": "ok"}]
        name_tool_results(messages)
        assert [m["name"] for m in messages[1:]] == [UNKNOWN_TOOL_NAME, UNKNOWN_TOOL_NAME]


class TestOnlyKimiK3:
    @pytest.mark.parametrize("model", ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "moonshotai/kimi-k2"])
    def test_other_models_get_the_messages_unchanged(self, model):
        payload = _payload([USER, _call("call_1"), _output("call_1")], model=model)
        assert _tool_messages(payload) == [("call_1", None)]
        assert all("name" not in m for m in payload["messages"] if m["role"] == "tool")

    @pytest.mark.parametrize("model", [KIMI, "Kimi-K3-NVFP4", "moonshotai/kimi_k3"])
    def test_every_k3_spelling(self, model):
        assert _tool_messages(_payload([USER, _call("call_1"), _output("call_1")], model=model)) == [
            ("call_1", "shell")]

    def test_the_ordinary_case_is_otherwise_unchanged(self):
        # Naming adds one key; roles, ids, content and order stay as they were.
        items = [USER, _call("call_1"), _output("call_1")]
        kimi, other = _payload(items), _payload(items, model="qwen/qwen3.8-27b")
        for m in kimi["messages"]:
            m.pop("name", None) if m["role"] == "tool" else None
        assert kimi["messages"] == other["messages"]
