"""Structured output translation and validation tests.

Tests structured output format translation between Ollama, vLLM, and canonical
representations, plus schema validation logic.
"""

import json
import pytest

from backend.app.core.translators import (
    OpenAIInTranslator,
    OllamaInTranslator,
    OllamaOutTranslator,
    VLLMOutTranslator,
)
from backend.app.core.canonical_schemas import (
    CanonicalChatRequest,
    CanonicalMessage,
    MessageRole,
    ResponseFormat,
    ResponseFormatType,
)
from backend.app.core.validators import StructuredOutputValidator, ValidationError


# --- Helper to build minimal canonical request with given response_format ---


def _make_canonical(response_format=None):
    return CanonicalChatRequest(
        model="test-model",
        messages=[CanonicalMessage(role=MessageRole.USER, content="test")],
        response_format=response_format,
    )


class TestOllamaStructuredOutputTranslation:
    """Ollama format field -> Canonical ResponseFormat and back."""

    def test_format_json_string(self):
        """format:"json" -> ResponseFormat(type=JSON_OBJECT)."""
        data = {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "Give JSON"}],
            "format": "json",
        }
        result = OllamaInTranslator.translate_chat_request(data)
        assert result.response_format is not None
        assert result.response_format.type == ResponseFormatType.JSON_OBJECT

    def test_format_json_case_insensitive(self):
        """format:"JSON" -> JSON_OBJECT (case-insensitive)."""
        data = {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "test"}],
            "format": "JSON",
        }
        result = OllamaInTranslator.translate_chat_request(data)
        assert result.response_format is not None
        assert result.response_format.type == ResponseFormatType.JSON_OBJECT

    def test_format_schema_object(self, simple_json_schema):
        """format:{schema} -> ResponseFormat(type=JSON_SCHEMA, json_schema=...)."""
        data = {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "test"}],
            "format": simple_json_schema,
        }
        result = OllamaInTranslator.translate_chat_request(data)
        assert result.response_format is not None
        assert result.response_format.type == ResponseFormatType.JSON_SCHEMA
        assert result.response_format.json_schema["schema"] == simple_json_schema

    def test_format_none(self):
        """No format field -> response_format=None."""
        data = {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "test"}],
        }
        result = OllamaInTranslator.translate_chat_request(data)
        assert result.response_format is None

    def test_format_invalid_string(self):
        """format:"xml" -> None."""
        data = {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "test"}],
            "format": "xml",
        }
        result = OllamaInTranslator.translate_chat_request(data)
        assert result.response_format is None

    def test_format_complex_schema(self, nested_json_schema):
        """Nested schema with arrays preserved."""
        data = {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "test"}],
            "format": nested_json_schema,
        }
        result = OllamaInTranslator.translate_chat_request(data)
        assert result.response_format.type == ResponseFormatType.JSON_SCHEMA
        stored_schema = result.response_format.json_schema["schema"]
        assert "user" in stored_schema["properties"]
        assert "tags" in stored_schema["properties"]

    def test_outbound_json_object(self):
        """Canonical JSON_OBJECT -> Ollama format:"json"."""
        canonical = _make_canonical(
            ResponseFormat(type=ResponseFormatType.JSON_OBJECT)
        )
        payload = OllamaOutTranslator.translate_chat_request(canonical)
        assert payload["format"] == "json"

    def test_outbound_json_schema(self, simple_json_schema):
        """Canonical JSON_SCHEMA -> Ollama format:{schema_object}."""
        canonical = _make_canonical(
            ResponseFormat(
                type=ResponseFormatType.JSON_SCHEMA,
                json_schema={"schema": simple_json_schema},
            )
        )
        payload = OllamaOutTranslator.translate_chat_request(canonical)
        assert isinstance(payload["format"], dict)
        assert payload["format"]["type"] == "object"

    def test_outbound_text_format(self):
        """Canonical TEXT -> no format key in Ollama payload."""
        canonical = _make_canonical(
            ResponseFormat(type=ResponseFormatType.TEXT)
        )
        payload = OllamaOutTranslator.translate_chat_request(canonical)
        assert "format" not in payload


class TestVLLMStructuredOutputTranslation:
    """OpenAI response_format -> Canonical and back to vLLM."""

    def test_response_format_json_object(self):
        """OpenAI {"type":"json_object"} -> Canonical JSON_OBJECT."""
        data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "test"}],
            "response_format": {"type": "json_object"},
        }
        result = OpenAIInTranslator.translate_chat_request(data)
        assert result.response_format.type == ResponseFormatType.JSON_OBJECT

    def test_response_format_json_schema(self, simple_json_schema):
        """OpenAI json_schema format -> Canonical JSON_SCHEMA."""
        data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "test"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": simple_json_schema},
            },
        }
        result = OpenAIInTranslator.translate_chat_request(data)
        assert result.response_format.type == ResponseFormatType.JSON_SCHEMA
        assert result.response_format.json_schema is not None

    def test_response_format_text(self):
        """OpenAI {"type":"text"} -> Canonical TEXT."""
        data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "test"}],
            "response_format": {"type": "text"},
        }
        result = OpenAIInTranslator.translate_chat_request(data)
        assert result.response_format.type == ResponseFormatType.TEXT

    def test_outbound_json_object(self):
        """Canonical JSON_OBJECT -> vLLM promoted to json_schema (vLLM nightly compat)."""
        canonical = _make_canonical(
            ResponseFormat(type=ResponseFormatType.JSON_OBJECT)
        )
        payload = VLLMOutTranslator.translate_chat_request(canonical)
        assert payload["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "json_response", "schema": {"type": "object"}},
        }

    def test_outbound_json_schema(self, simple_json_schema):
        """Canonical JSON_SCHEMA -> vLLM {"type":"json_schema","json_schema":{...}}."""
        canonical = _make_canonical(
            ResponseFormat(
                type=ResponseFormatType.JSON_SCHEMA,
                json_schema={"schema": simple_json_schema},
            )
        )
        payload = VLLMOutTranslator.translate_chat_request(canonical)
        assert payload["response_format"]["type"] == "json_schema"
        assert "json_schema" in payload["response_format"]

    def test_outbound_text(self):
        """Canonical TEXT -> vLLM {"type":"text"}."""
        canonical = _make_canonical(
            ResponseFormat(type=ResponseFormatType.TEXT)
        )
        payload = VLLMOutTranslator.translate_chat_request(canonical)
        assert payload["response_format"] == {"type": "text"}

    def test_no_response_format(self):
        """No response_format -> not in vLLM payload."""
        canonical = _make_canonical()
        payload = VLLMOutTranslator.translate_chat_request(canonical)
        assert "response_format" not in payload


class TestStructuredOutputCrossEngine:
    """Cross-engine structured output preservation."""

    def test_ollama_json_to_vllm_json(self):
        """Ollama format:"json" -> Canonical -> vLLM promoted to json_schema."""
        data = {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "test"}],
            "format": "json",
        }
        canonical = OllamaInTranslator.translate_chat_request(data)
        vllm_payload = VLLMOutTranslator.translate_chat_request(canonical)
        assert vllm_payload["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "json_response", "schema": {"type": "object"}},
        }

    def test_ollama_schema_to_vllm_schema(self, simple_json_schema):
        """Ollama format:{schema} -> Canonical -> vLLM json_schema."""
        data = {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "test"}],
            "format": simple_json_schema,
        }
        canonical = OllamaInTranslator.translate_chat_request(data)
        vllm_payload = VLLMOutTranslator.translate_chat_request(canonical)
        assert vllm_payload["response_format"]["type"] == "json_schema"

    def test_vllm_json_to_ollama_json(self):
        """vLLM {"type":"json_object"} -> Canonical -> Ollama format:"json"."""
        data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "test"}],
            "response_format": {"type": "json_object"},
        }
        canonical = OpenAIInTranslator.translate_chat_request(data)
        ollama_payload = OllamaOutTranslator.translate_chat_request(canonical)
        assert ollama_payload["format"] == "json"

    def test_vllm_schema_to_ollama_schema(self, simple_json_schema):
        """vLLM json_schema -> Canonical -> Ollama format:{schema}."""
        data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "test"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": simple_json_schema},
            },
        }
        canonical = OpenAIInTranslator.translate_chat_request(data)
        ollama_payload = OllamaOutTranslator.translate_chat_request(canonical)
        assert isinstance(ollama_payload["format"], dict)

    def test_complex_schema_both_directions(self, complex_schema):
        """Realistic nested schema preserved through both directions."""
        # Ollama -> vLLM
        ollama_data = {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "test"}],
            "format": complex_schema,
        }
        canonical = OllamaInTranslator.translate_chat_request(ollama_data)
        vllm_payload = VLLMOutTranslator.translate_chat_request(canonical)
        assert vllm_payload["response_format"]["type"] == "json_schema"

        # vLLM -> Ollama
        openai_data = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "test"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"schema": complex_schema},
            },
        }
        canonical2 = OpenAIInTranslator.translate_chat_request(openai_data)
        ollama_payload = OllamaOutTranslator.translate_chat_request(canonical2)
        assert isinstance(ollama_payload["format"], dict)

    def test_schema_with_required_fields(self, simple_json_schema):
        """required array preserved through Ollama -> Canonical -> vLLM."""
        data = {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "test"}],
            "format": simple_json_schema,
        }
        canonical = OllamaInTranslator.translate_chat_request(data)
        schema = canonical.response_format.json_schema["schema"]
        assert "required" in schema
        assert "name" in schema["required"]
        assert "age" in schema["required"]

    def test_schema_with_enum(self, enum_json_schema):
        """enum values preserved through Ollama -> Canonical."""
        data = {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "test"}],
            "format": enum_json_schema,
        }
        canonical = OllamaInTranslator.translate_chat_request(data)
        schema = canonical.response_format.json_schema["schema"]
        assert schema["properties"]["status"]["enum"] == ["active", "inactive"]

    def test_schema_with_arrays(self, array_schema):
        """array items schema preserved through Ollama -> Canonical."""
        data = {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "test"}],
            "format": array_schema,
        }
        canonical = OllamaInTranslator.translate_chat_request(data)
        schema = canonical.response_format.json_schema["schema"]
        items_schema = schema["properties"]["items"]["items"]
        assert items_schema["type"] == "object"
        assert "name" in items_schema["properties"]
        assert "quantity" in items_schema["properties"]

    def test_schema_with_additional_properties(self):
        """additionalProperties:false preserved."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": False,
        }
        data = {
            "model": "llama3.2",
            "messages": [{"role": "user", "content": "test"}],
            "format": schema,
        }
        canonical = OllamaInTranslator.translate_chat_request(data)
        stored = canonical.response_format.json_schema["schema"]
        assert stored["additionalProperties"] is False


class TestStructuredOutputValidation:
    """StructuredOutputValidator tests."""

    def setup_method(self):
        self.validator = StructuredOutputValidator()

    def test_valid_json_object(self):
        """Valid JSON object passes JSON_OBJECT validation."""
        rf = ResponseFormat(type=ResponseFormatType.JSON_OBJECT)
        valid, errors = self.validator.validate('{"key": "value"}', rf)
        assert valid is True
        assert len(errors) == 0

    def test_invalid_json(self):
        """Malformed JSON fails."""
        rf = ResponseFormat(type=ResponseFormatType.JSON_OBJECT)
        valid, errors = self.validator.validate("not json", rf)
        assert valid is False
        assert len(errors) > 0

    def test_valid_schema_match(self, simple_json_schema):
        """Output matches schema -> passes."""
        rf = ResponseFormat(
            type=ResponseFormatType.JSON_SCHEMA,
            json_schema=simple_json_schema,
        )
        content = json.dumps({"name": "Alice", "age": 30})
        valid, errors = self.validator.validate(content, rf)
        assert valid is True
        assert len(errors) == 0

    def test_missing_required_field(self, simple_json_schema):
        """Output missing required field -> fails."""
        rf = ResponseFormat(
            type=ResponseFormatType.JSON_SCHEMA,
            json_schema=simple_json_schema,
        )
        content = json.dumps({"name": "Alice"})  # missing "age"
        valid, errors = self.validator.validate(content, rf)
        assert valid is False
        assert any("required" in e.message.lower() or "age" in e.message.lower() for e in errors)

    def test_wrong_type(self, simple_json_schema):
        """Field with wrong type -> fails."""
        rf = ResponseFormat(
            type=ResponseFormatType.JSON_SCHEMA,
            json_schema=simple_json_schema,
        )
        content = json.dumps({"name": "Alice", "age": "thirty"})
        valid, errors = self.validator.validate(content, rf)
        assert valid is False

    def test_nested_schema_validation(self, nested_json_schema):
        """Nested object validation."""
        rf = ResponseFormat(
            type=ResponseFormatType.JSON_SCHEMA,
            json_schema=nested_json_schema,
        )
        content = json.dumps({
            "user": {"name": "Alice", "address": {"city": "NYC", "zip": "10001"}},
            "tags": ["admin"],
        })
        valid, errors = self.validator.validate(content, rf)
        assert valid is True

    def test_array_validation(self):
        """Array minItems/maxItems/items validation."""
        schema = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 3,
        }
        rf = ResponseFormat(
            type=ResponseFormatType.JSON_SCHEMA,
            json_schema=schema,
        )

        # Valid
        valid, errors = self.validator.validate('["a", "b"]', rf)
        assert valid is True

        # Too few
        valid, errors = self.validator.validate("[]", rf)
        assert valid is False

        # Too many
        valid, errors = self.validator.validate('["a", "b", "c", "d"]', rf)
        assert valid is False

    def test_enum_validation(self, enum_json_schema):
        """Value not in enum -> fails."""
        rf = ResponseFormat(
            type=ResponseFormatType.JSON_SCHEMA,
            json_schema=enum_json_schema,
        )
        # Invalid enum value
        content = json.dumps({"status": "deleted", "priority": 1})
        valid, errors = self.validator.validate(content, rf)
        assert valid is False
        assert any("enum" in e.message.lower() or "one of" in e.message.lower() for e in errors)

        # Valid enum value
        content = json.dumps({"status": "active", "priority": 1})
        valid, errors = self.validator.validate(content, rf)
        assert valid is True

    def test_text_format_always_valid(self):
        """TEXT format passes everything."""
        rf = ResponseFormat(type=ResponseFormatType.TEXT)
        valid, errors = self.validator.validate("anything goes", rf)
        assert valid is True

    def test_none_format_always_valid(self):
        """None response_format passes everything."""
        valid, errors = self.validator.validate("anything goes", None)
        assert valid is True

    def test_json_object_rejects_array(self):
        """JSON_OBJECT mode rejects arrays (must be object)."""
        rf = ResponseFormat(type=ResponseFormatType.JSON_OBJECT)
        valid, errors = self.validator.validate("[1, 2, 3]", rf)
        assert valid is False

    def test_additional_properties_false(self):
        """additionalProperties:false rejects extra props."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": False,
        }
        rf = ResponseFormat(
            type=ResponseFormatType.JSON_SCHEMA,
            json_schema=schema,
        )
        content = json.dumps({"name": "Alice", "extra": "field"})
        valid, errors = self.validator.validate(content, rf)
        assert valid is False
        assert any("additional" in e.message.lower() for e in errors)
