############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# test_validators.py: Unit tests for structured output validation
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Unit tests for structured output validators."""

import json
import pytest

from backend.app.core.validators import StructuredOutputValidator, ValidationError
from backend.app.core.canonical_schemas import ResponseFormat, ResponseFormatType


class TestValidationError:
    """Tests for ValidationError class."""

    def test_create_error(self):
        """Test creating a validation error."""
        error = ValidationError(
            path="$.name",
            message="Missing required property",
            expected="string",
            actual=None,
        )

        assert error.path == "$.name"
        assert error.message == "Missing required property"
        assert error.expected == "string"
        assert error.actual is None

    def test_error_to_dict(self):
        """Test converting error to dictionary."""
        error = ValidationError(
            path="$.items[0]",
            message="Expected type string",
            expected="string",
            actual="integer",
        )

        d = error.to_dict()

        assert d["path"] == "$.items[0]"
        assert d["message"] == "Expected type string"
        assert d["expected"] == "string"
        assert d["actual"] == "integer"


class TestStructuredOutputValidatorBasic:
    """Basic tests for StructuredOutputValidator."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return StructuredOutputValidator()

    def test_no_format_always_valid(self, validator):
        """Test that no format specified always validates."""
        is_valid, errors = validator.validate("any content", None)
        assert is_valid is True
        assert len(errors) == 0

    def test_text_format_always_valid(self, validator):
        """Test that text format always validates."""
        response_format = ResponseFormat(type=ResponseFormatType.TEXT)
        is_valid, errors = validator.validate("any text content", response_format)
        assert is_valid is True
        assert len(errors) == 0


class TestJsonObjectValidation:
    """Tests for JSON object validation."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return StructuredOutputValidator()

    @pytest.fixture
    def json_format(self):
        """Create JSON object format."""
        return ResponseFormat(type=ResponseFormatType.JSON_OBJECT)

    def test_valid_json_object(self, validator, json_format):
        """Test valid JSON object passes."""
        content = '{"name": "John", "age": 30}'
        is_valid, errors = validator.validate(content, json_format)

        assert is_valid is True
        assert len(errors) == 0

    def test_valid_complex_json_object(self, validator, json_format):
        """Test valid complex JSON object passes."""
        content = json.dumps({
            "user": {
                "name": "John",
                "emails": ["john@example.com", "j.doe@example.com"],
            },
            "active": True,
            "count": 42,
        })
        is_valid, errors = validator.validate(content, json_format)

        assert is_valid is True
        assert len(errors) == 0

    def test_invalid_json_syntax(self, validator, json_format):
        """Test invalid JSON syntax fails."""
        content = '{"name": "John", age: 30}'  # Missing quotes around age
        is_valid, errors = validator.validate(content, json_format)

        assert is_valid is False
        assert len(errors) == 1
        assert "Invalid JSON" in errors[0].message

    def test_json_array_not_object(self, validator, json_format):
        """Test JSON array fails (not an object)."""
        content = '[1, 2, 3]'
        is_valid, errors = validator.validate(content, json_format)

        assert is_valid is False
        assert len(errors) == 1
        assert "must be a JSON object" in errors[0].message

    def test_json_primitive_not_object(self, validator, json_format):
        """Test JSON primitive fails (not an object)."""
        content = '"just a string"'
        is_valid, errors = validator.validate(content, json_format)

        assert is_valid is False
        assert len(errors) == 1
        assert "must be a JSON object" in errors[0].message


class TestJsonSchemaValidation:
    """Tests for JSON schema validation."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return StructuredOutputValidator()

    def test_valid_against_simple_schema(self, validator):
        """Test valid object against simple schema."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }
        response_format = ResponseFormat(
            type=ResponseFormatType.JSON_SCHEMA,
            json_schema=schema,
        )

        content = '{"name": "John", "age": 30}'
        is_valid, errors = validator.validate(content, response_format)

        assert is_valid is True
        assert len(errors) == 0

    def test_missing_required_property(self, validator):
        """Test missing required property fails."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
            },
            "required": ["name", "email"],
        }
        response_format = ResponseFormat(
            type=ResponseFormatType.JSON_SCHEMA,
            json_schema=schema,
        )

        content = '{"name": "John"}'  # Missing email
        is_valid, errors = validator.validate(content, response_format)

        assert is_valid is False
        assert len(errors) == 1
        assert "email" in errors[0].message

    def test_wrong_property_type(self, validator):
        """Test wrong property type fails."""
        schema = {
            "type": "object",
            "properties": {
                "age": {"type": "integer"},
            },
        }
        response_format = ResponseFormat(
            type=ResponseFormatType.JSON_SCHEMA,
            json_schema=schema,
        )

        content = '{"age": "thirty"}'  # String instead of integer
        is_valid, errors = validator.validate(content, response_format)

        assert is_valid is False
        assert len(errors) == 1
        assert "type" in errors[0].message.lower()

    def test_nested_object_validation(self, validator):
        """Test nested object validation."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name"],
                },
            },
            "required": ["user"],
        }
        response_format = ResponseFormat(
            type=ResponseFormatType.JSON_SCHEMA,
            json_schema=schema,
        )

        # Valid nested object
        content = '{"user": {"name": "John", "age": 30}}'
        is_valid, errors = validator.validate(content, response_format)
        assert is_valid is True

        # Missing nested required property
        content = '{"user": {"age": 30}}'  # Missing name
        is_valid, errors = validator.validate(content, response_format)
        assert is_valid is False

    def test_array_validation(self, validator):
        """Test array validation."""
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
            },
        }
        response_format = ResponseFormat(
            type=ResponseFormatType.JSON_SCHEMA,
            json_schema=schema,
        )

        # Valid array
        content = '{"items": ["apple", "banana", "cherry"]}'
        is_valid, errors = validator.validate(content, response_format)
        assert is_valid is True

        # Empty array (violates minItems)
        content = '{"items": []}'
        is_valid, errors = validator.validate(content, response_format)
        assert is_valid is False

        # Wrong item type
        content = '{"items": ["apple", 123, "cherry"]}'  # 123 is not a string
        is_valid, errors = validator.validate(content, response_format)
        assert is_valid is False

    def test_enum_validation(self, validator):
        """Test enum validation."""
        schema = {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["pending", "active", "completed"],
                },
            },
        }
        response_format = ResponseFormat(
            type=ResponseFormatType.JSON_SCHEMA,
            json_schema=schema,
        )

        # Valid enum value
        content = '{"status": "active"}'
        is_valid, errors = validator.validate(content, response_format)
        assert is_valid is True

        # Invalid enum value
        content = '{"status": "unknown"}'
        is_valid, errors = validator.validate(content, response_format)
        assert is_valid is False

    def test_string_length_validation(self, validator):
        """Test string length validation."""
        schema = {
            "type": "object",
            "properties": {
                "username": {
                    "type": "string",
                    "minLength": 3,
                    "maxLength": 20,
                },
            },
        }
        response_format = ResponseFormat(
            type=ResponseFormatType.JSON_SCHEMA,
            json_schema=schema,
        )

        # Valid length
        content = '{"username": "john_doe"}'
        is_valid, errors = validator.validate(content, response_format)
        assert is_valid is True

        # Too short
        content = '{"username": "ab"}'
        is_valid, errors = validator.validate(content, response_format)
        assert is_valid is False

        # Too long
        content = '{"username": "this_username_is_way_too_long_for_validation"}'
        is_valid, errors = validator.validate(content, response_format)
        assert is_valid is False

    def test_number_range_validation(self, validator):
        """Test number range validation."""
        schema = {
            "type": "object",
            "properties": {
                "age": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 150,
                },
                "score": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 100,
                },
            },
        }
        response_format = ResponseFormat(
            type=ResponseFormatType.JSON_SCHEMA,
            json_schema=schema,
        )

        # Valid numbers
        content = '{"age": 30, "score": 85.5}'
        is_valid, errors = validator.validate(content, response_format)
        assert is_valid is True

        # Below minimum
        content = '{"age": -5, "score": 50}'
        is_valid, errors = validator.validate(content, response_format)
        assert is_valid is False

        # Above maximum
        content = '{"age": 30, "score": 150}'
        is_valid, errors = validator.validate(content, response_format)
        assert is_valid is False

    def test_additional_properties_false(self, validator):
        """Test additionalProperties: false validation."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
            "additionalProperties": False,
        }
        response_format = ResponseFormat(
            type=ResponseFormatType.JSON_SCHEMA,
            json_schema=schema,
        )

        # Only defined properties
        content = '{"name": "John"}'
        is_valid, errors = validator.validate(content, response_format)
        assert is_valid is True

        # Extra property
        content = '{"name": "John", "extra": "field"}'
        is_valid, errors = validator.validate(content, response_format)
        assert is_valid is False
        assert "extra" in errors[0].message

    def test_const_validation(self, validator):
        """Test const validation."""
        schema = {
            "type": "object",
            "properties": {
                "version": {"const": "1.0"},
            },
        }
        response_format = ResponseFormat(
            type=ResponseFormatType.JSON_SCHEMA,
            json_schema=schema,
        )

        # Matching const
        content = '{"version": "1.0"}'
        is_valid, errors = validator.validate(content, response_format)
        assert is_valid is True

        # Non-matching value
        content = '{"version": "2.0"}'
        is_valid, errors = validator.validate(content, response_format)
        assert is_valid is False


class TestComplexSchemas:
    """Tests for complex real-world schemas."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return StructuredOutputValidator()

    def test_user_profile_schema(self, validator):
        """Test validation of user profile schema."""
        schema = {
            "type": "object",
            "properties": {
                "id": {"type": "integer"},
                "name": {"type": "string", "minLength": 1},
                "email": {"type": "string"},
                "role": {"type": "string", "enum": ["user", "admin", "moderator"]},
                "settings": {
                    "type": "object",
                    "properties": {
                        "theme": {"type": "string", "enum": ["light", "dark"]},
                        "notifications": {"type": "boolean"},
                    },
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["id", "name", "email"],
        }
        response_format = ResponseFormat(
            type=ResponseFormatType.JSON_SCHEMA,
            json_schema=schema,
        )

        # Valid full profile
        content = json.dumps({
            "id": 1,
            "name": "John Doe",
            "email": "john@example.com",
            "role": "admin",
            "settings": {
                "theme": "dark",
                "notifications": True,
            },
            "tags": ["developer", "team-lead"],
        })
        is_valid, errors = validator.validate(content, response_format)
        assert is_valid is True

        # Valid minimal profile
        content = json.dumps({
            "id": 2,
            "name": "Jane",
            "email": "jane@example.com",
        })
        is_valid, errors = validator.validate(content, response_format)
        assert is_valid is True

        # Invalid: empty name
        content = json.dumps({
            "id": 3,
            "name": "",
            "email": "test@example.com",
        })
        is_valid, errors = validator.validate(content, response_format)
        assert is_valid is False

    def test_api_response_schema(self, validator):
        """Test validation of API response schema."""
        schema = {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "data": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "title": {"type": "string"},
                            "completed": {"type": "boolean"},
                        },
                        "required": ["id", "title"],
                    },
                },
                "pagination": {
                    "type": "object",
                    "properties": {
                        "page": {"type": "integer", "minimum": 1},
                        "total_pages": {"type": "integer", "minimum": 0},
                        "total_items": {"type": "integer", "minimum": 0},
                    },
                },
            },
            "required": ["success", "data"],
        }
        response_format = ResponseFormat(
            type=ResponseFormatType.JSON_SCHEMA,
            json_schema=schema,
        )

        # Valid response
        content = json.dumps({
            "success": True,
            "data": [
                {"id": 1, "title": "Task 1", "completed": False},
                {"id": 2, "title": "Task 2", "completed": True},
            ],
            "pagination": {
                "page": 1,
                "total_pages": 5,
                "total_items": 50,
            },
        })
        is_valid, errors = validator.validate(content, response_format)
        assert is_valid is True

        # Invalid: item missing required field
        content = json.dumps({
            "success": True,
            "data": [
                {"id": 1, "completed": False},  # Missing title
            ],
        })
        is_valid, errors = validator.validate(content, response_format)
        assert is_valid is False
        assert "title" in str(errors)
