############################################################
#
# mindrouter2 - LLM Inference Translator and Load Balancer
#
# validators.py: Structured output validation for JSON schemas
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Structured output validation for MindRouter2."""

import json
from typing import Any, Dict, List, Optional, Tuple, Union

from backend.app.core.canonical_schemas import ResponseFormat, ResponseFormatType


class ValidationError:
    """Represents a validation error."""

    def __init__(self, path: str, message: str, expected: Any = None, actual: Any = None):
        self.path = path
        self.message = message
        self.expected = expected
        self.actual = actual

    def __repr__(self) -> str:
        return f"ValidationError(path={self.path!r}, message={self.message!r})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "path": self.path,
            "message": self.message,
            "expected": self.expected,
            "actual": self.actual,
        }


class StructuredOutputValidator:
    """
    Validates structured output against JSON schemas.

    Supports validation for:
    - JSON object mode (any valid JSON object)
    - JSON schema mode (validates against provided schema)
    """

    def validate(
        self,
        content: str,
        response_format: Optional[ResponseFormat],
    ) -> Tuple[bool, List[ValidationError]]:
        """
        Validate response content against the specified format.

        Args:
            content: The response content to validate
            response_format: The format specification to validate against

        Returns:
            Tuple of (is_valid, list of validation errors)
        """
        if response_format is None:
            return True, []

        if response_format.type == ResponseFormatType.TEXT:
            return True, []

        if response_format.type == ResponseFormatType.JSON_OBJECT:
            return self._validate_json_object(content)

        if response_format.type == ResponseFormatType.JSON_SCHEMA:
            return self._validate_json_schema(content, response_format.json_schema)

        return True, []

    def _validate_json_object(self, content: str) -> Tuple[bool, List[ValidationError]]:
        """
        Validate that content is valid JSON.

        Args:
            content: The content to validate

        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []

        try:
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                errors.append(
                    ValidationError(
                        path="$",
                        message="Response must be a JSON object",
                        expected="object",
                        actual=type(parsed).__name__,
                    )
                )
        except json.JSONDecodeError as e:
            errors.append(
                ValidationError(
                    path="$",
                    message=f"Invalid JSON: {e.msg}",
                    expected="valid JSON",
                    actual=content[:100] if len(content) > 100 else content,
                )
            )

        return len(errors) == 0, errors

    def _validate_json_schema(
        self,
        content: str,
        schema: Optional[Dict[str, Any]],
    ) -> Tuple[bool, List[ValidationError]]:
        """
        Validate content against a JSON schema.

        Args:
            content: The content to validate
            schema: The JSON schema to validate against

        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []

        # First validate it's valid JSON
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            errors.append(
                ValidationError(
                    path="$",
                    message=f"Invalid JSON: {e.msg}",
                    expected="valid JSON",
                    actual=content[:100] if len(content) > 100 else content,
                )
            )
            return False, errors

        if schema is None:
            return True, []

        # Validate against schema
        schema_errors = self._validate_against_schema(parsed, schema, "$")
        errors.extend(schema_errors)

        return len(errors) == 0, errors

    def _validate_against_schema(
        self,
        data: Any,
        schema: Dict[str, Any],
        path: str,
    ) -> List[ValidationError]:
        """
        Recursively validate data against a JSON schema.

        This is a simplified validator that handles common cases.
        For production use, consider using jsonschema library.

        Args:
            data: The data to validate
            schema: The JSON schema
            path: Current path in the data structure

        Returns:
            List of validation errors
        """
        errors = []

        # Type validation
        if "type" in schema:
            expected_type = schema["type"]
            if not self._check_type(data, expected_type):
                errors.append(
                    ValidationError(
                        path=path,
                        message=f"Expected type {expected_type}",
                        expected=expected_type,
                        actual=type(data).__name__,
                    )
                )
                return errors  # Stop here if type is wrong

        # Object validation
        if schema.get("type") == "object":
            if isinstance(data, dict):
                errors.extend(self._validate_object(data, schema, path))

        # Array validation
        elif schema.get("type") == "array":
            if isinstance(data, list):
                errors.extend(self._validate_array(data, schema, path))

        # String validation
        elif schema.get("type") == "string":
            if isinstance(data, str):
                errors.extend(self._validate_string(data, schema, path))

        # Number validation
        elif schema.get("type") in ("number", "integer"):
            if isinstance(data, (int, float)):
                errors.extend(self._validate_number(data, schema, path))

        # Enum validation
        if "enum" in schema:
            if data not in schema["enum"]:
                errors.append(
                    ValidationError(
                        path=path,
                        message=f"Value must be one of: {schema['enum']}",
                        expected=schema["enum"],
                        actual=data,
                    )
                )

        # Const validation
        if "const" in schema:
            if data != schema["const"]:
                errors.append(
                    ValidationError(
                        path=path,
                        message=f"Value must be {schema['const']}",
                        expected=schema["const"],
                        actual=data,
                    )
                )

        return errors

    def _check_type(self, data: Any, expected_type: Union[str, List[str]]) -> bool:
        """Check if data matches the expected JSON schema type."""
        if isinstance(expected_type, list):
            return any(self._check_type(data, t) for t in expected_type)

        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None),
        }

        expected_python_type = type_map.get(expected_type)
        if expected_python_type is None:
            return True  # Unknown type, assume valid

        return isinstance(data, expected_python_type)

    def _validate_object(
        self,
        data: Dict[str, Any],
        schema: Dict[str, Any],
        path: str,
    ) -> List[ValidationError]:
        """Validate an object against its schema."""
        errors = []

        # Check required properties
        required = schema.get("required", [])
        for prop in required:
            if prop not in data:
                errors.append(
                    ValidationError(
                        path=f"{path}.{prop}",
                        message=f"Missing required property: {prop}",
                        expected=prop,
                        actual=None,
                    )
                )

        # Validate properties
        properties = schema.get("properties", {})
        for prop, value in data.items():
            if prop in properties:
                prop_schema = properties[prop]
                prop_path = f"{path}.{prop}"
                errors.extend(
                    self._validate_against_schema(value, prop_schema, prop_path)
                )

        # Check additionalProperties
        additional_properties = schema.get("additionalProperties", True)
        if additional_properties is False:
            extra_props = set(data.keys()) - set(properties.keys())
            for prop in extra_props:
                errors.append(
                    ValidationError(
                        path=f"{path}.{prop}",
                        message=f"Additional property not allowed: {prop}",
                        expected=None,
                        actual=prop,
                    )
                )

        return errors

    def _validate_array(
        self,
        data: List[Any],
        schema: Dict[str, Any],
        path: str,
    ) -> List[ValidationError]:
        """Validate an array against its schema."""
        errors = []

        # Check minItems
        if "minItems" in schema and len(data) < schema["minItems"]:
            errors.append(
                ValidationError(
                    path=path,
                    message=f"Array must have at least {schema['minItems']} items",
                    expected=schema["minItems"],
                    actual=len(data),
                )
            )

        # Check maxItems
        if "maxItems" in schema and len(data) > schema["maxItems"]:
            errors.append(
                ValidationError(
                    path=path,
                    message=f"Array must have at most {schema['maxItems']} items",
                    expected=schema["maxItems"],
                    actual=len(data),
                )
            )

        # Validate items
        if "items" in schema:
            items_schema = schema["items"]
            for i, item in enumerate(data):
                item_path = f"{path}[{i}]"
                errors.extend(
                    self._validate_against_schema(item, items_schema, item_path)
                )

        return errors

    def _validate_string(
        self,
        data: str,
        schema: Dict[str, Any],
        path: str,
    ) -> List[ValidationError]:
        """Validate a string against its schema."""
        errors = []

        # Check minLength
        if "minLength" in schema and len(data) < schema["minLength"]:
            errors.append(
                ValidationError(
                    path=path,
                    message=f"String must be at least {schema['minLength']} characters",
                    expected=schema["minLength"],
                    actual=len(data),
                )
            )

        # Check maxLength
        if "maxLength" in schema and len(data) > schema["maxLength"]:
            errors.append(
                ValidationError(
                    path=path,
                    message=f"String must be at most {schema['maxLength']} characters",
                    expected=schema["maxLength"],
                    actual=len(data),
                )
            )

        # Check pattern (simplified - no regex validation here)
        # For full pattern support, use the jsonschema library

        return errors

    def _validate_number(
        self,
        data: Union[int, float],
        schema: Dict[str, Any],
        path: str,
    ) -> List[ValidationError]:
        """Validate a number against its schema."""
        errors = []

        # Check minimum
        if "minimum" in schema:
            if "exclusiveMinimum" in schema and schema["exclusiveMinimum"]:
                if data <= schema["minimum"]:
                    errors.append(
                        ValidationError(
                            path=path,
                            message=f"Number must be greater than {schema['minimum']}",
                            expected=f"> {schema['minimum']}",
                            actual=data,
                        )
                    )
            elif data < schema["minimum"]:
                errors.append(
                    ValidationError(
                        path=path,
                        message=f"Number must be at least {schema['minimum']}",
                        expected=f">= {schema['minimum']}",
                        actual=data,
                    )
                )

        # Check maximum
        if "maximum" in schema:
            if "exclusiveMaximum" in schema and schema["exclusiveMaximum"]:
                if data >= schema["maximum"]:
                    errors.append(
                        ValidationError(
                            path=path,
                            message=f"Number must be less than {schema['maximum']}",
                            expected=f"< {schema['maximum']}",
                            actual=data,
                        )
                    )
            elif data > schema["maximum"]:
                errors.append(
                    ValidationError(
                        path=path,
                        message=f"Number must be at most {schema['maximum']}",
                        expected=f"<= {schema['maximum']}",
                        actual=data,
                    )
                )

        return errors
