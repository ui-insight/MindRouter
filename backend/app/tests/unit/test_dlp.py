############################################################
#
# mindrouter - LLM Inference Translator and Load Balancer
#
# test_dlp.py: Unit tests for DLP scanning logic
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Unit tests for DLP scanner.

Covers:
- Regex scanner: SSN, credit card, email, phone, keywords, clean text
- Severity classification: major from SSN, minor from email, highest wins, unknown defaults
- Text extraction: chat messages, images skipped, response included, empty returns None
- LLM prompt construction
- ScanResult dataclass
"""

import importlib
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ----------------------------------------------------------------
# Direct-load dlp_scanner.py to avoid the DB / telemetry import chain.
# ----------------------------------------------------------------

_svc_dir = Path(__file__).resolve().parents[2] / "services"

sys.modules.setdefault("backend", MagicMock())
sys.modules.setdefault("backend.app", MagicMock())
sys.modules.setdefault("backend.app.logging_config", MagicMock(get_logger=MagicMock(return_value=MagicMock())))

_scanner_spec = importlib.util.spec_from_file_location(
    "dlp_scanner", _svc_dir / "dlp_scanner.py",
    submodule_search_locations=[],
)
_scanner_mod = importlib.util.module_from_spec(_scanner_spec)
_scanner_spec.loader.exec_module(_scanner_mod)

# Import functions from loaded module
scan_regex = _scanner_mod.scan_regex
classify_severity = _scanner_mod.classify_severity
extract_scannable_text = _scanner_mod.extract_scannable_text
ScanFinding = _scanner_mod.ScanFinding
ScanResult = _scanner_mod.ScanResult


# ===================================================================
# Regex Scanner Tests
# ===================================================================

class TestRegexScanner:
    """Tests for the regex/keyword scanner."""

    def test_detects_ssn(self):
        text = "My SSN is 123-45-6789 and I need help."
        findings = scan_regex(text)
        assert len(findings) >= 1
        ssn_findings = [f for f in findings if f.category == "social security number"]
        assert len(ssn_findings) == 1
        assert ssn_findings[0].text == "123-45-6789"
        assert ssn_findings[0].confidence == 1.0
        assert ssn_findings[0].scanner == "regex"

    def test_detects_email(self):
        text = "Contact me at user@example.com for details."
        findings = scan_regex(text)
        email_findings = [f for f in findings if f.category == "email"]
        assert len(email_findings) == 1
        assert "user@example.com" in email_findings[0].text

    def test_detects_credit_card(self):
        text = "Card number: 4111 1111 1111 1111"
        findings = scan_regex(text)
        cc_findings = [f for f in findings if f.category == "credit card number"]
        assert len(cc_findings) >= 1

    def test_clean_text_no_findings(self):
        text = "The weather is nice today. Let's discuss the project timeline."
        findings = scan_regex(text)
        assert len(findings) == 0

    def test_multiple_patterns_match(self):
        text = "SSN: 123-45-6789, email: test@example.com"
        findings = scan_regex(text)
        categories = {f.category for f in findings}
        assert "social security number" in categories
        assert "email" in categories

    def test_custom_patterns(self):
        text = "Student ID: STUD-12345 is enrolled."
        custom = [{"name": "Student ID", "pattern": r"STUD-\d+", "category": "student_id"}]
        findings = scan_regex(text, custom_patterns=custom)
        custom_findings = [f for f in findings if f.category == "student_id"]
        assert len(custom_findings) == 1
        assert custom_findings[0].text == "STUD-12345"

    def test_keywords(self):
        text = "This document is CONFIDENTIAL and must not be shared."
        findings = scan_regex(text, keywords=["confidential"])
        kw_findings = [f for f in findings if f.category == "keyword"]
        assert len(kw_findings) == 1
        assert kw_findings[0].confidence == 0.9

    def test_keyword_case_insensitive(self):
        text = "TOP SECRET information follows."
        findings = scan_regex(text, keywords=["top secret"])
        kw_findings = [f for f in findings if f.category == "keyword"]
        assert len(kw_findings) == 1

    def test_invalid_regex_skipped(self):
        text = "Some text"
        custom = [{"name": "Bad", "pattern": r"[invalid", "category": "bad"}]
        findings = scan_regex(text, custom_patterns=custom)
        # Should not raise, just skip the bad pattern
        assert isinstance(findings, list)

    def test_empty_keywords_ignored(self):
        text = "Hello world"
        findings = scan_regex(text, keywords=["", "  ", None])
        # None keyword should be handled gracefully
        assert isinstance(findings, list)


# ===================================================================
# Severity Classification Tests
# ===================================================================

class TestSeverityClassification:
    """Tests for severity classification logic."""

    def test_major_from_ssn(self):
        findings = [ScanFinding("regex", "social security number", "123-45-6789", 1.0)]
        rules = {"social security number": "major", "email": "minor"}
        assert classify_severity(findings, rules) == "major"

    def test_minor_from_email(self):
        findings = [ScanFinding("regex", "email", "test@example.com", 1.0)]
        rules = {"email": "minor"}
        assert classify_severity(findings, rules) == "minor"

    def test_highest_wins(self):
        findings = [
            ScanFinding("regex", "email", "test@example.com", 1.0),
            ScanFinding("regex", "social security number", "123-45-6789", 1.0),
        ]
        rules = {"email": "minor", "social security number": "major"}
        assert classify_severity(findings, rules) == "major"

    def test_unknown_category_defaults_moderate(self):
        findings = [ScanFinding("regex", "unknown_category", "data", 0.8)]
        assert classify_severity(findings, {}) == "moderate"

    def test_empty_findings_returns_minor(self):
        assert classify_severity([], {}) == "minor"

    def test_no_rules_defaults_moderate(self):
        findings = [ScanFinding("gliner", "person", "John Doe", 0.9)]
        assert classify_severity(findings) == "moderate"


# ===================================================================
# Text Extraction Tests
# ===================================================================

class TestTextExtraction:
    """Tests for extracting scannable text from request/response data."""

    def test_chat_messages_concatenated(self):
        messages = [
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = extract_scannable_text(messages=messages)
        assert "Hello world" in result
        assert "Hi there" in result

    def test_images_skipped(self):
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "Describe this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ]},
        ]
        result = extract_scannable_text(messages=messages)
        assert "Describe this" in result
        assert "image" not in result.lower() or "data:image" not in result

    def test_response_included(self):
        result = extract_scannable_text(response_content="The answer is 42.")
        assert "The answer is 42." in result

    def test_empty_returns_none(self):
        result = extract_scannable_text()
        assert result is None

    def test_prompt_extracted(self):
        result = extract_scannable_text(prompt="Complete this sentence:")
        assert "Complete this sentence:" in result

    def test_all_sources_combined(self):
        result = extract_scannable_text(
            messages=[{"role": "user", "content": "Part 1"}],
            prompt="Part 2",
            response_content="Part 3",
        )
        assert "Part 1" in result
        assert "Part 2" in result
        assert "Part 3" in result

    def test_messages_dict_format(self):
        """Test messages in dict format with 'messages' key."""
        messages = {"messages": [
            {"role": "user", "content": "Hello from dict format"},
        ]}
        result = extract_scannable_text(messages=messages)
        assert "Hello from dict format" in result

    def test_whitespace_only_returns_none(self):
        messages = [{"role": "user", "content": "   "}]
        result = extract_scannable_text(messages=messages)
        # "   " is not empty after strip check
        # The function joins and checks strip()
        assert result is None or result.strip() == ""


# ===================================================================
# LLM Prompt Construction Tests
# ===================================================================

class TestLLMPromptConstruction:
    """Tests for LLM scanner input construction."""

    @pytest.mark.asyncio
    async def test_llm_scan_includes_text_in_user_message(self):
        """Verify the text is included in the user message sent to the LLM."""
        import httpx

        captured_json = {}

        async def mock_post(self, url, **kwargs):
            captured_json.update(kwargs.get("json", {}))
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "[]"}}]
            }
            return mock_resp

        with patch.object(httpx.AsyncClient, "post", mock_post):
            scan_llm = _scanner_mod.scan_llm
            await scan_llm(
                "My SSN is 123-45-6789",
                system_prompt="Analyze for PII",
                model="test-model",
                api_key="test-key",
                base_url="http://localhost:8000",
            )

        assert "messages" in captured_json
        user_msg = [m for m in captured_json["messages"] if m["role"] == "user"][0]
        assert "123-45-6789" in user_msg["content"]

    @pytest.mark.asyncio
    async def test_llm_scan_uses_system_prompt(self):
        """Verify the system prompt is passed correctly."""
        import httpx

        captured_json = {}

        async def mock_post(self, url, **kwargs):
            captured_json.update(kwargs.get("json", {}))
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.raise_for_status = MagicMock()
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": "[]"}}]
            }
            return mock_resp

        with patch.object(httpx.AsyncClient, "post", mock_post):
            scan_llm = _scanner_mod.scan_llm
            await scan_llm(
                "test text",
                system_prompt="Custom system prompt for DLP",
                model="test-model",
                api_key="test-key",
            )

        sys_msg = [m for m in captured_json["messages"] if m["role"] == "system"][0]
        assert sys_msg["content"] == "Custom system prompt for DLP"


# ===================================================================
# ScanResult Tests
# ===================================================================

class TestScanResult:
    """Tests for ScanResult dataclass."""

    def test_default_values(self):
        result = ScanResult()
        assert result.findings == []
        assert result.severity == "minor"
        assert result.scan_latency_ms == 0
        assert result.scanner == "regex"
        assert result.detail is None

    def test_with_findings(self):
        findings = [
            ScanFinding("regex", "ssn", "123-45-6789", 1.0),
            ScanFinding("gliner", "person", "John", 0.9),
        ]
        result = ScanResult(
            findings=findings,
            severity="major",
            scan_latency_ms=15,
            scanner="gliner",
            detail="2 findings",
        )
        assert len(result.findings) == 2
        assert result.severity == "major"
        assert result.scan_latency_ms == 15


# ===================================================================
# ScanFinding Tests
# ===================================================================

class TestScanFinding:
    """Tests for ScanFinding dataclass."""

    def test_default_offsets(self):
        f = ScanFinding("regex", "ssn", "123-45-6789", 1.0)
        assert f.start == 0
        assert f.end == 0

    def test_with_offsets(self):
        f = ScanFinding("regex", "ssn", "123-45-6789", 1.0, start=10, end=21)
        assert f.start == 10
        assert f.end == 21
