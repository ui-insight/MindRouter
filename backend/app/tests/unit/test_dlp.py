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

import asyncio
import importlib
import json
import secrets
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
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

class TestIgnoreSeverity:
    """A category mapped to "ignore" vanishes: it never raises the alert
    level, and run_dlp_scan drops the findings before anything downstream
    (worker, email, inline block/redact) can see them."""

    def test_ignored_category_does_not_raise_severity(self):
        findings = [
            ScanFinding("regex", "email", "a@b.com", 1.0, 0, 7),
            ScanFinding("regex", "social security number", "123-45-6789", 1.0, 0, 11),
        ]
        rules = {"social security number": "ignore", "email": "minor"}
        assert classify_severity(findings, rules) == "minor"

    def test_ignore_is_not_treated_as_unknown_moderate(self):
        # _SEVERITY_ORDER lacks "ignore"; the old .get(x, 1) default would have
        # silently promoted an ignored category to moderate.
        findings = [ScanFinding("regex", "email", "a@b.com", 1.0, 0, 7)]
        assert classify_severity(findings, {"email": "ignore"}) == "minor"

    def test_drop_ignored_findings(self):
        findings = [
            ScanFinding("regex", "email", "a@b.com", 1.0, 0, 7),
            ScanFinding("gliner", "person", "Bob", 0.9, 0, 3),
        ]
        kept = _scanner_mod.drop_ignored_findings(findings, {"person": "ignore"})
        assert [f.category for f in kept] == ["email"]
        # no rules -> untouched copy
        assert _scanner_mod.drop_ignored_findings(findings, {}) == findings

    @pytest.mark.asyncio
    async def test_run_dlp_scan_treats_all_ignored_as_clean(self):
        result = await _scanner_mod.run_dlp_scan(
            "mail me at a@b.com",
            {"regex.enabled": True, "gliner.enabled": False, "llm.enabled": False,
             "severity_rules": {"email": "ignore"}},
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_run_dlp_scan_keeps_non_ignored_findings(self):
        result = await _scanner_mod.run_dlp_scan(
            "a@b.com and 123-45-6789",
            {"regex.enabled": True, "gliner.enabled": False, "llm.enabled": False,
             "severity_rules": {"email": "ignore", "social security number": "major"}},
        )
        assert result is not None
        assert {f.category for f in result.findings} == {"social security number"}
        assert result.severity == "major"
        assert "email" not in result.detail

    @pytest.mark.asyncio
    async def test_gliner_not_asked_for_ignored_categories(self, monkeypatch):
        seen = {}

        async def fake_gliner(text, categories=None, threshold=0.5, max_chars=None):
            seen["categories"] = categories
            return []

        monkeypatch.setattr(_scanner_mod, "scan_gliner", fake_gliner)
        await _scanner_mod.run_dlp_scan(
            "hello",
            {"regex.enabled": False, "gliner.enabled": True, "llm.enabled": False,
             "gliner.categories": ["person", "social security number"],
             "severity_rules": {"person": "ignore"}},
        )
        assert seen["categories"] == ["social security number"]

    @pytest.mark.asyncio
    async def test_gliner_skipped_when_every_category_ignored(self, monkeypatch):
        called = []

        async def fake_gliner(*a, **k):
            called.append(1)
            return []

        monkeypatch.setattr(_scanner_mod, "scan_gliner", fake_gliner)
        result = await _scanner_mod.run_dlp_scan(
            "hello",
            {"regex.enabled": False, "gliner.enabled": True, "llm.enabled": False,
             "gliner.categories": ["person"], "severity_rules": {"person": "ignore"}},
        )
        assert called == [] and result is None


class TestCreditCardDecimalGuard:
    """A decimal number is not a credit card.

    `.` is not a word character, so a bare \\b let the card pattern start
    immediately after a decimal point and swallow the fractional digits.
    Game telemetry like `"dist":2.808820224788294` produced the 15-digit run
    808820224788294, which passes Luhn by chance (~1 random run in 10 does)
    and raised a MAJOR alert. Six fired in production before it was caught.
    """

    # Verbatim from the production payload that triggered the alerts.
    REAL_FALSE_POSITIVES = [
        '{"x":20,"y":9,"z":21,"face":[0,1,0],"dist":2.808820224788294}',
        '{"x":33,"y":12,"z":33,"face":[0,-1,0],"dist":0.181321289864317}',
        'S2 target: {"x":29,"y":10,"z":30,"dist":5.213401882041463}',
        'S3 target: {"x":30,"y":10,"z":30,"dist":4.8068097965929635}',
    ]

    @pytest.mark.parametrize("text", REAL_FALSE_POSITIVES)
    def test_decimal_fractions_are_not_cards(self, text):
        assert [f for f in scan_regex(text) if f.category == "credit card number"] == []

    def test_the_specific_luhn_passing_run_is_rejected(self):
        """808820224788294 really does pass Luhn — the guard, not the
        validator, is what has to reject it."""
        assert _scanner_mod._luhn_ok("808820224788294") is True
        assert [f for f in scan_regex('"dist":2.808820224788294')
                if f.category == "credit card number"] == []
        # the same digits standing alone are still a candidate
        assert [f.text for f in scan_regex("card 808820224788294 here")
                if f.category == "credit card number"] == ["808820224788294"]

    @pytest.mark.parametrize("text,expected", [
        ("card 4111 1111 1111 1111 ok", "4111 1111 1111 1111"),
        ("4111-1111-1111-1111", "4111-1111-1111-1111"),
        ("pan=4111111111111111.", "4111111111111111"),      # trailing period
        ("my card is 5500005555555559", "5500005555555559"),
        ("(4111111111111111)", "4111111111111111"),
    ])
    def test_real_cards_still_match(self, text, expected):
        hits = [f.text for f in scan_regex(text) if f.category == "credit card number"]
        assert hits == [expected], f"regression: {text!r} no longer detected"

    def test_guard_does_not_let_a_run_restart_mid_number(self):
        """Every later start inside the run is preceded by a digit, so the
        whole fractional part stays immune rather than matching one char in."""
        text = '"dist":2.8088202247882948888'
        assert [f for f in scan_regex(text) if f.category == "credit card number"] == []


class TestScanLimit:
    """dlp.max_scan_chars (2.9.48): admin-tunable global scan window; 0 means
    no limit so a long prompt cannot push sensitive text past the window."""

    def test_default_is_the_constant(self):
        assert _scanner_mod.effective_scan_limit({}) == _scanner_mod.MAX_SCAN_CHARS
        assert _scanner_mod.effective_scan_limit(None) == _scanner_mod.MAX_SCAN_CHARS

    def test_zero_or_negative_means_unlimited(self):
        assert _scanner_mod.effective_scan_limit({"max_scan_chars": 0}) is None
        assert _scanner_mod.effective_scan_limit({"max_scan_chars": "0"}) is None
        assert _scanner_mod.effective_scan_limit({"max_scan_chars": -5}) is None

    def test_garbage_falls_back_to_default_not_unlimited(self):
        assert _scanner_mod.effective_scan_limit({"max_scan_chars": "lots"}) == _scanner_mod.MAX_SCAN_CHARS
        assert _scanner_mod.effective_scan_limit({"max_scan_chars": None}) == _scanner_mod.MAX_SCAN_CHARS

    @pytest.mark.asyncio
    async def test_ssn_past_the_limit_is_missed_with_a_limit(self):
        text = "x" * 5000 + " SSN 123-45-6789"
        cfg = {"regex.enabled": True, "gliner.enabled": False, "llm.enabled": False,
               "max_scan_chars": 1000}
        assert await _scanner_mod.run_dlp_scan(text, cfg) is None

    @pytest.mark.asyncio
    async def test_no_limit_scans_the_entire_document(self):
        # Well past the old hard ceiling: the SSN sits at ~300k chars.
        text = "x" * (_scanner_mod.MAX_SCAN_CHARS + 100_000) + " SSN 123-45-6789"
        cfg = {"regex.enabled": True, "gliner.enabled": False, "llm.enabled": False,
               "max_scan_chars": 0}
        result = await _scanner_mod.run_dlp_scan(text, cfg)
        assert result is not None
        assert [f.category for f in result.findings] == ["social security number"]

    @pytest.mark.asyncio
    async def test_unset_keeps_the_old_ceiling(self):
        text = "x" * (_scanner_mod.MAX_SCAN_CHARS + 10) + " SSN 123-45-6789"
        cfg = {"regex.enabled": True, "gliner.enabled": False, "llm.enabled": False}
        assert await _scanner_mod.run_dlp_scan(text, cfg) is None


class TestAuthoritativePatternList:
    """Once the admin saves the rule list, the stored patterns ARE the
    scanner's set — built-ins are no longer implicitly prepended."""

    def test_builtins_prepended_by_default(self):
        assert any(f.category == "social security number" for f in scan_regex("SSN 123-45-6789"))

    def test_include_builtins_false_runs_only_the_given_list(self):
        findings = scan_regex("SSN 123-45-6789 id V00123456", custom_patterns=[
            {"name": "Vandal ID", "pattern": r"\bV\d{8}\b", "category": "student id"},
        ], include_builtins=False)
        assert [f.category for f in findings] == ["student id"]

    @pytest.mark.asyncio
    async def test_run_dlp_scan_honours_builtins_in_list_flag(self):
        cfg = {"regex.enabled": True, "gliner.enabled": False, "llm.enabled": False,
               "regex.patterns": [], "regex.builtins_in_list": True}
        assert await _scanner_mod.run_dlp_scan("SSN 123-45-6789", cfg) is None
        cfg["regex.builtins_in_list"] = False
        assert await _scanner_mod.run_dlp_scan("SSN 123-45-6789", cfg) is not None

    def test_stored_validator_key_is_applied(self):
        card = {"name": "Credit Card", "pattern": r"\b(?:\d[ -]*?){13,19}\b",
                "category": "credit card number", "validator": "luhn"}
        ok = scan_regex("4111 1111 1111 1111", custom_patterns=[card], include_builtins=False)
        bad = scan_regex("4111 1111 1111 1112", custom_patterns=[card], include_builtins=False)
        assert len(ok) == 1 and bad == []

    def test_builtin_helpers(self):
        names = [p["name"] for p in _scanner_mod.builtin_patterns()]
        assert names == ["SSN", "Credit Card", "Email Address", "Phone (US)", "Date of Birth"]
        assert _scanner_mod.builtin_validator_for("credit card") == "luhn"
        assert _scanner_mod.builtin_validator_for("SSN") is None
        assert _scanner_mod.builtin_validator_for("Vandal ID") is None
        # copies — mutating the result must not touch the module constant
        _scanner_mod.builtin_patterns()[0]["pattern"] = "x"
        assert _scanner_mod.builtin_patterns()[0]["pattern"] != "x"


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

    def test_earlier_reasoning_is_scanned(self):
        # Reasoning sent back by the client is client-supplied text in every
        # spelling: canonical/vLLM, OpenAI, Ollama, Anthropic and Responses.
        messages = [
            {"role": "assistant", "content": "a", "reasoning": "canonical-r"},
            {"role": "assistant", "content": "b", "reasoning_content": "openai-r"},
            {"role": "assistant", "content": "c", "thinking": "ollama-r"},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "anthropic-r", "signature": "s"}]},
            {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "responses-r"}]},
        ]
        result = extract_scannable_text(messages=messages)
        for text in ("canonical-r", "openai-r", "ollama-r", "anthropic-r", "responses-r"):
            assert text in result

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

def _fake_complete(reply, captured=None):
    """Build a completion callable standing in for a backend dispatch."""
    async def _complete(model, messages):
        if captured is not None:
            captured["model"] = model
            captured["messages"] = messages
        return reply
    return _complete


class TestLLMPromptConstruction:
    """Tests for LLM scanner input construction."""

    @pytest.mark.asyncio
    async def test_llm_scan_includes_text_in_user_message(self):
        """Verify the text is included in the user message sent to the LLM."""
        captured = {}
        await _scanner_mod.scan_llm(
            "My SSN is 123-45-6789",
            system_prompt="Analyze for PII",
            model="test-model",
            complete=_fake_complete("[]", captured),
        )

        user_msg = [m for m in captured["messages"] if m["role"] == "user"][0]
        assert "123-45-6789" in user_msg["content"]
        assert captured["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_llm_scan_uses_system_prompt(self):
        """Verify the system prompt is passed correctly."""
        captured = {}
        await _scanner_mod.scan_llm(
            "test text",
            system_prompt="Custom system prompt for DLP",
            model="test-model",
            complete=_fake_complete("[]", captured),
        )

        sys_msg = [m for m in captured["messages"] if m["role"] == "system"][0]
        assert sys_msg["content"] == "Custom system prompt for DLP"


class TestLLMCredentialRemoved:
    """2.9.9: the scanner holds no credential and speaks no HTTP.

    Before 2.9.9 scan_llm authenticated to MindRouter's own /v1 endpoint with a
    key whose RAW value was stored in app_config. These tests encode the
    property that the credential is gone, not merely relocated.
    """

    def test_scan_llm_takes_no_credential(self):
        import inspect

        params = inspect.signature(_scanner_mod.scan_llm).parameters
        assert "api_key" not in params
        assert "base_url" not in params
        assert "complete" in params

    def test_scan_llm_cannot_speak_http(self):
        """The LLM scanner must not be able to dispatch on its own.

        The module now speaks httpx for the OPTIONAL off-host GLiNER path
        (scan_gliner_remote), so this invariant is scoped to scan_llm itself:
        it holds no credential and owns no HTTP client — it dispatches only
        through the injected `complete` callable.
        """
        import inspect

        src = inspect.getsource(_scanner_mod.scan_llm)
        assert "httpx" not in src
        assert "localhost:8000" not in src

    @pytest.mark.asyncio
    async def test_run_dlp_scan_skips_llm_without_callable(self):
        """llm.enabled with no dispatcher must no-op, not crash."""
        result = await _scanner_mod.run_dlp_scan(
            "nothing sensitive here",
            {"regex.enabled": False, "llm.enabled": True, "llm.complete": None},
        )
        assert result is None


class TestLLMResponseParsing:
    """Parse-path hardening for the direct-dispatch scanner."""

    @pytest.mark.asyncio
    async def test_strips_think_block_before_parsing(self):
        """Reasoning models wrap the answer; the gateway used to strip this.

        Dispatching straight to a backend skips that, so scan_llm must strip it
        itself or every reasoning-model scan silently yields zero findings.
        """
        reply = (
            "<think>The user text contains what looks like an SSN.</think>"
            '[{"category": "social security number", "text": "123-45-6789", "confidence": 0.95}]'
        )
        findings = await _scanner_mod.scan_llm(
            "x", system_prompt="p", model="m", complete=_fake_complete(reply),
        )
        assert len(findings) == 1
        assert findings[0].category == "social security number"

    @pytest.mark.asyncio
    async def test_strips_markdown_fences(self):
        reply = '```json\n[{"category": "email", "text": "a@b.com", "confidence": 0.8}]\n```'
        findings = await _scanner_mod.scan_llm(
            "x", system_prompt="p", model="m", complete=_fake_complete(reply),
        )
        assert len(findings) == 1

    @pytest.mark.asyncio
    async def test_dispatch_failure_raises_scanner_error(self):
        """A dispatch failure is a DEGRADED scan, not a clean one: scan_llm
        raises DlpScannerError (fail-closed) so run_dlp_scan can surface the
        failure rather than silently passing the request as clean."""
        async def _boom(model, messages):
            raise RuntimeError("no healthy backend")

        with pytest.raises(_scanner_mod.DlpScannerError):
            await _scanner_mod.scan_llm(
                "x", system_prompt="p", model="m", complete=_boom,
            )

    @pytest.mark.asyncio
    async def test_non_dict_items_are_skipped(self):
        """A malformed array element must not abort the whole parse."""
        reply = '["junk", {"category": "email", "text": "a@b.com", "confidence": 0.8}]'
        findings = await _scanner_mod.scan_llm(
            "x", system_prompt="p", model="m", complete=_fake_complete(reply),
        )
        assert len(findings) == 1
        assert findings[0].category == "email"

    @pytest.mark.asyncio
    async def test_parse_failure_never_logs_scanned_content(self, caplog):
        """The parse-failure log used to emit 200 chars of the model's reply,
        which quotes the very content being scanned."""
        import logging

        records = []

        class _Capture:
            def warning(self, event, **kw):
                records.append((event, kw))

            def __getattr__(self, _):
                return lambda *a, **k: None

        secret = f"SENTINEL-{secrets.token_hex(8)}"
        orig = _scanner_mod.logger
        _scanner_mod.logger = _Capture()
        try:
            # Unparseable non-empty output is now fail-closed (raises), but the
            # privacy contract is unchanged: the log must never quote the reply.
            with pytest.raises(_scanner_mod.DlpScannerError):
                await _scanner_mod.scan_llm(
                    "x", system_prompt="p", model="m",
                    complete=_fake_complete(f"I found {secret} in the text"),
                )
        finally:
            _scanner_mod.logger = orig

        assert records, "expected a parse-failure warning"
        emitted = repr(records)
        assert secret not in emitted, f"scanned content leaked into logs: {emitted}"


class TestMaskSnippet:
    """Alert rows must not become a second copy of the sensitive data."""

    def test_masks_middle_keeps_ends(self):
        assert _scanner_mod.mask_snippet("123-45-6789") == "12*******89"

    def test_full_value_never_survives(self):
        for raw in ("123-45-6789", "4111111111111111", "person@example.edu"):
            masked = _scanner_mod.mask_snippet(raw)
            assert raw not in masked
            assert len(masked) == len(raw)

    def test_short_values_fully_masked(self):
        assert _scanner_mod.mask_snippet("abcd") == "****"
        assert _scanner_mod.mask_snippet("") == ""

    def test_long_values_truncated(self):
        assert len(_scanner_mod.mask_snippet("x" * 500)) == 64


class TestRegexRobustness:
    """One malformed custom pattern must not disable DLP entirely."""

    @pytest.mark.asyncio
    async def test_malformed_pattern_entry_does_not_abort_scan(self):
        """A list of strings (valid JSON, wrong shape) used to raise TypeError
        out of scan_regex, killing the whole scan for every request."""
        result = await _scanner_mod.run_dlp_scan(
            "my ssn is 123-45-6789",
            {"regex.enabled": True, "regex.patterns": ["not-a-dict", 42, None]},
        )
        assert result is not None, "built-in patterns must still run"
        assert any(f.category == "social security number" for f in result.findings)

    def test_invalid_regex_is_skipped_not_raised(self):
        findings = _scanner_mod.scan_regex(
            "my ssn is 123-45-6789",
            custom_patterns=[{"name": "bad", "pattern": "([unclosed", "category": "x"}],
        )
        assert any(f.category == "social security number" for f in findings)

    @pytest.mark.asyncio
    async def test_scan_text_is_capped(self):
        """A pathological request must not hand unbounded text to the engines."""
        huge = "a" * (_scanner_mod.MAX_SCAN_CHARS + 5000)
        result = await _scanner_mod.run_dlp_scan(
            huge, {"regex.enabled": True, "regex.keywords": ["aaa"]},
        )
        assert result is not None
        assert all(f.end <= _scanner_mod.MAX_SCAN_CHARS for f in result.findings)


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


class TestRunDlpScanFailOpen:
    """F53: a scanner that ERRORS must not look like clean traffic.

    run_dlp_scan records the failure in ``scanner_errors`` and returns a
    ScanResult (not None) even with zero findings, so the worker can surface a
    degraded scanner instead of silently passing the request as clean.
    """

    @pytest.mark.asyncio
    async def test_gliner_error_is_surfaced_not_swallowed(self, monkeypatch):
        async def _boom(text, categories=None, threshold=0.5, max_chars=None):
            raise _scanner_mod.DlpScannerError("gliner model unavailable: OSError")

        monkeypatch.setattr(_scanner_mod, "scan_gliner", _boom)
        result = await _scanner_mod.run_dlp_scan(
            "nothing sensitive here",
            {"regex.enabled": False, "gliner.enabled": True},
        )
        assert result is not None, "an errored scan must not return None (clean)"
        assert result.scanner_errors, "the gliner failure must be recorded"
        assert "gliner" in result.scanner_errors[0]
        assert result.findings == []

    @pytest.mark.asyncio
    async def test_clean_scan_still_returns_none(self, monkeypatch):
        # No findings AND no errors is the only genuinely-clean outcome.
        result = await _scanner_mod.run_dlp_scan(
            "nothing sensitive here",
            {"regex.enabled": True, "regex.patterns": [], "regex.keywords": []},
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_one_broken_scanner_does_not_blind_the_others(self, monkeypatch):
        async def _boom(text, categories=None, threshold=0.5, max_chars=None):
            raise _scanner_mod.DlpScannerError("gliner predict failed: RuntimeError")

        monkeypatch.setattr(_scanner_mod, "scan_gliner", _boom)
        # Regex still finds the SSN even though GLiNER is down.
        result = await _scanner_mod.run_dlp_scan(
            "my ssn is 123-45-6789",
            {"regex.enabled": True, "gliner.enabled": True},
        )
        assert result is not None
        assert any(f.category == "social security number" for f in result.findings)
        assert result.scanner_errors  # the gliner failure is still reported


class TestGlinerScanCap:
    """GLiNER (CPU-bound) gets its own text cap so a long chat can't pin a
    thread for tens of seconds. Regex is unaffected."""

    @pytest.mark.asyncio
    async def test_explicit_cap_truncates_text(self, monkeypatch):
        seen = {}

        class _FakeModel:
            def predict_entities(self, text, categories, threshold=0.5):
                seen["len"] = len(text)
                return []

        async def _fake_load():
            return _FakeModel()

        monkeypatch.setattr(_scanner_mod, "_load_gliner", _fake_load)
        await _scanner_mod.scan_gliner("x" * 50000, max_chars=1000)
        assert seen["len"] == 1000

    @pytest.mark.asyncio
    async def test_default_cap_applied_when_unset(self, monkeypatch):
        seen = {}

        class _FakeModel:
            def predict_entities(self, text, categories, threshold=0.5):
                seen["len"] = len(text)
                return []

        async def _fake_load():
            return _FakeModel()

        monkeypatch.setattr(_scanner_mod, "_load_gliner", _fake_load)
        await _scanner_mod.scan_gliner("x" * 50000)  # no max_chars -> default
        assert seen["len"] == _scanner_mod.GLINER_DEFAULT_MAX_CHARS

    @pytest.mark.asyncio
    async def test_short_text_not_truncated(self, monkeypatch):
        seen = {}

        class _FakeModel:
            def predict_entities(self, text, categories, threshold=0.5):
                seen["len"] = len(text)
                return []

        async def _fake_load():
            return _FakeModel()

        monkeypatch.setattr(_scanner_mod, "_load_gliner", _fake_load)
        await _scanner_mod.scan_gliner("just a short message", max_chars=10000)
        assert seen["len"] == len("just a short message")


class TestLuhnValidation:
    """Built-in credit-card pattern requires a Luhn-valid digit sequence."""

    def test_luhn_valid_cards_still_detected(self):
        for text in (
            "Card: 4111 1111 1111 1111",       # visa, spaced
            "card=4111-1111-1111-1111",        # visa, hyphenated
            "amex 378282246310005 on file",    # 15-digit amex
        ):
            findings = scan_regex(text)
            cats = {f.category for f in findings}
            assert "credit card number" in cats, text

    def test_luhn_invalid_lookalikes_not_flagged(self):
        for text in (
            "order id 4111111111111112",        # last digit off -> Luhn fails
            "EAN barcode 4006381333931",        # EAN-13-style digits failing Luhn
            # (note: ~10% of EAN-13s coincidentally PASS Luhn — the validator
            # removes most, not all, barcode false positives)
            "tracking 9400 1000 0000 0000 0001",
        ):
            findings = scan_regex(text)
            cats = {f.category for f in findings}
            assert "credit card number" not in cats, text

    def test_custom_patterns_bypass_luhn(self):
        # Admin-supplied patterns keep raw regex semantics even for the same category.
        custom = [{"name": "raw16", "pattern": r"\b\d{16}\b",
                   "category": "credit card number", "severity": "major"}]
        findings = scan_regex("val 4111111111111112", custom_patterns=custom)
        assert any(f.category == "credit card number" for f in findings)

    def test_luhn_helper(self):
        assert _scanner_mod._luhn_ok("4111 1111 1111 1111")
        assert not _scanner_mod._luhn_ok("4111111111111112")
        assert not _scanner_mod._luhn_ok("1234")           # too short


class TestGlinerDefaultCategories:
    def test_person_not_in_defaults(self):
        # 'person' measured at precision 0.34 (headers/greetings) — opt-in only.
        import inspect
        src = inspect.getsource(_scanner_mod.scan_gliner)
        assert '"person",' not in src.split("categories = [")[1].split("]")[0]


class TestLuhnGluedPrefixRecovery:
    """A validator failure must re-attempt INSIDE the span, not skip past it —
    the greedy card pattern glues short digit prefixes ('cvv 123 <card>') onto
    the card, and the glued span fails Luhn."""

    def test_card_after_short_digit_token_still_detected(self):
        for text in (
            "cvv 123 4111111111111111",
            "id 12 4111 1111 1111 1111",
            "pin 1 378282246310005",
        ):
            findings = scan_regex(text)
            cc = [f for f in findings if f.category == "credit card number"]
            assert cc, text
            assert any("4111" in f.text or "3782" in f.text for f in cc), text

    def test_luhn_invalid_after_prefix_still_not_flagged(self):
        findings = scan_regex("order 99 4111111111111112")
        assert not any(f.category == "credit card number" for f in findings)


class TestGlinerEmptyCategories:
    def test_explicit_empty_list_scans_nothing(self):
        # Explicitly-empty admin list means "no categories" — returns []
        # before any model load, so this runs without gliner installed.
        result = asyncio.run(_scanner_mod.scan_gliner("ssn 123-45-6789", categories=[]))
        assert result == []


# ===================================================================
# Off-host GLiNER scanner (scan_gliner_remote)
# ===================================================================

def _mock_client(handler):
    """An httpx.AsyncClient whose transport is driven by ``handler``."""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


class TestScanGlinerRemote:
    """The off-host GLiNER HTTP client — wire contract and error mapping."""

    @pytest.mark.asyncio
    async def test_success_maps_findings(self):
        """A 200 response maps to ScanFinding(scanner='gliner'), spans preserved."""
        def handler(request):
            assert request.url.path == "/scan"
            assert request.headers["X-Worker-Key"] == "secret"
            return httpx.Response(200, json={
                "findings": [
                    {"category": "email", "text": "a@b.com", "confidence": 0.91, "start": 6, "end": 13},
                    {"category": "phone number", "text": "208-555-0100", "confidence": 0.77, "start": 20, "end": 32},
                ],
                "latency_ms": 3.1, "queued_ms": 0.0, "batch_size": 1,
            })
        client = _mock_client(handler)
        findings = await _scanner_mod.scan_gliner_remote(
            "hello a@b.com call 208-555-0100", url="https://svc", key="secret", client=client,
        )
        await client.aclose()
        assert len(findings) == 2
        assert all(f.scanner == "gliner" for f in findings)
        assert findings[0].category == "email"
        assert findings[0].text == "a@b.com"
        assert abs(findings[0].confidence - 0.91) < 1e-9
        assert (findings[0].start, findings[0].end) == (6, 13)
        assert (findings[1].start, findings[1].end) == (20, 32)

    @pytest.mark.asyncio
    async def test_sends_contract_body(self):
        """The request body carries text/categories/threshold/max_chars."""
        captured = {}
        def handler(request):
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={
                "findings": [], "latency_ms": 0.0, "queued_ms": 0.0, "batch_size": 0,
            })
        client = _mock_client(handler)
        await _scanner_mod.scan_gliner_remote(
            "hello", url="https://svc/", key="k",
            categories=["email"], threshold=0.6, max_chars=50, client=client,
        )
        await client.aclose()
        assert captured["body"] == {
            "text": "hello", "categories": ["email"], "threshold": 0.6, "max_chars": 50,
        }

    @pytest.mark.asyncio
    async def test_nonpositive_max_chars_sent_as_null(self):
        """max_chars=0/None is sent as null (service uses its default)."""
        captured = {}
        def handler(request):
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={
                "findings": [], "latency_ms": 0.0, "queued_ms": 0.0, "batch_size": 0,
            })
        client = _mock_client(handler)
        await _scanner_mod.scan_gliner_remote(
            "hello", url="https://svc", key="k", max_chars=0, client=client,
        )
        await client.aclose()
        assert captured["body"]["max_chars"] is None

    @pytest.mark.asyncio
    async def test_503_raises_oversubscribed(self):
        """A 503 raises the DISTINCT DlpRemoteOversubscribed (a DlpScannerError)."""
        assert issubclass(_scanner_mod.DlpRemoteOversubscribed, _scanner_mod.DlpScannerError)
        def handler(request):
            return httpx.Response(503, json={
                "error": "oversubscribed", "queue_depth": 64, "max_queue": 64,
            })
        client = _mock_client(handler)
        with pytest.raises(_scanner_mod.DlpRemoteOversubscribed):
            await _scanner_mod.scan_gliner_remote("x", url="https://svc", key="k", client=client)
        await client.aclose()

    @pytest.mark.asyncio
    async def test_401_raises_scanner_error_not_oversubscribed(self):
        def handler(request):
            return httpx.Response(401, json={"error": "unauthorized"})
        client = _mock_client(handler)
        with pytest.raises(_scanner_mod.DlpScannerError) as exc:
            await _scanner_mod.scan_gliner_remote("x", url="https://svc", key="bad", client=client)
        await client.aclose()
        assert not isinstance(exc.value, _scanner_mod.DlpRemoteOversubscribed)

    @pytest.mark.asyncio
    async def test_non200_status_raises_scanner_error(self):
        def handler(request):
            return httpx.Response(500, json={"error": "boom"})
        client = _mock_client(handler)
        with pytest.raises(_scanner_mod.DlpScannerError):
            await _scanner_mod.scan_gliner_remote("x", url="https://svc", key="k", client=client)
        await client.aclose()

    @pytest.mark.asyncio
    async def test_connect_error_raises_scanner_error(self):
        """A transport error (connect refused / timeout) is a DlpScannerError."""
        def handler(request):
            raise httpx.ConnectError("connection refused")
        client = _mock_client(handler)
        with pytest.raises(_scanner_mod.DlpScannerError):
            await _scanner_mod.scan_gliner_remote("x", url="https://svc", key="k", client=client)
        await client.aclose()

    @pytest.mark.asyncio
    async def test_timeout_raises_scanner_error(self):
        def handler(request):
            raise httpx.ReadTimeout("timed out")
        client = _mock_client(handler)
        with pytest.raises(_scanner_mod.DlpScannerError):
            await _scanner_mod.scan_gliner_remote("x", url="https://svc", key="k", client=client)
        await client.aclose()

    @pytest.mark.asyncio
    async def test_unparseable_body_raises_scanner_error(self):
        def handler(request):
            return httpx.Response(200, content=b"this is not json")
        client = _mock_client(handler)
        with pytest.raises(_scanner_mod.DlpScannerError):
            await _scanner_mod.scan_gliner_remote("x", url="https://svc", key="k", client=client)
        await client.aclose()

    @pytest.mark.asyncio
    async def test_missing_findings_array_raises(self):
        def handler(request):
            return httpx.Response(200, json={"latency_ms": 1.0})
        client = _mock_client(handler)
        with pytest.raises(_scanner_mod.DlpScannerError):
            await _scanner_mod.scan_gliner_remote("x", url="https://svc", key="k", client=client)
        await client.aclose()


# ===================================================================
# run_dlp_scan remote dispatch + fallback
# ===================================================================

def _gliner_finding(cat="email"):
    return _scanner_mod.ScanFinding(
        scanner="gliner", category=cat, text="a@b.com", confidence=0.9, start=0, end=7,
    )


class TestRunDlpScanRemote:
    """run_dlp_scan honors dlp.gliner.remote.* and the fallback policy."""

    _BASE = {
        "regex.enabled": False,
        "gliner.enabled": True,
        "gliner.remote.enabled": True,
        "gliner.remote.url": "https://svc",
        "gliner.remote.key": "k",
    }

    @pytest.mark.asyncio
    async def test_remote_success_does_not_run_local(self, monkeypatch):
        async def fake_remote(*a, **k):
            return [_gliner_finding()]
        async def fake_local(*a, **k):
            raise AssertionError("local scanner must not run when remote succeeds")
        monkeypatch.setattr(_scanner_mod, "scan_gliner_remote", fake_remote)
        monkeypatch.setattr(_scanner_mod, "scan_gliner", fake_local)
        result = await _scanner_mod.run_dlp_scan("hi a@b.com", dict(self._BASE))
        assert result is not None
        assert any(f.category == "email" for f in result.findings)
        assert not result.scanner_errors

    @pytest.mark.asyncio
    async def test_fallback_local_runs_local_on_remote_failure(self, monkeypatch):
        called = {}
        async def fake_remote(*a, **k):
            raise _scanner_mod.DlpRemoteOversubscribed("queue full")
        async def fake_local(text, categories=None, threshold=0.5, max_chars=None):
            called["local"] = True
            return [_gliner_finding()]
        monkeypatch.setattr(_scanner_mod, "scan_gliner_remote", fake_remote)
        monkeypatch.setattr(_scanner_mod, "scan_gliner", fake_local)
        cfg = dict(self._BASE, **{"gliner.remote.fallback": "local"})
        result = await _scanner_mod.run_dlp_scan("hi a@b.com", cfg)
        assert called.get("local") is True
        assert result is not None
        assert any(f.category == "email" for f in result.findings)
        # Fallback is NON-fatal: no degraded-scan error surfaced.
        assert not result.scanner_errors

    @pytest.mark.asyncio
    async def test_fallback_skip_records_error_and_skips_local(self, monkeypatch):
        called = {}
        async def fake_remote(*a, **k):
            raise _scanner_mod.DlpScannerError("remote down")
        async def fake_local(*a, **k):
            called["local"] = True
            return []
        monkeypatch.setattr(_scanner_mod, "scan_gliner_remote", fake_remote)
        monkeypatch.setattr(_scanner_mod, "scan_gliner", fake_local)
        cfg = dict(self._BASE, **{"gliner.remote.fallback": "skip"})
        result = await _scanner_mod.run_dlp_scan("hi a@b.com", cfg)
        assert "local" not in called
        assert result is not None  # degraded scan is surfaced, not clean
        assert any("gliner" in e for e in result.scanner_errors)

    @pytest.mark.asyncio
    async def test_remote_disabled_uses_local(self, monkeypatch):
        """With remote disabled, behavior is EXACTLY today's: local scan only."""
        called = {}
        async def fake_local(text, categories=None, threshold=0.5, max_chars=None):
            called["local"] = True
            return []
        async def fake_remote(*a, **k):
            raise AssertionError("remote must not be called when disabled")
        monkeypatch.setattr(_scanner_mod, "scan_gliner", fake_local)
        monkeypatch.setattr(_scanner_mod, "scan_gliner_remote", fake_remote)
        result = await _scanner_mod.run_dlp_scan("nothing sensitive", {
            "regex.enabled": False,
            "gliner.enabled": True,
            # gliner.remote.enabled absent -> off
        })
        assert called.get("local") is True
        assert result is None  # clean scan, nothing found, nothing errored


class TestRemoteEndpointPool:
    """Multi-endpoint pool: parsing, round-robin, failover, oversubscription."""

    def test_parse_endpoints_list_and_string(self):
        pe = _scanner_mod.parse_remote_endpoints
        assert pe(["https://a:1/", "https://b:2"]) == ["https://a:1", "https://b:2"]
        assert pe("https://a:1\n https://b:2 , https://a:1") == ["https://a:1", "https://b:2"]
        assert pe([], legacy_url="https://legacy:9") == ["https://legacy:9"]
        assert pe("") == []
        assert pe(None) == []

    def test_pool_first_success_wins(self):
        calls = []
        async def fake_remote(text, url, key, **kw):
            calls.append(url)
            return [_scanner_mod.ScanFinding(scanner="gliner", category="email", text="x@y", confidence=0.9)]
        orig = _scanner_mod.scan_gliner_remote
        _scanner_mod.scan_gliner_remote = fake_remote
        try:
            r = asyncio.run(_scanner_mod.scan_gliner_pool("t", ["https://a", "https://b"], "k"))
            assert len(r) == 1 and len(calls) == 1  # only one endpoint hit
        finally:
            _scanner_mod.scan_gliner_remote = orig

    def test_pool_fails_over_to_next(self):
        calls = []
        async def fake_remote(text, url, key, **kw):
            calls.append(url)
            if url == "https://a":
                raise _scanner_mod.DlpScannerError("down")
            return []
        orig = _scanner_mod.scan_gliner_remote
        _scanner_mod._remote_cooldown.clear()
        _scanner_mod._remote_rr = 0
        _scanner_mod.scan_gliner_remote = fake_remote
        try:
            asyncio.run(_scanner_mod.scan_gliner_pool("t", ["https://a", "https://b"], "k"))
            assert "https://a" in calls and "https://b" in calls  # failed over
        finally:
            _scanner_mod.scan_gliner_remote = orig

    def test_pool_all_fail_raises(self):
        async def fake_remote(text, url, key, **kw):
            raise _scanner_mod.DlpScannerError("down")
        orig = _scanner_mod.scan_gliner_remote
        _scanner_mod._remote_cooldown.clear()
        _scanner_mod.scan_gliner_remote = fake_remote
        try:
            with pytest.raises(_scanner_mod.DlpScannerError):
                asyncio.run(_scanner_mod.scan_gliner_pool("t", ["https://a", "https://b"], "k"))
        finally:
            _scanner_mod.scan_gliner_remote = orig

    def test_pool_all_oversubscribed_preserves_signal(self):
        async def fake_remote(text, url, key, **kw):
            raise _scanner_mod.DlpRemoteOversubscribed("full")
        orig = _scanner_mod.scan_gliner_remote
        _scanner_mod._remote_cooldown.clear()
        _scanner_mod.scan_gliner_remote = fake_remote
        try:
            with pytest.raises(_scanner_mod.DlpRemoteOversubscribed):
                asyncio.run(_scanner_mod.scan_gliner_pool("t", ["https://a"], "k"))
        finally:
            _scanner_mod.scan_gliner_remote = orig

    def test_pool_empty_raises(self):
        with pytest.raises(_scanner_mod.DlpScannerError):
            asyncio.run(_scanner_mod.scan_gliner_pool("t", [], "k"))
