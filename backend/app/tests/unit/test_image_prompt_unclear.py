############################################################
#
# mindrouter - unit tests for the UNCLEAR image-prompt verdict
#
# Regression: a chat follow-up such as "the same image, unchanged" or
# "make it red" was reported to the user as a CONTENT POLICY VIOLATION.
# Nothing unsafe was requested — the prompt simply is not an image
# description. Ambiguity is a request problem, not a safety finding, and
# saying otherwise tells users they did something wrong when they did not.
#
############################################################

"""Unit tests separating 'unusable prompt' from 'policy violation'."""

import pytest

import backend.app.services.image_policy as ip


# --------------------------------------------------------------------------
# The judge is instructed to distinguish the two
# --------------------------------------------------------------------------


def test_system_prompt_offers_unclear_and_reserves_fail_for_policy():
    text = ip._JUDGE_SYSTEM_PROMPT
    assert "UNCLEAR" in text
    low = text.lower()
    # FAIL must be reserved for real violations...
    assert "use fail only when" in low
    # ...and ambiguity must route to UNCLEAR rather than FAIL.
    assert "not fail" in low or "— not fail" in low
    assert "unclear means" in low


def test_unclear_is_not_a_safety_bypass():
    """UNCLEAR must still deny generation; it only changes how we report it."""
    v = ip._parse_verdict('{"verdict": "UNCLEAR", "reason": "no visual content"}', "judge")
    assert v.passed is False
    assert v.is_unclear is True


# --------------------------------------------------------------------------
# Verdict parsing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,passed,category",
    [
        ('{"verdict": "PASS", "reason": "fine"}', True, ip.CATEGORY_OK),
        ('{"verdict": "FAIL", "reason": "real person"}', False, ip.CATEGORY_POLICY),
        ('{"verdict": "UNCLEAR", "reason": "not an image"}', False, ip.CATEGORY_UNCLEAR),
        # Unknown verdicts stay fail-closed AND are treated as policy denials,
        # never silently downgraded to the softer "unclear" message.
        ('{"verdict": "MAYBE", "reason": "?"}', False, ip.CATEGORY_POLICY),
        ("not json at all", False, ip.CATEGORY_POLICY),
    ],
)
def test_parse_verdict_categories(raw, passed, category):
    v = ip._parse_verdict(raw, "judge")
    assert v.passed is passed
    assert v.category == category


def test_judge_outage_is_a_policy_denial_not_unclear():
    """Fail-closed on outage must not present as a mere 'unclear prompt'."""
    v = ip.PolicyVerdict(False, "Policy check unavailable", "", "", ip.CATEGORY_POLICY)
    assert v.is_unclear is False


def test_category_defaults_preserve_existing_call_sites():
    """PolicyVerdict is constructed without a category in several places."""
    assert ip.PolicyVerdict(True, "ok").category == ip.CATEGORY_OK
    assert ip.PolicyVerdict(False, "bad").category == ip.CATEGORY_POLICY
    assert ip.PolicyVerdict(False, "bad").is_unclear is False


def test_to_dict_carries_category_for_the_audit_trail():
    d = ip._parse_verdict('{"verdict": "UNCLEAR", "reason": "x"}', "judge").to_dict()
    assert d["category"] == ip.CATEGORY_UNCLEAR
    assert d["passed"] is False


# --------------------------------------------------------------------------
# Edit-intent hint — advisory wording only, never a denial
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt",
    [
        "the same image, unchanged",       # rhendricks, 2026-09-01 (x2)
        "Make it more red",
        "make it red slime",
        "Make this shredded",
        "Add a mustache to this image",
        "make him sit on a park bench",
        "same, but bigger",
    ],
)
def test_edit_intent_detected_for_real_followups(prompt):
    assert ip.looks_like_edit_instruction(prompt) is True


@pytest.mark.parametrize(
    "prompt",
    [
        "Oil painting in the heroic-romantic fantasy tradition of Frank Frazetta",
        "A goddess, a vaguely feminine form who radiates with light",
        "a halfling looks away from his campfire in a savannah",
        "a red barn at sunset",
        "make a poster about potato sprout suppression",
        "",
    ],
)
def test_edit_intent_not_triggered_by_generation_prompts(prompt):
    assert ip.looks_like_edit_instruction(prompt) is False


# --------------------------------------------------------------------------
# Both call sites must branch on the category
# --------------------------------------------------------------------------


def _source(*parts):
    import pathlib

    return (pathlib.Path(__file__).resolve().parents[2].joinpath(*parts)).read_text()


def test_api_path_reports_unclear_as_a_request_error():
    src = _source("api", "v1_openai.py")
    body = src[src.index("if not policy_verdict.passed:"):]
    body = body[: body.index("# Enforce guardrails")]
    assert "is_unclear" in body
    assert '"prompt_unclear"' in body
    # The unclear branch must NOT be labelled a content policy violation.
    unclear_branch = body[body.index("if unclear:\n                raise HTTPException"):]
    unclear_branch = unclear_branch[: unclear_branch.index("raise HTTPException", 40)]
    assert "content_policy_violation" not in unclear_branch
    assert "invalid_request_error" in unclear_branch


def test_api_path_stores_a_distinct_error_code():
    src = _source("api", "v1_openai.py")
    assert 'denied_req.error_code = "prompt_unclear"' in src
    # The policy path is unchanged.
    assert 'denied_req.error_code = "policy_violation"' in src


def test_dashboard_path_reports_unclear_as_a_request_error():
    src = _source("dashboard", "images.py")
    body = src[src.index("if not verdict.passed:"):]
    body = body[: body.index("# Attach verdict for audit trail")]
    assert "is_unclear" in body
    assert '"prompt_unclear"' in body
    assert "looks_like_edit_instruction" in body


def test_unclear_message_points_at_the_edits_endpoint():
    """The whole point: tell the caller what to do instead."""
    src = _source("api", "v1_openai.py")
    assert "/v1/images/edits" in src
    # (the sentence wraps across source lines, so match a single-line fragment)
    assert "remember previous images" in src
