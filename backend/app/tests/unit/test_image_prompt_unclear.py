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


# --------------------------------------------------------------------------
# /v1/moderations must not conflate the two either
# --------------------------------------------------------------------------


def test_moderations_result_exposes_the_category():
    """VandalChat uses /v1/moderations as a pre-check; `flagged` alone cannot
    tell "unsafe" from "not an image description"."""
    from backend.app.api.v1_openai import _moderation_result

    unclear = _moderation_result(True, "not an image", "judge", ip.CATEGORY_UNCLEAR)
    violation = _moderation_result(True, "real person", "judge", ip.CATEGORY_POLICY)

    # `flagged` keeps its OpenAI meaning so existing gates behave unchanged...
    assert unclear["flagged"] is True and violation["flagged"] is True
    # ...and the category is what tells them apart.
    assert unclear["policy_category"] == ip.CATEGORY_UNCLEAR
    assert violation["policy_category"] == ip.CATEGORY_POLICY


def test_moderations_no_policy_path_is_ok_category():
    from backend.app.api.v1_openai import _moderation_result

    r = _moderation_result(False, "No policy configured", "", ip.CATEGORY_OK)
    assert r["flagged"] is False
    assert r["policy_category"] == ip.CATEGORY_OK


def test_moderations_handler_forwards_the_verdict_category():
    src = _source("api", "v1_openai.py")
    assert "_moderation_result(not v.passed, v.reason, v.judge_model, v.category)" in src
    # the no-policy early return is explicit, not relying on the default
    assert "CATEGORY_OK)" in src


def test_openai_shape_is_preserved():
    """The extension field must not disturb the fields the SDK requires."""
    from backend.app.api.v1_openai import _moderation_result, _MODERATION_CATEGORIES

    r = _moderation_result(True, "x", "j", ip.CATEGORY_UNCLEAR)
    for key in ("flagged", "categories", "category_scores", "category_applied_input_types"):
        assert key in r
    assert set(r["categories"]) == set(_MODERATION_CATEGORIES)
    assert all(v is False for v in r["categories"].values())


# --------------------------------------------------------------------------
# Public documentation must describe what the API actually does
# --------------------------------------------------------------------------


def _docs():
    import pathlib

    return (
        pathlib.Path(__file__).resolve().parents[2]
        / "dashboard" / "templates" / "public" / "documentation.html"
    ).read_text()


def test_docs_document_both_denial_envelopes():
    d = _docs()
    assert "prompt_unclear" in d
    assert "policy_violation" in d
    assert "invalid_request_error" in d
    # and say plainly that the second is not a safety finding
    assert "not</strong> a safety finding" in d or "not a safety finding" in d


def test_docs_document_the_moderations_category_field():
    assert "policy_category" in _docs()


def test_docs_do_not_duplicate_the_image_edits_section():
    """Regression: a second edits section/table row was added alongside the
    pre-existing #image-edits one."""
    d = _docs()
    assert d.count('id="image-edits"') == 1
    assert 'id="image-editing"' not in d
    assert d.count("<code>/v1/images/edits</code></td>") == 1


# --------------------------------------------------------------------------
# A genuinely-unusable EDIT prompt gets edit-appropriate advice
# --------------------------------------------------------------------------


def test_api_unclear_branch_has_an_edit_specific_message():
    """With a source image attached, "describe the complete image you want" is
    nonsense advice — the caller wants to change the image they sent."""
    src = _source("api", "v1_openai.py")
    body = src[src.index("if not policy_verdict.passed:"):]
    body = body[: body.index("# Enforce guardrails")]
    # the edit case is decided by the presence of a reference image
    assert "if images_b64:" in body
    assert "Describe the change you would like to make" in body
    # and the no-image cases still exist
    assert "looks_like_edit_instruction" in body
    assert "does not describe an image to generate" in body


def test_dashboard_unclear_branch_has_an_edit_specific_message():
    src = _source("dashboard", "images.py")
    body = src[src.index("if not verdict.passed:"):]
    body = body[: body.index("# Attach verdict for audit trail")]
    assert 'if body.get("image"):' in body
    assert "Describe the change you would like to make" in body


def test_edit_guidance_still_uses_the_prompt_unclear_code():
    """Wording changed; the machine-readable contract did not."""
    src = _source("api", "v1_openai.py")
    assert '"code": "prompt_unclear"' in src
    assert 'denied_req.error_code = "prompt_unclear"' in src
