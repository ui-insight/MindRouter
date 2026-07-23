############################################################
#
# mindrouter - unit tests for edit-aware image policy judging
#
# Regression: img2img prompts like "put glasses on this man" were FAILED by the
# text-only policy judge as "ambiguous / no image provided". For edits the judge
# must be told the prompt targets a user-supplied reference image so deictic
# references are expected, not grounds for an ambiguity FAIL.
#
############################################################

"""Unit tests for is_edit-aware content-policy evaluation."""

from unittest.mock import AsyncMock, patch

import pytest

import backend.app.services.image_policy as ip


def test_edit_template_defuses_ambiguity():
    content = ip._JUDGE_USER_TEMPLATE_EDIT.format(
        prompt="put glasses on this man", edit_note=ip._JUDGE_EDIT_NOTE
    )
    low = content.lower()
    assert "edit" in low
    assert "reference image" in low
    # Must explicitly tell the judge not to FAIL for ambiguity / missing image.
    assert "ambiguous" in low
    assert "put glasses on this man" in content


def test_plain_template_has_no_edit_note():
    content = ip._JUDGE_USER_TEMPLATE.format(prompt="a cat on a table")
    assert "reference image" not in content.lower()


@pytest.mark.asyncio
async def test_evaluate_prompt_forwards_is_edit_true():
    stub = AsyncMock(return_value=ip.PolicyVerdict(True, "ok", "judge", ""))
    with patch.object(ip, "_call_judge", new=stub):
        await ip.evaluate_prompt("put glasses on this man", "policy", "judge", "", is_edit=True)
    assert stub.await_args.kwargs.get("is_edit") is True


@pytest.mark.asyncio
async def test_evaluate_prompt_defaults_is_edit_false():
    stub = AsyncMock(return_value=ip.PolicyVerdict(True, "ok", "judge", ""))
    with patch.object(ip, "_call_judge", new=stub):
        await ip.evaluate_prompt("a cat", "policy", "judge", "")
    assert stub.await_args.kwargs.get("is_edit") is False


@pytest.mark.asyncio
async def test_no_policy_short_circuits_pass_even_for_edit():
    verdict = await ip.evaluate_prompt("put glasses on this man", "", "judge", "", is_edit=True)
    assert verdict.passed
