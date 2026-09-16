from __future__ import annotations

from app.graph import _validated_next


def test_a_guardrail_block_finishes_regardless_of_everything_else():
    assert (
        _validated_next(
            guardrail_checked=True,
            guardrail_allowed=False,
            has_draft=True,
            reviewed=True,
            verdict="approved",
            revision_count=0,
            proposed="researcher",
        )
        == "finish"
    )


def test_pre_draft_pre_guardrail_honors_a_proposed_guardrail_route():
    assert (
        _validated_next(
            guardrail_checked=False,
            guardrail_allowed=True,
            has_draft=False,
            reviewed=False,
            verdict="",
            revision_count=0,
            proposed="guardrail",
        )
        == "guardrail"
    )


def test_pre_draft_pre_guardrail_honors_a_proposed_researcher_route():
    assert (
        _validated_next(
            guardrail_checked=False,
            guardrail_allowed=True,
            has_draft=False,
            reviewed=False,
            verdict="",
            revision_count=0,
            proposed="researcher",
        )
        == "researcher"
    )


def test_pre_draft_pre_guardrail_rejects_an_invalid_proposal():
    assert (
        _validated_next(
            guardrail_checked=False,
            guardrail_allowed=True,
            has_draft=False,
            reviewed=False,
            verdict="",
            revision_count=0,
            proposed="critic",
        )
        == "researcher"
    )


def test_guardrail_already_cleared_still_needs_a_draft():
    assert (
        _validated_next(
            guardrail_checked=True,
            guardrail_allowed=True,
            has_draft=False,
            reviewed=False,
            verdict="",
            revision_count=0,
            proposed="finish",
        )
        == "researcher"
    )


def test_an_unreviewed_draft_always_goes_to_critic():
    assert (
        _validated_next(
            guardrail_checked=True,
            guardrail_allowed=True,
            has_draft=True,
            reviewed=False,
            verdict="",
            revision_count=0,
            proposed="finish",
        )
        == "critic"
    )


def test_the_revision_cap_forces_finish_regardless_of_proposal():
    assert (
        _validated_next(
            guardrail_checked=True,
            guardrail_allowed=True,
            has_draft=True,
            reviewed=True,
            verdict="needs_revision",
            revision_count=2,
            proposed="researcher",
        )
        == "finish"
    )


def test_needs_revision_under_the_cap_honors_a_proposed_revise():
    assert (
        _validated_next(
            guardrail_checked=True,
            guardrail_allowed=True,
            has_draft=True,
            reviewed=True,
            verdict="needs_revision",
            revision_count=1,
            proposed="researcher",
        )
        == "researcher"
    )


def test_needs_revision_under_the_cap_honors_a_proposed_finish():
    """The Supervisor may judge a minor critique good enough to ship as-is."""
    assert (
        _validated_next(
            guardrail_checked=True,
            guardrail_allowed=True,
            has_draft=True,
            reviewed=True,
            verdict="needs_revision",
            revision_count=1,
            proposed="finish",
        )
        == "finish"
    )


def test_needs_revision_under_the_cap_rejects_an_invalid_proposal():
    assert (
        _validated_next(
            guardrail_checked=True,
            guardrail_allowed=True,
            has_draft=True,
            reviewed=True,
            verdict="needs_revision",
            revision_count=1,
            proposed="guardrail",
        )
        == "researcher"
    )


def test_an_approved_draft_always_finishes():
    assert (
        _validated_next(
            guardrail_checked=True,
            guardrail_allowed=True,
            has_draft=True,
            reviewed=True,
            verdict="approved",
            revision_count=0,
            proposed="researcher",
        )
        == "finish"
    )
