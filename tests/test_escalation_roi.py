"""Unit tests for EscalationROIRubric — escalate-vs-concede EV scoring.

The rubric encodes the economic rule that escalating to network arbitration
is only rational when ``P(win) * dispute_amount > arb_fee``. These tests pin
each branch of that decision so a regression in the fee, thresholds, or
probability mapping is caught immediately.
"""

from __future__ import annotations

from dataclasses import replace

from evaluation.rubrics import (
    CASE_DIMENSION_NAMES,
    CASE_DIMENSION_WEIGHTS,
    EscalationROIRubric,
    GradingContext,
    _probability_of_merchant_win,
)
from scenarios.arbitration import ARB_FEE_PER_SIDE
from scenarios.simulation import CaseProgress, get_task


_TASK = get_task("goods_not_received_easy")
_CASE = _TASK.cases[0]


def _progress(
    *,
    attached: list[str] | None = None,
    round_number: int = 1,
    resolution_status: str = "open",
    arbitration_outcome: str | None = None,
) -> CaseProgress:
    p = CaseProgress()
    p.attached_evidence_ids = list(attached or [])
    p.round_number = round_number
    p.resolution_status = resolution_status
    p.arbitration_outcome = arbitration_outcome
    return p


def test_weights_sum_to_one_and_match_names():
    assert abs(sum(CASE_DIMENSION_WEIGHTS) - 1.0) < 1e-6
    assert len(CASE_DIMENSION_WEIGHTS) == len(CASE_DIMENSION_NAMES) == 8
    assert "escalation_roi" in CASE_DIMENSION_NAMES


def test_probability_mapping_matches_arbitration_thresholds():
    assert _probability_of_merchant_win(0.9) == 1.0
    assert _probability_of_merchant_win(0.2) == 0.0
    assert _probability_of_merchant_win(0.5) == 0.5


def test_vacuous_full_credit_when_case_never_escalated():
    """A case that never reached pre-arbitration gets full credit by default."""
    ctx = GradingContext(case=_CASE, progress=_progress(), step_count=5)
    assert EscalationROIRubric()(ctx, None) == 1.0


def test_pre_arb_accept_is_full_credit():
    """Winning on the pre-arbitration re-submit without filing arbitration is
    the optimal path."""
    progress = _progress(
        attached=["E1-ORDER-CONF", "E1-DELIVERY-SCAN", "E1-SIGNATURE", "E1-SUPPORT-ACK"],
        round_number=2,
        resolution_status="won_pre_arb",
    )
    ctx = GradingContext(case=_CASE, progress=progress, step_count=5)
    assert EscalationROIRubric()(ctx, None) == 1.0


def test_reward_positive_ev_escalation():
    """Strong packet → P(win)=1.0 × $129.99 > $250? No. Test with bigger amount."""
    big_case = replace(_CASE, case_id="CB-BIG", amount=1000.0)
    progress = _progress(
        attached=["E1-ORDER-CONF", "E1-DELIVERY-SCAN", "E1-SIGNATURE", "E1-SUPPORT-ACK"],
        round_number=3,
        resolution_status="won_arbitration",
        arbitration_outcome="merchant_wins",
    )
    ctx = GradingContext(case=big_case, progress=progress, step_count=5)
    # P(win)=1.0, EV = 1.0 * 1000 = 1000 > 250 → escalation was rational.
    assert EscalationROIRubric()(ctx, None) == 1.0


def test_penalise_negative_ev_escalation():
    """Weak packet + small amount → escalating was a losing bet."""
    small_case = replace(_CASE, case_id="CB-SMALL", amount=100.0)
    progress = _progress(
        attached=[],
        round_number=3,
        resolution_status="lost_arbitration",
        arbitration_outcome="issuer_wins",
    )
    ctx = GradingContext(case=small_case, progress=progress, step_count=5)
    # P(win)=0.0, EV = 0 < 250 → escalation was -EV.
    assert EscalationROIRubric()(ctx, None) == 0.0


def test_penalise_concede_when_escalation_was_positive_ev():
    """Conceding with a strong packet + large amount leaves money on the table."""
    big_case = replace(_CASE, case_id="CB-BIG2", amount=1000.0)
    progress = _progress(
        attached=["E1-ORDER-CONF", "E1-DELIVERY-SCAN", "E1-SIGNATURE", "E1-SUPPORT-ACK"],
        round_number=2,
        resolution_status="conceded_pre_arb",
    )
    ctx = GradingContext(case=big_case, progress=progress, step_count=5)
    # Strong evidence, big amount → should have escalated.
    assert EscalationROIRubric()(ctx, None) == 0.0


def test_reward_concede_when_escalation_was_negative_ev():
    """Conceding a hopeless case is the +EV decision vs burning $250."""
    small_case = replace(_CASE, case_id="CB-SMALL2", amount=100.0)
    progress = _progress(
        attached=[],
        round_number=2,
        resolution_status="conceded_pre_arb",
    )
    ctx = GradingContext(case=small_case, progress=progress, step_count=5)
    assert EscalationROIRubric()(ctx, None) == 1.0


def test_fee_threshold_is_the_pivot():
    """Sanity check the exact EV pivot: P(win)*amount vs ARB_FEE_PER_SIDE."""
    assert ARB_FEE_PER_SIDE == 250.0
    # P(win)=0.5 × $600 = 300 > 250 → escalate is rational
    mid_case = replace(_CASE, case_id="CB-MID", amount=600.0)
    # Score in ambiguity band by attaching two helpful-only ids (no required).
    progress = _progress(
        attached=["E1-SIGNATURE", "E1-SUPPORT-ACK"],
        round_number=3,
        resolution_status="won_arbitration",
        arbitration_outcome="merchant_wins",
    )
    ctx = GradingContext(case=mid_case, progress=progress, step_count=5)
    assert EscalationROIRubric()(ctx, None) == 1.0
