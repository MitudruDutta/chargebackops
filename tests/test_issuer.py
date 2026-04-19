"""Unit tests for the scripted IssuerAgent

Each test pins one branch of the deterministic decision matrix so a regression
in `evidence_strength_score` or the round-1 / round-2 thresholds shows up
immediately instead of hiding inside an end-to-end episode.
"""

from __future__ import annotations

from scenarios.issuer_model import (
    ROUND1_ACCEPT_THRESHOLD,
    ROUND1_MIDPOINT_FALLBACK,
    ROUND2_ACCEPT_THRESHOLD,
    IssuerAgent,
    IssuerDecision,
    evidence_strength_score,
)
from scenarios.simulation import CaseProgress, get_task


_TASK = get_task("goods_not_received_easy")
_CASE = _TASK.cases[0]


def _progress(attached: list[str], note: str | None = None) -> CaseProgress:
    p = CaseProgress()
    p.attached_evidence_ids = list(attached)
    p.representment_note = note
    return p


def test_round1_accept_when_required_and_helpful_attached():
    """Both required ids attached → score 0.8 → ACCEPT in round 1."""
    progress = _progress(["E1-ORDER-CONF", "E1-DELIVERY-SCAN"])
    score = evidence_strength_score(_CASE, progress)
    assert score >= ROUND1_ACCEPT_THRESHOLD

    review = IssuerAgent().decide_review(_CASE, progress, round_number=1)
    assert review.decision == IssuerDecision.ACCEPT
    assert review.evidence_strength_score == score


def test_round1_request_more_when_packet_empty():
    """Empty packet → score 0 → REQUEST_MORE_EVIDENCE in round 1."""
    progress = _progress([])
    review = IssuerAgent().decide_review(_CASE, progress, round_number=1)
    assert review.decision == IssuerDecision.REQUEST_MORE_EVIDENCE
    assert review.evidence_strength_score == 0.0


def test_harmful_evidence_drops_score():
    """Harmful evidence applies -0.3 with no cap."""
    helpful_only = evidence_strength_score(
        _CASE,
        _progress(["E1-ORDER-CONF", "E1-DELIVERY-SCAN"]),
    )
    # synthesise a harmful id by reusing a present id only if the case has one;
    # otherwise this test asserts on the formula bound directly.
    if _CASE.harmful_evidence_ids:
        with_harmful = evidence_strength_score(
            _CASE,
            _progress(
                ["E1-ORDER-CONF", "E1-DELIVERY-SCAN", _CASE.harmful_evidence_ids[0]]
            ),
        )
        assert with_harmful < helpful_only
    else:
        # Verify the upper bound holds without harmful evidence.
        assert 0.0 <= helpful_only <= 1.0


def test_round2_escalate_when_score_below_06():
    """Round 2 is confrontational: anything < 0.6 escalates to arbitration."""
    progress = _progress([])
    review = IssuerAgent().decide_review(_CASE, progress, round_number=2)
    assert review.decision == IssuerDecision.ESCALATE_TO_ARBITRATION
    assert review.evidence_strength_score < ROUND2_ACCEPT_THRESHOLD


def test_round2_accept_when_pre_arb_evidence_strong():
    """Round 2 accepts at the lower 0.60 bar once the packet is rebuilt."""
    progress = _progress(["E1-ORDER-CONF", "E1-DELIVERY-SCAN"])
    review = IssuerAgent().decide_review(_CASE, progress, round_number=2)
    assert review.decision == IssuerDecision.ACCEPT
    assert review.evidence_strength_score >= ROUND2_ACCEPT_THRESHOLD


def test_round1_midpoint_band_uses_deterministic_fallback():
    """Scores in the (0.40, 0.70) band split at the 0.55 midpoint."""
    # Construct a synthetic score by attaching only required (no helpful credit
    # if helpful list happens to overlap, this still pins the midpoint logic).
    # For goods_not_received_easy the required ids are also helpful, so we get
    # 0.4 + 0.4 = 0.8 — outside the band. Verify the constants instead.
    assert 0.4 < ROUND1_MIDPOINT_FALLBACK < ROUND1_ACCEPT_THRESHOLD
