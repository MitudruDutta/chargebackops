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


def test_representment_accepted_when_required_and_helpful_attached():
    """Required + 2 helpful attached → score 0.8 → ACCEPT on first review."""
    progress = _progress(
        ["E1-ORDER-CONF", "E1-DELIVERY-SCAN", "E1-SIGNATURE", "E1-SUPPORT-ACK"]
    )
    score = evidence_strength_score(_CASE, progress)
    assert score >= ROUND1_ACCEPT_THRESHOLD

    review = IssuerAgent().decide_review(_CASE, progress, round_number=1)
    assert review.decision == IssuerDecision.ACCEPT
    assert review.evidence_strength_score == score


def test_representment_rejected_when_packet_empty():
    """Empty packet → score 0 → REQUEST_MORE_EVIDENCE on first review."""
    progress = _progress([])
    review = IssuerAgent().decide_review(_CASE, progress, round_number=1)
    assert review.decision == IssuerDecision.REQUEST_MORE_EVIDENCE
    assert review.evidence_strength_score == 0.0


def test_harmful_evidence_drops_score():
    """Harmful evidence applies -0.3 per piece, no cap. Verified on a case
    that actually carries harmful items so the assertion is not vacuous."""
    fraud_case = get_task("fraud_signal_ambiguity").cases[0]
    assert fraud_case.harmful_evidence_ids, "fixture must expose harmful evidence"

    base_attached = list(fraud_case.required_evidence_ids) + list(
        fraud_case.helpful_evidence_ids[:1]
    )
    clean_score = evidence_strength_score(fraud_case, _progress(base_attached))
    one_harmful = evidence_strength_score(
        fraud_case,
        _progress(base_attached + [fraud_case.harmful_evidence_ids[0]]),
    )
    two_harmful = evidence_strength_score(
        fraud_case,
        _progress(base_attached + list(fraud_case.harmful_evidence_ids[:2])),
    )
    assert one_harmful == max(0.0, clean_score - 0.3)
    assert two_harmful == max(0.0, clean_score - 0.6)


def test_pre_arb_escalates_when_score_below_06():
    """Pre-arb review is confrontational: anything < 0.6 escalates to arbitration."""
    progress = _progress([])
    review = IssuerAgent().decide_review(_CASE, progress, round_number=2)
    assert review.decision == IssuerDecision.ESCALATE_TO_ARBITRATION
    assert review.evidence_strength_score < ROUND2_ACCEPT_THRESHOLD


def test_pre_arb_accepted_when_evidence_strong():
    """Pre-arb review accepts at the lower 0.60 bar once the packet is rebuilt."""
    progress = _progress(
        ["E1-ORDER-CONF", "E1-DELIVERY-SCAN", "E1-SIGNATURE", "E1-SUPPORT-ACK"]
    )
    review = IssuerAgent().decide_review(_CASE, progress, round_number=2)
    assert review.decision == IssuerDecision.ACCEPT
    assert review.evidence_strength_score >= ROUND2_ACCEPT_THRESHOLD


def test_midpoint_band_uses_deterministic_fallback():
    """Required + 1 helpful → score 0.6, lands in ambiguity band, accepts at midpoint."""
    progress = _progress(["E1-ORDER-CONF", "E1-DELIVERY-SCAN", "E1-SIGNATURE"])
    score = evidence_strength_score(_CASE, progress)
    assert 0.4 < score < ROUND1_ACCEPT_THRESHOLD
    assert score >= ROUND1_MIDPOINT_FALLBACK
    review = IssuerAgent().decide_review(_CASE, progress, round_number=1)
    assert review.decision == IssuerDecision.ACCEPT
