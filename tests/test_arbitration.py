"""Unit tests for scenarios.arbitration 

These tests pin the three bands of the arbitration ruling (merchant-wins,
issuer-wins, deterministic coin flip) and the $250-per-side fee accounting so
a regression in the terminal-round math shows up before the end-to-end env
tests do.
"""

from __future__ import annotations

from dataclasses import replace

from scenarios.arbitration import (
    ARB_FEE_PER_SIDE,
    ARB_ISSUER_WIN_THRESHOLD,
    ARB_MERCHANT_WIN_THRESHOLD,
    ArbitrationOutcome,
    _coin_flip_merchant_wins,
    arbitration_ruling,
)
from scenarios.simulation import CaseProgress, get_task


_TASK = get_task("goods_not_received_easy")
_CASE = _TASK.cases[0]


def _progress(attached: list[str]) -> CaseProgress:
    p = CaseProgress()
    p.attached_evidence_ids = list(attached)
    return p


def test_merchant_wins_on_strong_packet():
    """Required + 2 helpful → score 0.8 clears the 0.65 bar → MERCHANT_WINS."""
    progress = _progress(
        ["E1-ORDER-CONF", "E1-DELIVERY-SCAN", "E1-SIGNATURE", "E1-SUPPORT-ACK"]
    )
    ruling = arbitration_ruling(_CASE, progress)
    assert ruling.evidence_strength_score >= ARB_MERCHANT_WIN_THRESHOLD
    assert ruling.outcome == ArbitrationOutcome.MERCHANT_WINS
    assert ruling.arb_fee_per_side == ARB_FEE_PER_SIDE
    assert ruling.merchant_net_pnl == _CASE.amount - ARB_FEE_PER_SIDE


def test_issuer_wins_on_empty_packet():
    """Score 0 sits below the 0.35 floor → ISSUER_WINS, merchant eats amount + fee."""
    progress = _progress([])
    ruling = arbitration_ruling(_CASE, progress)
    assert ruling.evidence_strength_score <= ARB_ISSUER_WIN_THRESHOLD
    assert ruling.outcome == ArbitrationOutcome.ISSUER_WINS
    assert ruling.merchant_net_pnl == -_CASE.amount - ARB_FEE_PER_SIDE


def test_ambiguity_band_uses_deterministic_coin_flip():
    """Scores in (0.35, 0.65) map to a case_id-keyed coin flip — reproducible."""
    # Two helpful-only evidence ids → 0.4 band score, no required subset.
    progress = _progress(["E1-SIGNATURE", "E1-SUPPORT-ACK"])
    r1 = arbitration_ruling(_CASE, progress)
    r2 = arbitration_ruling(_CASE, progress)
    assert r1.outcome == r2.outcome
    assert ARB_ISSUER_WIN_THRESHOLD < r1.evidence_strength_score < ARB_MERCHANT_WIN_THRESHOLD
    expected = (
        ArbitrationOutcome.MERCHANT_WINS
        if _coin_flip_merchant_wins(_CASE.case_id)
        else ArbitrationOutcome.ISSUER_WINS
    )
    assert r1.outcome == expected


def test_coin_flip_varies_across_case_ids():
    """Changing only the case_id must change the coin-flip answer for some cases.

    If every case_id hashed to the same parity, the ambiguity band wouldn't
    actually be 50/50 across the benchmark — this test guards against that.
    """
    flips = {_coin_flip_merchant_wins(f"CB-TEST-{i}") for i in range(20)}
    assert flips == {True, False}


def test_ruling_is_pure():
    """Same inputs, same outputs — required for reproducible benchmarks."""
    progress = _progress(
        ["E1-ORDER-CONF", "E1-DELIVERY-SCAN", "E1-SIGNATURE", "E1-SUPPORT-ACK"]
    )
    r1 = arbitration_ruling(_CASE, progress)
    r2 = arbitration_ruling(_CASE, progress)
    assert r1 == r2

    # A second case_id clone with identical evidence should give the same
    # MERCHANT_WINS outcome (score is above 0.65, so no coin-flip involved).
    cloned = replace(_CASE, case_id="CB-CLONE-1")
    r3 = arbitration_ruling(cloned, progress)
    assert r3.outcome == r1.outcome
    assert r3.merchant_net_pnl == r1.merchant_net_pnl
