"""Unit tests for IssuerAgent LLM softening in the ambiguity band.

The deterministic score → decision mapping is pinned in tests/test_issuer.py.
These tests cover the LLM path specifically: verdict routing, fallback on
errors, and the ``used_llm_softening`` telemetry flag. An injected stub
decider stands in for the real provider chain so the tests are deterministic
and run offline.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from scenarios.issuer_model import (
    ROUND1_MIDPOINT_FALLBACK,
    IssuerAgent,
    IssuerDecision,
)
from scenarios.llm_softening import _parse_decision, build_default_llm_decider
from scenarios.simulation import CaseProgress, get_task


_TASK = get_task("goods_not_received_easy")
_CASE = _TASK.cases[0]

# Synthetic variant that can land in the ambiguity band deterministically.
# Stripping helpful-only ids down to a single entry means attaching both
# required ids + that single helpful-only id lands score at 0.6 — above the
# 0.55 midpoint but below the 0.70 accept bar.
_BAND_CASE = replace(
    _CASE,
    case_id="CB-E1-BAND",
    helpful_evidence_ids=("E1-SUPPORT-ACK",),
)


def _band_progress() -> CaseProgress:
    """Packet that scores 0.6 on ``_BAND_CASE`` — squarely in the ambiguity band."""
    p = CaseProgress()
    p.attached_evidence_ids = [
        "E1-ORDER-CONF",
        "E1-DELIVERY-SCAN",
        "E1-SUPPORT-ACK",
    ]
    return p


def test_softening_off_uses_midpoint_fallback():
    """Without softening, the midpoint rule still governs the band."""
    agent = IssuerAgent(enable_llm_softening=False)
    review = agent.decide_review(_BAND_CASE, _band_progress(), round_number=1)
    assert review.used_llm_softening is False
    assert review.decision == IssuerDecision.ACCEPT
    assert review.evidence_strength_score >= ROUND1_MIDPOINT_FALLBACK


def test_softening_accepts_when_llm_accepts():
    """LLM ACCEPT overrides the deterministic midpoint."""
    decider = lambda case, progress, score: IssuerDecision.ACCEPT
    agent = IssuerAgent(enable_llm_softening=True, llm_decider=decider)
    review = agent.decide_review(_BAND_CASE, _band_progress(), round_number=1)
    assert review.decision == IssuerDecision.ACCEPT
    assert review.used_llm_softening is True


def test_softening_rejects_when_llm_rejects():
    """LLM REQUEST_MORE_EVIDENCE also flows through."""
    decider = lambda case, progress, score: IssuerDecision.REQUEST_MORE_EVIDENCE
    agent = IssuerAgent(enable_llm_softening=True, llm_decider=decider)
    review = agent.decide_review(_BAND_CASE, _band_progress(), round_number=1)
    assert review.decision == IssuerDecision.REQUEST_MORE_EVIDENCE
    assert review.used_llm_softening is True


def test_softening_falls_back_on_none():
    """Decider returning None falls back to the deterministic midpoint."""
    decider = lambda case, progress, score: None
    agent = IssuerAgent(enable_llm_softening=True, llm_decider=decider)
    review = agent.decide_review(_BAND_CASE, _band_progress(), round_number=1)
    assert review.decision == IssuerDecision.ACCEPT
    assert review.used_llm_softening is False


def test_softening_falls_back_on_exception():
    """Decider raising an exception must not break the Issuer."""
    def _boom(case, progress, score):
        raise RuntimeError("provider down")

    agent = IssuerAgent(enable_llm_softening=True, llm_decider=_boom)
    review = agent.decide_review(_BAND_CASE, _band_progress(), round_number=1)
    assert review.used_llm_softening is False
    assert review.decision in {
        IssuerDecision.ACCEPT,
        IssuerDecision.REQUEST_MORE_EVIDENCE,
    }


def test_softening_not_invoked_outside_band():
    """Clear-cut scores (>=0.7 or <=0.4) bypass the LLM entirely."""
    calls: list[Any] = []

    def _tracking(case, progress, score):
        calls.append(score)
        return IssuerDecision.ACCEPT

    agent = IssuerAgent(enable_llm_softening=True, llm_decider=_tracking)

    strong = CaseProgress()
    strong.attached_evidence_ids = ["E1-ORDER-CONF", "E1-DELIVERY-SCAN"]
    agent.decide_review(_CASE, strong, round_number=1)

    empty = CaseProgress()
    agent.decide_review(_CASE, empty, round_number=1)

    assert calls == []


def test_softening_not_invoked_in_later_rounds():
    """Pre-arbitration review stays fully deterministic."""
    calls: list[int] = []

    def _tracking(case, progress, score):
        calls.append(1)
        return IssuerDecision.ACCEPT

    agent = IssuerAgent(enable_llm_softening=True, llm_decider=_tracking)
    progress = _band_progress()
    agent.decide_review(_CASE, progress, round_number=2)
    assert calls == []


def test_parse_decision_accepts_both_shapes():
    assert _parse_decision('{"decision": "accept"}') is IssuerDecision.ACCEPT
    assert (
        _parse_decision('{"decision": "request_more_evidence"}')
        is IssuerDecision.REQUEST_MORE_EVIDENCE
    )
    assert _parse_decision('{"decision": "ESCALATE"}') is None
    assert _parse_decision("not json") is None
    assert _parse_decision(json.dumps({})) is None


def test_default_decider_returns_none_without_api_keys(monkeypatch):
    """Offline environments must get a deterministic fallback."""
    for var in ("OPENROUTER_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    decider = build_default_llm_decider()
    assert decider(_BAND_CASE, _band_progress(), 0.5) is None


def test_default_decider_returns_none_when_all_providers_fail(monkeypatch):
    """Chain walks every configured provider and returns None on total failure."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "sk-google")
    monkeypatch.setenv("GROQ_API_KEY", "sk-groq")

    from scenarios import llm_softening

    calls: list[str] = []

    def _failing_try(*, base_url, api_key, model, case, progress, score):
        calls.append(base_url)
        return None

    monkeypatch.setattr(llm_softening, "_try_provider", _failing_try)

    decider = llm_softening.build_default_llm_decider()
    assert decider(_BAND_CASE, _band_progress(), 0.5) is None
    assert len(calls) == 3


def test_default_decider_returns_first_success(monkeypatch):
    """Chain short-circuits as soon as a provider returns a decision."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "sk-google")

    from scenarios import llm_softening

    calls: list[str] = []

    def _succeeding_try(*, base_url, api_key, model, case, progress, score):
        calls.append(base_url)
        return IssuerDecision.ACCEPT

    monkeypatch.setattr(llm_softening, "_try_provider", _succeeding_try)

    decider = llm_softening.build_default_llm_decider()
    assert decider(_BAND_CASE, _band_progress(), 0.5) is IssuerDecision.ACCEPT
    assert len(calls) == 1
