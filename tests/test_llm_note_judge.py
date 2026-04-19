"""Unit tests for the optional LLM-backed note judge.

The deterministic note scorer is pinned via ``test_grader.py`` and the
``EvidenceQuality`` / ``PacketValidity`` rubric tests. These tests cover the
LLM-backed wrapper specifically: opt-in activation through env var,
fallback on parse failure, and that the rubric still respects the
contest-only gate.
"""

from __future__ import annotations

import os
from typing import Any

from evaluation import llm_note_judge
from evaluation.llm_note_judge import LLMNoteJudgeRubric, llm_score_note
from evaluation.rubrics import (
    CASE_DIMENSION_NAMES,
    CaseRubric,
    GradingContext,
    NoteQualityRubric,
)
from scenarios.simulation import CaseProgress, get_task


_TASK = get_task("goods_not_received_easy")
_CASE = _TASK.cases[0]


def _strong_progress() -> CaseProgress:
    p = CaseProgress()
    p.attached_evidence_ids = [
        "E1-ORDER-CONF",
        "E1-DELIVERY-SCAN",
        "E1-SIGNATURE",
    ]
    p.final_resolution = "contest"
    p.representment_note = (
        "Order confirmation and carrier delivery confirmation establish "
        "fulfillment per policy."
    )
    return p


def _ctx(progress: CaseProgress | None = None) -> GradingContext:
    return GradingContext(case=_CASE, progress=progress or _strong_progress(), step_count=5)


def test_default_rubric_is_deterministic_when_flag_unset(monkeypatch):
    """Without USE_LLM_NOTE_JUDGE the case rubric uses the deterministic scorer."""
    monkeypatch.delenv("USE_LLM_NOTE_JUDGE", raising=False)
    rubric = CaseRubric()
    note_idx = CASE_DIMENSION_NAMES.index("note_quality")
    note_child = rubric.aggregator._rubric_list[note_idx]
    assert isinstance(note_child, NoteQualityRubric)


def test_rubric_swaps_to_llm_judge_when_flag_set(monkeypatch):
    """With USE_LLM_NOTE_JUDGE=1 the case rubric installs the LLM-backed one."""
    monkeypatch.setenv("USE_LLM_NOTE_JUDGE", "1")
    rubric = CaseRubric()
    note_idx = CASE_DIMENSION_NAMES.index("note_quality")
    note_child = rubric.aggregator._rubric_list[note_idx]
    assert isinstance(note_child, LLMNoteJudgeRubric)


def test_llm_judge_falls_back_when_provider_returns_none(monkeypatch):
    """No API keys → llm_score_note returns None → fallback to deterministic."""
    monkeypatch.setattr(llm_note_judge, "llm_score_note", lambda case, progress: None)
    rubric = LLMNoteJudgeRubric()
    score = rubric(_ctx(), None)
    assert 0.0 < score <= 1.0


def test_llm_judge_uses_provider_score_when_available(monkeypatch):
    """When the provider returns a score, the rubric returns it as-is."""
    monkeypatch.setattr(llm_note_judge, "llm_score_note", lambda case, progress: 0.42)
    rubric = LLMNoteJudgeRubric()
    score = rubric(_ctx(), None)
    assert score == 0.42


def test_llm_judge_returns_zero_when_not_contest(monkeypatch):
    """Non-contest cases skip note grading entirely."""
    monkeypatch.setattr(llm_note_judge, "llm_score_note", lambda case, progress: 0.99)
    progress = CaseProgress()
    progress.final_resolution = "accept_chargeback"
    progress.representment_note = "doesn't matter"
    rubric = LLMNoteJudgeRubric()
    assert rubric(_ctx(progress), None) == 0.0


def test_llm_judge_returns_zero_when_no_note(monkeypatch):
    """Empty note → zero, regardless of LLM availability."""
    monkeypatch.setattr(
        llm_note_judge, "llm_score_note", lambda case, progress: 0.99
    )
    progress = _strong_progress()
    progress.representment_note = ""
    rubric = LLMNoteJudgeRubric()
    assert rubric(_ctx(progress), None) == 0.0


def test_provider_chain_returns_none_when_no_keys(monkeypatch):
    """Empty env → walks the chain without ever calling OpenAI → None."""
    for var in ("OPENROUTER_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert llm_score_note(_CASE, _strong_progress()) is None
