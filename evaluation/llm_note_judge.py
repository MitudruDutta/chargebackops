"""Optional LLM-backed note grader that wraps :class:`NoteQualityRubric`.

The deterministic ``grade_representment_note`` checks keyword coverage,
substance, evidence references, and harmful term penalties. That heuristic
is reproducible and fast, but it can't tell whether a note is genuinely
persuasive — only whether it hits the right tokens.

This module exposes :class:`LLMNoteJudgeRubric`, an opt-in wrapper that
asks an LLM to score the note on a 0.0-1.0 scale. It mirrors the provider
chain pattern in :mod:`scenarios.llm_softening`: try OpenRouter, then
Google, then Groq; on any failure or with no API key, fall back to the
deterministic scorer so offline benchmarks stay reproducible.

Wire it in by setting ``USE_LLM_NOTE_JUDGE=1`` before constructing
:class:`CaseRubric`. The wrapper is intentionally thin — it does not
override any other dimension and does not change the rubric tree shape;
``case_rubric.aggregator.rubric_6`` simply becomes a different ``Rubric``
subclass with the same forward signature.
"""

from __future__ import annotations

import json
import os
from typing import Any

from openenv.core.rubrics import Rubric

try:
    from ..scenarios.simulation import CaseProgress, InternalCase
    from .rubrics import GradingContext, _final_resolution, grade_representment_note
except ImportError:  # pragma: no cover
    from evaluation.rubrics import (
        GradingContext,
        _final_resolution,
        grade_representment_note,
    )
    from scenarios.simulation import CaseProgress, InternalCase


_PROVIDER_CHAIN: tuple[tuple[str, str, str, str], ...] = (
    (
        "openrouter",
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        "openai/gpt-oss-120b",
    ),
    (
        "google",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GOOGLE_API_KEY",
        "gemini-2.5-flash",
    ),
    (
        "groq",
        "https://api.groq.com/openai/v1",
        "GROQ_API_KEY",
        "llama-3.3-70b-versatile",
    ),
)


_SYSTEM_PROMPT = (
    "You role-play as a card-network arbitration reviewer. A merchant has "
    "submitted a representment note alongside their evidence packet. Score the "
    "note's persuasiveness on a 0.0-1.0 scale, where 1.0 means the note "
    "clearly addresses the policy requirements, references the attached "
    "evidence, and avoids harmful admissions, and 0.0 means it is empty, "
    "off-topic, or actively damages the merchant's case. "
    'Return JSON only: {"score": <float>, "rationale": "one short sentence"}.'
)


def _build_user_prompt(case: InternalCase, progress: CaseProgress) -> str:
    return json.dumps(
        {
            "reason_code": case.reason_code,
            "policy_requirements": case.policy_requirements,
            "attached_evidence_ids": list(progress.attached_evidence_ids),
            "harmful_evidence_ids": list(case.harmful_evidence_ids),
            "representment_note": progress.representment_note or "",
        }
    )


def _parse_score(text: str) -> float | None:
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    raw = data.get("score")
    try:
        score = float(raw)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, score))


def _try_provider(
    base_url: str,
    api_key: str,
    model: str,
    case: InternalCase,
    progress: CaseProgress,
) -> float | None:
    try:
        from openai import OpenAI
    except ImportError:  # pragma: no cover
        return None

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=float(os.getenv("NOTE_JUDGE_TIMEOUT_SECONDS", "8")),
            max_retries=0,
        )
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=120,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(case, progress)},
            ],
        )
    except Exception:
        return None

    try:
        content = response.choices[0].message.content or ""
    except (AttributeError, IndexError):
        return None
    return _parse_score(content)


def llm_score_note(case: InternalCase, progress: CaseProgress) -> float | None:
    """Walk the provider chain. Return None if nothing succeeded."""

    for _name, base_url, env_var, default_model in _PROVIDER_CHAIN:
        api_key = os.getenv(env_var)
        if not api_key:
            continue
        model = os.getenv("NOTE_JUDGE_MODEL", default_model)
        score = _try_provider(
            base_url=base_url,
            api_key=api_key,
            model=model,
            case=case,
            progress=progress,
        )
        if score is not None:
            return score
    return None


class LLMNoteJudgeRubric(Rubric):
    """Drop-in replacement for :class:`NoteQualityRubric` that asks an LLM.

    Falls back to :func:`grade_representment_note` whenever no provider key
    is configured, every provider errors, or the response cannot be parsed.
    The fallback path is what the deterministic baseline benchmark uses, so
    offline runs match the no-LLM scores byte-for-byte.
    """

    def forward(self, action: Any, observation: Any) -> float:
        ctx: GradingContext = action
        progress = ctx.progress
        if (
            _final_resolution(progress) != "contest"
            or not progress.representment_note
        ):
            return 0.0

        llm_score = llm_score_note(ctx.case, progress)
        if llm_score is not None:
            return llm_score

        return grade_representment_note(
            progress.representment_note,
            ctx.case,
            set(progress.attached_evidence_ids),
        )


__all__ = ["LLMNoteJudgeRubric", "llm_score_note"]
