"""Deterministic grading adapters that delegate to OpenEnv Rubric subclasses.

The real scoring lives in :mod:`evaluation.rubrics`. This module keeps the
legacy call sites (``score_case`` / ``grade_episode`` / ``grade_representment_note``)
stable so the environment, tests, and audit tooling do not need to change.
"""

from __future__ import annotations

try:
    from ..core.models import CaseScoreBreakdown, GraderReport
    from ..scenarios.simulation import CaseProgress, InternalCase, TaskScenario
    from .rubrics import (
        CaseRubric,
        ChargebackOpsEpisodeRubric,
        EpisodeGradingContext,
        GradingContext,
        grade_representment_note,
    )
except ImportError:  # pragma: no cover
    from core.models import CaseScoreBreakdown, GraderReport
    from scenarios.simulation import CaseProgress, InternalCase, TaskScenario
    from evaluation.rubrics import (
        CaseRubric,
        ChargebackOpsEpisodeRubric,
        EpisodeGradingContext,
        GradingContext,
        grade_representment_note,
    )


__all__ = [
    "grade_representment_note",
    "score_case",
    "grade_episode",
]


_CASE_RUBRIC = CaseRubric()
_EPISODE_RUBRIC = ChargebackOpsEpisodeRubric()


def _build_case_notes(
    case: InternalCase, progress: CaseProgress, step_count: int
) -> str:
    final_resolution = progress.final_resolution or "unresolved"
    attached_set = set(progress.attached_evidence_ids)
    harmful_attached = len(attached_set.intersection(case.harmful_evidence_ids))

    note_parts = [case.resolution_summary]
    if harmful_attached:
        note_parts.append("Harmful evidence weakened the case.")
    if final_resolution == "unresolved":
        note_parts.append("Case was never resolved.")
    elif step_count > case.deadline_step:
        note_parts.append("Resolution happened after the deadline.")
    return " ".join(note_parts)


def score_case(
    case: InternalCase,
    progress: CaseProgress,
    step_count: int,
) -> CaseScoreBreakdown:
    """Score one case deterministically via the case rubric."""

    ctx = GradingContext(case=case, progress=progress, step_count=step_count)
    weighted = _CASE_RUBRIC(ctx, None)
    dims = _CASE_RUBRIC.dimension_scores()

    return CaseScoreBreakdown(
        case_id=case.case_id,
        strategy_correctness=round(dims["strategy_correctness"], 4),
        evidence_quality=round(dims["evidence_quality"], 4),
        packet_validity=round(dims["packet_validity"], 4),
        deadline_compliance=round(dims["deadline_compliance"], 4),
        efficiency=round(dims["efficiency"], 4),
        outcome_quality=round(dims["outcome_quality"], 4),
        note_quality=round(dims["note_quality"], 4),
        weighted_score=round(weighted * case.weight, 4),
        final_resolution=progress.final_resolution or "unresolved",
        notes=_build_case_notes(case, progress, step_count),
    )


def grade_episode(
    task: TaskScenario,
    progress_by_case: dict[str, CaseProgress],
    step_count: int,
    episode_id: str,
    completed: bool,
) -> GraderReport:
    """Grade a full episode via the episode-level rubric."""

    case_reports = [
        score_case(case, progress_by_case[case.case_id], step_count)
        for case in task.cases
    ]
    total_score = sum(report.weighted_score for report in case_reports)

    ctx = EpisodeGradingContext(
        task=task,
        progress_by_case=progress_by_case,
        step_count=step_count,
    )
    normalized = float(_EPISODE_RUBRIC(ctx, None))

    summary = (
        f"Resolved {sum(1 for report in case_reports if report.final_resolution != 'unresolved')}/"
        f"{len(case_reports)} cases with normalized score {normalized:.3f}."
    )
    return GraderReport(
        episode_id=episode_id,
        task_id=task.task_id,
        total_score=round(total_score, 4),
        normalized_score=round(normalized, 4),
        completed=completed,
        case_reports=case_reports,
        summary=summary,
    )
