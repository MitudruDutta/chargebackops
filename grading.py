"""Deterministic grading logic for ChargebackOps."""

from __future__ import annotations

try:
    from .models import CaseScoreBreakdown, GraderReport
    from .simulation import CaseProgress, InternalCase, TaskScenario
except ImportError:  # pragma: no cover
    from models import CaseScoreBreakdown, GraderReport
    from simulation import CaseProgress, InternalCase, TaskScenario


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 1.0
    return max(0.0, min(1.0, numerator / denominator))


def score_case(
    case: InternalCase,
    progress: CaseProgress,
    step_count: int,
) -> CaseScoreBreakdown:
    """Score one case deterministically."""

    final_resolution = progress.final_resolution or "unresolved"
    required_attached = len(
        set(progress.attached_evidence_ids).intersection(case.required_evidence_ids)
    )
    helpful_attached = len(
        set(progress.attached_evidence_ids).intersection(case.helpful_evidence_ids)
    )
    harmful_attached = len(
        set(progress.attached_evidence_ids).intersection(case.harmful_evidence_ids)
    )

    if final_resolution == case.optimal_strategy:
        strategy_correctness = 1.0
    elif final_resolution in case.acceptable_strategies:
        strategy_correctness = 0.55
    else:
        strategy_correctness = 0.0

    if final_resolution == "contest":
        base_evidence_quality = 0.7 * _ratio(required_attached, len(case.required_evidence_ids))
        bonus = 0.3 * _ratio(helpful_attached, max(1, len(case.helpful_evidence_ids)))
        penalty = 0.25 * harmful_attached
        evidence_quality = max(0.0, min(1.0, base_evidence_quality + bonus - penalty))
        packet_validity = (
            1.0
            if required_attached == len(case.required_evidence_ids) and harmful_attached == 0
            else 0.0
        )
    else:
        if final_resolution in {"accept_chargeback", "issue_refund"}:
            evidence_quality = 1.0 if helpful_attached == 0 and harmful_attached == 0 else 0.7
            packet_validity = 1.0
        else:
            evidence_quality = 0.0
            packet_validity = 0.0

    deadline_compliance = 1.0
    if final_resolution == "unresolved":
        deadline_compliance = 0.0
    elif step_count > case.deadline_step:
        deadline_compliance = 0.0

    wasted_actions = progress.duplicate_queries + progress.invalid_actions
    efficiency = max(0.0, 1.0 - min(0.9, wasted_actions * 0.1 + progress.submit_attempts * 0.05))

    if final_resolution == case.optimal_strategy:
        outcome_quality = 1.0
    elif final_resolution in case.acceptable_strategies:
        outcome_quality = 0.6
    else:
        outcome_quality = 0.0

    weighted_score = (
        0.25 * strategy_correctness
        + 0.25 * evidence_quality
        + 0.15 * packet_validity
        + 0.15 * deadline_compliance
        + 0.10 * efficiency
        + 0.10 * outcome_quality
    )

    note_parts = [case.resolution_summary]
    if harmful_attached:
        note_parts.append("Harmful evidence weakened the case.")
    if final_resolution == "unresolved":
        note_parts.append("Case was never resolved.")
    elif step_count > case.deadline_step:
        note_parts.append("Resolution happened after the deadline.")

    return CaseScoreBreakdown(
        case_id=case.case_id,
        strategy_correctness=round(strategy_correctness, 4),
        evidence_quality=round(evidence_quality, 4),
        packet_validity=round(packet_validity, 4),
        deadline_compliance=round(deadline_compliance, 4),
        efficiency=round(efficiency, 4),
        outcome_quality=round(outcome_quality, 4),
        weighted_score=round(weighted_score * case.weight, 4),
        final_resolution=final_resolution,
        notes=" ".join(note_parts),
    )


def grade_episode(
    task: TaskScenario,
    progress_by_case: dict[str, CaseProgress],
    step_count: int,
    episode_id: str,
    completed: bool,
) -> GraderReport:
    """Grade a full episode."""

    case_reports = [
        score_case(case, progress_by_case[case.case_id], step_count)
        for case in task.cases
    ]
    total_weight = sum(case.weight for case in task.cases)
    total_score = sum(report.weighted_score for report in case_reports)
    normalized = 0.0 if total_weight == 0 else min(1.0, total_score / total_weight)
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
