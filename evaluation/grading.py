"""Deterministic grading logic for ChargebackOps."""

from __future__ import annotations

try:
    from ..core.models import CaseScoreBreakdown, GraderReport
    from ..scenarios.simulation import CaseProgress, InternalCase, TaskScenario
except ImportError:  # pragma: no cover
    from core.models import CaseScoreBreakdown, GraderReport
    from scenarios.simulation import CaseProgress, InternalCase, TaskScenario


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 1.0
    return max(0.0, min(1.0, numerator / denominator))


def grade_representment_note(
    note: str | None,
    case: "InternalCase",
    attached_ids: set[str],
) -> float:
    """Score a representment note from 0.0 to 1.0.

    Evaluates whether the note:
    - References required claims from the policy requirements
    - Avoids mentioning harmful evidence
    - Has sufficient substance (length and specificity)
    """
    if not note or not note.strip():
        return 0.0

    text = note.lower()
    score = 0.0

    # Substance: minimum length for a coherent note
    word_count = len(text.split())
    if word_count >= 5:
        score += 0.2
    elif word_count >= 2:
        score += 0.1

    # Required claims coverage: does the note mention policy requirements?
    if case.policy_requirements:
        claims_hit = 0
        for req in case.policy_requirements:
            req_keywords = req.lower().split()
            if any(kw in text for kw in req_keywords if len(kw) > 3):
                claims_hit += 1
        score += 0.5 * _ratio(claims_hit, len(case.policy_requirements))
    else:
        score += 0.3  # No requirements to check

    # Evidence coherence: does the note reference attached evidence?
    evidence_refs = sum(1 for eid in attached_ids if eid.lower() in text or any(
        part in text for part in eid.lower().replace("-", " ").split() if len(part) > 3
    ))
    if evidence_refs > 0:
        score += 0.15

    # Harmful mention penalty: derived from the case's actual harmful evidence.
    # Each case defines its own harmful artifacts, so the penalty adapts to
    # the specific dispute rather than matching a static keyword list.
    harmful_terms: set[str] = set()
    for items in case.evidence_by_system.values():
        for item in items:
            if item.harmful:
                for word in (item.title + " " + item.summary).lower().split():
                    clean = word.strip(".,;:()")
                    if len(clean) > 3:
                        harmful_terms.add(clean)
    # Remove generic words that would cause false positives
    harmful_terms -= {"was", "the", "and", "for", "that", "with", "from", "time", "detail"}
    harmful_hits = sum(1 for term in harmful_terms if term in text)
    if harmful_hits > 0:
        score -= 0.12 * min(harmful_hits, 3)

    return max(0.0, min(1.0, score))


def score_case(
    case: InternalCase,
    progress: CaseProgress,
    step_count: int,
) -> CaseScoreBreakdown:
    """Score one case deterministically."""

    final_resolution = progress.final_resolution or "unresolved"
    attached_set = set(progress.attached_evidence_ids)
    required_attached = len(attached_set.intersection(case.required_evidence_ids))
    helpful_attached = len(attached_set.intersection(case.helpful_evidence_ids))
    harmful_attached = len(attached_set.intersection(case.harmful_evidence_ids))

    if final_resolution == case.optimal_strategy:
        strategy_correctness = 1.0
    elif final_resolution in case.acceptable_strategies:
        strategy_correctness = 0.35
    else:
        strategy_correctness = 0.0

    if final_resolution == "contest":
        if case.optimal_strategy != "contest" and "contest" not in case.acceptable_strategies:
            # Contesting a case that should not be contested — evidence is irrelevant
            evidence_quality = 0.0
            packet_validity = 0.0
        else:
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
            if case.optimal_strategy == "contest":
                # Conceded a contestable case — evidence gathering was abandoned
                evidence_quality = 0.15
                packet_validity = 0.0
            else:
                evidence_quality = 1.0 if helpful_attached == 0 and harmful_attached == 0 else 0.7
                packet_validity = 1.0
        else:
            evidence_quality = 0.0
            packet_validity = 0.0

    resolution_step = progress.resolved_at_step if progress.resolved_at_step is not None else step_count
    deadline_compliance = 1.0
    if final_resolution == "unresolved":
        deadline_compliance = 0.0
    elif resolution_step > case.deadline_step:
        deadline_compliance = 0.0

    # --- Efficiency: penalise shallow operational behaviour ---
    wasted_actions = progress.duplicate_queries + progress.invalid_actions
    efficiency = max(0.0, 1.0 - min(0.9, wasted_actions * 0.1 + progress.submit_attempts * 0.05))

    # Penalty: over-querying a concedable case wastes steps
    if final_resolution in {"accept_chargeback", "issue_refund"} and case.optimal_strategy != "contest":
        systems_queried = len(progress.revealed_systems)
        if systems_queried > 2:
            efficiency -= 0.15 * (systems_queried - 2)

    # Penalty: retrieving policy too late to change the outcome
    if progress.policy_retrieved and resolution_step is not None:
        # The case was already being resolved, policy retrieval was wasted
        if final_resolution in {"accept_chargeback", "issue_refund"} and case.optimal_strategy in {
            "accept_chargeback", "issue_refund"
        }:
            # Correct concession but wasted a step on policy retrieval
            efficiency -= 0.08

    # Reward: early correct concession on a clearly bad case (≤3 steps used)
    if (
        final_resolution in {"accept_chargeback", "issue_refund"}
        and case.optimal_strategy in {"accept_chargeback", "issue_refund"}
        and resolution_step is not None
        and resolution_step <= 3
    ):
        efficiency = min(1.0, efficiency + 0.1)

    efficiency = max(0.0, min(1.0, efficiency))

    if final_resolution == case.optimal_strategy:
        outcome_quality = 1.0
    elif final_resolution in case.acceptable_strategies:
        outcome_quality = 0.4
    else:
        outcome_quality = 0.0

    # Representment note quality (only relevant for contested cases)
    if final_resolution == "contest" and progress.representment_note:
        note_quality = grade_representment_note(progress.representment_note, case, attached_set)
    else:
        note_quality = 0.0

    weighted_score = (
        0.25 * strategy_correctness
        + 0.20 * evidence_quality
        + 0.15 * packet_validity
        + 0.15 * deadline_compliance
        + 0.10 * efficiency
        + 0.10 * outcome_quality
        + 0.05 * note_quality
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
        note_quality=round(note_quality, 4),
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
