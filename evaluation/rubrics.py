"""OpenEnv Rubric subclasses that power ChargebackOps grading.

Every scoring dimension is a standalone :class:`openenv.core.rubrics.Rubric`
so the whole grader can be introspected via ``named_rubrics``, captured via
``state_dict``, and swapped piecewise (e.g. replace :class:`NoteQualityRubric`
with an ``LLMJudge``). The per-case composite uses :class:`WeightedSum` with
weights that must sum to 1.0.

The rubrics take their inputs via a :class:`GradingContext` dataclass passed
as the ``action`` argument of :meth:`Rubric.forward`. The ``observation``
argument is ignored — ChargebackOps grading operates over deterministic
episode progress, not on the last observation payload. This keeps the rubrics
pure and unit-testable without an environment instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openenv.core.rubrics import Gate, Rubric, WeightedSum

try:
    from ..scenarios.simulation import CaseProgress, InternalCase, TaskScenario
except ImportError:  # pragma: no cover
    from scenarios.simulation import CaseProgress, InternalCase, TaskScenario


@dataclass(frozen=True)
class GradingContext:
    """Inputs one per-case rubric evaluation needs."""

    case: InternalCase
    progress: CaseProgress
    step_count: int


@dataclass(frozen=True)
class EpisodeGradingContext:
    """Inputs the episode-level rubric needs."""

    task: TaskScenario
    progress_by_case: dict[str, CaseProgress]
    step_count: int


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 1.0
    return max(0.0, min(1.0, numerator / denominator))


def _final_resolution(progress: CaseProgress) -> str:
    return progress.final_resolution or "unresolved"


def _contest_is_valid(case: InternalCase) -> bool:
    return case.optimal_strategy == "contest" or "contest" in case.acceptable_strategies


class StrategyCorrectnessRubric(Rubric):
    """Score final strategy: optimal=1.0, acceptable=0.35, else 0.0."""

    def forward(self, action: Any, observation: Any) -> float:
        ctx: GradingContext = action
        final = _final_resolution(ctx.progress)
        if final == ctx.case.optimal_strategy:
            return 1.0
        if final in ctx.case.acceptable_strategies:
            return 0.35
        return 0.0


class EvidenceQualityRubric(Rubric):
    """Score the evidence packet attached to the case.

    Zeroes out (vacuous-truth fix) when the agent contests a case that was
    never contestable — no evidence quality can rescue a wrong strategy.
    """

    def forward(self, action: Any, observation: Any) -> float:
        ctx: GradingContext = action
        case = ctx.case
        progress = ctx.progress
        final = _final_resolution(progress)

        attached_set = set(progress.attached_evidence_ids)
        required_attached = len(attached_set.intersection(case.required_evidence_ids))
        helpful_attached = len(attached_set.intersection(case.helpful_evidence_ids))
        harmful_attached = len(attached_set.intersection(case.harmful_evidence_ids))

        if final == "contest":
            if not _contest_is_valid(case):
                return 0.0
            base = 0.7 * _ratio(required_attached, len(case.required_evidence_ids))
            bonus = 0.3 * _ratio(
                helpful_attached, max(1, len(case.helpful_evidence_ids))
            )
            penalty = 0.25 * harmful_attached
            return max(0.0, min(1.0, base + bonus - penalty))

        if final in {"accept_chargeback", "issue_refund"}:
            if case.optimal_strategy == "contest":
                return 0.15
            return 1.0 if helpful_attached == 0 and harmful_attached == 0 else 0.7

        return 0.0


class PacketValidityRubric(Rubric):
    """All-or-nothing: required evidence complete AND no harmful attached."""

    def forward(self, action: Any, observation: Any) -> float:
        ctx: GradingContext = action
        case = ctx.case
        progress = ctx.progress
        final = _final_resolution(progress)

        attached_set = set(progress.attached_evidence_ids)
        required_attached = len(attached_set.intersection(case.required_evidence_ids))
        harmful_attached = len(attached_set.intersection(case.harmful_evidence_ids))

        if final == "contest":
            if not _contest_is_valid(case):
                return 0.0
            if (
                required_attached == len(case.required_evidence_ids)
                and harmful_attached == 0
            ):
                return 1.0
            return 0.0

        if final in {"accept_chargeback", "issue_refund"}:
            if case.optimal_strategy == "contest":
                return 0.0
            return 1.0

        return 0.0


class DeadlineComplianceRubric(Rubric):
    """1.0 if resolved on time, else 0.0."""

    def forward(self, action: Any, observation: Any) -> float:
        ctx: GradingContext = action
        case = ctx.case
        progress = ctx.progress
        final = _final_resolution(progress)

        if final == "unresolved":
            return 0.0
        resolution_step = (
            progress.resolved_at_step
            if progress.resolved_at_step is not None
            else ctx.step_count
        )
        if resolution_step > case.deadline_step:
            return 0.0
        return 1.0


class CaseAbandonedRubric(Rubric):
    """Hard-constraint rubric: 0.0 if the case was abandoned past deadline.

    Used as the inner rubric for the :class:`Gate` wrapped around
    :class:`CaseRubric`. The distinction from :class:`DeadlineComplianceRubric`
    is intentional:

    * :class:`DeadlineComplianceRubric` is the *dimension* — penalises any
      late resolution with a 0 in the weighted sum (15% score drop).
    * :class:`CaseAbandonedRubric` is the *gate* — hard-zeros the entire
      case only when the agent never even attempted to resolve it before
      the deadline expired. In real chargeback operations this is the
      "no contest, case closed" outcome: the merchant forfeited.

    This split means a late-but-completed representment takes the 15%
    deadline penalty (and still gets graded on evidence, strategy, and
    packet quality), while a case left untouched after the deadline
    collapses the entire case score to 0.
    """

    def forward(self, action: Any, observation: Any) -> float:
        ctx: GradingContext = action
        progress = ctx.progress
        case = ctx.case
        if _final_resolution(progress) != "unresolved":
            return 1.0
        if ctx.step_count > case.deadline_step:
            return 0.0
        return 1.0


class EfficiencyRubric(Rubric):
    """Penalise wasted / redundant actions, reward early correct concessions."""

    def forward(self, action: Any, observation: Any) -> float:
        ctx: GradingContext = action
        case = ctx.case
        progress = ctx.progress
        final = _final_resolution(progress)

        wasted_actions = progress.duplicate_queries + progress.invalid_actions
        efficiency = max(
            0.0,
            1.0 - min(0.9, wasted_actions * 0.1 + progress.submit_attempts * 0.05),
        )

        # Over-querying a concedable case is wasted exploration.
        if (
            final in {"accept_chargeback", "issue_refund"}
            and case.optimal_strategy != "contest"
        ):
            systems_queried = len(progress.revealed_systems)
            if systems_queried > 2:
                efficiency -= 0.15 * (systems_queried - 2)

        # Retrieving policy after the decision was already made is wasted.
        if progress.policy_retrieved and progress.resolved_at_step is not None:
            if final in {
                "accept_chargeback",
                "issue_refund",
            } and case.optimal_strategy in {
                "accept_chargeback",
                "issue_refund",
            }:
                efficiency -= 0.08

        # Early correct concession bonus.
        if (
            final in {"accept_chargeback", "issue_refund"}
            and case.optimal_strategy in {"accept_chargeback", "issue_refund"}
            and progress.resolved_at_step is not None
            and progress.resolved_at_step <= 3
        ):
            efficiency = min(1.0, efficiency + 0.1)

        return max(0.0, min(1.0, efficiency))


class OutcomeQualityRubric(Rubric):
    """Discrete outcome quality: optimal=1.0, acceptable=0.4, else 0.0."""

    def forward(self, action: Any, observation: Any) -> float:
        ctx: GradingContext = action
        final = _final_resolution(ctx.progress)
        if final == ctx.case.optimal_strategy:
            return 1.0
        if final in ctx.case.acceptable_strategies:
            return 0.4
        return 0.0


class NoteQualityRubric(Rubric):
    """Text-based representment note scorer (contest-only)."""

    def forward(self, action: Any, observation: Any) -> float:
        ctx: GradingContext = action
        progress = ctx.progress
        if _final_resolution(progress) != "contest" or not progress.representment_note:
            return 0.0
        return grade_representment_note(
            progress.representment_note,
            ctx.case,
            set(progress.attached_evidence_ids),
        )


def grade_representment_note(
    note: str | None,
    case: InternalCase,
    attached_ids: set[str],
) -> float:
    """Score a representment note from 0.0 to 1.0.

    Evaluates whether the note references required policy claims, mentions
    attached evidence, has sufficient substance, and avoids harmful mentions.
    """

    if not note or not note.strip():
        return 0.0

    text = note.lower()
    score = 0.0

    # Substance: minimum length for a coherent note.
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
        score += 0.3

    # Evidence coherence: does the note reference attached evidence?
    evidence_refs = sum(
        1
        for eid in attached_ids
        if eid.lower() in text
        or any(
            part in text
            for part in eid.lower().replace("-", " ").split()
            if len(part) > 3
        )
    )
    if evidence_refs > 0:
        score += 0.15

    # Harmful mention penalty derived from each case's harmful evidence blobs.
    harmful_terms: set[str] = set()
    for items in case.evidence_by_system.values():
        for item in items:
            if item.harmful:
                for word in (item.title + " " + item.summary).lower().split():
                    clean = word.strip(".,;:()")
                    if len(clean) > 3:
                        harmful_terms.add(clean)
    harmful_terms -= {
        "was",
        "the",
        "and",
        "for",
        "that",
        "with",
        "from",
        "time",
        "detail",
    }
    harmful_hits = sum(1 for term in harmful_terms if term in text)
    if harmful_hits > 0:
        score -= 0.12 * min(harmful_hits, 3)

    return max(0.0, min(1.0, score))


# Weights must match the order of rubrics handed to WeightedSum and sum to 1.0.
CASE_DIMENSION_WEIGHTS: tuple[float, ...] = (0.25, 0.20, 0.15, 0.15, 0.10, 0.10, 0.05)
CASE_DIMENSION_NAMES: tuple[str, ...] = (
    "strategy_correctness",
    "evidence_quality",
    "packet_validity",
    "deadline_compliance",
    "efficiency",
    "outcome_quality",
    "note_quality",
)


class CaseRubric(Rubric):
    """Per-case composite — weighted sum of the seven scoring dimensions,
    hard-gated on deadline compliance.

    The weighted sum lives in :class:`WeightedSum`. On top of that, a
    :class:`Gate` wrapping :class:`DeadlineComplianceRubric` at
    ``threshold=1.0`` hard-zeros the whole case if the deadline is missed —
    in real chargeback operations the best evidence in the world can't save a
    case filed too late, so a late resolution must collapse the case score,
    not just reduce one dimension. This is a direct use of OpenEnv's
    :class:`Gate` primitive and exposes the hard-constraint pattern through
    :meth:`named_rubrics`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.aggregator = WeightedSum(
            rubrics=[
                StrategyCorrectnessRubric(),
                EvidenceQualityRubric(),
                PacketValidityRubric(),
                DeadlineComplianceRubric(),
                EfficiencyRubric(),
                OutcomeQualityRubric(),
                NoteQualityRubric(),
            ],
            weights=list(CASE_DIMENSION_WEIGHTS),
        )
        # Hard constraint: a case never even *attempted* before the deadline
        # expires collapses the entire case score. This wraps
        # :class:`CaseAbandonedRubric` (not :class:`DeadlineComplianceRubric`)
        # so late-but-completed representments still get dimension-level
        # credit while truly abandoned cases are zeroed.
        self.deadline_gate = Gate(CaseAbandonedRubric(), threshold=1.0)

    def forward(self, action: Any, observation: Any) -> float:
        # Always run the aggregator first so per-dimension ``last_score``
        # values are fresh for the grader breakdown, even when the gate
        # hard-fails the case.
        weighted = self.aggregator(action, observation)
        if self.deadline_gate(action, observation) < 1.0:
            return 0.0
        return weighted

    def dimension_scores(self) -> dict[str, float]:
        """Return per-dimension scores from the most recent forward pass."""

        scores: dict[str, float] = {}
        for name, child in zip(CASE_DIMENSION_NAMES, self.aggregator._rubric_list):
            scores[name] = (
                float(child.last_score) if child.last_score is not None else 0.0
            )
        return scores


class ChargebackOpsEpisodeRubric(Rubric):
    """Episode-level rubric: aggregate per-case scores weighted by case.weight."""

    def __init__(self) -> None:
        super().__init__()
        self.case_rubric = CaseRubric()

    def forward(self, action: Any, observation: Any) -> float:
        ctx: EpisodeGradingContext = action
        total = 0.0
        total_weight = 0.0
        for case in ctx.task.cases:
            case_ctx = GradingContext(
                case=case,
                progress=ctx.progress_by_case[case.case_id],
                step_count=ctx.step_count,
            )
            case_score = self.case_rubric(case_ctx, observation)
            total += case_score * case.weight
            total_weight += case.weight
        if total_weight == 0:
            return 0.0
        return min(1.0, total / total_weight)
