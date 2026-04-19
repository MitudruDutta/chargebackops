"""Scripted Issuer agent for ChargebackOps multi-round dispute lifecycle.

The Issuer reviews a merchant's representment packet and decides whether to
accept it, request more evidence (triggering pre-arbitration ), or
escalate to network arbitration. The decision is **deterministic** by default —
benchmarks must be reproducible — with optional LLM softening reserved for the
Day 4 milestone.

Decision rule:

1. Compute ``evidence_strength_score`` in [0, 1] from the attached packet.
2. Round 1 cutoffs:
       score >= 0.7 -> ACCEPT
       score <= 0.4 -> REQUEST_MORE_EVIDENCE
       else         -> deterministic midpoint fallback (>= 0.55 -> ACCEPT,
                       else -> REQUEST_MORE_EVIDENCE)
3. Round 2 cutoffs (issuer is more confrontational once it has rejected once):
       score >= 0.6 -> ACCEPT
       else         -> ESCALATE_TO_ARBITRATION

The scoring inputs come from ``InternalCase`` (immutable case definition) and
``CaseProgress`` (mutable per-episode state). The agent never reads the
merchant's hidden ``optimal_strategy`` label — it sees only the evidence packet
and the representment note, exactly the way a real card-issuing bank would.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

try:
    from .simulation import CaseProgress, InternalCase
except ImportError:  # pragma: no cover
    from scenarios.simulation import CaseProgress, InternalCase

if TYPE_CHECKING:
    from .llm_softening import IssuerDeciderFn


class IssuerDecision(str, Enum):
    """One of three discrete decisions the Issuer can make in any round."""

    ACCEPT = "accept"
    REQUEST_MORE_EVIDENCE = "request_more_evidence"
    ESCALATE_TO_ARBITRATION = "escalate_to_arbitration"


@dataclass(frozen=True)
class IssuerReview:
    """The Issuer's response to a single representment submission."""

    decision: IssuerDecision
    evidence_strength_score: float
    rationale: str
    used_llm_softening: bool = False


# Deterministic decision band edges 
ROUND1_ACCEPT_THRESHOLD: float = 0.7
ROUND1_REJECT_THRESHOLD: float = 0.4
ROUND1_MIDPOINT_FALLBACK: float = 0.55
ROUND2_ACCEPT_THRESHOLD: float = 0.6


def evidence_strength_score(case: InternalCase, progress: CaseProgress) -> float:
    """Score the merchant's attached packet from the Issuer's point of view.

    This is the single source of truth used by both ``IssuerAgent`` and the
    deterministic ``arbitration_ruling`` so that round-2 escalation odds match
    round-3 outcome probabilities.
    """

    attached = set(progress.attached_evidence_ids)
    required_ids = set(case.required_evidence_ids)
    helpful_ids = set(case.helpful_evidence_ids)
    harmful_ids = set(case.harmful_evidence_ids)

    score = 0.0

    # Required-evidence bonus: all-or-nothing 0.4.
    if required_ids and required_ids.issubset(attached):
        score += 0.4

    # Helpful-evidence bonus: capped at +0.4 (max 2 helpful pieces credited).
    helpful_attached = len(helpful_ids.intersection(attached))
    score += min(0.4, 0.2 * helpful_attached)

    # Harmful penalty: -0.3 per harmful piece, no cap.
    harmful_attached = len(harmful_ids.intersection(attached))
    score -= 0.3 * harmful_attached

    # Note quality bonus: +0.1 if the note references >= 2 policy requirements.
    note = (progress.representment_note or "").lower()
    if note and case.policy_requirements:
        hits = 0
        for req in case.policy_requirements:
            for keyword in req.lower().split():
                if len(keyword) > 3 and keyword in note:
                    hits += 1
                    break
        if hits >= 2:
            score += 0.1

    return max(0.0, min(1.0, score))


class IssuerAgent:
    """Scripted Issuer with deterministic decisions, optional LLM softening.

    When ``enable_llm_softening`` is True, ambiguity-band scores (0.4 < score
    < 0.7) in the first-round review are routed through an LLM decider that
    returns ACCEPT or REQUEST_MORE_EVIDENCE. If the decider is missing or
    errors out, the deterministic midpoint rule takes over so benchmarks stay
    reproducible without API keys.
    """

    def __init__(
        self,
        *,
        enable_llm_softening: bool = False,
        llm_decider: "IssuerDeciderFn | None" = None,
    ) -> None:
        self.enable_llm_softening = enable_llm_softening
        self._llm_decider = llm_decider
        if self.enable_llm_softening and self._llm_decider is None:
            try:
                from .llm_softening import build_default_llm_decider
            except ImportError:  # pragma: no cover
                from scenarios.llm_softening import build_default_llm_decider
            self._llm_decider = build_default_llm_decider()

    def decide_review(
        self,
        case: InternalCase,
        progress: CaseProgress,
        round_number: int,
    ) -> IssuerReview:
        """Return the Issuer's decision for the current round."""

        score = evidence_strength_score(case, progress)

        if round_number >= 2:
            if score >= ROUND2_ACCEPT_THRESHOLD:
                return IssuerReview(
                    decision=IssuerDecision.ACCEPT,
                    evidence_strength_score=score,
                    rationale=(
                        f"Round {round_number}: pre-arb evidence brings the packet "
                        f"to {score:.2f}, above the 0.60 acceptance bar."
                    ),
                )
            return IssuerReview(
                decision=IssuerDecision.ESCALATE_TO_ARBITRATION,
                evidence_strength_score=score,
                rationale=(
                    f"Round {round_number}: packet still scores {score:.2f}; "
                    f"escalating to network arbitration."
                ),
            )

        # decision matrix.
        if score >= ROUND1_ACCEPT_THRESHOLD:
            return IssuerReview(
                decision=IssuerDecision.ACCEPT,
                evidence_strength_score=score,
                rationale=(
                    f"Round 1: packet scores {score:.2f}, clearing the 0.70 acceptance bar."
                ),
            )
        if score <= ROUND1_REJECT_THRESHOLD:
            return IssuerReview(
                decision=IssuerDecision.REQUEST_MORE_EVIDENCE,
                evidence_strength_score=score,
                rationale=(
                    f"Round 1: packet scores {score:.2f}, below the 0.40 floor; "
                    f"requesting compelling evidence."
                ),
            )

        # Ambiguity band (0.40, 0.70). Prefer LLM softening when enabled and
        # the decider returns a verdict; otherwise fall back to the midpoint.
        if self.enable_llm_softening and self._llm_decider is not None:
            try:
                llm_decision = self._llm_decider(case, progress, score)
            except Exception:
                llm_decision = None
            if llm_decision is IssuerDecision.ACCEPT:
                return IssuerReview(
                    decision=IssuerDecision.ACCEPT,
                    evidence_strength_score=score,
                    rationale=(
                        f"Round 1 ambiguity band: packet scores {score:.2f} — "
                        f"LLM softening accepted."
                    ),
                    used_llm_softening=True,
                )
            if llm_decision is IssuerDecision.REQUEST_MORE_EVIDENCE:
                return IssuerReview(
                    decision=IssuerDecision.REQUEST_MORE_EVIDENCE,
                    evidence_strength_score=score,
                    rationale=(
                        f"Round 1 ambiguity band: packet scores {score:.2f} — "
                        f"LLM softening requested compelling evidence."
                    ),
                    used_llm_softening=True,
                )

        if score >= ROUND1_MIDPOINT_FALLBACK:
            return IssuerReview(
                decision=IssuerDecision.ACCEPT,
                evidence_strength_score=score,
                rationale=(
                    f"Round 1 ambiguity band: packet scores {score:.2f} "
                    f"(>= {ROUND1_MIDPOINT_FALLBACK:.2f} midpoint) — accepting."
                ),
            )
        return IssuerReview(
            decision=IssuerDecision.REQUEST_MORE_EVIDENCE,
            evidence_strength_score=score,
            rationale=(
                f"Round 1 ambiguity band: packet scores {score:.2f} "
                f"(< {ROUND1_MIDPOINT_FALLBACK:.2f} midpoint) — requesting more evidence."
            ),
        )


__all__ = [
    "IssuerAgent",
    "IssuerDecision",
    "IssuerReview",
    "evidence_strength_score",
    "ROUND1_ACCEPT_THRESHOLD",
    "ROUND1_REJECT_THRESHOLD",
    "ROUND1_MIDPOINT_FALLBACK",
    "ROUND2_ACCEPT_THRESHOLD",
]
