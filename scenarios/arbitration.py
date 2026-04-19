"""Network arbitration ruling for ChargebackOps

Arbitration is the terminal round of a disputed chargeback. The card network
(Visa / Mastercard) adjudicates and charges **both** sides a non-refundable
$250 filing fee. The loser additionally forfeits the disputed amount. The
function is pure: same inputs → same output, so benchmarks and the arbitration
EV rubric  stay reproducible.

Decision rule:

    score >= 0.65 → MERCHANT_WINS
    score <= 0.35 → ISSUER_WINS
    else          → deterministic coin flip keyed by case_id

The score comes from the shared ``evidence_strength_score`` so round-2
escalation odds line up with round-3 outcomes — a merchant that barely cleared
pre-arb shouldn't suddenly crush arbitration.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum

try:
    from .issuer_model import evidence_strength_score
    from .simulation import CaseProgress, InternalCase
except ImportError:  # pragma: no cover
    from scenarios.issuer_model import evidence_strength_score
    from scenarios.simulation import CaseProgress, InternalCase


class ArbitrationOutcome(str, Enum):
    """Terminal outcomes of network arbitration."""

    MERCHANT_WINS = "merchant_wins"
    ISSUER_WINS = "issuer_wins"


# Decision band edges 
ARB_MERCHANT_WIN_THRESHOLD: float = 0.65
ARB_ISSUER_WIN_THRESHOLD: float = 0.35
ARB_FEE_PER_SIDE: float = 250.0


@dataclass(frozen=True)
class ArbitrationRuling:
    """Pure result of one arbitration adjudication."""

    outcome: ArbitrationOutcome
    evidence_strength_score: float
    arb_fee_per_side: float
    dispute_amount: float
    merchant_net_pnl: float
    rationale: str


def _coin_flip_merchant_wins(case_id: str) -> bool:
    """Deterministic 50/50 coin flip keyed by ``case_id``.

    Python's built-in ``hash`` is salted per-process so we use SHA-256 to keep
    outcomes reproducible across runs and machines.
    """

    digest = hashlib.sha256(case_id.encode("utf-8")).digest()
    return digest[0] % 2 == 0


def arbitration_ruling(
    case: InternalCase, progress: CaseProgress
) -> ArbitrationRuling:
    """Adjudicate the case and return the terminal ruling.

    Both sides pay the fixed fee. Loser additionally forfeits the disputed
    amount. ``merchant_net_pnl`` is the merchant's net dollars from this
    arbitration alone (not the full episode).
    """

    score = evidence_strength_score(case, progress)

    if score >= ARB_MERCHANT_WIN_THRESHOLD:
        outcome = ArbitrationOutcome.MERCHANT_WINS
        rationale = (
            f"Arbitration: packet scores {score:.2f} (>= {ARB_MERCHANT_WIN_THRESHOLD:.2f}) "
            f"— network rules for the merchant."
        )
    elif score <= ARB_ISSUER_WIN_THRESHOLD:
        outcome = ArbitrationOutcome.ISSUER_WINS
        rationale = (
            f"Arbitration: packet scores {score:.2f} (<= {ARB_ISSUER_WIN_THRESHOLD:.2f}) "
            f"— network rules for the issuer."
        )
    else:
        merchant_wins = _coin_flip_merchant_wins(case.case_id)
        outcome = (
            ArbitrationOutcome.MERCHANT_WINS
            if merchant_wins
            else ArbitrationOutcome.ISSUER_WINS
        )
        rationale = (
            f"Arbitration: packet scores {score:.2f} in ambiguity band "
            f"({ARB_ISSUER_WIN_THRESHOLD:.2f}, {ARB_MERCHANT_WIN_THRESHOLD:.2f}) — "
            f"deterministic coin flip on case_id ruled for "
            f"{'merchant' if merchant_wins else 'issuer'}."
        )

    if outcome == ArbitrationOutcome.MERCHANT_WINS:
        merchant_net_pnl = case.amount - ARB_FEE_PER_SIDE
    else:
        merchant_net_pnl = -case.amount - ARB_FEE_PER_SIDE

    return ArbitrationRuling(
        outcome=outcome,
        evidence_strength_score=score,
        arb_fee_per_side=ARB_FEE_PER_SIDE,
        dispute_amount=case.amount,
        merchant_net_pnl=merchant_net_pnl,
        rationale=rationale,
    )


__all__ = [
    "ArbitrationOutcome",
    "ArbitrationRuling",
    "arbitration_ruling",
    "ARB_MERCHANT_WIN_THRESHOLD",
    "ARB_ISSUER_WIN_THRESHOLD",
    "ARB_FEE_PER_SIDE",
]
