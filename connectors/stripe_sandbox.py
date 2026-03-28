"""Stripe sandbox connector for ChargebackOps.

Maps Stripe test-mode dispute objects into ``InternalCase`` / ``TaskScenario``
so real Stripe dispute flows can be processed through the environment.

Usage::

    export STRIPE_API_KEY=sk_test_...
    from connectors.stripe_sandbox import fetch_disputes, build_stripe_task

    disputes = fetch_disputes(limit=10)
    task = build_stripe_task(disputes, difficulty="medium")
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Any

try:
    from ..simulation import (
        InternalCase,
        InternalEvidence,
        TaskScenario,
        SystemName,
        StrategyName,
    )
except ImportError:  # pragma: no cover
    from simulation import (
        InternalCase,
        InternalEvidence,
        TaskScenario,
        SystemName,
        StrategyName,
    )

_STRIPE_REASON_MAP: dict[str, str] = {
    "fraudulent": "fraud_cnp",
    "unrecognized": "fraud_cnp",
    "product_not_received": "goods_not_received",
    "product_unacceptable": "product_not_as_described",
    "duplicate": "duplicate_processing",
    "subscription_canceled": "credit_not_processed",
    "credit_not_processed": "credit_not_processed",
    "general": "goods_not_received",
    "service_not_as_described": "service_not_provided",
}

_STRIPE_STATUS_WON = {"won"}
_STRIPE_STATUS_LOST = {"lost"}
_STRIPE_STATUS_OPEN = {
    "needs_response",
    "under_review",
    "warning_needs_response",
    "warning_under_review",
    "warning_closed",
    "charge_refunded",
}

_POLICY_GUIDANCE: dict[str, str] = {
    "goods_not_received": "Prove fulfillment with order confirmation and carrier delivery evidence.",
    "fraud_cnp": "Contest only with prior account linkage and device history. Do not attach mismatch artifacts.",
    "product_not_as_described": "Contest when listing accurately represents the product and customer bypassed returns.",
    "service_not_provided": "Contest when provider records confirm service delivery.",
    "credit_not_processed": "Refund immediately or concede. Contesting is not supportable.",
    "duplicate_processing": "Refund the duplicate charge immediately. Do not contest.",
}

_POLICY_REQS: dict[str, tuple[str, ...]] = {
    "goods_not_received": ("order confirmation", "carrier delivery confirmation"),
    "fraud_cnp": ("prior good order linkage", "customer account confirmation"),
    "product_not_as_described": ("product listing verification", "return policy documentation"),
    "service_not_provided": ("service completion record", "customer acknowledgment"),
    "credit_not_processed": ("proof of cancellation request", "refund status check"),
    "duplicate_processing": ("payment transaction log", "duplicate confirmation"),
}


def _ev(eid: str, system: SystemName, title: str, summary: str,
        *, helpful: bool = False, harmful: bool = False, required: bool = False) -> InternalEvidence:
    return InternalEvidence(
        evidence_id=eid, source_system=system, title=title,
        summary=summary, helpful=helpful, harmful=harmful, required=required,
    )


def _infer_strategy(reason_code: str, stripe_status: str) -> tuple[str, tuple[str, ...]]:
    """Infer optimal strategy from Stripe dispute status."""
    # These reason codes should always refund — contesting is never supportable.
    if reason_code in ("credit_not_processed", "duplicate_processing"):
        return "issue_refund", ("accept_chargeback",)
    if stripe_status in _STRIPE_STATUS_WON:
        return "contest", ()
    if stripe_status in _STRIPE_STATUS_LOST:
        return "accept_chargeback", ("issue_refund",)
    return "contest", ()


def _build_evidence(
    prefix: str,
    reason_code: str,
    amount: float,
    currency: str,
    metadata: dict[str, Any],
    optimal: str,
    rng: random.Random,
) -> tuple[dict[SystemName, tuple[InternalEvidence, ...]], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    by_sys: dict[SystemName, list[InternalEvidence]] = {
        s: [] for s in ("orders", "payment", "shipping", "support", "refunds", "risk")
    }
    req: list[str] = []
    hlp: list[str] = []
    hrm: list[str] = []

    desc = metadata.get("description", f"Stripe dispute for {amount} {currency}")

    if reason_code == "goods_not_received":
        e = _ev(f"{prefix}-ORDER", "orders", "Order confirmation", f"Order for {amount} {currency}.", helpful=True, required=True)
        by_sys["orders"].append(e); req.append(e.evidence_id); hlp.append(e.evidence_id)
        by_sys["payment"].append(_ev(f"{prefix}-AUTH", "payment", "Payment capture", "Stripe charge captured."))
        if optimal == "contest":
            e = _ev(f"{prefix}-DELIVERY", "shipping", "Delivery confirmation", "Carrier confirms delivery.", helpful=True, required=True)
            by_sys["shipping"].append(e); req.append(e.evidence_id); hlp.append(e.evidence_id)
        else:
            by_sys["shipping"].append(_ev(f"{prefix}-NOTRACK", "shipping", "Tracking", "No delivery confirmation."))
        by_sys["refunds"].append(_ev(f"{prefix}-REFUND", "refunds", "Refund ledger", "No refund issued."))

    elif reason_code == "fraud_cnp":
        by_sys["orders"].append(_ev(f"{prefix}-ORDER", "orders", "Order receipt", f"Order for {amount} {currency}.", helpful=True))
        hlp.append(f"{prefix}-ORDER")
        e_avs = _ev(f"{prefix}-AVS", "payment", "AVS check", "AVS mismatch at authorization.", harmful=True)
        by_sys["payment"].append(e_avs); hrm.append(e_avs.evidence_id)
        by_sys["payment"].append(_ev(f"{prefix}-AUTH", "payment", "Payment capture", "Stripe charge captured."))
        if optimal == "contest":
            e = _ev(f"{prefix}-PRIOR", "risk", "Prior account activity", "Same account with prior fulfilled orders.", helpful=True, required=True)
            by_sys["risk"].append(e); req.append(e.evidence_id); hlp.append(e.evidence_id)
            e = _ev(f"{prefix}-CHAT", "support", "Customer verification", "Customer confirmed order via support.", helpful=True, required=True)
            by_sys["support"].append(e); req.append(e.evidence_id); hlp.append(e.evidence_id)
        else:
            by_sys["risk"].append(_ev(f"{prefix}-RISK", "risk", "Risk summary", "No positive account history."))
        by_sys["refunds"].append(_ev(f"{prefix}-REFUND", "refunds", "Refund ledger", "No refund issued."))

    elif reason_code == "product_not_as_described":
        e = _ev(f"{prefix}-ORDER", "orders", "Order details", f"Order for {amount} {currency} — SKU matches.", helpful=True, required=True)
        by_sys["orders"].append(e); req.append(e.evidence_id); hlp.append(e.evidence_id)
        e = _ev(f"{prefix}-LISTING", "orders", "Product listing", "Listing matches manufacturer specs.", helpful=True, required=True)
        by_sys["orders"].append(e); req.append(e.evidence_id); hlp.append(e.evidence_id)
        by_sys["payment"].append(_ev(f"{prefix}-AUTH", "payment", "Payment capture", "Settled at listed price."))
        by_sys["shipping"].append(_ev(f"{prefix}-DELIVERY", "shipping", "Delivery confirmation", "Delivered.", helpful=True))
        hlp.append(f"{prefix}-DELIVERY")
        by_sys["refunds"].append(_ev(f"{prefix}-REFUND", "refunds", "Refund ledger", "No refund processed."))

    elif reason_code == "service_not_provided":
        e = _ev(f"{prefix}-BOOKING", "orders", "Service booking", f"Booking for {amount} {currency}.", helpful=True, required=True)
        by_sys["orders"].append(e); req.append(e.evidence_id); hlp.append(e.evidence_id)
        by_sys["payment"].append(_ev(f"{prefix}-AUTH", "payment", "Payment record", "Stripe charge captured."))
        if optimal == "contest":
            e = _ev(f"{prefix}-COMPLETION", "support", "Service completion", "Service marked completed.", helpful=True, required=True)
            by_sys["support"].append(e); req.append(e.evidence_id); hlp.append(e.evidence_id)
        by_sys["refunds"].append(_ev(f"{prefix}-REFUND", "refunds", "Refund ledger", "No refund issued."))

    elif reason_code in ("credit_not_processed", "duplicate_processing"):
        by_sys["orders"].append(_ev(f"{prefix}-ORDER", "orders", "Invoice", f"Charge of {amount} {currency}."))
        by_sys["payment"].append(_ev(f"{prefix}-PAYMENT", "payment", "Payment", "Stripe charge settled."))
        by_sys["support"].append(_ev(f"{prefix}-REQ", "support", "Customer request", desc[:100], helpful=True))
        hlp.append(f"{prefix}-REQ")
        by_sys["refunds"].append(_ev(f"{prefix}-NOREFUND", "refunds", "Refund ledger", "No refund processed.", helpful=True))
        hlp.append(f"{prefix}-NOREFUND")

    frozen = {k: tuple(v) for k, v in by_sys.items()}
    return frozen, tuple(req), tuple(hlp), tuple(hrm)


def dispute_to_case(dispute: dict[str, Any], case_index: int, *, deadline_step: int = 8) -> InternalCase | None:
    """Convert a Stripe dispute object to an InternalCase."""
    stripe_reason = dispute.get("reason", "general")
    reason_code = _STRIPE_REASON_MAP.get(stripe_reason)
    if reason_code is None:
        return None

    amount = dispute.get("amount", 0) / 100.0  # Stripe amounts are in cents
    currency = dispute.get("currency", "usd").upper()
    status = dispute.get("status", "needs_response")
    metadata = dispute.get("metadata", {})
    dispute_id = dispute.get("id", f"dp_{case_index}")

    optimal, acceptable = _infer_strategy(reason_code, status)
    rng = random.Random(int(hashlib.sha256(dispute_id.encode()).hexdigest()[:8], 16))
    prefix = f"STRIPE{case_index}"

    evidence, req_ids, hlp_ids, hrm_ids = _build_evidence(
        prefix, reason_code, amount, currency, metadata, optimal, rng,
    )

    guidance = _POLICY_GUIDANCE.get(reason_code, "")
    if optimal in ("accept_chargeback", "issue_refund") and reason_code not in ("credit_not_processed", "duplicate_processing"):
        guidance = f"Do not contest this {reason_code.replace('_', ' ')} dispute. Concede to avoid wasting resources."

    return InternalCase(
        case_id=f"CB-STRIPE{case_index}",
        order_id=dispute.get("charge", f"ch_stripe{case_index}"),
        customer_id=f"CUST-STRIPE{case_index}",
        amount=amount,
        currency=currency,
        reason_code=reason_code,
        summary=dispute.get("evidence_details", {}).get("due_by_reason", f"Stripe dispute: {stripe_reason}"),
        inspection_notes=f"Stripe dispute {dispute_id} — {stripe_reason}. Status: {status}.",
        deadline_step=deadline_step,
        optimal_strategy=optimal,
        acceptable_strategies=acceptable,
        policy_guidance=guidance,
        policy_requirements=_POLICY_REQS.get(reason_code, ()),
        recommended_strategy=optimal,
        resolution_summary=f"Stripe dispute status: {status}.",
        weight=round(1.0 + (amount / 5000.0), 2),
        required_evidence_ids=req_ids,
        helpful_evidence_ids=hlp_ids,
        harmful_evidence_ids=hrm_ids,
        evidence_by_system=evidence,
    )


def build_stripe_task(
    disputes: list[dict[str, Any]],
    *,
    difficulty: str = "medium",
    task_index: int = 0,
) -> TaskScenario | None:
    """Build a TaskScenario from a list of Stripe dispute objects."""
    case_count = {"easy": 1, "medium": 2, "hard": 3}.get(difficulty, 2)
    max_steps = {"easy": 10, "medium": 12, "hard": max(12, case_count * 5)}.get(difficulty, 12)
    deadline = {"easy": 8, "medium": 7, "hard": 5}.get(difficulty, 7)

    cases: list[InternalCase] = []
    for i, dispute in enumerate(disputes):
        if len(cases) >= case_count:
            break
        case = dispute_to_case(dispute, i + 1, deadline_step=deadline)
        if case is not None:
            cases.append(case)

    if not cases:
        return None

    codes = ", ".join(list({c.reason_code for c in cases})[:3])
    return TaskScenario(
        task_id=f"stripe_{difficulty}_{task_index}",
        title=f"Stripe Dispute {'Queue' if len(cases) > 1 else 'Case'} ({difficulty.title()})",
        difficulty=difficulty,
        objective=f"Handle {len(cases)} Stripe dispute(s) ({codes}).",
        description=f"Real Stripe sandbox dispute scenario with {len(cases)} case(s). Codes: {codes}.",
        max_steps=max_steps,
        cases=tuple(cases),
    )


def fetch_disputes(*, limit: int = 10, api_key: str | None = None) -> list[dict[str, Any]]:
    """Fetch disputes from Stripe test mode.

    Requires ``stripe`` package and a test-mode API key.
    Falls back to synthetic test disputes if Stripe is unavailable.
    """
    key = api_key or os.environ.get("STRIPE_API_KEY", "")
    if not key or not key.startswith("sk_test_"):
        return _synthetic_test_disputes(limit)

    try:
        import stripe
        stripe.api_key = key
        result = stripe.Dispute.list(limit=limit)
        return [d.to_dict() if hasattr(d, "to_dict") else dict(d) for d in result.data]
    except Exception:
        return _synthetic_test_disputes(limit)


def _synthetic_test_disputes(count: int) -> list[dict[str, Any]]:
    """Generate synthetic Stripe-format dispute objects for testing without API access."""
    rng = random.Random(42)
    reasons = list(_STRIPE_REASON_MAP.keys())
    statuses = ["needs_response", "won", "lost", "under_review"]
    disputes = []

    for i in range(count):
        reason = rng.choice(reasons)
        status = rng.choice(statuses)
        amount = rng.randint(500, 50000)  # cents
        disputes.append({
            "id": f"dp_test_{i:04d}",
            "amount": amount,
            "currency": "usd",
            "reason": reason,
            "status": status,
            "charge": f"ch_test_{i:04d}",
            "metadata": {"description": f"Test dispute {i} — {reason}"},
            "evidence_details": {"due_by_reason": f"Dispute for {reason}"},
        })

    return disputes
