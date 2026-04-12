"""Adapter that converts real ISO 20022 chargeback CSV rows into environment cases.

Reads ``data/iso20022-card-chargeback-casr-003.csv`` and produces
``InternalCase`` / ``TaskScenario`` objects so real dispute data flows
through the benchmark.
"""

from __future__ import annotations

import csv
import hashlib
import random
from pathlib import Path

try:
    from .simulation import InternalCase, InternalEvidence, TaskScenario, SystemName
except ImportError:  # pragma: no cover
    from simulation import InternalCase, InternalEvidence, TaskScenario, SystemName

ISO_CSV_PATH = Path("data/iso20022-card-chargeback-casr-003.csv")

_REASON_MAP: dict[str, str] = {
    "goods_not_received": "goods_not_received",
    "GOODS_NOT_RECEIVED": "goods_not_received",
    "NR02": "goods_not_received",
    "FRAUD": "fraud_cnp",
    "fraud": "fraud_cnp",
    "fraudulent_transaction": "fraud_cnp",
    "FR01": "fraud_cnp",
    "FR02": "fraud_cnp",
    "goods_not_as_described": "product_not_as_described",
    "GOODS_NOT_AS_DESCRIBED": "product_not_as_described",
    "not_as_described": "product_not_as_described",
    "NR04": "product_not_as_described",
    "SERVICE_NOT_RENDERED": "service_not_provided",
    "services_not_rendered": "service_not_provided",
    "NR03": "credit_not_processed",
    "duplicate": "duplicate_processing",
    "DUPLICATE_PROCESSING": "duplicate_processing",
    "duplicate_processing": "duplicate_processing",
}

_MERCHANT_WON = {"merchant_won", "chargeback_reversed", "chargeback_declined"}
_CONCEDED = {"chargeback_accepted"}

_POLICY_GUIDANCE: dict[str, str] = {
    "goods_not_received": "For goods-not-received disputes, prove fulfillment with order confirmation and carrier delivery evidence.",
    "fraud_cnp": "For CNP fraud disputes, contest only when you can link the cardholder to the account or device history. Do not attach mismatch artifacts.",
    "product_not_as_described": "Contest product-not-as-described disputes when the listing accurately represents the product and the customer bypassed the return process.",
    "service_not_provided": "Contest service-not-provided disputes when provider records confirm the service was delivered.",
    "credit_not_processed": "If the merchant failed to process a promised credit, refund immediately or concede. Contesting is not supportable.",
    "duplicate_processing": "When a duplicate charge is confirmed, refund the extra amount immediately. Do not contest.",
}

_POLICY_REQS: dict[str, tuple[str, ...]] = {
    "goods_not_received": ("order confirmation", "carrier delivery confirmation"),
    "fraud_cnp": ("prior good order linkage", "customer account confirmation"),
    "product_not_as_described": (
        "product listing verification",
        "return policy documentation",
    ),
    "service_not_provided": ("service completion record", "customer acknowledgment"),
    "credit_not_processed": ("proof of cancellation request", "refund status check"),
    "duplicate_processing": ("payment transaction log", "duplicate confirmation"),
}


def _ev(eid, system, title, summary, *, helpful=False, harmful=False, required=False):
    return InternalEvidence(
        evidence_id=eid,
        source_system=system,
        title=title,
        summary=summary,
        helpful=helpful,
        harmful=harmful,
        required=required,
    )


def _infer_strategy(reason_code, final_decision, notes):
    nl = notes.lower()
    if final_decision in _MERCHANT_WON:
        return "contest", ()
    if final_decision in _CONCEDED:
        if reason_code in ("credit_not_processed", "duplicate_processing"):
            return "issue_refund", ("accept_chargeback",)
        return "accept_chargeback", ("issue_refund",)
    if reason_code in ("credit_not_processed", "duplicate_processing"):
        return "issue_refund", ("accept_chargeback",)
    if reason_code == "fraud_cnp" and (
        "stolen" in nl or "no evidence" in nl or "unable" in nl
    ):
        return "accept_chargeback", ("issue_refund",)
    return "contest", ()


def _build_evidence(prefix, reason_code, merchant, amount, notes, optimal, rng):
    by_sys: dict[SystemName, list[InternalEvidence]] = {
        s: [] for s in ("orders", "payment", "shipping", "support", "refunds", "risk")
    }
    req, hlp, hrm = [], [], []

    if reason_code == "goods_not_received":
        e = _ev(
            f"{prefix}-ORDER",
            "orders",
            "Order confirmation",
            f"Order with {merchant} for ${amount:.2f}.",
            helpful=True,
            required=True,
        )
        by_sys["orders"].append(e)
        req.append(e.evidence_id)
        hlp.append(e.evidence_id)
        by_sys["payment"].append(
            _ev(
                f"{prefix}-AUTH",
                "payment",
                "Authorization",
                "Payment authorized and captured.",
            )
        )
        if optimal == "contest":
            e = _ev(
                f"{prefix}-DELIVERY",
                "shipping",
                "Carrier delivery confirmation",
                "Carrier confirms delivery to customer address.",
                helpful=True,
                required=True,
            )
            by_sys["shipping"].append(e)
            req.append(e.evidence_id)
            hlp.append(e.evidence_id)
            if rng.random() > 0.4:
                e2 = _ev(
                    f"{prefix}-SIG",
                    "shipping",
                    "Delivery signature",
                    "Recipient signature on file.",
                    helpful=True,
                )
                by_sys["shipping"].append(e2)
                hlp.append(e2.evidence_id)
        else:
            by_sys["shipping"].append(
                _ev(
                    f"{prefix}-NOTRACK",
                    "shipping",
                    "Tracking status",
                    "No confirmed delivery scan.",
                )
            )
        by_sys["support"].append(
            _ev(
                f"{prefix}-SUPPORT",
                "support",
                "Support notes",
                notes[:120] if notes else "No support interactions.",
            )
        )
        by_sys["refunds"].append(
            _ev(
                f"{prefix}-REFUND",
                "refunds",
                "Refund ledger",
                "No refund issued before dispute.",
            )
        )

    elif reason_code == "fraud_cnp":
        by_sys["orders"].append(
            _ev(
                f"{prefix}-ORDER",
                "orders",
                "Order receipt",
                f"Order with {merchant} for ${amount:.2f}.",
                helpful=True,
            )
        )
        hlp.append(f"{prefix}-ORDER")
        e_avs = _ev(
            f"{prefix}-AVS",
            "payment",
            "AVS mismatch",
            "Street mismatch at authorization.",
            harmful=True,
        )
        by_sys["payment"].append(e_avs)
        hrm.append(e_avs.evidence_id)
        if rng.random() > 0.5:
            e_cvv = _ev(
                f"{prefix}-CVV",
                "payment",
                "CVV mismatch",
                "CVV verification failed.",
                harmful=True,
            )
            by_sys["payment"].append(e_cvv)
            hrm.append(e_cvv.evidence_id)
        by_sys["payment"].append(
            _ev(f"{prefix}-AUTH", "payment", "Authorization", "Payment captured.")
        )
        if optimal == "contest":
            e = _ev(
                f"{prefix}-PRIOR",
                "risk",
                "Prior account activity",
                "Same account/device with prior fulfilled orders.",
                helpful=True,
                required=True,
            )
            by_sys["risk"].append(e)
            req.append(e.evidence_id)
            hlp.append(e.evidence_id)
            e = _ev(
                f"{prefix}-CHAT",
                "support",
                "Authenticated chat",
                "Customer logged in and confirmed order.",
                helpful=True,
                required=True,
            )
            by_sys["support"].append(e)
            req.append(e.evidence_id)
            hlp.append(e.evidence_id)
        else:
            by_sys["risk"].append(
                _ev(
                    f"{prefix}-RISK",
                    "risk",
                    "Risk summary",
                    "Elevated risk. No positive account history.",
                )
            )
            by_sys["support"].append(
                _ev(
                    f"{prefix}-SUPPORT",
                    "support",
                    "Support log",
                    "No authenticated interactions.",
                )
            )
        by_sys["shipping"].append(
            _ev(
                f"{prefix}-DELIVERY",
                "shipping",
                "Delivery confirmation",
                "Delivered to address on file.",
                helpful=True,
            )
        )
        hlp.append(f"{prefix}-DELIVERY")
        by_sys["refunds"].append(
            _ev(f"{prefix}-REFUND", "refunds", "Refund ledger", "No refund issued.")
        )

    elif reason_code == "product_not_as_described":
        e = _ev(
            f"{prefix}-ORDER",
            "orders",
            "Order details",
            f"Order with {merchant} — SKU matches listing.",
            helpful=True,
            required=True,
        )
        by_sys["orders"].append(e)
        req.append(e.evidence_id)
        hlp.append(e.evidence_id)
        e = _ev(
            f"{prefix}-LISTING",
            "orders",
            "Product listing",
            "Listing matches manufacturer specs.",
            helpful=True,
            required=True,
        )
        by_sys["orders"].append(e)
        req.append(e.evidence_id)
        hlp.append(e.evidence_id)
        by_sys["payment"].append(
            _ev(
                f"{prefix}-AUTH",
                "payment",
                "Payment capture",
                "Settled for listed price.",
            )
        )
        by_sys["shipping"].append(
            _ev(
                f"{prefix}-DELIVERY",
                "shipping",
                "Delivery confirmation",
                "Delivered within window.",
                helpful=True,
            )
        )
        hlp.append(f"{prefix}-DELIVERY")
        by_sys["support"].append(
            _ev(
                f"{prefix}-RETURN",
                "support",
                "Return policy",
                "No return initiated before dispute.",
                helpful=True,
            )
        )
        hlp.append(f"{prefix}-RETURN")
        by_sys["refunds"].append(
            _ev(f"{prefix}-REFUND", "refunds", "Refund ledger", "No refund processed.")
        )

    elif reason_code == "service_not_provided":
        e = _ev(
            f"{prefix}-BOOKING",
            "orders",
            "Service booking",
            f"Booking with {merchant} for ${amount:.2f}.",
            helpful=True,
            required=True,
        )
        by_sys["orders"].append(e)
        req.append(e.evidence_id)
        hlp.append(e.evidence_id)
        by_sys["payment"].append(
            _ev(f"{prefix}-AUTH", "payment", "Payment record", "Payment captured.")
        )
        if optimal == "contest":
            e = _ev(
                f"{prefix}-COMPLETION",
                "support",
                "Service completion",
                "Provider marked service completed.",
                helpful=True,
                required=True,
            )
            by_sys["support"].append(e)
            req.append(e.evidence_id)
            hlp.append(e.evidence_id)
        else:
            by_sys["support"].append(
                _ev(
                    f"{prefix}-CANCEL",
                    "support",
                    "Cancellation",
                    notes[:100] if notes else "Service cancelled.",
                    helpful=True,
                )
            )
            hlp.append(f"{prefix}-CANCEL")
        by_sys["refunds"].append(
            _ev(f"{prefix}-REFUND", "refunds", "Refund ledger", "No refund issued.")
        )

    elif reason_code in ("credit_not_processed", "duplicate_processing"):
        by_sys["orders"].append(
            _ev(
                f"{prefix}-ORDER",
                "orders",
                "Invoice",
                f"Charge of ${amount:.2f} from {merchant}.",
            )
        )
        by_sys["payment"].append(
            _ev(f"{prefix}-PAYMENT", "payment", "Payment", "Payment settled.")
        )
        by_sys["support"].append(
            _ev(
                f"{prefix}-REQ",
                "support",
                "Customer request",
                notes[:100] if notes else "Customer requested credit.",
                helpful=True,
            )
        )
        hlp.append(f"{prefix}-REQ")
        by_sys["refunds"].append(
            _ev(
                f"{prefix}-NOREFUND",
                "refunds",
                "Refund ledger",
                "No refund processed.",
                helpful=True,
            )
        )
        hlp.append(f"{prefix}-NOREFUND")

    frozen = {k: tuple(v) for k, v in by_sys.items()}
    return frozen, tuple(req), tuple(hlp), tuple(hrm)


def _concedable_guidance(reason_code: str, optimal: str) -> str:
    """Return guidance that signals concede when the optimal strategy isn't contest."""
    if optimal in ("accept_chargeback", "issue_refund") and reason_code not in (
        "credit_not_processed",
        "duplicate_processing",
    ):
        if optimal == "accept_chargeback":
            return (
                f"Do not contest this {reason_code.replace('_', ' ')} dispute. "
                "The merchant's position is not supportable. Concede to avoid wasting resources."
            )
        return (
            f"Refund immediately for this {reason_code.replace('_', ' ')} dispute. "
            "Contesting is not supportable."
        )
    return _POLICY_GUIDANCE.get(reason_code, "")


def row_to_case(row, case_index, *, deadline_step=8):
    raw_code = row.get("chargeback_reason_code", "")
    reason_code = _REASON_MAP.get(raw_code)
    if reason_code is None:
        return None

    amount = float(row.get("transaction_amount", "0") or "0")
    merchant = row.get("merchant_name", "Unknown")
    notes = row.get("notes", "")
    final_decision = row.get("final_decision", "")

    optimal, acceptable = _infer_strategy(reason_code, final_decision, notes)
    rng = random.Random(
        int(hashlib.sha256(row["chargeback_id"].encode()).hexdigest()[:8], 16)
    )
    prefix = f"ISO{case_index}"

    evidence, req_ids, hlp_ids, hrm_ids = _build_evidence(
        prefix, reason_code, merchant, amount, notes, optimal, rng
    )

    return InternalCase(
        case_id=f"CB-ISO{case_index}",
        order_id=row.get("original_transaction_id", f"TX-ISO{case_index}"),
        customer_id=f"CUST-ISO{case_index}",
        amount=amount,
        currency=row.get("transaction_currency", "USD"),
        reason_code=reason_code,
        summary=row.get("chargeback_reason_description", "Chargeback filed."),
        inspection_notes=notes or f"Chargeback against {merchant} for ${amount:.2f}.",
        deadline_step=deadline_step,
        optimal_strategy=optimal,
        acceptable_strategies=acceptable,
        policy_guidance=_concedable_guidance(reason_code, optimal),
        policy_requirements=_POLICY_REQS.get(reason_code, ()),
        recommended_strategy=optimal,
        resolution_summary=f"Real case outcome: {final_decision or 'pending'}.",
        weight=round(1.0 + (amount / 5000.0), 2),
        required_evidence_ids=req_ids,
        helpful_evidence_ids=hlp_ids,
        harmful_evidence_ids=hrm_ids,
        evidence_by_system=evidence,
    )


def load_iso_rows(csv_path=None):
    path = csv_path or ISO_CSV_PATH
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_iso_task(
    rows, *, difficulty="medium", start_index=0, case_count=None, task_index=0
):
    if case_count is None:
        case_count = {"easy": 1, "medium": 2, "hard": 3}[difficulty]
    max_steps = {"easy": 10, "medium": 12, "hard": max(12, case_count * 5)}[difficulty]

    cases = []
    idx = start_index
    while len(cases) < case_count and idx < len(rows):
        deadline = {"easy": 8, "medium": 7, "hard": max(4, 8 - len(cases))}[difficulty]
        case = row_to_case(rows[idx], idx + 1, deadline_step=deadline)
        idx += 1
        if case is not None:
            cases.append(case)

    if not cases:
        return None

    codes = ", ".join(list({c.reason_code for c in cases})[:3])
    return TaskScenario(
        task_id=f"iso_{difficulty}_{task_index}",
        title=f"ISO Dispute {'Queue' if len(cases) > 1 else 'Case'} ({difficulty.title()})",
        difficulty=difficulty,
        objective=f"Handle {len(cases)} real dispute(s) ({codes}) from ISO 20022 chargeback data.",
        description=f"Real-world-derived scenario with {len(cases)} case(s). Reason codes: {codes}.",
        max_steps=max_steps,
        cases=tuple(cases),
    )


def generate_iso_suite(csv_path=None, *, easy_count=3, medium_count=3, hard_count=3):
    rows = load_iso_rows(csv_path)
    if not rows:
        return []
    rng = random.Random(42)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    tasks, offset, idx = [], 0, 0
    for diff, count in [
        ("easy", easy_count),
        ("medium", medium_count),
        ("hard", hard_count),
    ]:
        for _ in range(count):
            task = build_iso_task(
                shuffled, difficulty=diff, start_index=offset, task_index=idx
            )
            if task is not None:
                tasks.append(task)
                offset += len(task.cases) + 1
                idx += 1
    return tasks
