"""Internal task definitions and runtime types for ChargebackOps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

SystemName = Literal["orders", "payment", "shipping", "support", "refunds", "risk"]
StrategyName = Literal["contest", "accept_chargeback", "issue_refund"]


@dataclass(frozen=True)
class InternalEvidence:
    """Evidence item stored in a synthetic merchant system."""

    evidence_id: str
    source_system: SystemName
    title: str
    summary: str
    helpful: bool = False
    harmful: bool = False
    required: bool = False


@dataclass(frozen=True)
class InternalCase:
    """Synthetic chargeback case definition."""

    case_id: str
    order_id: str
    customer_id: str
    amount: float
    currency: str
    reason_code: str
    summary: str
    inspection_notes: str
    deadline_step: int
    optimal_strategy: StrategyName
    acceptable_strategies: tuple[StrategyName, ...]
    policy_guidance: str
    policy_requirements: tuple[str, ...]
    recommended_strategy: StrategyName
    resolution_summary: str
    weight: float
    evidence_by_system: dict[SystemName, tuple[InternalEvidence, ...]]
    required_evidence_ids: tuple[str, ...] = ()
    helpful_evidence_ids: tuple[str, ...] = ()
    harmful_evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskScenario:
    """One benchmark task."""

    task_id: str
    title: str
    difficulty: Literal["easy", "medium", "hard"]
    objective: str
    description: str
    max_steps: int
    cases: tuple[InternalCase, ...]


@dataclass
class CaseProgress:
    """Mutable runtime state for one case."""

    inspected: bool = False
    policy_retrieved: bool = False
    revealed_systems: set[SystemName] = field(default_factory=set)
    retrieved_evidence_ids: set[str] = field(default_factory=set)
    attached_evidence_ids: list[str] = field(default_factory=list)
    current_strategy: StrategyName | None = None
    final_resolution: str | None = None
    resolution_status: str = "open"
    resolved_at_step: int | None = None
    duplicate_queries: int = 0
    invalid_actions: int = 0
    submit_attempts: int = 0
    deadline_penalized: bool = False
    notes: list[str] = field(default_factory=list)
    representment_note: str | None = None


@dataclass
class ActionRecord:
    """Runtime action history."""

    step_index: int
    action_type: str
    case_id: str | None
    outcome: str
    reward: float


def _ev(
    evidence_id: str,
    source_system: SystemName,
    title: str,
    summary: str,
    *,
    helpful: bool = False,
    harmful: bool = False,
    required: bool = False,
) -> InternalEvidence:
    return InternalEvidence(
        evidence_id=evidence_id,
        source_system=source_system,
        title=title,
        summary=summary,
        helpful=helpful,
        harmful=harmful,
        required=required,
    )


TASKS: dict[str, TaskScenario] = {
    "goods_not_received_easy": TaskScenario(
        task_id="goods_not_received_easy",
        title="Delivered But Disputed",
        difficulty="easy",
        objective="Contest a goods-not-received chargeback with the right delivery proof before the deadline.",
        description=(
            "A single e-commerce dispute where carrier confirmation and the order confirmation "
            "are enough to win. The task teaches the standard representment loop."
        ),
        max_steps=10,
        cases=(
            InternalCase(
                case_id="CB-E1",
                order_id="ORD-7410",
                customer_id="CUST-1001",
                amount=129.99,
                currency="USD",
                reason_code="goods_not_received",
                summary="Cardholder claims the package never arrived.",
                inspection_notes=(
                    "Order shipped the same day. Merchant policy requires carrier proof plus the original order confirmation "
                    "for goods-not-received disputes."
                ),
                deadline_step=8,
                optimal_strategy="contest",
                acceptable_strategies=(),
                policy_guidance=(
                    "For goods-not-received disputes, prove the merchandise was fulfilled to the billed customer with "
                    "order confirmation and carrier delivery evidence."
                ),
                policy_requirements=("order confirmation", "carrier delivery confirmation"),
                recommended_strategy="contest",
                resolution_summary="Strong delivery proof exists. Contesting should recover the funds.",
                weight=1.0,
                required_evidence_ids=("E1-ORDER-CONF", "E1-DELIVERY-SCAN"),
                helpful_evidence_ids=("E1-ORDER-CONF", "E1-DELIVERY-SCAN", "E1-SUPPORT-ACK"),
                harmful_evidence_ids=(),
                evidence_by_system={
                    "orders": (
                        _ev(
                            "E1-ORDER-CONF",
                            "orders",
                            "Order confirmation",
                            "Order confirmation email and checkout receipt showing the billed customer, shipping address, and SKU.",
                            helpful=True,
                            required=True,
                        ),
                    ),
                    "payment": (
                        _ev(
                            "E1-AUTH",
                            "payment",
                            "Authorization record",
                            "Authorization approved and captured successfully.",
                        ),
                    ),
                    "shipping": (
                        _ev(
                            "E1-DELIVERY-SCAN",
                            "shipping",
                            "Carrier delivery scan",
                            "Carrier tracking shows delivered to the customer address two days after shipment.",
                            helpful=True,
                            required=True,
                        ),
                        _ev(
                            "E1-SIGNATURE",
                            "shipping",
                            "Doorstep photo confirmation",
                            "Carrier stored a package photo at the delivery location.",
                            helpful=True,
                        ),
                    ),
                    "support": (
                        _ev(
                            "E1-SUPPORT-ACK",
                            "support",
                            "Support ticket acknowledgement",
                            "Customer contacted support to ask if the package was left at the front desk after delivery.",
                            helpful=True,
                        ),
                    ),
                    "refunds": (
                        _ev(
                            "E1-NO-REFUND",
                            "refunds",
                            "Refund ledger",
                            "No refund or goodwill credit was issued before the dispute opened.",
                        ),
                    ),
                    "risk": (
                        _ev(
                            "E1-RISK",
                            "risk",
                            "Risk summary",
                            "Low-risk order with no fraud flags.",
                        ),
                    ),
                },
            ),
        ),
    ),
    "fraud_signal_ambiguity": TaskScenario(
        task_id="fraud_signal_ambiguity",
        title="Fraud Signal Ambiguity",
        difficulty="medium",
        objective="Choose whether to contest a CNP fraud dispute and curate only the evidence that helps.",
        description=(
            "A card-not-present fraud dispute with mixed signals. Strong account-linkage evidence exists, "
            "but payment mismatch artifacts will hurt the case if attached."
        ),
        max_steps=10,
        cases=(
            InternalCase(
                case_id="CB-M1",
                order_id="ORD-8821",
                customer_id="CUST-2048",
                amount=480.0,
                currency="USD",
                reason_code="fraud_cnp",
                summary="Issuer filed a card-not-present fraud dispute on a high-value electronics order.",
                inspection_notes=(
                    "The order used a known account and device, but AVS/CVV mismatches were present. "
                    "Winning requires emphasizing customer-account linkage and avoiding mismatch artifacts."
                ),
                deadline_step=7,
                optimal_strategy="contest",
                acceptable_strategies=("accept_chargeback",),
                policy_guidance=(
                    "For CNP fraud disputes, contest only when you can link the cardholder to the account or device history. "
                    "Do not attach evidence that strengthens the issuer's fraud narrative."
                ),
                policy_requirements=("prior good order linkage", "customer account confirmation"),
                recommended_strategy="contest",
                resolution_summary="Contest only with strong account-linkage evidence. Conceding is acceptable but suboptimal.",
                weight=1.1,
                required_evidence_ids=("M1-PRIOR-ORDERS", "M1-ACCOUNT-CHAT"),
                helpful_evidence_ids=("M1-PRIOR-ORDERS", "M1-ACCOUNT-CHAT", "M1-DELIVERY"),
                harmful_evidence_ids=("M1-AVS-MISMATCH", "M1-CVV-MISMATCH"),
                evidence_by_system={
                    "orders": (
                        _ev(
                            "M1-ORDER",
                            "orders",
                            "Order receipt",
                            "Checkout receipt showing customer account id, shipping address, and same email as prior purchases.",
                            helpful=True,
                        ),
                    ),
                    "payment": (
                        _ev(
                            "M1-AVS-MISMATCH",
                            "payment",
                            "AVS mismatch detail",
                            "Street-number mismatch was recorded at authorization time.",
                            harmful=True,
                        ),
                        _ev(
                            "M1-CVV-MISMATCH",
                            "payment",
                            "CVV mismatch detail",
                            "CVV did not fully match at authorization time.",
                            harmful=True,
                        ),
                        _ev(
                            "M1-AUTH",
                            "payment",
                            "Authorization capture",
                            "Payment was successfully authorized and captured.",
                        ),
                    ),
                    "shipping": (
                        _ev(
                            "M1-DELIVERY",
                            "shipping",
                            "Carrier delivery confirmation",
                            "Package was delivered to the saved customer address two days later.",
                            helpful=True,
                        ),
                    ),
                    "support": (
                        _ev(
                            "M1-ACCOUNT-CHAT",
                            "support",
                            "Authenticated support chat",
                            "Customer logged into the account and confirmed the delivery window in chat before shipment.",
                            helpful=True,
                            required=True,
                        ),
                    ),
                    "refunds": (
                        _ev(
                            "M1-NO-REFUND",
                            "refunds",
                            "Refund ledger",
                            "No refund or cancellation was issued prior to the dispute.",
                        ),
                    ),
                    "risk": (
                        _ev(
                            "M1-PRIOR-ORDERS",
                            "risk",
                            "Prior account activity",
                            "Same account, same device fingerprint, and three prior fulfilled orders without disputes.",
                            helpful=True,
                            required=True,
                        ),
                        _ev(
                            "M1-VELOCITY",
                            "risk",
                            "Velocity check",
                            "No abnormal velocity or proxy usage detected.",
                            helpful=True,
                        ),
                    ),
                },
            ),
        ),
    ),
    "queue_optimization_hard": TaskScenario(
        task_id="queue_optimization_hard",
        title="Dispute Queue Optimization",
        difficulty="hard",
        objective="Maximize recovery across a queue of disputes while respecting deadlines and avoiding weak contests.",
        description=(
            "A real operations queue with three disputes. Two should be actioned quickly, and one should be conceded. "
            "The step budget leaves little room for waste."
        ),
        max_steps=15,
        cases=(
            InternalCase(
                case_id="CB-H1",
                order_id="ORD-9901",
                customer_id="CUST-4100",
                amount=860.0,
                currency="USD",
                reason_code="goods_not_received",
                summary="High-value furniture delivery disputed as not received.",
                inspection_notes=(
                    "Carrier stored both a delivery scan and signature. This is the highest-value recoverable case in the queue."
                ),
                deadline_step=7,
                optimal_strategy="contest",
                acceptable_strategies=(),
                policy_guidance=(
                    "Use merchant receipt plus carrier proof for goods-not-received disputes. This case is strong if contested on time."
                ),
                policy_requirements=("order confirmation", "signature-backed delivery proof"),
                recommended_strategy="contest",
                resolution_summary="Contest immediately with the signature-backed delivery packet.",
                weight=1.7,
                required_evidence_ids=("H1-ORDER-CONF", "H1-SIGNATURE"),
                helpful_evidence_ids=("H1-ORDER-CONF", "H1-SIGNATURE", "H1-DELIVERY-SCAN"),
                harmful_evidence_ids=(),
                evidence_by_system={
                    "orders": (
                        _ev(
                            "H1-ORDER-CONF",
                            "orders",
                            "Order invoice",
                            "Signed furniture order invoice with billing and delivery address.",
                            helpful=True,
                            required=True,
                        ),
                    ),
                    "payment": (
                        _ev(
                            "H1-AUTH",
                            "payment",
                            "Captured payment",
                            "Payment authorization and capture both succeeded.",
                        ),
                    ),
                    "shipping": (
                        _ev(
                            "H1-SIGNATURE",
                            "shipping",
                            "Delivery signature",
                            "Carrier recorded a recipient signature at the shipping address.",
                            helpful=True,
                            required=True,
                        ),
                        _ev(
                            "H1-DELIVERY-SCAN",
                            "shipping",
                            "Final-mile delivery scan",
                            "Tracking confirms delivery within the promised window.",
                            helpful=True,
                        ),
                    ),
                    "support": (
                        _ev(
                            "H1-SUPPORT",
                            "support",
                            "Support history",
                            "No delivery complaint was opened before the dispute.",
                        ),
                    ),
                    "refunds": (
                        _ev(
                            "H1-NO-REFUND",
                            "refunds",
                            "Refund ledger",
                            "No refund was issued.",
                        ),
                    ),
                    "risk": (
                        _ev(
                            "H1-RISK",
                            "risk",
                            "Risk summary",
                            "Low-risk order. No notable fraud flags.",
                        ),
                    ),
                },
            ),
            InternalCase(
                case_id="CB-H2",
                order_id="ORD-9902",
                customer_id="CUST-4101",
                amount=240.0,
                currency="USD",
                reason_code="fraud_cnp",
                summary="Apparel order disputed as unauthorized.",
                inspection_notes=(
                    "The account is new, there is no durable linkage to the cardholder, and the payment record contains mismatch artifacts. "
                    "This case should be conceded."
                ),
                deadline_step=14,
                optimal_strategy="accept_chargeback",
                acceptable_strategies=("issue_refund",),
                policy_guidance=(
                    "Do not contest when you lack durable account or device linkage. Avoid wasting steps on weak fraud disputes."
                ),
                policy_requirements=("cardholder linkage evidence"),
                recommended_strategy="accept_chargeback",
                resolution_summary="Concede the dispute. Contesting wastes portfolio value.",
                weight=0.8,
                required_evidence_ids=(),
                helpful_evidence_ids=(),
                harmful_evidence_ids=("H2-AVS", "H2-CVV"),
                evidence_by_system={
                    "orders": (
                        _ev(
                            "H2-ORDER",
                            "orders",
                            "Order receipt",
                            "Guest checkout with a new shipping address and no prior order history.",
                        ),
                    ),
                    "payment": (
                        _ev(
                            "H2-AVS",
                            "payment",
                            "AVS mismatch detail",
                            "Street and postal code mismatches were present.",
                            harmful=True,
                        ),
                        _ev(
                            "H2-CVV",
                            "payment",
                            "CVV mismatch detail",
                            "CVV did not match.",
                            harmful=True,
                        ),
                    ),
                    "shipping": (
                        _ev(
                            "H2-DELIVERY",
                            "shipping",
                            "Carrier delivery confirmation",
                            "Delivered to a new address without signature.",
                        ),
                    ),
                    "support": (
                        _ev(
                            "H2-SUPPORT",
                            "support",
                            "Support log",
                            "No authenticated support interactions were recorded.",
                        ),
                    ),
                    "refunds": (
                        _ev(
                            "H2-NO-REFUND",
                            "refunds",
                            "Refund ledger",
                            "No refund issued before the chargeback.",
                        ),
                    ),
                    "risk": (
                        _ev(
                            "H2-RISK",
                            "risk",
                            "Risk summary",
                            "Elevated risk score and no positive account history.",
                        ),
                    ),
                },
            ),
            InternalCase(
                case_id="CB-H3",
                order_id="ORD-9903",
                customer_id="CUST-4102",
                amount=320.0,
                currency="USD",
                reason_code="credit_not_processed",
                summary="Subscriber canceled before renewal and says the credit was never processed.",
                inspection_notes=(
                    "The merchant missed the promised refund SLA. This should be resolved fast with a refund, not a contest."
                ),
                deadline_step=4,
                optimal_strategy="issue_refund",
                acceptable_strategies=("accept_chargeback",),
                policy_guidance=(
                    "If the merchant failed to process a promised credit, refund immediately or concede. Contesting is not supportable."
                ),
                policy_requirements=("proof of cancellation request", "refund status check"),
                recommended_strategy="issue_refund",
                resolution_summary="Refund immediately. Delay turns a manageable loss into a deadline miss.",
                weight=1.2,
                required_evidence_ids=(),
                helpful_evidence_ids=("H3-CANCEL", "H3-NO-REFUND"),
                harmful_evidence_ids=(),
                evidence_by_system={
                    "orders": (
                        _ev(
                            "H3-ORDER",
                            "orders",
                            "Renewal invoice",
                            "Subscription renewed automatically for the annual plan.",
                        ),
                    ),
                    "payment": (
                        _ev(
                            "H3-PAYMENT",
                            "payment",
                            "Captured renewal payment",
                            "Renewal payment settled successfully.",
                        ),
                    ),
                    "shipping": (),
                    "support": (
                        _ev(
                            "H3-CANCEL",
                            "support",
                            "Cancellation request",
                            "Customer requested cancellation before renewal and support promised a refund within five business days.",
                            helpful=True,
                        ),
                    ),
                    "refunds": (
                        _ev(
                            "H3-NO-REFUND",
                            "refunds",
                            "Refund ledger",
                            "No refund has been issued as of the dispute open date.",
                            helpful=True,
                        ),
                    ),
                    "risk": (),
                },
            ),
        ),
    ),
}


def get_task(task_id: str) -> TaskScenario:
    """Look up a built-in task or generate one from a ``generated_*`` id."""

    if task_id in TASKS:
        return TASKS[task_id]

    # Support generated task ids: generated_{difficulty}_s{seed}
    import re

    m = re.match(r"^generated_(easy|medium|hard)_s(\d+)$", task_id)
    if m:
        try:
            from .case_generator import generate_task
        except ImportError:  # pragma: no cover
            from case_generator import generate_task
        difficulty = m.group(1)
        seed = int(m.group(2))
        return generate_task(seed, difficulty=difficulty)  # type: ignore[arg-type]

    # Support ISO-derived task ids: iso_{difficulty}_{index}
    m_iso = re.match(r"^iso_(easy|medium|hard)_(\d+)$", task_id)
    if m_iso:
        try:
            from .iso_adapter import build_iso_task, load_iso_rows
        except ImportError:  # pragma: no cover
            from iso_adapter import build_iso_task, load_iso_rows
        difficulty = m_iso.group(1)
        task_index = int(m_iso.group(2))
        rows = load_iso_rows()
        if rows:
            import random as _rng_mod
            shuffled = list(rows)
            _rng_mod.Random(42).shuffle(shuffled)
            task = build_iso_task(shuffled, difficulty=difficulty, start_index=task_index * 4, task_index=task_index)
            if task is not None:
                return task

    raise ValueError(f"Unknown task_id '{task_id}'. Available: {', '.join(TASKS)}")


def list_tasks() -> list[TaskScenario]:
    """Return built-in tasks in a stable order."""

    ordered_ids = [
        "goods_not_received_easy",
        "fraud_signal_ambiguity",
        "queue_optimization_hard",
    ]
    return [TASKS[task_id] for task_id in ordered_ids]
