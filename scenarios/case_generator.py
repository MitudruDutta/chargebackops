"""Parametric case generator for ChargebackOps.

Generates reproducible chargeback cases from reason-code templates using a
seeded RNG.  Every seed produces the same cases, so benchmarks are replayable
while the scenario space is effectively infinite.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

try:
    from .simulation import (
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


# ---------------------------------------------------------------------------
# Evidence blueprint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _EvidenceBlueprint:
    """Template for one evidence item inside a case template."""

    id_suffix: str
    source_system: SystemName
    title: str
    summaries: tuple[str, ...]
    helpful: bool = False
    harmful: bool = False
    required: bool = False
    probability: float = 1.0  # 0-1, chance of appearing in a generated case


# ---------------------------------------------------------------------------
# Case template
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CaseTemplate:
    """Blueprint for one reason-code family."""

    reason_code: str
    summaries: tuple[str, ...]
    inspection_notes: tuple[str, ...]
    policy_guidance: str
    policy_requirements: tuple[str, ...]
    optimal_strategy: StrategyName
    acceptable_strategies: tuple[StrategyName, ...]
    resolution_summary: str
    base_weight: float
    evidence_blueprints: tuple[_EvidenceBlueprint, ...]
    # If set, the template can "flip" to this strategy when key evidence is
    # missing (probability-gated blueprints not generated).
    weak_variant_strategy: StrategyName | None = None


# ---------------------------------------------------------------------------
# Amount / ID generation helpers
# ---------------------------------------------------------------------------

_FIRST_NAMES = (
    "Alex", "Jordan", "Sam", "Morgan", "Casey", "Riley", "Taylor",
    "Quinn", "Avery", "Dakota", "Reese", "Blake", "Skyler", "Drew",
)


def _amount(rng: random.Random, lo: float, hi: float) -> float:
    return round(rng.uniform(lo, hi), 2)


def _customer_id(rng: random.Random) -> str:
    return f"CUST-{rng.randint(1000, 9999)}"


def _order_id(rng: random.Random) -> str:
    return f"ORD-{rng.randint(10000, 99999)}"


# ---------------------------------------------------------------------------
# Reason-code template library  (6 families)
# ---------------------------------------------------------------------------


_GOODS_NOT_RECEIVED = _CaseTemplate(
    reason_code="goods_not_received",
    summaries=(
        "Cardholder claims the package was never delivered.",
        "Customer says the order never arrived at the shipping address.",
        "Buyer disputes the charge, stating the parcel was not received.",
    ),
    inspection_notes=(
        "Order shipped promptly. Carrier tracking and order confirmation are available.",
        "Merchant fulfilled and shipped within SLA. Delivery proof exists in the shipping system.",
        "Same-day shipment with standard carrier. Proof of delivery should be retrievable.",
    ),
    policy_guidance=(
        "For goods-not-received disputes, prove the merchandise was fulfilled "
        "to the billed customer with order confirmation and carrier delivery evidence."
    ),
    policy_requirements=("order confirmation", "carrier delivery confirmation"),
    optimal_strategy="contest",
    acceptable_strategies=(),
    resolution_summary="Strong delivery proof exists. Contesting should recover the funds.",
    base_weight=1.0,
    evidence_blueprints=(
        _EvidenceBlueprint(
            "ORDER-CONF", "orders", "Order confirmation",
            (
                "Order confirmation email and checkout receipt showing the billed customer, shipping address, and SKU.",
                "Original order receipt with billing name, address, and itemized products.",
            ),
            helpful=True, required=True,
        ),
        _EvidenceBlueprint(
            "AUTH", "payment", "Authorization record",
            ("Authorization approved and captured successfully.",),
        ),
        _EvidenceBlueprint(
            "DELIVERY", "shipping", "Carrier delivery scan",
            (
                "Carrier tracking shows delivered to the customer address.",
                "Delivery scan confirms package arrived at the registered shipping address.",
            ),
            helpful=True, required=True,
        ),
        _EvidenceBlueprint(
            "SIGNATURE", "shipping", "Delivery signature",
            (
                "Carrier recorded a recipient signature at the delivery address.",
                "Signed-for delivery confirmation at destination.",
            ),
            helpful=True, probability=0.6,
        ),
        _EvidenceBlueprint(
            "SUPPORT", "support", "Support interaction",
            (
                "Customer contacted support asking about delivery status after tracking showed delivered.",
                "Customer inquired about the package location post-delivery.",
            ),
            helpful=True, probability=0.5,
        ),
        _EvidenceBlueprint(
            "NO-REFUND", "refunds", "Refund ledger",
            ("No refund or goodwill credit was issued before the dispute opened.",),
        ),
        _EvidenceBlueprint(
            "RISK", "risk", "Risk summary",
            ("Low-risk order with no fraud flags.",),
        ),
    ),
)


_FRAUD_CNP_STRONG = _CaseTemplate(
    reason_code="fraud_cnp",
    summaries=(
        "Issuer filed a card-not-present fraud dispute on an online order.",
        "Card-not-present transaction disputed as unauthorized by the issuing bank.",
        "CNP fraud claim on an e-commerce purchase.",
    ),
    inspection_notes=(
        "The order used a known account and device. Account linkage evidence is available.",
        "Returning customer with consistent device fingerprint. Good account history exists.",
        "Established customer account with prior successful orders from the same device.",
    ),
    policy_guidance=(
        "For CNP fraud disputes, contest only when you can link the cardholder "
        "to the account or device history. Do not attach evidence that strengthens "
        "the issuer's fraud narrative."
    ),
    policy_requirements=("prior good order linkage", "customer account confirmation"),
    optimal_strategy="contest",
    acceptable_strategies=("accept_chargeback",),
    resolution_summary="Contest with strong account-linkage evidence. Conceding is acceptable but suboptimal.",
    base_weight=1.1,
    evidence_blueprints=(
        _EvidenceBlueprint(
            "ORDER", "orders", "Order receipt",
            (
                "Checkout receipt with customer account id and shipping address matching prior purchases.",
                "Order confirmation linked to an established customer account.",
            ),
            helpful=True,
        ),
        _EvidenceBlueprint(
            "AVS-MISMATCH", "payment", "AVS mismatch detail",
            (
                "Street-number mismatch was recorded at authorization time.",
                "AVS partial match — zip matched but street did not.",
            ),
            harmful=True, probability=0.7,
        ),
        _EvidenceBlueprint(
            "CVV-MISMATCH", "payment", "CVV mismatch detail",
            (
                "CVV did not fully match at authorization time.",
                "CVV verification returned a mismatch result.",
            ),
            harmful=True, probability=0.5,
        ),
        _EvidenceBlueprint(
            "AUTH", "payment", "Authorization capture",
            ("Payment was successfully authorized and captured.",),
        ),
        _EvidenceBlueprint(
            "DELIVERY", "shipping", "Carrier delivery confirmation",
            (
                "Package was delivered to the saved customer address.",
                "Carrier tracking confirms delivery to the account's shipping address.",
            ),
            helpful=True,
        ),
        _EvidenceBlueprint(
            "ACCOUNT-CHAT", "support", "Authenticated support chat",
            (
                "Customer logged into the account and confirmed the order in chat before shipment.",
                "Authenticated support session where the customer discussed this order.",
            ),
            helpful=True, required=True,
        ),
        _EvidenceBlueprint(
            "NO-REFUND", "refunds", "Refund ledger",
            ("No refund or cancellation was issued prior to the dispute.",),
        ),
        _EvidenceBlueprint(
            "PRIOR-ORDERS", "risk", "Prior account activity",
            (
                "Same account, same device fingerprint, and prior fulfilled orders without disputes.",
                "Established account with consistent device and address history.",
            ),
            helpful=True, required=True,
        ),
        _EvidenceBlueprint(
            "VELOCITY", "risk", "Velocity check",
            ("No abnormal velocity or proxy usage detected.",),
            helpful=True, probability=0.5,
        ),
    ),
)


_FRAUD_CNP_WEAK = _CaseTemplate(
    reason_code="fraud_cnp",
    summaries=(
        "CNP fraud dispute on a guest-checkout order with no account history.",
        "Issuer flagged an unauthorized charge from a first-time buyer.",
        "Fraud claim on an order placed without an established customer account.",
    ),
    inspection_notes=(
        "New account with no prior history. No durable device or address linkage. Concede this dispute.",
        "Guest checkout with mismatched AVS and CVV. No evidence supports contesting.",
        "First-time buyer, no account linkage, multiple verification mismatches.",
    ),
    policy_guidance=(
        "Do not contest when you lack durable account or device linkage. "
        "Avoid wasting steps on weak fraud disputes."
    ),
    policy_requirements=("cardholder linkage evidence",),
    optimal_strategy="accept_chargeback",
    acceptable_strategies=("issue_refund",),
    resolution_summary="Concede the dispute. Contesting wastes portfolio value.",
    base_weight=0.8,
    evidence_blueprints=(
        _EvidenceBlueprint(
            "ORDER", "orders", "Order receipt",
            (
                "Guest checkout with a new shipping address and no prior order history.",
                "Order placed without account login, new address, no purchase history.",
            ),
        ),
        _EvidenceBlueprint(
            "AVS", "payment", "AVS mismatch detail",
            (
                "Street and postal code mismatches were present.",
                "AVS returned a full mismatch for this transaction.",
            ),
            harmful=True,
        ),
        _EvidenceBlueprint(
            "CVV", "payment", "CVV mismatch detail",
            (
                "CVV did not match.",
                "CVV verification failed at authorization.",
            ),
            harmful=True,
        ),
        _EvidenceBlueprint(
            "DELIVERY", "shipping", "Carrier delivery confirmation",
            ("Delivered to a new address without signature.",),
        ),
        _EvidenceBlueprint(
            "SUPPORT", "support", "Support log",
            ("No authenticated support interactions were recorded.",),
        ),
        _EvidenceBlueprint(
            "NO-REFUND", "refunds", "Refund ledger",
            ("No refund issued before the chargeback.",),
        ),
        _EvidenceBlueprint(
            "RISK", "risk", "Risk summary",
            (
                "Elevated risk score and no positive account history.",
                "High-risk transaction with no account-level trust signals.",
            ),
        ),
    ),
)


_CREDIT_NOT_PROCESSED = _CaseTemplate(
    reason_code="credit_not_processed",
    summaries=(
        "Customer canceled a subscription but says the promised refund was never processed.",
        "Cardholder claims the credit for a returned item was never applied.",
        "Subscriber disputes a renewal charge after requesting cancellation.",
    ),
    inspection_notes=(
        "The merchant missed the promised refund SLA. This should be resolved fast with a refund.",
        "Cancellation was received before the charge, but the refund was never issued.",
        "Return was accepted but the credit has not appeared on the customer's statement.",
    ),
    policy_guidance=(
        "If the merchant failed to process a promised credit, refund immediately or concede. "
        "Contesting is not supportable."
    ),
    policy_requirements=("proof of cancellation request", "refund status check"),
    optimal_strategy="issue_refund",
    acceptable_strategies=("accept_chargeback",),
    resolution_summary="Refund immediately. Delay turns a manageable loss into a deadline miss.",
    base_weight=1.2,
    evidence_blueprints=(
        _EvidenceBlueprint(
            "ORDER", "orders", "Invoice",
            (
                "Subscription renewed automatically for the plan period.",
                "Recurring charge invoice for the billing cycle.",
            ),
        ),
        _EvidenceBlueprint(
            "PAYMENT", "payment", "Captured payment",
            ("Renewal payment settled successfully.",),
        ),
        _EvidenceBlueprint(
            "CANCEL", "support", "Cancellation request",
            (
                "Customer requested cancellation before renewal and support promised a refund.",
                "Support ticket confirms the customer asked to cancel before the charge date.",
            ),
            helpful=True,
        ),
        _EvidenceBlueprint(
            "NO-REFUND", "refunds", "Refund ledger",
            (
                "No refund has been issued as of the dispute open date.",
                "Refund ledger shows no credit processed for this transaction.",
            ),
            helpful=True,
        ),
    ),
)


_DUPLICATE_PROCESSING = _CaseTemplate(
    reason_code="duplicate_processing",
    summaries=(
        "Customer was charged twice for the same order.",
        "Cardholder reports a duplicate charge on their statement.",
        "Two identical charges appeared for a single purchase.",
    ),
    inspection_notes=(
        "Payment system shows two captured authorizations for the same order. One charge is valid.",
        "Duplicate settlement detected in the payment logs. The second charge is erroneous.",
        "Order was charged twice due to a retry in the payment flow. Refund the duplicate.",
    ),
    policy_guidance=(
        "When a duplicate charge is confirmed, refund the extra amount immediately. "
        "Do not contest duplicate processing disputes where the error is on the merchant side."
    ),
    policy_requirements=("payment transaction log", "duplicate confirmation"),
    optimal_strategy="issue_refund",
    acceptable_strategies=("accept_chargeback",),
    resolution_summary="Refund the duplicate charge. The error is on the merchant side.",
    base_weight=1.0,
    evidence_blueprints=(
        _EvidenceBlueprint(
            "ORDER", "orders", "Order record",
            (
                "Single order with one expected charge amount.",
                "Order confirmation showing one purchase at the disputed amount.",
            ),
        ),
        _EvidenceBlueprint(
            "DUP-AUTH", "payment", "Duplicate authorization",
            (
                "Two authorization captures recorded for the same order ID and amount.",
                "Payment log shows duplicate settlement for this transaction.",
            ),
            helpful=True,
        ),
        _EvidenceBlueprint(
            "ORIGINAL-AUTH", "payment", "Original authorization",
            ("First authorization and capture succeeded normally.",),
        ),
        _EvidenceBlueprint(
            "SUPPORT", "support", "Customer complaint",
            (
                "Customer reported the double charge to support before filing the dispute.",
                "Support ticket opened about duplicate billing.",
            ),
            helpful=True, probability=0.6,
        ),
        _EvidenceBlueprint(
            "NO-REFUND", "refunds", "Refund ledger",
            ("No refund for the duplicate charge has been issued yet.",),
            helpful=True,
        ),
    ),
)


_PRODUCT_NOT_AS_DESCRIBED = _CaseTemplate(
    reason_code="product_not_as_described",
    summaries=(
        "Customer says the item received was materially different from the listing.",
        "Buyer disputes the charge because the product did not match the description.",
        "Cardholder claims the delivered item was not as advertised.",
    ),
    inspection_notes=(
        "Product listing matches manufacturer specs. Customer did not attempt a return within the return window.",
        "Item shipped matches the SKU ordered. No return request was filed before the dispute.",
        "Listing description is accurate. Customer bypassed the return process and went straight to a chargeback.",
    ),
    policy_guidance=(
        "Contest product-not-as-described disputes when the listing accurately represents "
        "the product and the customer did not follow the return process. "
        "Attach product listing proof and return policy evidence."
    ),
    policy_requirements=("product listing verification", "return policy documentation"),
    optimal_strategy="contest",
    acceptable_strategies=("issue_refund",),
    resolution_summary="Contest with listing accuracy proof and return policy documentation.",
    base_weight=1.0,
    evidence_blueprints=(
        _EvidenceBlueprint(
            "ORDER", "orders", "Order details",
            (
                "Order with correct SKU and product description matching the listing.",
                "Checkout receipt showing the exact product title and specifications ordered.",
            ),
            helpful=True, required=True,
        ),
        _EvidenceBlueprint(
            "LISTING", "orders", "Product listing snapshot",
            (
                "Archived product listing matches manufacturer specifications.",
                "Product page snapshot shows accurate description and images.",
            ),
            helpful=True, required=True,
        ),
        _EvidenceBlueprint(
            "AUTH", "payment", "Payment capture",
            ("Payment authorized and settled for the listed price.",),
        ),
        _EvidenceBlueprint(
            "DELIVERY", "shipping", "Delivery confirmation",
            (
                "Item delivered to the customer address within the estimated window.",
                "Carrier tracking confirms successful delivery.",
            ),
            helpful=True, probability=0.7,
        ),
        _EvidenceBlueprint(
            "RETURN-POLICY", "support", "Return policy record",
            (
                "Return window was still open but no return was initiated before the dispute.",
                "Merchant return policy allows returns within 30 days; no return request was filed.",
            ),
            helpful=True,
        ),
        _EvidenceBlueprint(
            "NO-REFUND", "refunds", "Refund ledger",
            ("No return or refund was processed before the chargeback.",),
        ),
        _EvidenceBlueprint(
            "RISK", "risk", "Risk assessment",
            (
                "Standard-risk transaction with no prior dispute history.",
                "Low-risk order from a repeat customer.",
            ),
            probability=0.5,
        ),
    ),
)


_SERVICE_NOT_PROVIDED = _CaseTemplate(
    reason_code="service_not_provided",
    summaries=(
        "Customer claims the paid service was never rendered.",
        "Cardholder disputes a charge for a service appointment that was missed.",
        "Buyer says the booked service was canceled but the charge remained.",
    ),
    inspection_notes=(
        "Service was completed according to provider records. Customer did not raise a complaint until the dispute.",
        "Provider logs show the service was delivered on the scheduled date.",
        "Appointment records confirm the service was performed as booked.",
    ),
    policy_guidance=(
        "Contest service-not-provided disputes when provider records confirm the "
        "service was delivered. Attach service completion logs and any customer acknowledgment."
    ),
    policy_requirements=("service completion record", "customer acknowledgment or scheduling proof"),
    optimal_strategy="contest",
    acceptable_strategies=("issue_refund",),
    resolution_summary="Contest with service completion proof. The service was delivered as booked.",
    base_weight=1.0,
    evidence_blueprints=(
        _EvidenceBlueprint(
            "BOOKING", "orders", "Service booking",
            (
                "Booking confirmation for the service appointment with date and details.",
                "Service order with scheduled date, customer name, and service description.",
            ),
            helpful=True, required=True,
        ),
        _EvidenceBlueprint(
            "AUTH", "payment", "Payment record",
            ("Payment authorized and captured for the service fee.",),
        ),
        _EvidenceBlueprint(
            "COMPLETION", "support", "Service completion log",
            (
                "Provider marked the service as completed on the scheduled date.",
                "Service delivery confirmation logged by the provider.",
            ),
            helpful=True, required=True,
        ),
        _EvidenceBlueprint(
            "FEEDBACK", "support", "Customer feedback",
            (
                "Customer left a positive review after the service was completed.",
                "Post-service survey response received from the customer.",
            ),
            helpful=True, probability=0.4,
        ),
        _EvidenceBlueprint(
            "NO-REFUND", "refunds", "Refund ledger",
            ("No refund or credit was issued before the dispute.",),
        ),
        _EvidenceBlueprint(
            "RISK", "risk", "Risk summary",
            ("Standard transaction with no fraud indicators.",),
        ),
    ),
)


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------

# Templates are split into "contestable" and "concedable" families so the
# generator can control the mix in a multi-case queue.
_CONTESTABLE_TEMPLATES: tuple[_CaseTemplate, ...] = (
    _GOODS_NOT_RECEIVED,
    _FRAUD_CNP_STRONG,
    _PRODUCT_NOT_AS_DESCRIBED,
    _SERVICE_NOT_PROVIDED,
)

_CONCEDABLE_TEMPLATES: tuple[_CaseTemplate, ...] = (
    _FRAUD_CNP_WEAK,
    _CREDIT_NOT_PROCESSED,
    _DUPLICATE_PROCESSING,
)

_ALL_TEMPLATES: tuple[_CaseTemplate, ...] = _CONTESTABLE_TEMPLATES + _CONCEDABLE_TEMPLATES


# ---------------------------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------------------------


def _pick_summary(rng: random.Random, options: tuple[str, ...]) -> str:
    return rng.choice(options)


def _generate_evidence(
    rng: random.Random,
    case_prefix: str,
    blueprints: tuple[_EvidenceBlueprint, ...],
) -> tuple[
    dict[SystemName, tuple[InternalEvidence, ...]],
    tuple[str, ...],  # required_ids
    tuple[str, ...],  # helpful_ids
    tuple[str, ...],  # harmful_ids
]:
    """Generate evidence items from blueprints, gating by probability."""

    by_system: dict[SystemName, list[InternalEvidence]] = {
        s: [] for s in ("orders", "payment", "shipping", "support", "refunds", "risk")
    }
    required_ids: list[str] = []
    helpful_ids: list[str] = []
    harmful_ids: list[str] = []

    for bp in blueprints:
        if bp.probability < 1.0 and rng.random() > bp.probability:
            continue

        eid = f"{case_prefix}-{bp.id_suffix}"
        summary = _pick_summary(rng, bp.summaries)

        ev = InternalEvidence(
            evidence_id=eid,
            source_system=bp.source_system,
            title=bp.title,
            summary=summary,
            helpful=bp.helpful,
            harmful=bp.harmful,
            required=bp.required,
        )
        by_system[bp.source_system].append(ev)

        if bp.required:
            required_ids.append(eid)
        if bp.helpful:
            helpful_ids.append(eid)
        if bp.harmful:
            harmful_ids.append(eid)

    frozen: dict[SystemName, tuple[InternalEvidence, ...]] = {
        k: tuple(v) for k, v in by_system.items()
    }
    return frozen, tuple(required_ids), tuple(helpful_ids), tuple(harmful_ids)


def generate_case(
    rng: random.Random,
    template: _CaseTemplate,
    case_index: int,
    *,
    deadline_step: int = 8,
) -> InternalCase:
    """Generate a single case from a template."""

    prefix = f"G{case_index}"
    amount = _amount(rng, 50.0, 2000.0)

    evidence_by_system, required_ids, helpful_ids, harmful_ids = _generate_evidence(
        rng, prefix, template.evidence_blueprints,
    )

    # If required evidence was gated out and a weak variant exists, flip strategy
    strategy = template.optimal_strategy
    acceptable = template.acceptable_strategies
    if template.weak_variant_strategy and not required_ids:
        strategy = template.weak_variant_strategy
        # When strategy flips, the original optimal becomes acceptable
        acceptable = (template.optimal_strategy,) + tuple(
            s for s in template.acceptable_strategies if s != template.weak_variant_strategy
        )

    return InternalCase(
        case_id=f"CB-G{case_index}",
        order_id=_order_id(rng),
        customer_id=_customer_id(rng),
        amount=amount,
        currency="USD",
        reason_code=template.reason_code,
        summary=_pick_summary(rng, template.summaries),
        inspection_notes=_pick_summary(rng, template.inspection_notes),
        deadline_step=deadline_step,
        optimal_strategy=strategy,
        acceptable_strategies=acceptable,
        policy_guidance=template.policy_guidance,
        policy_requirements=template.policy_requirements,
        recommended_strategy=strategy,
        resolution_summary=template.resolution_summary,
        weight=round(template.base_weight + rng.uniform(-0.2, 0.2), 2),
        required_evidence_ids=required_ids,
        helpful_evidence_ids=helpful_ids,
        harmful_evidence_ids=harmful_ids,
        evidence_by_system=evidence_by_system,
    )


# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------


def generate_task(
    seed: int,
    *,
    difficulty: Literal["easy", "medium", "hard"] = "medium",
    case_count: int | None = None,
) -> TaskScenario:
    """Generate a full task scenario from a seed.

    Parameters
    ----------
    seed:
        Deterministic seed — same seed always produces the same task.
    difficulty:
        Controls step budget, deadline pressure, and case count defaults.
    case_count:
        Override the number of cases (default: 1 for easy, 1-2 for medium, 2-4 for hard).
    """

    rng = random.Random(seed)

    # Defaults per difficulty
    if case_count is None:
        case_count = {"easy": 1, "medium": rng.choice([1, 2]), "hard": rng.choice([2, 3, 4])}[difficulty]

    max_steps = {"easy": 10, "medium": 12, "hard": max(12, case_count * 5)}[difficulty]

    # Build the case list
    cases: list[InternalCase] = []
    used_templates: list[_CaseTemplate] = []

    for i in range(case_count):
        if difficulty == "easy":
            # Easy: always a clean contestable case
            template = rng.choice(_CONTESTABLE_TEMPLATES)
        elif difficulty == "hard" and case_count > 1:
            # Hard: mix of contestable and concedable — ensure at least one of each
            if i == 0:
                template = rng.choice(_CONTESTABLE_TEMPLATES)
            elif i == 1:
                template = rng.choice(_CONCEDABLE_TEMPLATES)
            else:
                template = rng.choice(_ALL_TEMPLATES)
        else:
            # Medium: any template
            template = rng.choice(_ALL_TEMPLATES)

        used_templates.append(template)

        # Deadline tightens with difficulty
        base_deadline = {"easy": 8, "medium": 7, "hard": max(4, 8 - i)}[difficulty]
        deadline = base_deadline + rng.randint(-1, 1)
        deadline = max(3, min(deadline, max_steps - 1))

        case = generate_case(rng, template, i + 1, deadline_step=deadline)
        cases.append(case)

    # Build task metadata
    task_id = f"generated_{difficulty}_s{seed}"
    reason_codes = list({t.reason_code for t in used_templates})
    rng.shuffle(reason_codes)
    code_list = ", ".join(reason_codes[:3])

    title_pool = {
        "easy": [
            "Single Dispute Resolution",
            "Straightforward Chargeback",
            "Quick Recovery Case",
        ],
        "medium": [
            "Mixed Signal Dispute",
            "Evidence Curation Challenge",
            "Strategy Selection Test",
        ],
        "hard": [
            "Multi-Case Queue Triage",
            "Portfolio Optimization",
            "Deadline Pressure Queue",
            "Complex Dispute Portfolio",
        ],
    }
    title = rng.choice(title_pool[difficulty])

    objective_pool = {
        "easy": f"Resolve a {code_list} dispute correctly with the right evidence before the deadline.",
        "medium": f"Handle {case_count} dispute(s) involving {code_list}, choosing the right strategy and evidence.",
        "hard": (
            f"Optimize outcomes across {case_count} disputes ({code_list}) under tight deadlines. "
            "Prioritize high-value recoverable cases and concede weak ones efficiently."
        ),
    }

    description = (
        f"A {'seeded ' if seed else ''}scenario with {case_count} case(s) spanning "
        f"{code_list} reason codes at {difficulty} difficulty."
    )

    return TaskScenario(
        task_id=task_id,
        title=title,
        difficulty=difficulty,
        objective=objective_pool[difficulty],
        description=description,
        max_steps=max_steps,
        cases=tuple(cases),
    )


# ---------------------------------------------------------------------------
# Convenience: batch generation
# ---------------------------------------------------------------------------


def generate_task_suite(
    base_seed: int = 42,
    *,
    easy_count: int = 2,
    medium_count: int = 2,
    hard_count: int = 2,
) -> list[TaskScenario]:
    """Generate a balanced suite of tasks across difficulties."""

    tasks: list[TaskScenario] = []
    seed = base_seed
    for difficulty, count in [("easy", easy_count), ("medium", medium_count), ("hard", hard_count)]:
        for _ in range(count):
            tasks.append(generate_task(seed, difficulty=difficulty))
            seed += 1
    return tasks
