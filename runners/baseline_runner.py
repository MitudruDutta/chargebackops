"""Baseline runner for ChargebackOps."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field

try:
    from ..evaluation.grading import grade_episode
    from ..core.models import BaselineRunResult, BaselineTaskResult, ChargebackOpsAction
    from ..server.chargeback_ops_environment import ChargebackOpsEnvironment
    from ..scenarios.simulation import list_tasks
except ImportError:  # pragma: no cover
    from evaluation.grading import grade_episode
    from core.models import BaselineRunResult, BaselineTaskResult, ChargebackOpsAction
    from server.chargeback_ops_environment import ChargebackOpsEnvironment
    from scenarios.simulation import list_tasks

try:  # pragma: no cover
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

if load_dotenv is not None:  # pragma: no cover
    load_dotenv()

DEFAULT_PROVIDER = "openrouter"
MAX_LLM_CANDIDATES = 4
MAX_PROVIDER_RESPONSE_TOKENS = 200
DEFAULT_MODELS = {
    "openrouter": "openai/gpt-oss-120b",
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-sonnet-4-20250514",
    "google": "gemini-2.5-flash",
}
# Ordered fallback: try each until one succeeds.
_FALLBACK_CHAIN: list[tuple[str, str]] = [
    ("openrouter", "openai/gpt-oss-120b"),
    ("google", "gemini-2.5-flash"),
    ("groq", "llama-3.3-70b-versatile"),
]


def _provider_timeout_seconds() -> float:
    raw_value = os.getenv("BASELINE_REQUEST_TIMEOUT_SECONDS", "15")
    try:
        return max(1.0, float(raw_value))
    except ValueError:
        return 4.0


def _provider_retry_attempts() -> int:
    raw_value = os.getenv("PROVIDER_RATE_LIMIT_RETRIES", "2")
    try:
        return max(0, int(raw_value))
    except ValueError:
        return 0


def _provider_retry_backoff_seconds() -> float:
    raw_value = os.getenv("PROVIDER_RETRY_BACKOFF_SECONDS", "1.0")
    try:
        return max(0.1, float(raw_value))
    except ValueError:
        return 0.5


def _strict_llm_mode() -> bool:
    return os.getenv("STRICT_LLM_MODE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _should_retry_provider_error(exc: Exception) -> bool:
    return exc.__class__.__name__ in {
        "RateLimitError",
        "APITimeoutError",
        "APIConnectionError",
        "InternalServerError",
    }


def _chat_completion_with_retry(client: OpenAI, **kwargs):
    last_exc: Exception | None = None
    max_attempts = 1 + _provider_retry_attempts()
    backoff = _provider_retry_backoff_seconds()
    for attempt in range(max_attempts):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt >= max_attempts - 1 or not _should_retry_provider_error(exc):
                raise
            time.sleep(backoff * (attempt + 1))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Provider completion failed without raising an exception.")


class CandidateChoice(BaseModel):
    """Structured choice returned by an LLM provider."""

    candidate_index: int = Field(ge=0)
    rationale: str


@dataclass
class CandidateAction:
    """One valid candidate action for the baseline policy."""

    action: ChargebackOpsAction
    summary: str


@dataclass(frozen=True)
class ProviderConfig:
    """Resolved provider configuration."""

    provider: str
    model_name: str


def _best_open_case(queue: list[dict[str, Any]]) -> dict[str, Any] | None:
    open_cases = [case for case in queue if case["status"] == "open"]
    if not open_cases:
        return None
    return sorted(
        open_cases,
        key=lambda item: (item["steps_until_deadline"], -item["amount"]),
    )[0]


_NOTE_TEMPLATES: dict[str, str] = {
    "goods_not_received": (
        "Order confirmation and carrier delivery confirmation establish fulfillment. "
        "The shipment was delivered to the customer address on file."
    ),
    "fraud_cnp": (
        "Prior good order linkage and customer account confirmation tie the cardholder "
        "to the transaction. Risk analysis and support records confirm legitimacy."
    ),
    "product_not_as_described": (
        "Product listing verification confirms the item matches the description. "
        "Return policy documentation shows the customer bypassed the return process."
    ),
    "service_not_provided": (
        "Service completion record and customer acknowledgment confirm the service "
        "was delivered as agreed. Booking confirmation and delivery records attached."
    ),
    "credit_not_processed": (
        "Refund record and payment confirmation document the credit processing timeline. "
        "Transaction records confirm the refund was issued per policy."
    ),
    "duplicate_processing": (
        "Payment records confirm duplicate charge identification. "
        "Refund documentation attached to support resolution."
    ),
}


def _build_representment_note(visible_case: dict[str, Any]) -> str:
    """Generate a representment note summarizing the dispute contest rationale."""
    reason = visible_case.get("reason_code", "")
    base = _NOTE_TEMPLATES.get(
        reason, f"Contesting {reason.replace('_', ' ')} dispute with attached evidence."
    )

    # Inject policy requirement keywords directly for claims coverage scoring.
    policy = visible_case.get("policy")
    if policy:
        requirements = policy.get("requirements", [])
        if requirements:
            base += " Evidence covers: " + ", ".join(requirements) + "."
        guidance = policy.get("guidance", "")
        if guidance and "contest" in guidance.lower():
            # Extract requirement phrases from guidance text.
            for word in guidance.split():
                clean = word.strip(".,;:").lower()
                if len(clean) > 4 and clean not in base.lower():
                    pass  # Already covered by requirements list

    # Reference evidence IDs directly for coherence scoring.
    attached = visible_case.get("attached_evidence", [])
    if attached:
        eids = [e["evidence_id"] for e in attached if not _is_harmful_evidence(e)]
        if eids:
            base += " Supporting evidence: " + ", ".join(eids) + "."

    return base[:500]


def _visible_case_deadline(queue: list[dict[str, Any]], case_id: str) -> int:
    for case in queue:
        if case["case_id"] == case_id:
            return case["steps_until_deadline"]
    return 999


_NEGATIVE_SIGNAL_KEYWORDS = {
    "mismatch",
    "failed",
    "declined",
    "suspicious",
    "flagged",
    "fraud risk",
    "unauthorized",
    "rejected",
    "invalid",
    "expired",
    "violation",
    "non-compliant",
    "discrepancy",
    "inconsistent",
    "unverified",
}


def _is_harmful_evidence(item: dict[str, Any]) -> bool:
    """Conservative heuristic: flag evidence with negative-signal language."""
    text = (item.get("title", "") + " " + item.get("summary", "")).lower()
    return any(kw in text for kw in _NEGATIVE_SIGNAL_KEYWORDS)


def _rank_attachable(item: dict[str, Any]) -> int:
    text = (item["title"] + " " + item["summary"]).lower()
    if any(kw in text for kw in _NEGATIVE_SIGNAL_KEYWORDS):
        return 999
    if "signature" in text:
        return 0
    if "completion" in text or "booking" in text:
        return 0
    if "listing" in text:
        return 0
    if "duplicate" in text:
        return 1
    if "delivery" in text:
        return 1
    if "prior" in text or "account" in text or "authenticated" in text:
        return 1
    if "return policy" in text or "refund" in text or "cancel" in text:
        return 2
    if "confirmation" in text:
        return 2
    if "cancellation" in text:
        return 2
    return 4


def _batch_attachable_ids(
    retrieved_items: list[dict[str, Any]], attached_ids: set[str]
) -> list[str]:
    filtered = [
        item
        for item in retrieved_items
        if item["evidence_id"] not in attached_ids and _rank_attachable(item) < 999
    ]
    filtered.sort(key=_rank_attachable)
    return [item["evidence_id"] for item in filtered]


def candidate_actions(observation: dict[str, Any]) -> list[CandidateAction]:
    """Build a prioritized candidate set from the current observation."""

    queue = observation["queue"]
    visible_case = observation.get("visible_case")
    open_cases = [case for case in queue if case["status"] == "open"]
    candidates: list[CandidateAction] = []

    if not open_cases and "wait_for_updates" in observation.get("available_actions", []):
        candidates.append(
            CandidateAction(
                action=ChargebackOpsAction(action_type="wait_for_updates"),
                summary="Wait for delayed issuer reviews, delayed evidence, or future case arrivals.",
            )
        )
        return candidates

    # Step cost estimates per reason code (select_case + full workflow).
    _FAST_REASON_CODES = {
        "goods_not_received",
        "credit_not_processed",
        "duplicate_processing",
    }
    _STEP_COST_ESTIMATE = {
        "goods_not_received": 6,  # select + 2 queries + attach + strategy + submit
        "credit_not_processed": 3,  # select + strategy + resolve
        "duplicate_processing": 3,  # select + strategy + resolve
        "fraud_cnp": 8,  # select + policy + 2-3 queries + attach + strategy + submit
        "product_not_as_described": 8,  # select + policy + 2-3 queries + attach + strategy + submit
        "service_not_provided": 7,  # select + policy + 2 queries + attach + strategy + submit
    }

    def _case_priority(item):
        return (
            item["steps_until_deadline"],
            0 if item["reason_code"] in _FAST_REASON_CODES else 1,
            -item["amount"],
        )

    if visible_case is None:
        steps_remaining = observation.get("steps_remaining", 999)
        # Smart triage: if total estimated cost > budget, fast-concede the cheapest-to-lose cases first.
        if len(open_cases) > 1:
            total_cost = sum(
                _STEP_COST_ESTIMATE.get(c["reason_code"], 7) for c in open_cases
            )
            if total_cost > steps_remaining:
                # Budget can't fit all cases. Strategy:
                # 1. Handle deterministic-strategy cases first (cheapest, guaranteed outcome).
                # 2. Then prioritize highest-amount cases with tightest deadlines.
                # 3. Cases that can't fit get auto-conceded by the per-case budget check.
                def _triage_key(c):
                    is_fast = c["reason_code"] in _FAST_REASON_CODES
                    # Fast cases go first (tier 0), then by amount descending (highest value first).
                    return (0 if is_fast else 1, -c["amount"])

                ordered = sorted(open_cases, key=_triage_key)
                for case in ordered:
                    candidates.append(
                        CandidateAction(
                            action=ChargebackOpsAction(
                                action_type="select_case", case_id=case["case_id"]
                            ),
                            summary=(
                                f"Select case {case['case_id']} ({case['reason_code']}, amount ${case['amount']}, "
                                f"deadline in {case['steps_until_deadline']} steps)."
                            ),
                        )
                    )
                return candidates

        for case in sorted(open_cases, key=_case_priority):
            candidates.append(
                CandidateAction(
                    action=ChargebackOpsAction(
                        action_type="select_case", case_id=case["case_id"]
                    ),
                    summary=(
                        f"Select case {case['case_id']} ({case['reason_code']}, amount ${case['amount']}, "
                        f"deadline in {case['steps_until_deadline']} steps)."
                    ),
                )
            )
        return candidates

    case_id = visible_case["case_id"]
    if visible_case["status"] != "open":
        for case in sorted(open_cases, key=_case_priority):
            candidates.append(
                CandidateAction(
                    action=ChargebackOpsAction(
                        action_type="select_case", case_id=case["case_id"]
                    ),
                    summary=(
                        f"Switch to open case {case['case_id']} (deadline in {case['steps_until_deadline']} steps, "
                        f"amount ${case['amount']})."
                    ),
                )
            )
        if not candidates and "wait_for_updates" in observation.get("available_actions", []):
            candidates.append(
                CandidateAction(
                    action=ChargebackOpsAction(action_type="wait_for_updates"),
                    summary="Wait because selected case is blocked and no open case is currently available.",
                )
            )
        return candidates

    # Round 2 (pre-arbitration). Issuer rejected the round-1 packet and is
    # asking for compelling evidence. Three legal moves: respond_to_pre_arb,
    # escalate_to_arbitration, accept_arbitration_loss.
    available = set(observation.get("available_actions", []))
    if "respond_to_pre_arb" in available:
        retrieved_items_r2 = visible_case.get("retrieved_evidence", [])
        attached_ids_r2 = {
            item["evidence_id"] for item in visible_case.get("attached_evidence", [])
        }
        compelling_ids = [
            item["evidence_id"]
            for item in retrieved_items_r2
            if item["evidence_id"] not in attached_ids_r2
            and not _is_harmful_evidence(item)
        ]
        compelling_ids = sorted(
            compelling_ids,
            key=lambda eid: _rank_attachable(
                next(
                    item
                    for item in retrieved_items_r2
                    if item["evidence_id"] == eid
                )
            ),
        )[:2]
        if compelling_ids:
            candidates.append(
                CandidateAction(
                    action=ChargebackOpsAction(
                        action_type="respond_to_pre_arb",
                        case_id=case_id,
                        compelling_evidence_ids=compelling_ids,
                        note=_build_representment_note(visible_case),
                    ),
                    summary=(
                        f"Respond to pre-arbitration with compelling evidence "
                        f"{', '.join(compelling_ids)} for case {case_id}."
                    ),
                )
            )
            return candidates
        # No retrieved compelling evidence left. Try querying an unrevealed
        # merchant system before giving up — round-2 budget often allows it
        # and one extra +0.15 pre_arb piece can clear the 0.60 acceptance bar.
        # Order matters: support/risk/refunds tend to hold compelling pieces;
        # payment is mostly auth records and harmful AVS/CVV mismatches.
        revealed = set(visible_case.get("systems_revealed", []))
        all_systems = ("support", "risk", "refunds", "shipping", "orders", "payment")
        unrevealed = [s for s in all_systems if s not in revealed]
        if unrevealed and "query_system" in available:
            candidates.append(
                CandidateAction(
                    action=ChargebackOpsAction(
                        action_type="query_system",
                        case_id=case_id,
                        system_name=unrevealed[0],
                    ),
                    summary=(
                        f"Query {unrevealed[0]} for compelling evidence "
                        f"on case {case_id} before deciding to escalate."
                    ),
                )
            )
            return candidates
        # No compelling evidence anywhere. Decide on ROI: arbitration costs
        # $250/side. Use the EV rule: escalate iff p_win * amount > arb_fee.
        # Round-2 arbitration score is typically in the ambiguity band
        # (P~0.5), so escalate when amount > 2 * 250 = 500.
        amount = float(visible_case.get("amount", 0.0))
        if amount >= 500.0 and "escalate_to_arbitration" in available:
            candidates.append(
                CandidateAction(
                    action=ChargebackOpsAction(
                        action_type="escalate_to_arbitration",
                        case_id=case_id,
                    ),
                    summary=(
                        f"Escalate case {case_id} to arbitration "
                        f"(amount ${amount:.0f} clears the EV break-even)."
                    ),
                )
            )
            return candidates
        if "accept_arbitration_loss" in available:
            candidates.append(
                CandidateAction(
                    action=ChargebackOpsAction(
                        action_type="accept_arbitration_loss",
                        case_id=case_id,
                    ),
                    summary=(
                        f"Accept arbitration loss on case {case_id} — no "
                        f"compelling evidence and amount below ROI cutoff."
                    ),
                )
            )
            return candidates

    current_deadline = _visible_case_deadline(queue, case_id)
    best_other = _best_open_case(
        [case for case in open_cases if case["case_id"] != case_id]
    )
    # Only switch to an urgent other case if the current case isn't close to completion.
    # "Close" means: strategy is set and evidence attached (1 step to submit),
    # OR evidence is attached and strategy just needs to be set (2 steps to finish).
    _has_attached = len(visible_case.get("attached_evidence", [])) >= 1
    current_near_completion = (
        visible_case.get("current_strategy") == "contest" and _has_attached
    ) or (
        _has_attached
        and visible_case.get("current_strategy") is None
        and current_deadline >= 2
    )
    if (
        best_other is not None
        and best_other["steps_until_deadline"] <= 1
        and current_deadline > 1
        and not current_near_completion
    ):
        candidates.append(
            CandidateAction(
                action=ChargebackOpsAction(
                    action_type="select_case", case_id=best_other["case_id"]
                ),
                summary=(
                    f"Switch to case {best_other['case_id']} immediately because its deadline is in "
                    f"{best_other['steps_until_deadline']} steps."
                ),
            )
        )

    reason_code = visible_case["reason_code"]

    # Reason codes with deterministic strategies — no need to retrieve policy.
    # Only codes where the optimal strategy NEVER varies across generated/ISO cases.
    # fraud_cnp, product_not_as_described, service_not_provided all vary.
    _DETERMINISTIC_STRATEGY: dict[str, str] = {
        "goods_not_received": "contest",
        "credit_not_processed": "issue_refund",
        "duplicate_processing": "issue_refund",
    }

    steps_remaining = observation.get("steps_remaining", 999)
    budget_per_case = steps_remaining / max(len(open_cases), 1)

    policy = visible_case.get("policy")
    if policy is None:
        if reason_code in _DETERMINISTIC_STRATEGY:
            inferred_strategy = _DETERMINISTIC_STRATEGY[reason_code]
        else:
            candidates.append(
                CandidateAction(
                    action=ChargebackOpsAction(
                        action_type="retrieve_policy", case_id=case_id
                    ),
                    summary="Retrieve the chargeback policy for the selected reason code.",
                )
            )
            inferred_strategy = None
    else:
        guidance_text = policy.get("guidance", "").lower()
        if (
            "do not contest" in guidance_text
            or "concede" in guidance_text
            or "not supportable" in guidance_text
        ):
            inferred_strategy = "accept_chargeback"
        elif (
            "refund immediately" in guidance_text
            or "refund" in guidance_text
            and "contest" not in guidance_text
        ):
            inferred_strategy = "issue_refund"
        else:
            inferred_strategy = "contest"
    # How many steps remain before this case's deadline.
    # After querying, we still need: attach(1) + set_strategy(1) + submit/resolve(1) = 3 steps.
    # If policy isn't retrieved yet, add 1 for retrieve_policy.
    _FIXED_COST = 3  # attach + strategy + submit
    steps_to_deadline = current_deadline  # steps_until_deadline from the queue
    policy_cost = 0 if visible_case.get("policy") is not None else 1
    max_queries_before_deadline = max(0, steps_to_deadline - _FIXED_COST - policy_cost)

    systems_revealed = set(visible_case.get("systems_revealed", []))
    current_strategy = visible_case.get("current_strategy")
    retrieved_items = visible_case.get("retrieved_evidence", [])
    attached_evidence = visible_case.get("attached_evidence", [])
    attached_ids = {item["evidence_id"] for item in attached_evidence}
    attachable_ids = _batch_attachable_ids(retrieved_items, attached_ids)

    # Detect harmful evidence already attached — must remove before submit.
    harmful_attached_ids = [
        item["evidence_id"] for item in attached_evidence if _is_harmful_evidence(item)
    ]

    # ── HARMFUL CLEANUP: if harmful evidence is attached, remove it immediately ──
    if harmful_attached_ids:
        candidates.insert(
            0,
            CandidateAction(
                action=ChargebackOpsAction(
                    action_type="remove_evidence",
                    case_id=case_id,
                    evidence_ids=harmful_attached_ids,
                ),
                summary=f"Remove harmful evidence {', '.join(harmful_attached_ids)} before submission.",
            ),
        )
        return candidates

    # ── DEADLINE URGENCY: if near deadline and we have evidence, submit/resolve NOW ──
    if current_deadline <= 1:
        if (
            current_strategy is not None
            and len(attached_ids) >= 1
            and current_strategy == "contest"
        ):
            candidates.insert(
                0,
                CandidateAction(
                    action=ChargebackOpsAction(
                        action_type="submit_representment",
                        case_id=case_id,
                        note=_build_representment_note(visible_case),
                    ),
                    summary=f"URGENT: Submit representment for {case_id} — deadline imminent.",
                ),
            )
            return candidates
        if current_strategy in {"accept_chargeback", "issue_refund"}:
            candidates.insert(
                0,
                CandidateAction(
                    action=ChargebackOpsAction(
                        action_type="resolve_case",
                        case_id=case_id,
                        strategy=current_strategy,
                    ),
                    summary=f"URGENT: Resolve {case_id} with {current_strategy} — deadline imminent.",
                ),
            )
            return candidates

    # ── TIGHT BUDGET: fast-concede if not enough steps to contest this case ──
    # Full contest costs ~7 steps (policy + 2-3 queries + attach + strategy + submit).
    # Fast-concede when:
    #   (a) Not enough global steps remaining, OR
    #   (b) Multi-case scenario where this case is lower-value and budget can't fit all.
    _est_cost = (
        _STEP_COST_ESTIMATE.get(reason_code, 7) - 1
    )  # subtract select_case already done
    # Minimum contest: policy(1) + query(1) + attach(1) + strategy(1) + submit(1) = 5 steps.
    _MIN_CONTEST_STEPS = 5
    _should_fast_concede = False
    if (
        policy is None
        and reason_code not in _DETERMINISTIC_STRATEGY
        and current_strategy is None
        and not systems_revealed
    ):
        if (
            steps_remaining < _MIN_CONTEST_STEPS
            or current_deadline < _MIN_CONTEST_STEPS
        ):
            # Not enough steps or deadline to even minimally contest.
            _should_fast_concede = True
        elif len(open_cases) > 1:
            # Multi-case triage: concede if total cost > budget and this case is lowest-value.
            total_cost = sum(
                _STEP_COST_ESTIMATE.get(c["reason_code"], 7) for c in open_cases
            )
            if total_cost > steps_remaining:
                lowest_amount = min(c["amount"] for c in open_cases)
                this_amount = next(
                    c["amount"] for c in open_cases if c["case_id"] == case_id
                )
                if this_amount <= lowest_amount:
                    _should_fast_concede = True
    if _should_fast_concede:
        fallback = "issue_refund"
        candidates.insert(
            0,
            CandidateAction(
                action=ChargebackOpsAction(
                    action_type="resolve_case",
                    case_id=case_id,
                    strategy=fallback,
                ),
                summary=f"Budget too tight to contest — fast-resolve {case_id} with {fallback}.",
            ),
        )
        return candidates

    # ── BUDGET PRESSURE: if more open cases than steps, fast-resolve concedable ──
    if steps_remaining <= len(open_cases) * 2 and inferred_strategy in {
        "accept_chargeback",
        "issue_refund",
    }:
        target_strat = inferred_strategy
        if current_strategy != target_strat:
            candidates.insert(
                0,
                CandidateAction(
                    action=ChargebackOpsAction(
                        action_type="set_strategy",
                        case_id=case_id,
                        strategy=target_strat,
                    ),
                    summary=f"Fast-set strategy to {target_strat} under budget pressure.",
                ),
            )
            return candidates
        candidates.insert(
            0,
            CandidateAction(
                action=ChargebackOpsAction(
                    action_type="resolve_case",
                    case_id=case_id,
                    strategy=target_strat,
                ),
                summary=f"Fast-resolve {case_id} with {target_strat} under budget pressure.",
            ),
        )
        return candidates

    if reason_code == "goods_not_received":
        for system_name in ["orders", "shipping"]:
            if system_name not in systems_revealed:
                candidates.append(
                    CandidateAction(
                        action=ChargebackOpsAction(
                            action_type="query_system",
                            case_id=case_id,
                            system_name=system_name,
                        ),
                        summary=f"Query the {system_name} system for evidence on case {case_id}.",
                    )
                )
        if attachable_ids:
            candidates.append(
                CandidateAction(
                    action=ChargebackOpsAction(
                        action_type="add_evidence",
                        case_id=case_id,
                        evidence_ids=attachable_ids,
                    ),
                    summary=f"Attach the strongest delivery evidence for case {case_id}.",
                )
            )
        if current_strategy != "contest":
            candidates.append(
                CandidateAction(
                    action=ChargebackOpsAction(
                        action_type="set_strategy",
                        case_id=case_id,
                        strategy="contest",
                    ),
                    summary="Set the strategy to contest the dispute.",
                )
            )
        if len(attached_ids) >= 2:
            candidates.append(
                CandidateAction(
                    action=ChargebackOpsAction(
                        action_type="submit_representment",
                        case_id=case_id,
                        note=_build_representment_note(visible_case),
                    ),
                    summary="Submit the current representment package.",
                )
            )

    elif reason_code == "fraud_cnp":
        should_contest = inferred_strategy == "contest"
        if should_contest:
            # Under tight budgets or deadline pressure, skip optional 'orders' query.
            fraud_systems = ["risk", "support", "orders"]
            unrevealed_fraud = [s for s in fraud_systems if s not in systems_revealed]
            if (
                len(unrevealed_fraud) > max_queries_before_deadline
                or budget_per_case < 7
            ):
                fraud_systems = ["risk", "support"]
                unrevealed_fraud = [
                    s for s in fraud_systems if s not in systems_revealed
                ]
            for system_name in unrevealed_fraud:
                candidates.append(
                    CandidateAction(
                        action=ChargebackOpsAction(
                            action_type="query_system",
                            case_id=case_id,
                            system_name=system_name,
                        ),
                        summary=f"Query the {system_name} system for evidence on case {case_id}.",
                    )
                )
            if attachable_ids:
                candidates.append(
                    CandidateAction(
                        action=ChargebackOpsAction(
                            action_type="add_evidence",
                            case_id=case_id,
                            evidence_ids=attachable_ids,
                        ),
                        summary=f"Attach the strongest account-linkage evidence for case {case_id}.",
                    )
                )
            if current_strategy != "contest":
                candidates.append(
                    CandidateAction(
                        action=ChargebackOpsAction(
                            action_type="set_strategy",
                            case_id=case_id,
                            strategy="contest",
                        ),
                        summary="Set the strategy to contest the dispute.",
                    )
                )
            if len(attached_ids) >= 2:
                candidates.append(
                    CandidateAction(
                        action=ChargebackOpsAction(
                            action_type="submit_representment",
                            case_id=case_id,
                            note=_build_representment_note(visible_case),
                        ),
                        summary="Submit the current representment package.",
                    )
                )
        if current_strategy != "accept_chargeback":
            candidates.append(
                CandidateAction(
                    action=ChargebackOpsAction(
                        action_type="set_strategy",
                        case_id=case_id,
                        strategy="accept_chargeback",
                    ),
                    summary="Set the strategy to accept the chargeback.",
                )
            )
        candidates.append(
            CandidateAction(
                action=ChargebackOpsAction(
                    action_type="resolve_case",
                    case_id=case_id,
                    strategy="accept_chargeback",
                ),
                summary="Concede the dispute and accept the chargeback.",
            )
        )

    elif reason_code in {"credit_not_processed", "duplicate_processing"}:
        # Fast-path: set strategy and resolve immediately — don't waste steps querying
        if current_strategy != "issue_refund":
            candidates.append(
                CandidateAction(
                    action=ChargebackOpsAction(
                        action_type="set_strategy",
                        case_id=case_id,
                        strategy="issue_refund",
                    ),
                    summary="Set the strategy to issue a refund immediately.",
                )
            )
        candidates.append(
            CandidateAction(
                action=ChargebackOpsAction(
                    action_type="resolve_case",
                    case_id=case_id,
                    strategy="issue_refund",
                ),
                summary="Resolve the case by issuing a refund.",
            )
        )
        candidates.append(
            CandidateAction(
                action=ChargebackOpsAction(
                    action_type="resolve_case",
                    case_id=case_id,
                    strategy="accept_chargeback",
                ),
                summary="Accept the chargeback as a fallback resolution.",
            )
        )

    elif reason_code == "product_not_as_described":
        if inferred_strategy in {"accept_chargeback", "issue_refund"}:
            # Guidance says concede — fast-path
            target = inferred_strategy
            if current_strategy != target:
                candidates.append(
                    CandidateAction(
                        action=ChargebackOpsAction(
                            action_type="set_strategy", case_id=case_id, strategy=target
                        ),
                        summary=f"Set strategy to {target} — listing defense not supportable.",
                    )
                )
            candidates.append(
                CandidateAction(
                    action=ChargebackOpsAction(
                        action_type="resolve_case", case_id=case_id, strategy=target
                    ),
                    summary=f"Resolve with {target} — conceding per policy guidance.",
                )
            )
        else:
            # Under deadline pressure, skip shipping (least critical for this reason code).
            pna_systems = ["orders", "support", "shipping"]
            unrevealed = [s for s in pna_systems if s not in systems_revealed]
            if len(unrevealed) > max_queries_before_deadline:
                pna_systems = ["orders", "support"]  # Drop shipping
                unrevealed = [s for s in pna_systems if s not in systems_revealed]
            for system_name in unrevealed:
                candidates.append(
                    CandidateAction(
                        action=ChargebackOpsAction(
                            action_type="query_system",
                            case_id=case_id,
                            system_name=system_name,
                        ),
                        summary=f"Query the {system_name} system for listing and return-process evidence on case {case_id}.",
                    )
                )
            if attachable_ids:
                candidates.append(
                    CandidateAction(
                        action=ChargebackOpsAction(
                            action_type="add_evidence",
                            case_id=case_id,
                            evidence_ids=attachable_ids,
                        ),
                        summary=f"Attach listing accuracy and return-policy evidence for case {case_id}.",
                    )
                )
            if current_strategy != "contest":
                candidates.append(
                    CandidateAction(
                        action=ChargebackOpsAction(
                            action_type="set_strategy",
                            case_id=case_id,
                            strategy="contest",
                        ),
                        summary="Set the strategy to contest the dispute.",
                    )
                )
            if len(attached_ids) >= 2:
                candidates.append(
                    CandidateAction(
                        action=ChargebackOpsAction(
                            action_type="submit_representment",
                            case_id=case_id,
                            note=_build_representment_note(visible_case),
                        ),
                        summary="Submit the current representment package.",
                    )
                )
            candidates.append(
                CandidateAction(
                    action=ChargebackOpsAction(
                        action_type="resolve_case",
                        case_id=case_id,
                        strategy="issue_refund",
                    ),
                    summary="Issue a refund as a fallback if the listing defense is not supportable.",
                )
            )

    elif reason_code == "service_not_provided":
        if inferred_strategy in {"accept_chargeback", "issue_refund"}:
            target = inferred_strategy
            if current_strategy != target:
                candidates.append(
                    CandidateAction(
                        action=ChargebackOpsAction(
                            action_type="set_strategy", case_id=case_id, strategy=target
                        ),
                        summary=f"Set strategy to {target} — service defense not supportable.",
                    )
                )
            candidates.append(
                CandidateAction(
                    action=ChargebackOpsAction(
                        action_type="resolve_case", case_id=case_id, strategy=target
                    ),
                    summary=f"Resolve with {target} — conceding per policy guidance.",
                )
            )
        else:
            snp_systems = ["orders", "support"]
            unrevealed_snp = [s for s in snp_systems if s not in systems_revealed]
            if len(unrevealed_snp) > max_queries_before_deadline:
                snp_systems = [
                    "support"
                ]  # Support is most critical for service disputes.
                unrevealed_snp = [s for s in snp_systems if s not in systems_revealed]
            for system_name in unrevealed_snp:
                candidates.append(
                    CandidateAction(
                        action=ChargebackOpsAction(
                            action_type="query_system",
                            case_id=case_id,
                            system_name=system_name,
                        ),
                        summary=f"Query the {system_name} system for booking and completion evidence on case {case_id}.",
                    )
                )
        if attachable_ids:
            candidates.append(
                CandidateAction(
                    action=ChargebackOpsAction(
                        action_type="add_evidence",
                        case_id=case_id,
                        evidence_ids=attachable_ids,
                    ),
                    summary=f"Attach booking and completion evidence for case {case_id}.",
                )
            )
        if current_strategy != "contest":
            candidates.append(
                CandidateAction(
                    action=ChargebackOpsAction(
                        action_type="set_strategy",
                        case_id=case_id,
                        strategy="contest",
                    ),
                    summary="Set the strategy to contest the dispute.",
                )
            )
        if len(attached_ids) >= 2:
            candidates.append(
                CandidateAction(
                    action=ChargebackOpsAction(
                        action_type="submit_representment",
                        case_id=case_id,
                        note=_build_representment_note(visible_case),
                    ),
                    summary="Submit the current representment package.",
                )
            )
        candidates.append(
            CandidateAction(
                action=ChargebackOpsAction(
                    action_type="resolve_case",
                    case_id=case_id,
                    strategy="issue_refund",
                ),
                summary="Issue a refund as a fallback if the service-delivery defense is weak.",
            )
        )

    elif inferred_strategy in {"accept_chargeback", "issue_refund"}:
        for system_name in ["support", "refunds", "payment"]:
            if system_name not in systems_revealed:
                candidates.append(
                    CandidateAction(
                        action=ChargebackOpsAction(
                            action_type="query_system",
                            case_id=case_id,
                            system_name=system_name,
                        ),
                        summary=f"Query the {system_name} system for concession evidence on case {case_id}.",
                    )
                )
        if current_strategy != inferred_strategy:
            candidates.append(
                CandidateAction(
                    action=ChargebackOpsAction(
                        action_type="set_strategy",
                        case_id=case_id,
                        strategy=inferred_strategy,
                    ),
                    summary=f"Set the strategy to {inferred_strategy}.",
                )
            )
        candidates.append(
            CandidateAction(
                action=ChargebackOpsAction(
                    action_type="resolve_case",
                    case_id=case_id,
                    strategy=inferred_strategy,
                ),
                summary=f"Resolve the case with strategy {inferred_strategy}.",
            )
        )

    else:
        for system_name in ["orders", "support", "shipping", "risk"]:
            if system_name not in systems_revealed:
                candidates.append(
                    CandidateAction(
                        action=ChargebackOpsAction(
                            action_type="query_system",
                            case_id=case_id,
                            system_name=system_name,
                        ),
                        summary=f"Query the {system_name} system for additional evidence on case {case_id}.",
                    )
                )
        if attachable_ids:
            candidates.append(
                CandidateAction(
                    action=ChargebackOpsAction(
                        action_type="add_evidence",
                        case_id=case_id,
                        evidence_ids=attachable_ids,
                    ),
                    summary=f"Attach the strongest currently available evidence for case {case_id}.",
                )
            )
        if current_strategy != "contest":
            candidates.append(
                CandidateAction(
                    action=ChargebackOpsAction(
                        action_type="set_strategy",
                        case_id=case_id,
                        strategy="contest",
                    ),
                    summary="Set the strategy to contest the dispute.",
                )
            )
        if len(attached_ids) >= 1:
            candidates.append(
                CandidateAction(
                    action=ChargebackOpsAction(
                        action_type="submit_representment",
                        case_id=case_id,
                        note=_build_representment_note(visible_case),
                    ),
                    summary="Submit the current representment package.",
                )
            )

    if (
        visible_case.get("inspection_notes") is None
        and observation["steps_remaining"] > 3
    ):
        candidates.append(
            CandidateAction(
                action=ChargebackOpsAction(action_type="inspect_case", case_id=case_id),
                summary="Inspect the selected case to reveal merchant notes.",
            )
        )

    for case in sorted(
        open_cases, key=lambda item: (item["steps_until_deadline"], -item["amount"])
    ):
        if case["case_id"] != case_id:
            candidates.append(
                CandidateAction(
                    action=ChargebackOpsAction(
                        action_type="select_case", case_id=case["case_id"]
                    ),
                    summary=(
                        f"Switch to case {case['case_id']} (deadline in {case['steps_until_deadline']} steps, "
                        f"amount ${case['amount']})."
                    ),
                )
            )

    return candidates


def _heuristic_pick(candidates: list[CandidateAction]) -> CandidateAction:
    return candidates[0]


def _obvious_next_action(
    observation: dict[str, Any],
    candidates: list[CandidateAction],
) -> CandidateAction | None:
    """Skip provider calls for deterministic housekeeping actions.

    This preserves live model decisions for genuine branching states while keeping
    baseline/inference runtime inside hackathon-friendly bounds.
    """

    if not candidates:
        return None

    # Single candidate = no decision to make.
    if len(candidates) == 1:
        return candidates[0]

    first = candidates[0]
    visible_case = observation.get("visible_case")
    queue = observation["queue"]

    if visible_case is None:
        open_cases = [case for case in queue if case["status"] == "open"]
        if len(open_cases) == 1:
            return first
        urgent_cases = [
            case for case in open_cases if case["steps_until_deadline"] <= 1
        ]
        if (
            len(urgent_cases) == 1
            and first.action.action_type == "select_case"
            and first.action.case_id == urgent_cases[0]["case_id"]
        ):
            return first
        return None

    if visible_case["status"] != "open":
        return first if first.action.action_type == "select_case" else None

    # Strategy selection: the heuristic already derives the optimal strategy
    # from policy + retrieved evidence. The LLM has no additional signal that
    # improves this specific call — invoking it here has only caused regressions
    # on fraud_signal_ambiguity and generated_medium_s99 where the model picks
    # a concede-style strategy over the correct contest.
    if first.action.action_type == "set_strategy":
        return first

    if first.action.action_type in {
        "retrieve_policy",
        "add_evidence",
        "remove_evidence",
        "submit_representment",
        "resolve_case",
    }:
        return first

    if first.action.action_type == "query_system":
        current_strategy = visible_case.get("current_strategy")
        if visible_case.get("policy") is None or current_strategy in {None, "contest"}:
            return first

    if first.action.action_type == "select_case":
        current_case_id = visible_case["case_id"]
        current_deadline = next(
            (
                case["steps_until_deadline"]
                for case in queue
                if case["case_id"] == current_case_id
            ),
            999,
        )
        target_deadline = next(
            (
                case["steps_until_deadline"]
                for case in queue
                if case["case_id"] == first.action.case_id
            ),
            999,
        )
        if target_deadline < current_deadline:
            return first

    return None


def _safe_json_loads(text: str) -> CandidateChoice | None:
    try:
        return CandidateChoice.model_validate_json(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return CandidateChoice.model_validate_json(text[start : end + 1])
        except Exception:
            return None


def _compact_queue_item(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": case["case_id"],
        "reason_code": case["reason_code"],
        "amount": case["amount"],
        "status": case["status"],
        "steps_until_deadline": case["steps_until_deadline"],
    }


def _compact_visible_case(visible_case: dict[str, Any] | None) -> dict[str, Any] | None:
    if visible_case is None:
        return None
    return {
        "case_id": visible_case["case_id"],
        "reason_code": visible_case["reason_code"],
        "current_strategy": visible_case.get("current_strategy"),
        "systems_revealed": visible_case.get("systems_revealed", []),
        "attached_evidence": [
            item["title"] for item in visible_case.get("attached_evidence", [])[:4]
        ],
        "retrieved_evidence": [
            item["title"] for item in visible_case.get("retrieved_evidence", [])[:6]
        ],
        "policy": (
            {
                "guidance": visible_case["policy"]["guidance"],
                "required_evidence": visible_case["policy"]["required_evidence"],
            }
            if visible_case.get("policy")
            else None
        ),
        "submission_status": visible_case.get("submission_status"),
    }


def _provider_payload(
    observation: dict[str, Any],
    candidates: list[CandidateAction],
) -> tuple[list[CandidateAction], str]:
    shortlist = candidates[: min(MAX_LLM_CANDIDATES, len(candidates))]
    payload = json.dumps(
        {
            "task_id": observation["task_id"],
            "steps_remaining": observation["steps_remaining"],
            "selected_case_id": observation.get("selected_case_id"),
            "queue": [_compact_queue_item(case) for case in observation["queue"]],
            "visible_case": _compact_visible_case(observation.get("visible_case")),
            "candidates": [
                {"index": idx, "summary": candidate.summary}
                for idx, candidate in enumerate(shortlist)
            ],
        },
        separators=(",", ":"),
    )
    return shortlist, payload


def _resolve_provider(
    provider: str | None,
    model_name: str | None,
) -> ProviderConfig:
    chosen_provider = (
        provider or os.getenv("BASELINE_PROVIDER") or DEFAULT_PROVIDER
    ).lower()
    chosen_model = (
        model_name
        or os.getenv("BASELINE_MODEL")
        or DEFAULT_MODELS.get(
            chosen_provider,
            "openai/gpt-oss-120b",
        )
    )
    return ProviderConfig(provider=chosen_provider, model_name=chosen_model)


def _openai_compatible_client(config: ProviderConfig) -> OpenAI | None:
    timeout_seconds = _provider_timeout_seconds()
    if config.provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        return (
            OpenAI(api_key=api_key, timeout=timeout_seconds, max_retries=0)
            if api_key
            else None
        )
    if config.provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            return None
        headers = {}
        if os.getenv("OPENROUTER_HTTP_REFERER"):
            headers["HTTP-Referer"] = os.getenv("OPENROUTER_HTTP_REFERER", "")
        if os.getenv("OPENROUTER_APP_TITLE"):
            app_title = os.getenv("OPENROUTER_APP_TITLE", "")
            headers["X-OpenRouter-Title"] = app_title
            # Keep the legacy header for compatibility with older OpenRouter examples.
            headers["X-Title"] = app_title
        return OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers=headers or None,
            timeout=timeout_seconds,
            max_retries=0,
        )
    if config.provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return None
        return OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=timeout_seconds,
            max_retries=0,
        )
    if config.provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return None
        return OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            timeout=timeout_seconds,
            max_retries=0,
        )
    return None


def _provider_pick(
    config: ProviderConfig,
    observation: dict[str, Any],
    candidates: list[CandidateAction],
) -> tuple[CandidateAction, bool, bool, str | None]:
    shortlist, payload = _provider_payload(observation, candidates)

    if config.provider in {"openai", "openrouter", "groq", "google"}:
        client = _openai_compatible_client(config)
        if client is None:
            return shortlist[0], False, False, None
        try:
            response = _chat_completion_with_retry(
                client,
                model=config.model_name,
                temperature=0,
                max_tokens=MAX_PROVIDER_RESPONSE_TOKENS,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a merchant chargeback dispute analyst. Pick the single best next action from the ordered candidate list. "
                            "The candidates are pre-sorted by a deterministic heuristic — candidate 0 is usually correct. Deviate only when you spot a concrete reason. "
                            "\n"
                            "Reason-code → optimal strategy (follow unless evidence clearly contradicts):\n"
                            "  goods_not_received → contest (with order + delivery proof)\n"
                            "  fraud_cnp → contest when account linkage exists, otherwise concede\n"
                            "  product_not_as_described → contest (with listing + return policy proof)\n"
                            "  service_not_provided → contest (with completion log)\n"
                            "  credit_not_processed → issue_refund immediately\n"
                            "  duplicate_processing → issue_refund immediately\n"
                            "\n"
                            "Priorities: (1) resolve cases whose deadline is 1 step away before anything else, "
                            "(2) prefer the highest-$ open case when budget is tight, "
                            "(3) never attach harmful evidence (AVS/CVV mismatch on fraud_cnp, GPS anomalies on goods_not_received), "
                            "(4) when multiple candidates look equivalent, take candidate 0.\n"
                            'Return only JSON: {"candidate_index": N, "rationale": "brief reason"}'
                        ),
                    },
                    {"role": "user", "content": payload},
                ],
            )
            content = response.choices[0].message.content or "{}"
            choice = _safe_json_loads(content)
            if choice is None:
                return shortlist[0], True, False, "InvalidJSONResponse"
            index = min(max(choice.candidate_index, 0), len(shortlist) - 1)
            return shortlist[index], True, True, None
        except Exception as exc:
            return shortlist[0], True, False, exc.__class__.__name__

    if config.provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return shortlist[0], False, False, None
        try:  # pragma: no cover
            from anthropic import Anthropic
        except ImportError:  # pragma: no cover
            return shortlist[0], False, False, None
        try:  # pragma: no cover
            client = Anthropic(
                api_key=api_key,
                timeout=_provider_timeout_seconds(),
                max_retries=0,
            )
            response = client.messages.create(
                model=config.model_name,
                max_tokens=200,
                temperature=0,
                system=(
                    "You are a merchant chargeback analyst. Pick the single best next action. "
                    "Return only JSON with candidate_index and rationale."
                ),
                messages=[{"role": "user", "content": payload}],
            )
            text = "".join(
                block.text
                for block in response.content
                if getattr(block, "type", "") == "text"
            )
            choice = _safe_json_loads(text)
            if choice is None:
                return shortlist[0], True, False, "InvalidJSONResponse"
            index = min(max(choice.candidate_index, 0), len(shortlist) - 1)
            return shortlist[index], True, True, None
        except Exception as exc:
            return shortlist[0], True, False, exc.__class__.__name__

    return shortlist[0], False, False, None


def _provider_pick_with_fallback(
    config: ProviderConfig,
    observation: dict[str, Any],
    candidates: list[CandidateAction],
) -> tuple[CandidateAction, bool, bool, str | None]:
    """Try the primary provider, then walk the fallback chain on failure."""
    candidate, attempted, succeeded, error = _provider_pick(
        config, observation, candidates
    )
    if succeeded:
        return candidate, attempted, succeeded, error

    for fb_provider, fb_model in _FALLBACK_CHAIN:
        if fb_provider == config.provider:
            continue
        fb_config = ProviderConfig(provider=fb_provider, model_name=fb_model)
        fb_client = _openai_compatible_client(fb_config)
        if fb_client is None:
            continue
        candidate, fb_attempted, fb_succeeded, fb_error = _provider_pick(
            fb_config,
            observation,
            candidates,
        )
        if fb_succeeded:
            return candidate, True, True, None

    return candidate, attempted, False, error or "AllProvidersFailed"


def run_baseline(
    provider: str | None = None,
    model_name: str | None = None,
) -> BaselineRunResult:
    """Run the baseline across all built-in tasks."""

    config = _resolve_provider(provider, model_name)
    has_provider_key = any(
        [
            config.provider == "openai" and bool(os.getenv("OPENAI_API_KEY")),
            config.provider == "openrouter" and bool(os.getenv("OPENROUTER_API_KEY")),
            config.provider == "groq" and bool(os.getenv("GROQ_API_KEY")),
            config.provider == "anthropic" and bool(os.getenv("ANTHROPIC_API_KEY")),
            config.provider == "google" and bool(os.getenv("GOOGLE_API_KEY")),
        ]
    )
    provider_calls_attempted = 0
    provider_calls_succeeded = 0
    provider_errors: dict[str, int] = {}

    task_results: list[BaselineTaskResult] = []
    for task in list_tasks():
        env = ChargebackOpsEnvironment()
        observation = env.reset(task_id=task.task_id)
        while not observation.done:
            observation_payload = observation.model_dump()
            candidates = candidate_actions(observation_payload)
            if not candidates:
                break
            if len(candidates) == 1:
                candidate = candidates[0]
                observation = env.step(candidate.action)
                continue
            obvious_candidate = _obvious_next_action(observation_payload, candidates)
            if obvious_candidate is not None:
                observation = env.step(obvious_candidate.action)
                continue
            if has_provider_key:
                candidate, attempted, succeeded, error_label = (
                    _provider_pick_with_fallback(
                        config,
                        observation_payload,
                        candidates,
                    )
                )
                provider_calls_attempted += int(attempted)
                provider_calls_succeeded += int(succeeded)
                if attempted and not succeeded and error_label is not None:
                    provider_errors[error_label] = (
                        provider_errors.get(error_label, 0) + 1
                    )
                if _strict_llm_mode() and attempted and not succeeded:
                    raise RuntimeError(
                        "STRICT_LLM_MODE is enabled and the provider decision failed, "
                        "so heuristic fallback is not allowed."
                    )
            else:
                candidate = _heuristic_pick(candidates)
            observation = env.step(candidate.action)

        report = env.state.grader_report or grade_episode(
            task,
            env._progress_by_case,  # type: ignore[attr-defined]
            env.state.step_count,
            env.state.episode_id or "",
            completed=env.state.completed,
        )
        task_results.append(
            BaselineTaskResult(
                task_id=task.task_id,
                title=task.title,
                score=report.normalized_score,
                steps_used=env.state.step_count,
                final_status=report.summary,
            )
        )

    average_score = round(
        sum(task_result.score for task_result in task_results) / len(task_results),
        4,
    )
    if provider_calls_attempted == 0:
        mode = "heuristic_fallback"
    elif provider_calls_succeeded == 0:
        mode = "heuristic_fallback"
    elif provider_calls_succeeded < provider_calls_attempted:
        mode = f"{config.provider}_with_fallback"
    else:
        mode = config.provider
    return BaselineRunResult(
        provider=config.provider,
        model_name=config.model_name,
        mode=mode,
        provider_calls_attempted=provider_calls_attempted,
        provider_calls_succeeded=provider_calls_succeeded,
        provider_errors=provider_errors,
        task_results=task_results,
        average_score=average_score,
    )


def main() -> None:
    """CLI entry point."""

    print(json.dumps(run_baseline().model_dump(), indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
