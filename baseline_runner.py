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
    from .grading import grade_episode
    from .models import BaselineRunResult, BaselineTaskResult, ChargebackOpsAction
    from .server.chargeback_ops_environment import ChargebackOpsEnvironment
    from .simulation import list_tasks
except ImportError:  # pragma: no cover
    from grading import grade_episode
    from models import BaselineRunResult, BaselineTaskResult, ChargebackOpsAction
    from server.chargeback_ops_environment import ChargebackOpsEnvironment
    from simulation import list_tasks

try:  # pragma: no cover
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

if load_dotenv is not None:  # pragma: no cover
    load_dotenv()

DEFAULT_PROVIDER = "groq"
MAX_LLM_CANDIDATES = 4
MAX_PROVIDER_RESPONSE_TOKENS = 80
DEFAULT_MODELS = {
    "openrouter": "nvidia/nemotron-3-super-120b-a12b:free",
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-5-mini",
    "anthropic": "claude-3-5-haiku-latest",
}


def _provider_timeout_seconds() -> float:
    raw_value = os.getenv("BASELINE_REQUEST_TIMEOUT_SECONDS", "4")
    try:
        return max(1.0, float(raw_value))
    except ValueError:
        return 4.0


def _provider_retry_attempts() -> int:
    raw_value = os.getenv("PROVIDER_RATE_LIMIT_RETRIES", "0")
    try:
        return max(0, int(raw_value))
    except ValueError:
        return 0


def _provider_retry_backoff_seconds() -> float:
    raw_value = os.getenv("PROVIDER_RETRY_BACKOFF_SECONDS", "0.5")
    try:
        return max(0.1, float(raw_value))
    except ValueError:
        return 0.5


def _strict_llm_mode() -> bool:
    return os.getenv("STRICT_LLM_MODE", "").strip().lower() in {"1", "true", "yes", "on"}


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


def _visible_case_deadline(queue: list[dict[str, Any]], case_id: str) -> int:
    for case in queue:
        if case["case_id"] == case_id:
            return case["steps_until_deadline"]
    return 999


def _rank_attachable(item: dict[str, Any]) -> int:
    text = (item["title"] + " " + item["summary"]).lower()
    if "mismatch" in text:
        return 999
    if "signature" in text:
        return 0
    if "delivery" in text:
        return 1
    if "prior" in text or "account" in text or "authenticated" in text:
        return 1
    if "confirmation" in text:
        return 2
    if "refund" in text or "cancellation" in text:
        return 2
    return 4


def _batch_attachable_ids(retrieved_items: list[dict[str, Any]], attached_ids: set[str]) -> list[str]:
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

    if visible_case is None:
        for case in sorted(open_cases, key=lambda item: (item["steps_until_deadline"], -item["amount"])):
            candidates.append(
                CandidateAction(
                    action=ChargebackOpsAction(action_type="select_case", case_id=case["case_id"]),
                    summary=(
                        f"Select case {case['case_id']} ({case['reason_code']}, amount ${case['amount']}, "
                        f"deadline in {case['steps_until_deadline']} steps)."
                    ),
                )
            )
        return candidates

    case_id = visible_case["case_id"]
    if visible_case["status"] != "open":
        for case in sorted(open_cases, key=lambda item: (item["steps_until_deadline"], -item["amount"])):
            candidates.append(
                CandidateAction(
                    action=ChargebackOpsAction(action_type="select_case", case_id=case["case_id"]),
                    summary=(
                        f"Switch to open case {case['case_id']} (deadline in {case['steps_until_deadline']} steps, "
                        f"amount ${case['amount']})."
                    ),
                )
            )
        return candidates

    current_deadline = _visible_case_deadline(queue, case_id)
    best_other = _best_open_case([case for case in open_cases if case["case_id"] != case_id])
    if best_other is not None and best_other["steps_until_deadline"] <= 1 and current_deadline > 1:
        candidates.append(
            CandidateAction(
                action=ChargebackOpsAction(action_type="select_case", case_id=best_other["case_id"]),
                summary=(
                    f"Switch to case {best_other['case_id']} immediately because its deadline is in "
                    f"{best_other['steps_until_deadline']} steps."
                ),
            )
        )

    policy = visible_case.get("policy")
    if policy is None:
        candidates.append(
            CandidateAction(
                action=ChargebackOpsAction(action_type="retrieve_policy", case_id=case_id),
                summary="Retrieve the chargeback policy for the selected reason code.",
            )
        )
        recommended_strategy = None
    else:
        recommended_strategy = policy["recommended_strategy"]

    reason_code = visible_case["reason_code"]
    systems_revealed = set(visible_case.get("systems_revealed", []))
    current_strategy = visible_case.get("current_strategy")
    retrieved_items = visible_case.get("retrieved_evidence", [])
    attached_ids = {item["evidence_id"] for item in visible_case.get("attached_evidence", [])}
    attachable_ids = _batch_attachable_ids(retrieved_items, attached_ids)

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
        if attachable_ids and len(attached_ids) < 2:
            candidates.append(
                CandidateAction(
                    action=ChargebackOpsAction(
                        action_type="add_evidence",
                        case_id=case_id,
                        evidence_ids=attachable_ids[:2],
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
                    ),
                    summary="Submit the current representment package.",
                )
            )

    elif reason_code == "fraud_cnp":
        should_contest = recommended_strategy == "contest"
        if should_contest:
            for system_name in ["risk", "support", "orders"]:
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
            if attachable_ids and len(attached_ids) < 2:
                candidates.append(
                    CandidateAction(
                        action=ChargebackOpsAction(
                            action_type="add_evidence",
                            case_id=case_id,
                            evidence_ids=attachable_ids[:2],
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

    elif reason_code == "credit_not_processed":
        for system_name in ["support", "refunds"]:
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

    if visible_case.get("inspection_notes") is None and observation["steps_remaining"] > 3:
        candidates.append(
            CandidateAction(
                action=ChargebackOpsAction(action_type="inspect_case", case_id=case_id),
                summary="Inspect the selected case to reveal merchant notes.",
            )
        )

    for case in sorted(open_cases, key=lambda item: (item["steps_until_deadline"], -item["amount"])):
        if case["case_id"] != case_id:
            candidates.append(
                CandidateAction(
                    action=ChargebackOpsAction(action_type="select_case", case_id=case["case_id"]),
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

    first = candidates[0]
    visible_case = observation.get("visible_case")
    queue = observation["queue"]

    if visible_case is None:
        open_cases = [case for case in queue if case["status"] == "open"]
        if len(open_cases) == 1:
            return first
        urgent_cases = [case for case in open_cases if case["steps_until_deadline"] <= 1]
        if (
            len(urgent_cases) == 1
            and first.action.action_type == "select_case"
            and first.action.case_id == urgent_cases[0]["case_id"]
        ):
            return first
        return None

    if visible_case["status"] != "open":
        return first if first.action.action_type == "select_case" else None

    if first.action.action_type in {
        "retrieve_policy",
        "add_evidence",
        "submit_representment",
        "resolve_case",
    }:
        return first

    if first.action.action_type == "query_system":
        current_strategy = visible_case.get("current_strategy")
        if visible_case.get("policy") is None or current_strategy in {None, "contest"}:
            return first

    if first.action.action_type == "set_strategy":
        strategy = first.action.strategy
        competing_strategies = {
            candidate.action.strategy
            for candidate in candidates[1:]
            if candidate.action.action_type == "set_strategy"
        }
        if strategy in {"accept_chargeback", "issue_refund"} and "contest" not in competing_strategies:
            return first

    if first.action.action_type == "select_case":
        current_case_id = visible_case["case_id"]
        current_deadline = next(
            (case["steps_until_deadline"] for case in queue if case["case_id"] == current_case_id),
            999,
        )
        target_deadline = next(
            (case["steps_until_deadline"] for case in queue if case["case_id"] == first.action.case_id),
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
                "recommended_strategy": visible_case["policy"]["recommended_strategy"],
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
    chosen_provider = (provider or os.getenv("BASELINE_PROVIDER") or DEFAULT_PROVIDER).lower()
    chosen_model = model_name or os.getenv("BASELINE_MODEL") or DEFAULT_MODELS.get(
        chosen_provider,
        "nvidia/nemotron-3-super-120b-a12b:free",
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
    return None


def _provider_pick(
    config: ProviderConfig,
    observation: dict[str, Any],
    candidates: list[CandidateAction],
) -> tuple[CandidateAction, bool, bool, str | None]:
    shortlist, payload = _provider_payload(observation, candidates)

    if config.provider in {"openai", "openrouter", "groq"}:
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
                            "You are a merchant chargeback analyst. Pick the single best next action. "
                            "Prefer on-time, evidence-backed resolutions and avoid weak contests. "
                            "Return valid JSON with keys candidate_index and rationale."
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
                block.text for block in response.content if getattr(block, "type", "") == "text"
            )
            choice = _safe_json_loads(text)
            if choice is None:
                return shortlist[0], True, False, "InvalidJSONResponse"
            index = min(max(choice.candidate_index, 0), len(shortlist) - 1)
            return shortlist[index], True, True, None
        except Exception as exc:
            return shortlist[0], True, False, exc.__class__.__name__

    return shortlist[0], False, False, None


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
                candidate, attempted, succeeded, error_label = _provider_pick(
                    config,
                    observation_payload,
                    candidates,
                )
                provider_calls_attempted += int(attempted)
                provider_calls_succeeded += int(succeeded)
                if attempted and not succeeded and error_label is not None:
                    provider_errors[error_label] = provider_errors.get(error_label, 0) + 1
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
