"""Challenge-compatible inference entry point for ChargebackOps."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.parse import urlparse

from openai import OpenAI

try:  # pragma: no cover
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

try:
    from .baseline_runner import (
        MAX_PROVIDER_RESPONSE_TOKENS,
        _chat_completion_with_retry,
        _heuristic_pick,
        _obvious_next_action,
        _provider_payload,
        _safe_json_loads,
        _strict_llm_mode,
        candidate_actions,
    )
    from ..evaluation.grading import grade_episode
    from ..core.models import BaselineRunResult, BaselineTaskResult
    from ..server.chargeback_ops_environment import ChargebackOpsEnvironment
    from ..scenarios.simulation import list_tasks
except ImportError:  # pragma: no cover
    from runners.baseline_runner import (
        MAX_PROVIDER_RESPONSE_TOKENS,
        _chat_completion_with_retry,
        _heuristic_pick,
        _obvious_next_action,
        _provider_payload,
        _safe_json_loads,
        _strict_llm_mode,
        candidate_actions,
    )
    from evaluation.grading import grade_episode
    from core.models import BaselineRunResult, BaselineTaskResult
    from server.chargeback_ops_environment import ChargebackOpsEnvironment
    from scenarios.simulation import list_tasks

if load_dotenv is not None:  # pragma: no cover
    load_dotenv()


def _inference_timeout_seconds() -> float:
    raw_value = os.getenv("INFERENCE_TIMEOUT_SECONDS", os.getenv("BASELINE_REQUEST_TIMEOUT_SECONDS", "15"))
    try:
        return max(1.0, float(raw_value))
    except ValueError:
        return 4.0


def _provider_label(base_url: str | None) -> str:
    if not base_url:
        return "openai_client"
    host = urlparse(base_url).netloc.lower()
    if "openrouter" in host:
        return "openrouter"
    if "groq" in host:
        return "groq"
    if "openai" in host:
        return "openai"
    if "anthropic" in host:
        return "anthropic-compatible"
    if "googleapis" in host or "generativelanguage" in host:
        return "google"
    return host or "openai_client"


def _default_headers(base_url: str | None) -> dict[str, str] | None:
    if not base_url or "openrouter" not in base_url.lower():
        return None

    headers: dict[str, str] = {}
    if os.getenv("OPENROUTER_HTTP_REFERER"):
        headers["HTTP-Referer"] = os.getenv("OPENROUTER_HTTP_REFERER", "")
    if os.getenv("OPENROUTER_APP_TITLE"):
        app_title = os.getenv("OPENROUTER_APP_TITLE", "")
        headers["X-OpenRouter-Title"] = app_title
        headers["X-Title"] = app_title
    return headers or None


def _build_client() -> tuple[OpenAI | None, str, str]:
    api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
    base_url = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
    model_name = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
    if not api_key:
        return None, model_name, _provider_label(base_url)

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        default_headers=_default_headers(base_url),
        timeout=_inference_timeout_seconds(),
        max_retries=0,
    )
    return client, model_name, _provider_label(base_url)


def _build_fallback_client() -> tuple[OpenAI | None, str | None]:
    """Build a fallback client from Groq if the primary provider fails."""
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        return OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=groq_key,
            timeout=_inference_timeout_seconds(),
            max_retries=0,
        ), "llama-3.3-70b-versatile"
    return None, None


def _pick_with_openai_client(
    client: OpenAI,
    model_name: str,
    observation: dict[str, Any],
    candidates: list[Any],
) -> tuple[Any, bool, str | None]:
    shortlist, payload = _provider_payload(observation, candidates)

    try:
        response = _chat_completion_with_retry(
            client,
            model=model_name,
            temperature=0,
            max_tokens=MAX_PROVIDER_RESPONSE_TOKENS,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a merchant chargeback dispute analyst. Pick the single best next action from the candidates. "
                        "Prioritize: 1) deadline-urgent cases, 2) evidence-backed contests, 3) fast concedes for weak cases. "
                        "Avoid attaching harmful evidence. Return JSON: {\"candidate_index\": N, \"rationale\": \"brief reason\"}"
                    ),
                },
                {"role": "user", "content": payload},
            ],
        )
        content = response.choices[0].message.content or "{}"
        choice = _safe_json_loads(content)
        if choice is None:
            return shortlist[0], False, "InvalidJSONResponse"
        index = min(max(choice.candidate_index, 0), len(shortlist) - 1)
        return shortlist[index], True, None
    except Exception as exc:
        return shortlist[0], False, exc.__class__.__name__


def run_inference(*, structured_output: bool = True) -> BaselineRunResult:
    """Run the challenge-compatible inference baseline across all tasks.

    When *structured_output* is True (default), prints ``[START]``,
    ``[STEP]``, and ``[END]`` markers to stdout so the challenge
    validator can parse results.
    """

    client, model_name, provider = _build_client()
    provider_calls_attempted = 0
    provider_calls_succeeded = 0
    provider_errors: dict[str, int] = {}

    task_results: list[BaselineTaskResult] = []
    for task in list_tasks():
        if structured_output:
            print(f"[START] task={task.task_id} env=chargeback_ops model={model_name}", flush=True)

        env = ChargebackOpsEnvironment()
        observation = env.reset(task_id=task.task_id)
        step_num = 0
        rewards: list[float] = []

        while not observation.done:
            observation_payload = observation.model_dump()
            candidates = candidate_actions(observation_payload)
            if not candidates:
                break

            candidate = None
            if len(candidates) == 1:
                candidate = candidates[0]
            else:
                obvious_candidate = _obvious_next_action(observation_payload, candidates)
                if obvious_candidate is not None:
                    candidate = obvious_candidate
                elif client is not None and model_name:
                    candidate, succeeded, error_label = _pick_with_openai_client(
                        client,
                        model_name,
                        observation_payload,
                        candidates,
                    )
                    provider_calls_attempted += 1
                    if not succeeded:
                        fb_client, fb_model = _build_fallback_client()
                        if fb_client is not None and fb_model:
                            candidate, fb_ok, fb_err = _pick_with_openai_client(
                                fb_client, fb_model, observation_payload, candidates,
                            )
                            if fb_ok:
                                succeeded = True
                                error_label = None
                    provider_calls_succeeded += int(succeeded)
                    if not succeeded and error_label is not None:
                        provider_errors[error_label] = provider_errors.get(error_label, 0) + 1
                    if _strict_llm_mode() and not succeeded:
                        raise RuntimeError(
                            "STRICT_LLM_MODE is enabled and the provider decision failed, "
                            "so heuristic fallback is not allowed."
                        )
                else:
                    candidate = _heuristic_pick(candidates)

            action = candidate.action
            action_str = action.action_type
            if action.case_id:
                action_str += f"({action.case_id})"

            observation = env.step(action)
            step_num += 1
            reward = observation.reward or 0.0
            rewards.append(reward)

            if structured_output:
                error_val = "null"
                if observation.last_action_result and "error" in observation.last_action_result.lower():
                    error_val = observation.last_action_result
                print(
                    f"[STEP] step={step_num} action={action_str} "
                    f"reward={reward:.2f} done={str(observation.done).lower()} "
                    f"error={error_val}",
                    flush=True,
                )

        report = env.state.grader_report or grade_episode(
            task,
            env._progress_by_case,  # type: ignore[attr-defined]
            env.state.step_count,
            env.state.episode_id or "",
            completed=env.state.completed,
        )
        score = report.normalized_score
        task_results.append(
            BaselineTaskResult(
                task_id=task.task_id,
                title=task.title,
                score=score,
                steps_used=env.state.step_count,
                final_status=report.summary,
            )
        )
        if structured_output:
            rewards_str = ",".join(f"{r:.2f}" for r in rewards)
            print(
                f"[END] success={str(score >= 0.1).lower()} steps={step_num} "
                f"score={score:.2f} rewards={rewards_str}",
                flush=True,
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
        mode = "openai_client_with_fallback"
    else:
        mode = "openai_client"
    return BaselineRunResult(
        provider=provider,
        model_name=model_name or "heuristic_fallback",
        mode=mode,
        provider_calls_attempted=provider_calls_attempted,
        provider_calls_succeeded=provider_calls_succeeded,
        provider_errors=provider_errors,
        task_results=task_results,
        average_score=average_score,
    )


def main() -> None:
    """CLI entry point used by the challenge validator."""

    print(json.dumps(run_inference().model_dump(), indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
