"""Reward adapter for GRPO / RL training on ChargebackOps.

Exposes a callable shape compatible with TRL's GRPO reward function:

``reward_fn(prompts, completions, **kwargs) -> list[float]``

Each completion is parsed into an action sequence (one action per line
is the simplest case; the helper also accepts a single-action
completion and runs the remainder of the episode under the scripted
heuristic so training always produces a terminal score). The resulting
reward is the episode's deterministic normalized grade in ``[0, 1]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

try:
    from ..core.models import ChargebackOpsAction, ChargebackOpsObservation
    from ..scenarios.simulation import get_task, list_tasks
    from ..server.chargeback_ops_environment import ChargebackOpsEnvironment
    from .env_adapter import action_from_completion, build_prompt
except ImportError:  # pragma: no cover
    from core.models import ChargebackOpsAction, ChargebackOpsObservation
    from scenarios.simulation import get_task, list_tasks
    from server.chargeback_ops_environment import ChargebackOpsEnvironment
    from training.env_adapter import action_from_completion, build_prompt


TextPolicyFn = Callable[[str], str]


@dataclass(frozen=True)
class EpisodeResult:
    """Outcome of a single rollout."""

    task_id: str
    score: float
    steps_used: int
    invalid_actions: int
    prompts: tuple[str, ...] = field(default_factory=tuple)
    completions: tuple[str, ...] = field(default_factory=tuple)


def _fallback_action(
    observation: ChargebackOpsObservation,
) -> ChargebackOpsAction | None:
    """Scripted fallback when the model output is unparseable."""

    try:
        from ..runners.benchmark_runner import heuristic_policy
    except ImportError:  # pragma: no cover
        from runners.benchmark_runner import heuristic_policy
    return heuristic_policy(observation.model_dump())


def run_episode_with_text_policy(
    task_id: str,
    text_policy: TextPolicyFn,
    *,
    max_steps: int | None = None,
    capture_trace: bool = False,
) -> EpisodeResult:
    """Roll one episode forward under a text-in / text-out policy.

    The policy is invoked at every step. If the completion fails to
    parse into a valid action the scripted heuristic is used instead;
    this keeps early-training trajectories from deadlocking.
    """

    task = get_task(task_id)
    env = ChargebackOpsEnvironment()
    observation = env.reset(task_id=task_id)
    step_budget = (max_steps if max_steps is not None else task.max_steps) + 5
    steps = 0
    invalid = 0
    prompts: list[str] = []
    completions: list[str] = []

    while not observation.done and steps < step_budget:
        obs_dict = observation.model_dump()
        prompt = build_prompt(obs_dict)
        completion = text_policy(prompt)
        action = action_from_completion(completion)
        if action is None:
            invalid += 1
            action = _fallback_action(observation)
            if action is None:
                break
        observation = env.step(action)
        steps += 1
        if capture_trace:
            prompts.append(prompt)
            completions.append(completion)

    report = env.state.grader_report
    score = float(report.normalized_score) if report is not None else 0.0
    return EpisodeResult(
        task_id=task_id,
        score=score,
        steps_used=env.state.step_count,
        invalid_actions=invalid,
        prompts=tuple(prompts),
        completions=tuple(completions),
    )


def compute_reward(
    prompts: Sequence[str],
    completions: Sequence[str],
    *,
    task_ids: Sequence[str] | None = None,
    **_: Any,
) -> list[float]:
    """GRPO-style reward function.

    Each ``completion`` is replayed as a *single* action. The remainder
    of the episode is driven by the scripted heuristic, so the reward
    signal rewards the model for picking a good first move from a
    given observation. This matches the behaviour TRL expects: one
    ``(prompt, completion)`` pair → one scalar reward.

    ``task_ids`` optionally binds each prompt to a task id for env
    replay. When omitted, the headline catalog is cycled.
    """

    if task_ids is None:
        headline = [task.task_id for task in list_tasks()]
        task_ids = [headline[i % len(headline)] for i in range(len(prompts))]
    if len(task_ids) != len(prompts) or len(prompts) != len(completions):
        raise ValueError(
            "prompts, completions, and task_ids must all have the same length"
        )

    rewards: list[float] = []
    for task_id, completion in zip(task_ids, completions):
        first_action = action_from_completion(completion)

        def _once(_prompt: str, _used=[False], _action=first_action) -> str:
            if _used[0] or _action is None:
                return ""
            _used[0] = True
            return completion

        result = run_episode_with_text_policy(task_id, _once)
        rewards.append(result.score)
    return rewards


__all__ = [
    "EpisodeResult",
    "TextPolicyFn",
    "compute_reward",
    "run_episode_with_text_policy",
]
