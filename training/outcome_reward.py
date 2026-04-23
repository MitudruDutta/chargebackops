"""Outcome-based reward for ChargebackOps RL training.

Replaces per-action heuristic-match (distillation) with terminal $-PnL
from the multi-round dispute lifecycle. The merchant policy gets the
*economic outcome* of its action — not whether the action looked like the
scripted heuristic. That removes the specification-gaming risk: the model
cannot earn reward by mimicking the heuristic if mimicking the heuristic
loses money against the Issuer.

Reward construction per (prompt, completion, task_id, state_step):

1. Reset env to ``task_id``, replay the heuristic for ``state_step`` steps.
2. Take the model's parsed action at that state.
3. Continue the rollout with the heuristic until the episode terminates
   or the per-task step budget is hit.
4. Reward = sum of ``CaseProgress.final_economic_outcome`` across all
   resolved cases, normalized by sum of ``case.amount`` so the signal is
   in [-1, +1] regardless of dispute size. Unresolved cases are treated
   as -case.amount (merchant defaults to losing).

This is REINFORCE with a heuristic-rollout baseline: the only thing the
model controls is one action, so the credit assignment is clean.

A second reward function (``compute_format_reward``) returns +0.05 for
parseable JSON, -0.10 for parse-fail, providing dense early-training
signal so GRPO has gradient before the policy can produce winning packets.
"""

from __future__ import annotations

from typing import Any, Sequence

try:
    from ..scenarios.simulation import get_task, list_tasks
    from ..server.chargeback_ops_environment import ChargebackOpsEnvironment
    from .env_adapter import action_from_completion
    from .reward_adapter import _advance_to_state, _heuristic_policy
except ImportError:  # pragma: no cover
    from scenarios.simulation import get_task, list_tasks
    from server.chargeback_ops_environment import ChargebackOpsEnvironment
    from training.env_adapter import action_from_completion
    from training.reward_adapter import _advance_to_state, _heuristic_policy


# Per-step budget for the heuristic tail roll-forward after the model
# takes its single action. Capped so a stuck model can't drag training
# throughput down. The env's own ``task.max_steps`` is the real cap;
# this is a defensive secondary limit.
_TAIL_STEP_CAP: int = 80

# Format reward magnitudes. Positive on parseable JSON gives a non-zero
# signal even when outcome reward is the same across the group (e.g.
# every rollout loses arbitration). Negative on parse-fail keeps GRPO
# from rewarding pure noise.
_FORMAT_REWARD_VALID: float = 0.05
_FORMAT_REWARD_INVALID: float = -0.10


def _episode_pnl_normalized(env: ChargebackOpsEnvironment) -> float:
    """Sum final_economic_outcome across cases / sum of case.amount.

    Cases without a recorded outcome (still ``open`` at episode end, or
    resolved through a non-economic path) default to -case.amount: the
    merchant loses the dispute when no positive resolution was reached.
    """

    task = env._task  # type: ignore[attr-defined]
    progress_by_case = env._progress_by_case  # type: ignore[attr-defined]

    total_pnl = 0.0
    total_amount = 0.0
    for case in task.cases:
        progress = progress_by_case.get(case.case_id)
        total_amount += case.amount
        if progress is None or progress.final_economic_outcome is None:
            total_pnl -= case.amount
        else:
            total_pnl += progress.final_economic_outcome

    if total_amount <= 0:
        return 0.0
    return max(-1.0, min(1.0, total_pnl / total_amount))


def _rollout_terminal_reward(
    task_id: str,
    state_step: int,
    completion: str,
    *,
    tail_step_cap: int = _TAIL_STEP_CAP,
) -> float | None:
    """Reset env to (task_id, state_step), apply model action, run heuristic
    to termination, and return normalized terminal $-PnL.

    Returns None if the env can't be advanced to the requested state (dataset
    drift) or if the model action raises before the env can record it. The
    caller maps None to 0.0 reward.
    """

    advanced = _advance_to_state(task_id, int(state_step))
    if advanced is None:
        return None
    env, obs = advanced

    action = action_from_completion(completion)
    if action is None:
        return None

    try:
        obs = env.step(action)
    except Exception:
        return None

    steps = 0
    while not obs.done and steps < tail_step_cap:
        heur = _heuristic_policy(obs.model_dump())
        if heur is None:
            break
        try:
            obs = env.step(heur)
        except Exception:
            break
        steps += 1

    return _episode_pnl_normalized(env)


def compute_outcome_reward(
    prompts: Sequence[str],
    completions: Sequence[str],
    *,
    task_ids: Sequence[str] | None = None,
    state_steps: Sequence[int] | None = None,
    **_: Any,
) -> list[float]:
    """GRPO-compatible reward: terminal $-PnL after model action + heuristic tail.

    Drop-in replacement for ``training.reward_adapter.compute_reward``.
    Same signature, different reward source: outcome dollars vs heuristic
    match.
    """

    if task_ids is None:
        headline = [task.task_id for task in list_tasks()]
        task_ids = [headline[i % len(headline)] for i in range(len(prompts))]
    if len(task_ids) != len(prompts) or len(prompts) != len(completions):
        raise ValueError(
            "prompts, completions, and task_ids must all have the same length"
        )
    if state_steps is None:
        state_steps = [0] * len(prompts)
    if len(state_steps) != len(prompts):
        raise ValueError("state_steps must have the same length as prompts")

    rewards: list[float] = []
    for task_id, state_step, completion in zip(task_ids, state_steps, completions):
        pnl = _rollout_terminal_reward(task_id, int(state_step), completion)
        rewards.append(0.0 if pnl is None else float(pnl))
    return rewards


def compute_format_reward(
    prompts: Sequence[str],
    completions: Sequence[str],
    **_: Any,
) -> list[float]:
    """Dense parse-validity shaping reward.

    Sits alongside ``compute_outcome_reward`` in TRL's ``reward_funcs``
    list so the model gets non-zero gradient even when every rollout in
    the group lands on the same terminal outcome.
    """

    rewards: list[float] = []
    for completion in completions:
        action = action_from_completion(completion)
        rewards.append(_FORMAT_REWARD_VALID if action is not None else _FORMAT_REWARD_INVALID)
    return rewards


__all__ = [
    "compute_outcome_reward",
    "compute_format_reward",
]
