"""Reward adapter for GRPO / RL training on ChargebackOps.

Exposes a callable shape compatible with TRL's GRPO reward function:

``reward_fn(prompts, completions, **kwargs) -> list[float]``

The reward is a *per-action* match score against the scripted heuristic
oracle at the dataset's recorded environment state. Episode replay was
removed deliberately: previously every parse-failure fell back to the
heuristic and earned ~0.96 reward, which trained the model that emitting
garbage was optimal (group reward variance ≈ 0 → GRPO advantage = 0 →
loss collapsed). Per-action scoring against the oracle gives high
variance even for an untrained model: parse-fails earn 0.0, valid-but-
wrong actions earn 0.1-0.7, exact matches earn 1.0.
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


@dataclass(frozen=True)
class StateActionSample:
    """One (env_state, oracle_action) pair captured from a heuristic rollout."""

    task_id: str
    state_step: int
    prompt: str
    oracle_action_type: str


def _heuristic_policy(observation_dict: dict[str, Any]) -> ChargebackOpsAction | None:
    try:
        from ..runners.benchmark_runner import heuristic_policy
    except ImportError:  # pragma: no cover
        from runners.benchmark_runner import heuristic_policy
    return heuristic_policy(observation_dict)


def _fallback_action(
    observation: ChargebackOpsObservation,
) -> ChargebackOpsAction | None:
    """Scripted fallback used by the debug/eval rollout helper only.

    NOTE: deliberately *not* used by :func:`compute_reward` — falling back
    to the heuristic on parse-fail trains the model that garbage = good.
    """

    return _heuristic_policy(observation.model_dump())


def _state_signature(observation: ChargebackOpsObservation) -> tuple:
    """Stable hashable snapshot of the env state visible to the policy.

    Used to detect rollout stalls — if step() leaves this signature
    unchanged, the model picked an action the env silently no-op'd
    (e.g. ``select_case`` when a case is already selected) and is
    about to loop forever on the same prompt. ``steps_remaining`` is
    deliberately excluded: it decrements on every step regardless of
    whether the env actually progressed, so including it would mask
    every real stall.
    """

    visible = observation.visible_case
    visible_sig: tuple
    if visible is None:
        visible_sig = ()
    else:
        visible_sig = (
            visible.case_id,
            visible.status,
            visible.current_strategy,
            len(visible.attached_evidence),
            len(visible.retrieved_evidence),
        )
    return (
        observation.selected_case_id,
        tuple(sorted(observation.available_actions)),
        observation.done,
        visible_sig,
    )


def _action_key(action: ChargebackOpsAction) -> tuple:
    """Hashable identity for "have we tried this exact action here?" check."""

    return (
        action.action_type,
        action.case_id,
        action.system_name,
        tuple(action.evidence_ids),
        action.strategy,
    )


def _predicted_noop(
    action: ChargebackOpsAction,
    observation: ChargebackOpsObservation,
) -> bool:
    """Cheap upfront check that the env will silently no-op this action.

    Catches the dominant Qwen failure mode (always emit ``select_case``
    even after a case is already selected). Without this check the
    model burns an env step per state on the duplicate ``select_case``,
    blowing the per-task step budget before the heuristic fallback can
    finish the episode. We only hard-code rules we *know* the env
    treats as no-ops; everything else flows through the env and the
    post-hoc ``tried_at_state`` cache.
    """

    if (
        action.action_type == "select_case"
        and observation.selected_case_id is not None
        and action.case_id == observation.selected_case_id
    ):
        return True
    return False


def run_episode_with_text_policy(
    task_id: str,
    text_policy: TextPolicyFn,
    *,
    max_steps: int | None = None,
    capture_trace: bool = False,
) -> EpisodeResult:
    """Roll one episode forward under a text-in / text-out policy.

    Used for evaluation and debugging only. Falls back to the scripted
    heuristic when the policy returns unparseable output **or** when
    the model picks an action it has already tried from the current
    state (the env silently no-ops the duplicate, ``done`` never flips,
    score stays 0). The repeat-action guard catches the dominant Qwen
    failure mode where a checkpoint always emits ``select_case`` and
    the episode loops forever. **Not** used for training reward.
    """

    task = get_task(task_id)
    env = ChargebackOpsEnvironment()
    observation = env.reset(task_id=task_id)
    step_budget = (max_steps if max_steps is not None else task.max_steps) + 5
    steps = 0
    invalid = 0
    tried_at_state: dict[tuple, set[tuple]] = {}
    prompts: list[str] = []
    completions: list[str] = []

    while not observation.done and steps < step_budget:
        obs_dict = observation.model_dump()
        prompt = build_prompt(obs_dict)
        completion = text_policy(prompt)
        action = action_from_completion(completion)
        used_fallback = False
        if action is None:
            invalid += 1
            action = _fallback_action(observation)
            used_fallback = True
            if action is None:
                break

        if not used_fallback and _predicted_noop(action, observation):
            fallback = _fallback_action(observation)
            if fallback is not None:
                action = fallback
                used_fallback = True

        state_sig = _state_signature(observation)
        attempted = tried_at_state.setdefault(state_sig, set())
        action_key = _action_key(action)
        if action_key in attempted and not used_fallback:
            fallback = _fallback_action(observation)
            if fallback is None:
                break
            fallback_key = _action_key(fallback)
            if fallback_key in attempted:
                # Heuristic also stuck — bail out, score whatever we have.
                break
            action = fallback
            action_key = fallback_key
            used_fallback = True
        attempted.add(action_key)

        observation = env.step(action)
        steps += 1
        if capture_trace:
            prompts.append(prompt)
            tag = "<<fallback>> " if used_fallback else ""
            completions.append(f"{tag}{completion}")

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


def _advance_to_state(
    task_id: str, state_step: int
) -> tuple[ChargebackOpsEnvironment, ChargebackOpsObservation] | None:
    """Reset env and replay heuristic for ``state_step`` steps.

    Returns ``None`` if the heuristic terminates the episode before
    reaching ``state_step`` (e.g. dataset went stale).
    """

    env = ChargebackOpsEnvironment()
    obs = env.reset(task_id=task_id)
    for _ in range(state_step):
        if obs.done:
            return None
        heur = _heuristic_policy(obs.model_dump())
        if heur is None:
            return None
        obs = env.step(heur)
    if obs.done:
        return None
    return env, obs


def _score_action_match(
    action: ChargebackOpsAction,
    heuristic: ChargebackOpsAction,
    available_actions: list[str],
) -> float:
    """Score a model action against the oracle (heuristic) at this state.

    Tiers (chosen for non-degenerate reward variance under GRPO sampling):

    * 0.0 — parse-fail (handled by caller before calling this).
    * 0.1 — parses, but action_type not in the env's allowed set at this
      state. The model emitted valid JSON but picked an impossible move.
    * 0.4 — same valid action_type as heuristic neighbourhood but a
      different action_type than the oracle. Valid exploration, low credit.
    * 0.7 — right action_type, wrong target (e.g. picked a different case
      or system than the oracle). Right idea, wrong object.
    * 1.0 — exact match on action_type + targeted fields.
    """

    if action.action_type not in available_actions:
        return 0.1

    if action.action_type != heuristic.action_type:
        return 0.4

    if heuristic.case_id and action.case_id != heuristic.case_id:
        return 0.7

    if heuristic.system_name and action.system_name != heuristic.system_name:
        return 0.7

    return 1.0


def compute_reward(
    prompts: Sequence[str],
    completions: Sequence[str],
    *,
    task_ids: Sequence[str] | None = None,
    state_steps: Sequence[int] | None = None,
    **_: Any,
) -> list[float]:
    """GRPO-style per-action reward.

    For each ``(task_id, state_step, completion)`` triple:

    1. Reset env to ``task_id`` and replay the heuristic for
       ``state_step`` steps to land on the dataset state.
    2. Parse the completion into an action.
    3. Score the action against the heuristic oracle at that state via
       :func:`_score_action_match`.

    No fallback to the heuristic on parse-fail (the prior design did
    this; it created a reward floor that flattened group variance and
    starved GRPO of gradient signal).

    ``state_steps`` defaults to all-zero (initial state) when omitted, so
    legacy callers that only pass ``task_ids`` still work.
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
        action = action_from_completion(completion)
        if action is None:
            rewards.append(0.0)
            continue

        advanced = _advance_to_state(task_id, int(state_step))
        if advanced is None:
            rewards.append(0.0)
            continue
        _env, obs = advanced

        heur = _heuristic_policy(obs.model_dump())
        if heur is None:
            rewards.append(0.0)
            continue

        rewards.append(
            _score_action_match(action, heur, list(obs.available_actions))
        )
    return rewards


def build_state_action_dataset(
    task_ids: Sequence[str],
    *,
    max_states_per_task: int = 12,
) -> list[dict[str, Any]]:
    """Roll the heuristic on each task and capture (state, oracle) pairs.

    For each task we reset, then step the heuristic forward and record
    the prompt string + state_step at every state until termination or
    ``max_states_per_task``. The resulting list is suitable as a TRL
    dataset (each row carries ``prompt``, ``task_id``, ``state_step``).
    """

    samples: list[dict[str, Any]] = []
    for task_id in task_ids:
        env = ChargebackOpsEnvironment()
        obs = env.reset(task_id=task_id)
        for state_step in range(max_states_per_task):
            if obs.done:
                break
            samples.append(
                {
                    "task_id": task_id,
                    "state_step": state_step,
                    "prompt": build_prompt(obs.model_dump()),
                }
            )
            heur = _heuristic_policy(obs.model_dump())
            if heur is None:
                break
            obs = env.step(heur)
    return samples


__all__ = [
    "EpisodeResult",
    "StateActionSample",
    "TextPolicyFn",
    "build_state_action_dataset",
    "compute_reward",
    "run_episode_with_text_policy",
]
