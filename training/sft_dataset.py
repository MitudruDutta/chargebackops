"""Supervised fine-tuning dataset builder for ChargebackOps.

Rolls the scripted heuristic across each task and captures every
``(observation_prompt, oracle_completion)`` pair as a single-turn
training sample. The completion is the JSON serialisation of the
heuristic action, matching the format the merchant policy must emit
at inference time.

SFT before GRPO is the standard RLHF pattern. It teaches the base
model two things GRPO struggles to learn from sparse reward alone:

* The output schema (valid JSON, the right ``action_type`` strings,
  no extra prose).
* Per-state action variety — the heuristic emits a *different*
  action_type at each state, so an SFT-trained model stops
  collapsing to ``select_case`` at every step.

The module returns plain dicts so the notebook can wrap them in any
trainer's expected dataset format (TRL ``SFTTrainer``, HF
``Dataset.from_list``, etc.) without pulling those deps into the
package.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence

try:
    from ..core.models import ChargebackOpsAction
    from ..server.chargeback_ops_environment import ChargebackOpsEnvironment
    from .env_adapter import build_prompt
except ImportError:  # pragma: no cover
    from core.models import ChargebackOpsAction
    from server.chargeback_ops_environment import ChargebackOpsEnvironment
    from training.env_adapter import build_prompt


def _heuristic_policy(observation_dict: dict[str, Any]) -> ChargebackOpsAction | None:
    try:
        from ..runners.benchmark_runner import heuristic_policy
    except ImportError:  # pragma: no cover
        from runners.benchmark_runner import heuristic_policy
    return heuristic_policy(observation_dict)


def action_to_completion(action: ChargebackOpsAction) -> str:
    """Serialise an action as the canonical JSON completion string."""

    payload = action.model_dump(exclude_none=True)
    payload = {k: v for k, v in payload.items() if v not in ([], "")}
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


@dataclass(frozen=True)
class SFTSample:
    """One supervised training pair."""

    task_id: str
    state_step: int
    prompt: str
    completion: str
    action_type: str


def build_sft_dataset(
    task_ids: Sequence[str],
    *,
    max_states_per_task: int = 24,
) -> list[dict[str, Any]]:
    """Roll heuristic on each task; capture (prompt, oracle_completion) pairs.

    Goes deeper than :func:`training.reward_adapter.build_state_action_dataset`
    (default 24 vs 12 states per task) because SFT benefits from seeing
    the full trajectory — including terminal-resolution actions which
    are rare in the early-state distribution.
    """

    samples: list[dict[str, Any]] = []
    for task_id in task_ids:
        env = ChargebackOpsEnvironment()
        obs = env.reset(task_id=task_id)
        for state_step in range(max_states_per_task):
            if obs.done:
                break
            heur = _heuristic_policy(obs.model_dump())
            if heur is None:
                break
            samples.append(
                {
                    "task_id": task_id,
                    "state_step": state_step,
                    "prompt": build_prompt(obs.model_dump()),
                    "completion": action_to_completion(heur),
                    "action_type": heur.action_type,
                }
            )
            obs = env.step(heur)
    return samples


__all__ = [
    "SFTSample",
    "action_to_completion",
    "build_sft_dataset",
]
