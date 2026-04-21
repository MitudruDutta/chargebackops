"""Unit tests for the SFT dataset builder.

The supervised pre-training stage feeds (prompt, oracle_completion)
pairs into the base model so it learns the JSON schema and per-state
action variety *before* GRPO. These tests pin the contract so the
notebook's SFT cell stays stable.
"""

from __future__ import annotations

import json

from training.env_adapter import parse_completion
from training.sft_dataset import (
    action_to_completion,
    build_sft_dataset,
)


def test_action_to_completion_round_trips_through_parser():
    """Oracle completion must parse back into the same action dict."""

    from runners.benchmark_runner import heuristic_policy
    from server.chargeback_ops_environment import ChargebackOpsEnvironment

    env = ChargebackOpsEnvironment()
    obs = env.reset(task_id="goods_not_received_easy")
    action = heuristic_policy(obs.model_dump())
    completion = action_to_completion(action)

    parsed = parse_completion(completion)
    assert parsed is not None
    assert parsed["action_type"] == action.action_type
    if action.case_id:
        assert parsed["case_id"] == action.case_id


def test_build_sft_dataset_has_action_variety():
    """SFT dataset must include >1 distinct action_type per task.

    The whole point of SFT is to teach the model that different states
    require different action_types. If the heuristic only ever emits
    ``select_case`` we have no variety to teach and SFT is useless.
    """

    samples = build_sft_dataset(
        ["goods_not_received_easy"], max_states_per_task=24
    )
    assert len(samples) >= 4
    action_types = {s["action_type"] for s in samples}
    assert len(action_types) >= 3, f"only saw {action_types}"


def test_build_sft_dataset_completion_is_valid_json():
    samples = build_sft_dataset(
        ["goods_not_received_easy"], max_states_per_task=10
    )
    for s in samples:
        decoded = json.loads(s["completion"])
        assert decoded["action_type"] == s["action_type"]


def test_build_sft_dataset_state_steps_monotonic():
    samples = build_sft_dataset(
        ["goods_not_received_easy"], max_states_per_task=10
    )
    state_steps = [s["state_step"] for s in samples]
    assert state_steps == sorted(state_steps)
    assert state_steps[0] == 0


def test_build_sft_dataset_handles_multiple_tasks():
    samples = build_sft_dataset(
        ["goods_not_received_easy", "queue_optimization_hard"],
        max_states_per_task=6,
    )
    task_ids = {s["task_id"] for s in samples}
    assert task_ids == {"goods_not_received_easy", "queue_optimization_hard"}
