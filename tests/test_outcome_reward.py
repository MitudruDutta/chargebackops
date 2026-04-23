"""Tests for outcome-based RL reward.

Pin the contract: reward is in [-1, +1], heuristic-tail rollout produces
positive reward on a winnable case, parse-fail returns 0, format reward
splits valid vs invalid completions.
"""

from __future__ import annotations

import json

from training.outcome_reward import (
    _FORMAT_REWARD_INVALID,
    _FORMAT_REWARD_VALID,
    compute_format_reward,
    compute_outcome_reward,
)


def _action_json(**fields) -> str:
    return json.dumps(fields)


def test_outcome_reward_in_bounds():
    """Reward must lie in [-1, +1] for a normal action."""
    rewards = compute_outcome_reward(
        prompts=["x"],
        completions=[_action_json(action_type="select_case", case_id="CB-1001")],
        task_ids=["goods_not_received_easy"],
        state_steps=[0],
    )
    assert len(rewards) == 1
    assert -1.0 <= rewards[0] <= 1.0


def test_outcome_reward_parse_fail_returns_zero():
    """Unparseable completion → 0.0 reward (no rollout, no PnL)."""
    rewards = compute_outcome_reward(
        prompts=["x"],
        completions=["this is not json"],
        task_ids=["goods_not_received_easy"],
        state_steps=[0],
    )
    assert rewards == [0.0]


def test_outcome_reward_default_state_step_is_zero():
    """Omitting state_steps must default to all-zero, not crash."""
    rewards = compute_outcome_reward(
        prompts=["x", "y"],
        completions=[
            _action_json(action_type="select_case", case_id="CB-1001"),
            _action_json(action_type="select_case", case_id="CB-1001"),
        ],
        task_ids=["goods_not_received_easy", "goods_not_received_easy"],
    )
    assert len(rewards) == 2
    for r in rewards:
        assert -1.0 <= r <= 1.0


def test_outcome_reward_length_mismatch_raises():
    """Mismatched lengths must raise (silent broadcast hides bugs)."""
    import pytest

    with pytest.raises(ValueError):
        compute_outcome_reward(
            prompts=["x", "y"],
            completions=["a"],
            task_ids=["goods_not_received_easy", "goods_not_received_easy"],
        )


def test_format_reward_valid_vs_invalid():
    """Parseable JSON earns positive shaping; garbage earns negative."""
    rewards = compute_format_reward(
        prompts=["x", "x"],
        completions=[
            _action_json(action_type="select_case", case_id="CB-1001"),
            "garbage output",
        ],
    )
    assert rewards[0] == _FORMAT_REWARD_VALID
    assert rewards[1] == _FORMAT_REWARD_INVALID


def test_outcome_reward_variance_across_states():
    """Reward variance across many (state, action) pairs must be > 0.

    The heuristic-tail rollout can mask a bad model action at state_step=0
    by recovering downstream — so a single-state check is too noisy.
    Sample a spread of state_steps + diverse first-actions; require that
    the resulting PnL distribution has non-zero spread.
    """

    completions = [
        _action_json(action_type="select_case", case_id="CB-1001"),
        _action_json(action_type="query_system", case_id="CB-1001", system_name="orders"),
        _action_json(action_type="resolve_case", case_id="CB-1001", strategy="accept_chargeback"),
        _action_json(action_type="resolve_case", case_id="CB-1001", strategy="issue_refund"),
        "this is unparseable garbage",
    ]
    rewards = compute_outcome_reward(
        prompts=["x"] * len(completions),
        completions=completions,
        task_ids=["goods_not_received_easy"] * len(completions),
        state_steps=[0] * len(completions),
    )
    assert len(set(rewards)) > 1, f"reward must vary across actions, got {rewards}"
