"""Unit tests for the training adapter.

Pin the prompt/completion serialization and the episode-replay reward
signal so the training notebook has a stable offline contract.
"""

from __future__ import annotations

import json

from core.models import ChargebackOpsAction
from scenarios.simulation import get_task
from server.chargeback_ops_environment import ChargebackOpsEnvironment
from training.env_adapter import (
    action_from_completion,
    build_prompt,
    parse_completion,
)
from training.reward_adapter import (
    compute_reward,
    run_episode_with_text_policy,
)


def _fresh_observation(task_id: str = "goods_not_received_easy"):
    env = ChargebackOpsEnvironment()
    return env.reset(task_id=task_id).model_dump()


def test_build_prompt_is_deterministic_and_includes_available_actions():
    obs = _fresh_observation()
    a = build_prompt(obs)
    b = build_prompt(obs)
    assert a == b
    assert "available_actions" in a
    assert "OBSERVATION:" in a
    assert "ACTION:" in a


def test_parse_completion_accepts_plain_json():
    payload = '{"action_type": "select_case", "case_id": "CB-X"}'
    parsed = parse_completion(payload)
    assert parsed == {"action_type": "select_case", "case_id": "CB-X"}


def test_parse_completion_strips_code_fence():
    payload = '```json\n{"action_type": "select_case", "case_id": "CB-X"}\n```'
    parsed = parse_completion(payload)
    assert parsed == {"action_type": "select_case", "case_id": "CB-X"}


def test_parse_completion_returns_none_on_garbage():
    assert parse_completion("") is None
    assert parse_completion("not json at all") is None
    assert parse_completion("{not-valid-json}") is None


def test_parse_completion_drops_unknown_fields():
    payload = json.dumps({"action_type": "select_case", "hack_field": 42})
    parsed = parse_completion(payload)
    assert parsed == {"action_type": "select_case"}


def test_action_from_completion_returns_valid_action():
    payload = '{"action_type": "select_case", "case_id": "CB-X"}'
    action = action_from_completion(payload)
    assert isinstance(action, ChargebackOpsAction)
    assert action.action_type == "select_case"
    assert action.case_id == "CB-X"


def test_action_from_completion_returns_none_on_bad_type():
    payload = '{"action_type": "not_a_real_action"}'
    assert action_from_completion(payload) is None


def test_run_episode_falls_back_to_heuristic_on_empty_completion():
    """Unparseable completions must not deadlock the episode."""
    result = run_episode_with_text_policy(
        "goods_not_received_easy",
        text_policy=lambda _prompt: "",
    )
    assert result.steps_used > 0
    assert result.invalid_actions > 0
    assert result.score > 0.0  # heuristic fallback still scores


def test_compute_reward_matches_episode_score():
    """Single completion + heuristic tail reproduces the heuristic score."""
    task = get_task("goods_not_received_easy")
    prompts = ["unused"]
    completions = [""]  # triggers heuristic fallback on the first action
    rewards = compute_reward(
        prompts, completions, task_ids=[task.task_id]
    )
    assert len(rewards) == 1
    assert 0.0 <= rewards[0] <= 1.0
    assert rewards[0] > 0.5  # heuristic scores ~0.97 on this task


def test_compute_reward_rejects_mismatched_lengths():
    import pytest

    with pytest.raises(ValueError):
        compute_reward(["a"], ["b", "c"], task_ids=["goods_not_received_easy"])
