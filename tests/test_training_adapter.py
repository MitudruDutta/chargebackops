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
    build_state_action_dataset,
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


def test_parse_completion_handles_truncated_json():
    """Mid-string truncation: tolerate by closing at last balanced field."""
    payload = (
        '```json\n{"action_type": "select_case", "case_id": "CB-E1", '
        '"strategy": "Select the case ID to procee'
    )
    parsed = parse_completion(payload)
    assert parsed is not None
    assert parsed["action_type"] == "select_case"
    assert parsed["case_id"] == "CB-E1"


def test_parse_completion_strips_think_block():
    payload = (
        '<think>\nlet me think about this\n</think>\n'
        '{"action_type": "select_case", "case_id": "CB-1"}'
    )
    parsed = parse_completion(payload)
    assert parsed == {"action_type": "select_case", "case_id": "CB-1"}


def test_parse_completion_infers_action_type_from_prefix():
    """Model emits action name as prose then JSON without action_type field."""
    payload = ' select_case\n{"case_id": "CB-X", "strategy": "go"}'
    parsed = parse_completion(payload)
    assert parsed is not None
    assert parsed["action_type"] == "select_case"
    assert parsed["case_id"] == "CB-X"


def test_run_episode_falls_back_to_heuristic_on_empty_completion():
    """Unparseable completions must not deadlock the episode."""
    result = run_episode_with_text_policy(
        "goods_not_received_easy",
        text_policy=lambda _prompt: "",
    )
    assert result.steps_used > 0
    assert result.invalid_actions > 0
    assert result.score > 0.0  # heuristic fallback still scores


def test_compute_reward_unparseable_returns_zero():
    """Per-action scorer must NOT fall back to heuristic on parse-fail.

    The previous fallback design poisoned the GRPO signal: garbage
    completions earned ~0.96 reward (heuristic played the episode), so
    the model learned that emitting garbage was optimal and group
    reward variance collapsed to ~0.005, killing the gradient.
    """

    rewards = compute_reward(
        ["unused"], [""], task_ids=["goods_not_received_easy"]
    )
    assert rewards == [0.0]


def test_compute_reward_exact_match_scores_one():
    """Completion that matches the heuristic action exactly gets 1.0."""

    import json

    from runners.benchmark_runner import heuristic_policy
    from server.chargeback_ops_environment import ChargebackOpsEnvironment

    env = ChargebackOpsEnvironment()
    obs = env.reset(task_id="goods_not_received_easy")
    oracle = heuristic_policy(obs.model_dump())
    completion = json.dumps(oracle.model_dump(exclude_none=True))

    rewards = compute_reward(
        ["unused"], [completion], task_ids=["goods_not_received_easy"]
    )
    assert rewards == [1.0]


def test_compute_reward_unavailable_action_scores_low():
    """Valid JSON but action_type not allowed at this state → 0.1."""

    # First state on goods_not_received_easy only allows ``select_case``.
    completion = '{"action_type": "submit_representment", "case_id": "CB-E1"}'
    rewards = compute_reward(
        ["unused"], [completion], task_ids=["goods_not_received_easy"]
    )
    assert rewards == [0.1]


def test_compute_reward_has_real_variance_across_diverse_completions():
    """Diverse completions must produce distinct rewards (the whole point).

    The prior design produced std ≈ 0.005 across 6 wildly different
    completions because the heuristic dominated the episode. New design
    should give ≥ 3 distinct reward values across the same set.
    """

    import json

    from runners.benchmark_runner import heuristic_policy
    from server.chargeback_ops_environment import ChargebackOpsEnvironment

    env = ChargebackOpsEnvironment()
    obs = env.reset(task_id="goods_not_received_easy")
    oracle = heuristic_policy(obs.model_dump())

    completions = [
        "",  # parse-fail → 0.0
        "garbage no json",  # parse-fail → 0.0
        '{"action_type": "submit_representment", "case_id": "CB-E1"}',  # unavailable → 0.1
        json.dumps(oracle.model_dump(exclude_none=True)),  # exact → 1.0
    ]
    rewards = compute_reward(
        ["x"] * 4, completions, task_ids=["goods_not_received_easy"] * 4
    )
    assert len(set(rewards)) >= 3
    assert max(rewards) - min(rewards) >= 0.5


def test_compute_reward_state_steps_advance_env():
    """state_steps replays heuristic to reach mid-episode states."""

    rewards = compute_reward(
        ["x", "x"],
        ["", ""],
        task_ids=["goods_not_received_easy", "goods_not_received_easy"],
        state_steps=[0, 2],
    )
    # Both unparseable → both 0.0 regardless of state.
    assert rewards == [0.0, 0.0]


def test_build_state_action_dataset_covers_multiple_states():
    """Heuristic rollout must yield several (state, oracle) pairs per task."""

    samples = build_state_action_dataset(
        ["goods_not_received_easy"], max_states_per_task=8
    )
    assert len(samples) >= 2
    state_steps = [s["state_step"] for s in samples]
    assert state_steps == sorted(state_steps)
    assert state_steps[0] == 0
    for s in samples:
        assert s["task_id"] == "goods_not_received_easy"
        assert "OBSERVATION:" in s["prompt"]


def test_compute_reward_rejects_mismatched_lengths():
    import pytest

    with pytest.raises(ValueError):
        compute_reward(["a"], ["b", "c"], task_ids=["goods_not_received_easy"])


def test_run_episode_breaks_select_case_loop():
    """Degenerate model that always emits select_case must not deadlock.

    Real failure mode observed in Colab eval: a Qwen3.5 checkpoint
    after 300 GRPO steps emitted ``select_case`` at every state. The
    env silently no-ops the second ``select_case``, the prompt stays
    identical, the model emits the same string, score stays 0 because
    ``done`` never flips. Stall detection must force-fallback to the
    heuristic so the episode reaches grading.
    """

    import json

    select_case_payload = json.dumps(
        {"action_type": "select_case", "case_id": "CB-E1"}
    )

    result = run_episode_with_text_policy(
        "goods_not_received_easy",
        text_policy=lambda _prompt: select_case_payload,
    )
    assert result.steps_used > 0
    assert result.score > 0.0, (
        f"stall detection failed: score={result.score} "
        f"means episode never reached terminal grading"
    )
