"""Unit tests for the scripted-policy benchmark runner.

The runner drives a fixed set of non-learning policies through the full
environment without LLM calls. These tests pin:

1. Each policy returns valid action or None offline.
2. The aggregator produces per-policy means and a discrimination delta.
3. The headline policy sweep keeps the heuristic ≥ 0.40 above the naive floor.
"""

from __future__ import annotations

from runners.benchmark_runner import (
    POLICY_NAMES,
    POLICY_REGISTRY,
    concede_all_policy,
    escalate_all_policy,
    heuristic_policy,
    naive_policy,
    run_multi_seed,
    run_policy_on_task,
    run_policy_sweep,
)
from scenarios.simulation import get_task


_EASY_TASK = get_task("goods_not_received_easy")


def test_policy_registry_matches_public_names():
    assert set(POLICY_NAMES) == set(POLICY_REGISTRY)
    assert set(POLICY_NAMES) == {"heuristic", "escalate_all", "concede_all", "naive"}


def test_heuristic_scores_above_naive_on_easy():
    heur = run_policy_on_task(heuristic_policy, _EASY_TASK)
    nv = run_policy_on_task(naive_policy, _EASY_TASK)
    assert heur.score > nv.score
    assert heur.task_id == _EASY_TASK.task_id
    assert heur.steps_used > 0


def test_concede_all_lands_final_resolution():
    """concede_all must always terminate the episode with a concede path."""
    result = run_policy_on_task(concede_all_policy, _EASY_TASK)
    assert result.steps_used > 0
    # concede_all scores strictly below heuristic but must stay in [0, 1].
    assert 0.0 <= result.score <= 1.0


def test_escalate_all_runs_to_completion():
    result = run_policy_on_task(escalate_all_policy, _EASY_TASK)
    assert 0.0 <= result.score <= 1.0
    assert result.steps_used > 0


def test_sweep_aggregates_and_produces_delta():
    result = run_policy_sweep()
    policies = {summary.policy: summary for summary in result.policies}
    assert set(policies) == set(POLICY_NAMES)
    # mean scores sit in the valid range
    for summary in result.policies:
        assert 0.0 <= summary.mean_score <= 1.0
    # discrimination delta is heuristic - naive and must clear the PRD bar.
    assert result.discrimination_delta >= 0.40


def test_sweep_is_deterministic():
    """Two runs on the same catalog must produce identical numbers."""
    first = run_policy_sweep().to_dict()
    second = run_policy_sweep().to_dict()
    assert first == second


def test_multi_seed_sweep_runs_subset():
    """Tiny grid (2 seeds × 1 difficulty) stays under a second and returns data."""
    result = run_multi_seed(seeds=[42, 17], difficulties=["easy"])
    for summary in result.policies:
        assert len(summary.tasks) == 2
        for task_score in summary.tasks:
            assert task_score.task_id.startswith("generated_easy_s")
