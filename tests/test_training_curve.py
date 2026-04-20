"""Unit tests for the checkpoint eval + training curve plot helpers."""

from __future__ import annotations

import os
import tempfile

import pytest

from training.curve import (
    CheckpointEval,
    GroupedCheckpointEval,
    TaskOutcome,
    evaluate_checkpoint,
    evaluate_checkpoint_by_family,
    evaluate_policy_across_tasks,
    plot_training_curve,
    plot_training_curve_by_family,
)


def _empty_policy(_prompt: str) -> str:
    """Text policy that always forces the scripted heuristic fallback."""
    return ""


def test_evaluate_policy_across_tasks_returns_one_outcome_per_task():
    from scenarios.simulation import list_tasks

    outcomes = evaluate_policy_across_tasks(_empty_policy)
    assert len(outcomes) == len(list_tasks())
    for outcome in outcomes:
        assert 0.0 <= outcome.score <= 1.0
        assert outcome.steps_used > 0


def test_evaluate_checkpoint_aggregates_mean():
    from scenarios.simulation import get_task

    tasks = [get_task("goods_not_received_easy")]
    ckpt = evaluate_checkpoint(step=50, policy=_empty_policy, tasks=tasks)
    assert isinstance(ckpt, CheckpointEval)
    assert ckpt.step == 50
    assert len(ckpt.tasks) == 1
    assert ckpt.mean_score == round(ckpt.tasks[0].score, 4)


def test_plot_training_curve_writes_png_file():
    checkpoints = [
        CheckpointEval(step=0, mean_score=0.42, tasks=()),
        CheckpointEval(step=100, mean_score=0.61, tasks=()),
        CheckpointEval(step=200, mean_score=0.71, tasks=()),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "curve.png")
        plot_training_curve(
            checkpoints,
            path,
            baseline_scores={"heuristic": 0.77, "naive": 0.0},
        )
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


def test_plot_training_curve_raises_on_empty_input():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "curve.png")
        with pytest.raises(ValueError):
            plot_training_curve([], path)


def test_evaluate_checkpoint_by_family_buckets_by_difficulty():
    from scenarios.simulation import list_tasks

    grouped = evaluate_checkpoint_by_family(step=42, policy=_empty_policy)
    assert isinstance(grouped, GroupedCheckpointEval)
    assert grouped.step == 42

    expected_families = {task.difficulty for task in list_tasks()}
    assert set(grouped.by_family.keys()) == expected_families

    total_tasks = sum(len(ev.tasks) for ev in grouped.by_family.values())
    assert total_tasks == len(list_tasks())

    for family, ev in grouped.by_family.items():
        assert ev.step == 42
        assert 0.0 <= ev.mean_score <= 1.0


def test_evaluate_checkpoint_by_family_custom_key():
    from scenarios.simulation import get_task

    tasks = [
        get_task("goods_not_received_easy"),
        get_task("queue_optimization_hard"),
    ]
    grouped = evaluate_checkpoint_by_family(
        step=0,
        policy=_empty_policy,
        tasks=tasks,
        family_key=lambda t: "all",
    )
    assert set(grouped.by_family.keys()) == {"all"}
    assert len(grouped.by_family["all"].tasks) == 2


def test_grouped_overall_mean_matches_flat_mean():
    grouped = evaluate_checkpoint_by_family(step=0, policy=_empty_policy)
    flat = evaluate_checkpoint(step=0, policy=_empty_policy)
    assert abs(grouped.overall_mean - flat.mean_score) < 1e-3


def test_plot_training_curve_by_family_writes_png():
    grouped = [
        GroupedCheckpointEval(
            step=0,
            by_family={
                "easy": CheckpointEval(step=0, mean_score=0.5, tasks=()),
                "hard": CheckpointEval(step=0, mean_score=0.3, tasks=()),
            },
        ),
        GroupedCheckpointEval(
            step=100,
            by_family={
                "easy": CheckpointEval(step=100, mean_score=0.7, tasks=()),
                "hard": CheckpointEval(step=100, mean_score=0.5, tasks=()),
            },
        ),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "by_family.png")
        plot_training_curve_by_family(
            grouped,
            path,
            baseline_scores={
                "easy": {"heuristic": 0.92},
                "hard": {"heuristic": 0.83},
            },
            family_order=["easy", "hard"],
        )
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


def test_plot_training_curve_by_family_raises_on_empty():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.png")
        with pytest.raises(ValueError):
            plot_training_curve_by_family([], path)
