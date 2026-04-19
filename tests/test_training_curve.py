"""Unit tests for the checkpoint eval + training curve plot helpers."""

from __future__ import annotations

import os
import tempfile

import pytest

from training.curve import (
    CheckpointEval,
    TaskOutcome,
    evaluate_checkpoint,
    evaluate_policy_across_tasks,
    plot_training_curve,
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
