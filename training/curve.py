"""Checkpoint evaluation + training curve utilities.

Offline helpers that take a text policy (scripted or model-driven),
evaluate it across the headline catalog, and plot a training curve
from a sequence of such evaluations. Kept dependency-light so tests
stay fast; matplotlib is imported lazily inside the plot helper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Callable, Iterable, Sequence

try:
    from ..scenarios.simulation import TaskScenario, list_tasks
    from .reward_adapter import run_episode_with_text_policy
except ImportError:  # pragma: no cover
    from scenarios.simulation import TaskScenario, list_tasks
    from training.reward_adapter import run_episode_with_text_policy


TextPolicyFn = Callable[[str], str]


@dataclass(frozen=True)
class TaskOutcome:
    """One task's eval result."""

    task_id: str
    score: float
    steps_used: int
    invalid_actions: int


@dataclass(frozen=True)
class CheckpointEval:
    """Mean + per-task scores for one checkpoint."""

    step: int
    mean_score: float
    tasks: tuple[TaskOutcome, ...] = field(default_factory=tuple)


def evaluate_policy_across_tasks(
    policy: TextPolicyFn,
    tasks: Iterable[TaskScenario] | None = None,
) -> tuple[TaskOutcome, ...]:
    """Evaluate ``policy`` across ``tasks`` (default: headline catalog)."""

    task_list = list(tasks) if tasks is not None else list_tasks()
    outcomes: list[TaskOutcome] = []
    for task in task_list:
        result = run_episode_with_text_policy(task.task_id, policy)
        outcomes.append(
            TaskOutcome(
                task_id=task.task_id,
                score=result.score,
                steps_used=result.steps_used,
                invalid_actions=result.invalid_actions,
            )
        )
    return tuple(outcomes)


def evaluate_checkpoint(
    step: int,
    policy: TextPolicyFn,
    tasks: Iterable[TaskScenario] | None = None,
) -> CheckpointEval:
    """Evaluate one checkpoint and wrap into a :class:`CheckpointEval`."""

    outcomes = evaluate_policy_across_tasks(policy, tasks=tasks)
    scores = [outcome.score for outcome in outcomes]
    return CheckpointEval(
        step=step,
        mean_score=round(mean(scores), 4) if scores else 0.0,
        tasks=outcomes,
    )


def plot_training_curve(
    checkpoints: Sequence[CheckpointEval],
    output_path: str,
    *,
    baseline_scores: dict[str, float] | None = None,
    title: str = "Merchant agent training curve",
) -> str:
    """Render a mean-score-vs-step line plot and return the output path.

    ``baseline_scores`` draws a dashed horizontal line per scripted
    baseline (e.g. ``{"heuristic": 0.77, "naive": 0.0}``) so the
    trained curve is visually grounded against the benchmark floor.
    """

    if not checkpoints:
        raise ValueError("at least one checkpoint eval is required")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [ckpt.step for ckpt in checkpoints]
    means = [ckpt.mean_score for ckpt in checkpoints]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(steps, means, marker="o", linewidth=2, label="trained")
    if baseline_scores:
        for name, value in baseline_scores.items():
            ax.axhline(value, linestyle="--", alpha=0.6, label=name)
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("Mean normalised score (headline catalog)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


__all__ = [
    "CheckpointEval",
    "TaskOutcome",
    "TextPolicyFn",
    "evaluate_checkpoint",
    "evaluate_policy_across_tasks",
    "plot_training_curve",
]
