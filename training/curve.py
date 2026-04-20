"""Checkpoint evaluation + training curve utilities.

Offline helpers that take a text policy (scripted or model-driven),
evaluate it across the headline catalog, and plot a training curve
from a sequence of such evaluations. Kept dependency-light so tests
stay fast; matplotlib is imported lazily inside the plot helper.

Multi-task RL: ``evaluate_checkpoint_by_family`` groups task outcomes
by a chosen key (default: difficulty) so a single training run yields
one curve per family. ``plot_training_curve_by_family`` renders the
multi-line variant. Use this to verify the policy improves *uniformly*
rather than only on cheap easy tasks.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean
from typing import Callable, Iterable, Mapping, Sequence

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


FamilyKeyFn = Callable[[TaskScenario], str]


def _default_family_key(task: TaskScenario) -> str:
    return task.difficulty


@dataclass(frozen=True)
class GroupedCheckpointEval:
    """Per-family checkpoint scores from a single evaluation pass."""

    step: int
    by_family: Mapping[str, CheckpointEval]

    @property
    def overall_mean(self) -> float:
        all_scores: list[float] = []
        for fam_eval in self.by_family.values():
            all_scores.extend(o.score for o in fam_eval.tasks)
        return round(mean(all_scores), 4) if all_scores else 0.0


def evaluate_checkpoint_by_family(
    step: int,
    policy: TextPolicyFn,
    *,
    tasks: Iterable[TaskScenario] | None = None,
    family_key: FamilyKeyFn = _default_family_key,
) -> GroupedCheckpointEval:
    """Evaluate ``policy`` once and bucket per-task outcomes by family.

    One env pass per task — same cost as :func:`evaluate_checkpoint` —
    but yields one :class:`CheckpointEval` per family for per-cohort
    curves. Default ``family_key`` groups by difficulty.
    """

    task_list = list(tasks) if tasks is not None else list_tasks()
    outcomes = evaluate_policy_across_tasks(policy, tasks=task_list)
    family_of = {task.task_id: family_key(task) for task in task_list}

    bucketed: dict[str, list[TaskOutcome]] = defaultdict(list)
    for outcome in outcomes:
        bucketed[family_of[outcome.task_id]].append(outcome)

    by_family: dict[str, CheckpointEval] = {}
    for family, fam_outcomes in bucketed.items():
        scores = [o.score for o in fam_outcomes]
        by_family[family] = CheckpointEval(
            step=step,
            mean_score=round(mean(scores), 4) if scores else 0.0,
            tasks=tuple(fam_outcomes),
        )
    return GroupedCheckpointEval(step=step, by_family=by_family)


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


def plot_training_curve_by_family(
    grouped: Sequence[GroupedCheckpointEval],
    output_path: str,
    *,
    baseline_scores: Mapping[str, Mapping[str, float]] | None = None,
    title: str = "Merchant agent — per-family training curves",
    family_order: Sequence[str] | None = None,
) -> str:
    """Render one line per family + an ``overall`` line.

    ``baseline_scores`` maps ``family → {policy_name: score}`` and
    draws a dashed horizontal line per (family, baseline) pair tinted
    to match the family's solid line. Use this to ground each curve
    against its scripted floor.
    """

    if not grouped:
        raise ValueError("at least one grouped eval is required")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = list(family_order) if family_order is not None else sorted(
        {fam for g in grouped for fam in g.by_family}
    )
    steps = [g.step for g in grouped]

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    family_color: dict[str, str] = {}
    for idx, family in enumerate(families):
        color = color_cycle[idx % len(color_cycle)]
        family_color[family] = color
        ys = [g.by_family[family].mean_score if family in g.by_family else float("nan") for g in grouped]
        ax.plot(steps, ys, marker="o", linewidth=2, label=family, color=color)
        if baseline_scores and family in baseline_scores:
            for baseline_name, value in baseline_scores[family].items():
                ax.axhline(
                    value,
                    linestyle="--",
                    alpha=0.4,
                    color=color,
                    label=f"{family}/{baseline_name}",
                )

    overall = [g.overall_mean for g in grouped]
    ax.plot(steps, overall, marker="s", linewidth=2.5, color="black", label="overall")

    ax.set_xlabel("GRPO step")
    ax.set_ylabel("Mean normalised score")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


__all__ = [
    "CheckpointEval",
    "FamilyKeyFn",
    "GroupedCheckpointEval",
    "TaskOutcome",
    "TextPolicyFn",
    "evaluate_checkpoint",
    "evaluate_checkpoint_by_family",
    "evaluate_policy_across_tasks",
    "plot_training_curve",
    "plot_training_curve_by_family",
]
