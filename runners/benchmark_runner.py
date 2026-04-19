"""Scripted-policy benchmark runner for ChargebackOps.

Drives a fixed set of non-learning policies through the full environment so
the trained-merchant vs. baseline discrimination delta can be measured
without calling an LLM provider. Every policy returned here is deterministic
and offline.

Policies
--------
* ``heuristic`` — the Round 1 first-candidate pick (best scripted baseline).
* ``concede_all`` — always set strategy to ``accept_chargeback`` and resolve.
* ``escalate_all`` — contest like the heuristic, then escalate in the
  pre-arb and arbitration steps regardless of evidence strength.
* ``naive`` — submit an empty packet / take a minimal path to terminal.

The runner also exposes :func:`run_multi_seed` which sweeps each policy
over the headline catalog plus extra generator seeds so the benchmark
table in ``docs/RESULTS_V2.md`` is reproducible from one command.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Any, Callable, Iterable, Sequence

try:
    from ..core.models import ChargebackOpsAction
    from ..scenarios.simulation import TaskScenario, get_task, list_tasks
    from ..server.chargeback_ops_environment import ChargebackOpsEnvironment
    from .baseline_runner import candidate_actions
except ImportError:  # pragma: no cover
    from core.models import ChargebackOpsAction
    from scenarios.simulation import TaskScenario, get_task, list_tasks
    from server.chargeback_ops_environment import ChargebackOpsEnvironment
    from runners.baseline_runner import candidate_actions


PolicyFn = Callable[[dict[str, Any]], ChargebackOpsAction | None]


POLICY_NAMES: tuple[str, ...] = (
    "heuristic",
    "escalate_all",
    "concede_all",
    "naive",
)


# ---------------------------------------------------------------------------
# Scripted policies
# ---------------------------------------------------------------------------


def heuristic_policy(observation: dict[str, Any]) -> ChargebackOpsAction | None:
    """First-candidate pick from the existing candidate generator."""

    candidates = candidate_actions(observation)
    if not candidates:
        return None
    return candidates[0].action


def escalate_all_policy(observation: dict[str, Any]) -> ChargebackOpsAction | None:
    """Play like the heuristic, but always push terminal disputes into arbitration."""

    available = set(observation.get("available_actions", []))
    visible_case = observation.get("visible_case")
    if visible_case is not None and "escalate_to_arbitration" in available:
        return ChargebackOpsAction(
            action_type="escalate_to_arbitration",
            case_id=visible_case["case_id"],
        )
    return heuristic_policy(observation)


def concede_all_policy(observation: dict[str, Any]) -> ChargebackOpsAction | None:
    """Always accept the chargeback. Never contests, never escalates."""

    available = set(observation.get("available_actions", []))
    visible_case = observation.get("visible_case")
    queue = observation.get("queue", [])

    if visible_case is None:
        open_cases = [item for item in queue if item["status"] == "open"]
        if not open_cases:
            return None
        target = sorted(
            open_cases,
            key=lambda item: (item["steps_until_deadline"], -item["amount"]),
        )[0]
        return ChargebackOpsAction(
            action_type="select_case", case_id=target["case_id"]
        )

    case_id = visible_case["case_id"]
    if visible_case["status"] != "open":
        open_cases = [
            item
            for item in queue
            if item["status"] == "open" and item["case_id"] != case_id
        ]
        if not open_cases:
            return None
        target = sorted(
            open_cases,
            key=lambda item: (item["steps_until_deadline"], -item["amount"]),
        )[0]
        return ChargebackOpsAction(
            action_type="select_case", case_id=target["case_id"]
        )

    if "accept_arbitration_loss" in available:
        return ChargebackOpsAction(
            action_type="accept_arbitration_loss", case_id=case_id
        )

    if visible_case.get("current_strategy") != "accept_chargeback" and (
        "set_strategy" in available
    ):
        return ChargebackOpsAction(
            action_type="set_strategy",
            case_id=case_id,
            strategy="accept_chargeback",
        )

    if "resolve_case" in available:
        return ChargebackOpsAction(
            action_type="resolve_case",
            case_id=case_id,
            strategy="accept_chargeback",
        )

    return heuristic_policy(observation)


def naive_policy(observation: dict[str, Any]) -> ChargebackOpsAction | None:
    """Minimum-effort agent: select a case, submit without evidence or policy work."""

    available = set(observation.get("available_actions", []))
    visible_case = observation.get("visible_case")
    queue = observation.get("queue", [])

    if visible_case is None:
        open_cases = [item for item in queue if item["status"] == "open"]
        if not open_cases:
            return None
        return ChargebackOpsAction(
            action_type="select_case", case_id=open_cases[0]["case_id"]
        )

    case_id = visible_case["case_id"]
    if visible_case["status"] != "open":
        open_cases = [
            item
            for item in queue
            if item["status"] == "open" and item["case_id"] != case_id
        ]
        if not open_cases:
            return None
        return ChargebackOpsAction(
            action_type="select_case", case_id=open_cases[0]["case_id"]
        )

    if "accept_arbitration_loss" in available:
        return ChargebackOpsAction(
            action_type="accept_arbitration_loss", case_id=case_id
        )

    if "submit_representment" in available:
        return ChargebackOpsAction(
            action_type="submit_representment", case_id=case_id
        )

    if "respond_to_pre_arb" in available:
        return ChargebackOpsAction(
            action_type="respond_to_pre_arb", case_id=case_id
        )

    if "resolve_case" in available:
        return ChargebackOpsAction(
            action_type="resolve_case",
            case_id=case_id,
            strategy="accept_chargeback",
        )

    return heuristic_policy(observation)


POLICY_REGISTRY: dict[str, PolicyFn] = {
    "heuristic": heuristic_policy,
    "escalate_all": escalate_all_policy,
    "concede_all": concede_all_policy,
    "naive": naive_policy,
}


# ---------------------------------------------------------------------------
# Episode / sweep driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskScore:
    """One policy × task result."""

    policy: str
    task_id: str
    score: float
    steps_used: int


@dataclass(frozen=True)
class PolicySummary:
    """Aggregate of one policy across a task list."""

    policy: str
    mean_score: float
    stdev: float
    tasks: tuple[TaskScore, ...]


@dataclass(frozen=True)
class BenchmarkResult:
    """Output of a full policy sweep."""

    policies: tuple[PolicySummary, ...]
    discrimination_delta: float  # heuristic minus naive

    def to_dict(self) -> dict[str, Any]:
        return {
            "discrimination_delta": self.discrimination_delta,
            "policies": [
                {
                    "policy": summary.policy,
                    "mean_score": summary.mean_score,
                    "stdev": summary.stdev,
                    "tasks": [
                        {
                            "task_id": task.task_id,
                            "score": task.score,
                            "steps_used": task.steps_used,
                        }
                        for task in summary.tasks
                    ],
                }
                for summary in self.policies
            ],
        }


def run_policy_on_task(policy: PolicyFn, task: TaskScenario) -> TaskScore:
    """Drive one policy through one task. Fully offline, no LLM calls."""

    env = ChargebackOpsEnvironment()
    observation = env.reset(task_id=task.task_id)
    max_steps = task.max_steps + 5  # small safety margin
    steps = 0
    while not observation.done and steps < max_steps:
        action = policy(observation.model_dump())
        if action is None:
            break
        observation = env.step(action)
        steps += 1

    report = env.state.grader_report
    score = float(report.normalized_score) if report is not None else 0.0
    return TaskScore(
        policy=policy.__name__,
        task_id=task.task_id,
        score=score,
        steps_used=env.state.step_count,
    )


def run_policy_sweep(
    policy_names: Sequence[str] = POLICY_NAMES,
    tasks: Iterable[TaskScenario] | None = None,
) -> BenchmarkResult:
    """Run each named policy across the headline catalog (or provided tasks)."""

    task_list = list(tasks) if tasks is not None else list_tasks()

    summaries: list[PolicySummary] = []
    for name in policy_names:
        if name not in POLICY_REGISTRY:
            raise KeyError(f"Unknown policy '{name}'. Known: {sorted(POLICY_REGISTRY)}")
        policy = POLICY_REGISTRY[name]
        task_scores: list[TaskScore] = []
        for task in task_list:
            score = run_policy_on_task(policy, task)
            task_scores.append(
                TaskScore(
                    policy=name,
                    task_id=score.task_id,
                    score=score.score,
                    steps_used=score.steps_used,
                )
            )
        scores = [item.score for item in task_scores]
        summaries.append(
            PolicySummary(
                policy=name,
                mean_score=round(mean(scores), 4) if scores else 0.0,
                stdev=round(pstdev(scores), 4) if len(scores) > 1 else 0.0,
                tasks=tuple(task_scores),
            )
        )

    by_name = {summary.policy: summary for summary in summaries}
    delta = 0.0
    if "heuristic" in by_name and "naive" in by_name:
        delta = round(
            by_name["heuristic"].mean_score - by_name["naive"].mean_score, 4
        )
    return BenchmarkResult(policies=tuple(summaries), discrimination_delta=delta)


def run_multi_seed(
    seeds: Sequence[int],
    difficulties: Sequence[str] = ("easy", "medium", "hard", "nightmare"),
    policy_names: Sequence[str] = POLICY_NAMES,
) -> BenchmarkResult:
    """Sweep each policy over ``seeds × difficulties`` generated tasks.

    Used for the multi-seed grid cited in the PRD's Day-5 exit criteria.
    """

    tasks: list[TaskScenario] = []
    for difficulty in difficulties:
        for seed in seeds:
            task_id = f"generated_{difficulty}_s{seed}"
            tasks.append(get_task(task_id))
    return run_policy_sweep(policy_names, tasks=tasks)


__all__ = [
    "POLICY_NAMES",
    "POLICY_REGISTRY",
    "PolicyFn",
    "BenchmarkResult",
    "PolicySummary",
    "TaskScore",
    "heuristic_policy",
    "escalate_all_policy",
    "concede_all_policy",
    "naive_policy",
    "run_policy_on_task",
    "run_policy_sweep",
    "run_multi_seed",
]
