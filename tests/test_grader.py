from evaluation.grading import grade_episode
from evaluation.rubrics import (
    CASE_DIMENSION_WEIGHTS,
    ChargebackOpsEpisodeRubric,
)
from server.chargeback_ops_environment import ChargebackOpsEnvironment
from scenarios.simulation import get_task


def test_grade_episode_bounds():
    env = ChargebackOpsEnvironment()
    env.reset(task_id="queue_optimization_hard")
    report = grade_episode(
        get_task("queue_optimization_hard"),
        env._progress_by_case,  # type: ignore[attr-defined]
        env.state.step_count,
        env.state.episode_id or "",
        completed=False,
    )
    assert 0.0 <= report.normalized_score <= 1.0


def test_environment_exposes_rubric_tree():
    """The env must wire an OpenEnv Rubric that exposes all 7 scoring dimensions."""

    env = ChargebackOpsEnvironment()
    assert isinstance(env.rubric, ChargebackOpsEpisodeRubric)

    names = {name for name, _ in env.rubric.named_rubrics()}
    expected = {
        "case_rubric",
        "case_rubric.aggregator",
        *(f"case_rubric.aggregator.rubric_{i}" for i in range(8)),
    }
    assert expected.issubset(names)

    # Weights must sum to 1.0 (WeightedSum enforces this at construction but
    # we lock the constant here so weight changes stay intentional).
    assert abs(sum(CASE_DIMENSION_WEIGHTS) - 1.0) < 1e-6
    assert len(CASE_DIMENSION_WEIGHTS) == 8
