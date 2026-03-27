from grading import grade_episode
from server.chargeback_ops_environment import ChargebackOpsEnvironment
from simulation import get_task


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
