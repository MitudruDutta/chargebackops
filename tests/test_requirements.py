from __future__ import annotations

from pathlib import Path

from baseline_runner import _heuristic_pick, _obvious_next_action, candidate_actions
from grading import grade_episode
from models import ChargebackOpsAction
from server.chargeback_ops_environment import ChargebackOpsEnvironment
from simulation import get_task, list_tasks


def _run_heuristic_episode(task_id: str) -> tuple[float, float]:
    env = ChargebackOpsEnvironment()
    observation = env.reset(task_id=task_id)
    total_reward = 0.0
    while not observation.done:
        candidates = candidate_actions(observation.model_dump())
        assert candidates, f"No candidate actions available for task {task_id}"
        observation = env.step(_heuristic_pick(candidates).action)
        total_reward += observation.reward or 0.0
    assert observation.grader_report is not None
    return round(total_reward, 4), observation.grader_report.normalized_score


def _run_bad_episode(task_id: str) -> tuple[float, float]:
    env = ChargebackOpsEnvironment()
    observation = env.reset(task_id=task_id)
    total_reward = 0.0

    while not observation.done:
        if observation.selected_case_id is None:
            open_case = next(case for case in observation.queue if case.status == "open")
            action = ChargebackOpsAction(action_type="select_case", case_id=open_case.case_id)
        else:
            case_id = observation.selected_case_id
            visible_case = observation.visible_case
            if visible_case and visible_case.current_strategy is None:
                action = ChargebackOpsAction(
                    action_type="set_strategy",
                    case_id=case_id,
                    strategy="accept_chargeback",
                )
            elif visible_case and visible_case.current_strategy == "accept_chargeback":
                action = ChargebackOpsAction(
                    action_type="resolve_case",
                    case_id=case_id,
                    strategy="accept_chargeback",
                )
            else:
                action = ChargebackOpsAction(
                    action_type="query_system",
                    case_id=case_id,
                    system_name="payment",
                )
        observation = env.step(action)
        total_reward += observation.reward or 0.0

    assert observation.grader_report is not None
    return round(total_reward, 4), observation.grader_report.normalized_score


def test_problem_statement_task_catalog():
    tasks = list_tasks()
    assert len(tasks) >= 3
    assert {task.difficulty for task in tasks} == {"easy", "medium", "hard"}


def test_problem_statement_reset_and_state_cleanliness():
    env = ChargebackOpsEnvironment()
    first = env.reset(task_id="goods_not_received_easy")
    first_episode = env.state.episode_id
    env.step(ChargebackOpsAction(action_type="select_case", case_id="CB-E1"))
    second = env.reset(task_id="fraud_signal_ambiguity")

    assert first.done is False
    assert second.task_id == "fraud_signal_ambiguity"
    assert env.state.task_id == "fraud_signal_ambiguity"
    assert env.state.step_count == 0
    assert env.state.action_history == []
    assert env.state.selected_case_id is None
    assert env.state.episode_id != first_episode


def test_problem_statement_grader_is_deterministic():
    env = ChargebackOpsEnvironment()
    env.reset(task_id="queue_optimization_hard")
    task = get_task("queue_optimization_hard")

    report_a = grade_episode(
        task,
        env._progress_by_case,  # type: ignore[attr-defined]
        env.state.step_count,
        env.state.episode_id or "",
        completed=False,
    )
    report_b = grade_episode(
        task,
        env._progress_by_case,  # type: ignore[attr-defined]
        env.state.step_count,
        env.state.episode_id or "",
        completed=False,
    )
    assert report_a.model_dump() == report_b.model_dump()
    assert 0.0 <= report_a.normalized_score <= 1.0


def test_problem_statement_reward_signal_has_partial_progress_and_penalties():
    env = ChargebackOpsEnvironment()
    env.reset(task_id="fraud_signal_ambiguity")
    env.step(ChargebackOpsAction(action_type="select_case", case_id="CB-M1"))
    helpful = env.step(
        ChargebackOpsAction(
            action_type="query_system",
            case_id="CB-M1",
            system_name="orders",
        )
    )
    duplicate = env.step(
        ChargebackOpsAction(
            action_type="query_system",
            case_id="CB-M1",
            system_name="orders",
        )
    )
    harmful = env.step(
        ChargebackOpsAction(
            action_type="add_evidence",
            case_id="CB-M1",
            evidence_ids=["M1-AVS-MISMATCH"],
        )
    )

    assert (helpful.reward or 0.0) > 0
    assert (duplicate.reward or 0.0) < 0
    assert (harmful.reward or 0.0) < 0


def test_problem_statement_agent_signal_distinguishes_good_from_bad():
    heuristic_reward, heuristic_score = _run_heuristic_episode("queue_optimization_hard")
    bad_reward, bad_score = _run_bad_episode("queue_optimization_hard")

    assert heuristic_score > bad_score
    assert heuristic_reward > bad_reward


def test_problem_statement_live_agent_budget_targets_real_branches():
    ambiguous_states = 0

    for task in list_tasks():
        env = ChargebackOpsEnvironment()
        observation = env.reset(task_id=task.task_id)
        while not observation.done:
            payload = observation.model_dump()
            candidates = candidate_actions(payload)
            assert candidates
            if len(candidates) > 1 and _obvious_next_action(payload, candidates) is None:
                ambiguous_states += 1
            candidate = _obvious_next_action(payload, candidates) or _heuristic_pick(candidates)
            observation = env.step(candidate.action)

    assert ambiguous_states <= 6


def test_problem_statement_inference_contract_exists():
    content = Path("inference.py").read_text()
    assert "from openai import OpenAI" in content
    assert "API_BASE_URL" in content
    assert "MODEL_NAME" in content
    assert "HF_TOKEN" in content
