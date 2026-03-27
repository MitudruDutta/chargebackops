"""Requirement-focused audit for the ChargebackOps submission."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline_runner import _heuristic_pick, candidate_actions
from grading import grade_episode
from inference import run_inference
from models import ChargebackOpsAction
from server.app import baseline, tasks
from server.chargeback_ops_environment import ChargebackOpsEnvironment
from simulation import get_task, list_tasks


def _run_heuristic_episode(task_id: str) -> dict[str, float]:
    env = ChargebackOpsEnvironment()
    observation = env.reset(task_id=task_id)
    total_reward = 0.0
    while not observation.done:
        candidates = candidate_actions(observation.model_dump())
        observation = env.step(_heuristic_pick(candidates).action)
        total_reward += observation.reward or 0.0
    assert observation.grader_report is not None
    return {
        "reward": round(total_reward, 4),
        "score": observation.grader_report.normalized_score,
    }


def _run_bad_episode(task_id: str) -> dict[str, float]:
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
    return {
        "reward": round(total_reward, 4),
        "score": observation.grader_report.normalized_score,
    }


def _check(condition: bool, message: str, details: object | None = None) -> dict[str, object]:
    return {
        "pass": condition,
        "message": message,
        "details": details,
    }


@contextmanager
def _deterministic_provider_disabled():
    keys = [
        "HF_TOKEN",
        "API_BASE_URL",
        "MODEL_NAME",
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GROQ_API_KEY",
        "STRICT_LLM_MODE",
    ]
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ.pop(key, None)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main() -> None:
    tasks_payload = tasks()
    task_list = list_tasks()

    openenv_cli = shutil.which("openenv")
    openenv_validate = subprocess.run(
        [openenv_cli or "openenv", "validate", "."],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    files = {
        "Dockerfile": (PROJECT_ROOT / "Dockerfile").exists(),
        "README.md": (PROJECT_ROOT / "README.md").exists(),
        "openenv.yaml": (PROJECT_ROOT / "openenv.yaml").exists(),
        "inference.py": (PROJECT_ROOT / "inference.py").exists(),
    }

    heuristic_hard = _run_heuristic_episode("queue_optimization_hard")
    bad_hard = _run_bad_episode("queue_optimization_hard")

    env = ChargebackOpsEnvironment()
    reset_obs = env.reset(task_id="goods_not_received_easy")
    initial_episode = env.state.episode_id
    env.step(ChargebackOpsAction(action_type="select_case", case_id="CB-E1"))
    reset_obs_2 = env.reset(task_id="fraud_signal_ambiguity")

    env_reward = ChargebackOpsEnvironment()
    env_reward.reset(task_id="fraud_signal_ambiguity")
    env_reward.step(ChargebackOpsAction(action_type="select_case", case_id="CB-M1"))
    helpful = env_reward.step(
        ChargebackOpsAction(action_type="query_system", case_id="CB-M1", system_name="orders")
    )
    duplicate = env_reward.step(
        ChargebackOpsAction(action_type="query_system", case_id="CB-M1", system_name="orders")
    )
    harmful = env_reward.step(
        ChargebackOpsAction(
            action_type="add_evidence",
            case_id="CB-M1",
            evidence_ids=["M1-AVS-MISMATCH"],
        )
    )

    task = get_task("queue_optimization_hard")
    env_grader = ChargebackOpsEnvironment()
    env_grader.reset(task_id="queue_optimization_hard")
    grader_a = grade_episode(
        task,
        env_grader._progress_by_case,  # type: ignore[attr-defined]
        env_grader.state.step_count,
        env_grader.state.episode_id or "",
        completed=False,
    )
    grader_b = grade_episode(
        task,
        env_grader._progress_by_case,  # type: ignore[attr-defined]
        env_grader.state.step_count,
        env_grader.state.episode_id or "",
        completed=False,
    )

    with _deterministic_provider_disabled():
        baseline_payload = baseline()
        inference_payload = run_inference()
    source = (PROJECT_ROOT / "inference.py").read_text()

    report = {
        "task_catalog": _check(
            len(task_list) >= 3 and {task.difficulty for task in task_list} == {"easy", "medium", "hard"},
            "Environment exposes easy, medium, and hard tasks.",
            [task.task_id for task in task_list],
        ),
        "grader_range": _check(
            all(0.0 <= result.score <= 1.0 for result in baseline_payload.task_results),
            "Grader returns scores in [0.0, 1.0] for all baseline tasks.",
            [result.score for result in baseline_payload.task_results],
        ),
        "grader_determinism": _check(
            grader_a.model_dump() == grader_b.model_dump(),
            "Grader is deterministic on identical state.",
            {"score": grader_a.normalized_score},
        ),
        "reward_signal": _check(
            (helpful.reward or 0.0) > 0 and (duplicate.reward or 0.0) < 0 and (harmful.reward or 0.0) < 0,
            "Reward provides partial progress and penalty signals.",
            {
                "helpful_reward": helpful.reward,
                "duplicate_reward": duplicate.reward,
                "harmful_reward": harmful.reward,
            },
        ),
        "agent_separation": _check(
            heuristic_hard["score"] > bad_hard["score"] and heuristic_hard["reward"] > bad_hard["reward"],
            "A competent policy scores better than a bad control policy on the hard task.",
            {"heuristic": heuristic_hard, "bad": bad_hard},
        ),
        "reset_state": _check(
            reset_obs.done is False
            and reset_obs_2.task_id == "fraud_signal_ambiguity"
            and env.state.step_count == 0
            and env.state.action_history == []
            and env.state.episode_id != initial_episode,
            "reset() produces a clean episode state.",
            {
                "first_task": reset_obs.task_id,
                "second_task": reset_obs_2.task_id,
                "step_count": env.state.step_count,
            },
        ),
        "tasks_endpoint": _check(
            len(tasks_payload.tasks) >= 3 and "properties" in tasks_payload.action_schema,
            "/tasks exposes task metadata and a typed action schema.",
            {"task_count": len(tasks_payload.tasks)},
        ),
        "inference_contract": _check(
            all(token in source for token in ["from openai import OpenAI", "API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]),
            "inference.py uses the OpenAI client with the required environment variables.",
            None,
        ),
        "openenv_validate": _check(
            openenv_validate.returncode == 0,
            "openenv validate passes.",
            openenv_validate.stdout.strip() or openenv_validate.stderr.strip(),
        ),
        "baseline_runs": _check(
            len(baseline_payload.task_results) == 3,
            "Baseline endpoint runs across all tasks.",
            {
                "mode": baseline_payload.mode,
                "provider_calls_attempted": baseline_payload.provider_calls_attempted,
                "provider_calls_succeeded": baseline_payload.provider_calls_succeeded,
            },
        ),
        "inference_runs": _check(
            len(inference_payload.task_results) == 3,
            "inference.py runs across all tasks.",
            {
                "mode": inference_payload.mode,
                "provider_calls_attempted": inference_payload.provider_calls_attempted,
                "provider_calls_succeeded": inference_payload.provider_calls_succeeded,
            },
        ),
        "required_files": _check(
            all(files.values()),
            "Submission-critical files exist.",
            files,
        ),
    }

    report["all_passed"] = all(item["pass"] for item in report.values())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
