"""Gradio demo UI for ChargebackOps."""

from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import gradio as gr

try:
    from ..core.models import ChargebackOpsAction
    from ..runners.baseline_runner import _heuristic_pick, _obvious_next_action, candidate_actions
    from ..scenarios.simulation import list_tasks
    from .chargeback_ops_environment import ChargebackOpsEnvironment
except ImportError:  # pragma: no cover
    from core.models import ChargebackOpsAction
    from runners.baseline_runner import _heuristic_pick, _obvious_next_action, candidate_actions
    from scenarios.simulation import list_tasks
    from server.chargeback_ops_environment import ChargebackOpsEnvironment


def _resolve_demo_task(task_id: str, generated: bool, difficulty: str, seed: int) -> str:
    if generated:
        return f"generated_{difficulty}_s{seed}"
    return task_id


def _step_log_row(
    step_number: int,
    action: ChargebackOpsAction,
    observation,
) -> list[Any]:
    return [
        step_number,
        action.action_type,
        action.case_id or observation.selected_case_id or "",
        action.system_name or "",
        action.strategy or "",
        round(observation.reward or 0.0, 4),
        observation.last_action_result,
    ]


def _snapshot_rows(rows: list[list[Any]]) -> list[list[Any]]:
    return [row[:] for row in rows]


def run_demo_episode(task_id: str, generated: bool, difficulty: str, seed: int):
    resolved_task_id = _resolve_demo_task(task_id, generated, difficulty, int(seed))
    env = ChargebackOpsEnvironment()
    observation = env.reset(task_id=resolved_task_id, difficulty=difficulty, seed=int(seed))
    rows: list[list[Any]] = []

    intro = (
        f"Running `{observation.task_title}` (`{observation.task_id}`) "
        f"with {len(observation.queue)} case(s)."
    )
    yield intro, _snapshot_rows(rows), observation.model_dump(), None

    step_number = 0
    while not observation.done:
        payload = observation.model_dump()
        candidates = candidate_actions(payload)
        if not candidates:
            break
        candidate = _obvious_next_action(payload, candidates) or _heuristic_pick(candidates)
        step_number += 1
        observation = env.step(candidate.action)
        rows.append(_step_log_row(step_number, candidate.action, observation))
        status = (
            f"Step {step_number}: `{candidate.action.action_type}` -> "
            f"reward {round(observation.reward or 0.0, 4)}"
        )
        grader_payload = (
            observation.grader_report.model_dump() if observation.grader_report is not None else None
        )
        yield status, _snapshot_rows(rows), observation.model_dump(), grader_payload

    final_report = observation.grader_report.model_dump() if observation.grader_report else None
    final_text = (
        f"Completed `{observation.task_id}` in "
        f"{len(rows)} step(s). Final score: "
        f"{observation.grader_report.normalized_score if observation.grader_report else 'n/a'}"
    )
    yield final_text, _snapshot_rows(rows), env.state.model_dump(), final_report


def build_demo() -> gr.Blocks:
    task_choices = [task.task_id for task in list_tasks()]
    default_task = task_choices[0] if task_choices else "goods_not_received_easy"

    with gr.Blocks(title="ChargebackOps Demo") as demo:
        gr.Markdown(
            """
            # ChargebackOps Demo
            Run a full episode and watch the benchmark agent work through disputes step by step.
            """
        )
        with gr.Tabs():
            with gr.Tab("Run Episode"):
                with gr.Row():
                    task_id = gr.Dropdown(label="Built-in task", choices=task_choices, value=default_task)
                    generated = gr.Checkbox(label="Use generated task", value=False)
                    difficulty = gr.Radio(
                        label="Generated difficulty",
                        choices=["easy", "medium", "hard"],
                        value="easy",
                    )
                    seed = gr.Number(label="Seed", value=42, precision=0)
                run_button = gr.Button("Run Episode", variant="primary")
                status = gr.Markdown("Ready.")
                step_table = gr.Dataframe(
                    headers=["step", "action", "case_id", "system", "strategy", "reward", "result"],
                    datatype=["number", "str", "str", "str", "str", "number", "str"],
                    interactive=False,
                    row_count=8,
                    wrap=True,
                    label="Step Trace",
                )
                observation_json = gr.JSON(label="Current Observation / State")
                grader_json = gr.JSON(label="Grader Report")
                run_button.click(
                    fn=run_demo_episode,
                    inputs=[task_id, generated, difficulty, seed],
                    outputs=[status, step_table, observation_json, grader_json],
                )
            with gr.Tab("Task Catalog"):
                gr.JSON(
                    value=[
                        {
                            "task_id": task.task_id,
                            "title": task.title,
                            "difficulty": task.difficulty,
                            "objective": task.objective,
                            "max_steps": task.max_steps,
                            "case_count": len(task.cases),
                        }
                        for task in list_tasks()
                    ],
                    label="Built-in Tasks",
                )
    return demo
