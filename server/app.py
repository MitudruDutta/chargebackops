"""FastAPI application for ChargebackOps."""

from __future__ import annotations

from contextlib import suppress

from fastapi import HTTPException
from fastapi.responses import JSONResponse

try:
    from openenv.core.env_server.http_server import create_app
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "openenv-core is required to run ChargebackOps. Install project dependencies first."
    ) from exc

try:
    from ..runners.baseline_runner import run_baseline
    from ..core.episode_store import get_report, list_reports
    from ..runners.inference import run_inference
    from ..core.models import (
        BaselineRunResult,
        ChargebackOpsAction,
        ChargebackOpsObservation,
        TasksResponse,
        TaskSummary,
    )
    from ..scenarios.simulation import list_tasks
    from .chargeback_ops_environment import ChargebackOpsEnvironment
    from .demo_ui import build_demo
except ImportError:  # pragma: no cover
    from runners.baseline_runner import run_baseline
    from core.episode_store import get_report, list_reports
    from runners.inference import run_inference
    from core.models import (
        BaselineRunResult,
        ChargebackOpsAction,
        ChargebackOpsObservation,
        TasksResponse,
        TaskSummary,
    )
    from scenarios.simulation import list_tasks
    from server.chargeback_ops_environment import ChargebackOpsEnvironment
    from server.demo_ui import build_demo


app = create_app(
    ChargebackOpsEnvironment,
    ChargebackOpsAction,
    ChargebackOpsObservation,
    env_name="chargeback_ops",
    max_concurrent_envs=8,
)

with suppress(Exception):
    import gradio as gr

    app = gr.mount_gradio_app(app, build_demo(), path="/demo")


@app.get("/")
def root() -> JSONResponse:
    """Return a lightweight root response for HF Space and validator pings."""

    return JSONResponse(
        {
            "name": "ChargebackOps",
            "status": "ok",
            "docs_url": "/docs",
            "health_url": "/health",
            "tasks_url": "/tasks",
            "demo_url": "/demo",
        }
    )


@app.get("/tasks", response_model=TasksResponse)
def tasks() -> TasksResponse:
    """List built-in tasks and the action schema."""

    return TasksResponse(
        tasks=[
            TaskSummary(
                task_id=task.task_id,
                title=task.title,
                difficulty=task.difficulty,
                objective=task.objective,
                description=task.description,
                max_steps=task.max_steps,
                case_count=len(task.cases),
            )
            for task in list_tasks()
        ],
        action_schema=ChargebackOpsAction.model_json_schema(),
    )


@app.get("/generate")
def generate_tasks(
    seed: int = 42,
    easy: int = 2,
    medium: int = 2,
    hard: int = 2,
) -> list[dict]:
    """Generate parametric tasks from a seed for infinite scenario variety."""

    try:
        from scenarios.case_generator import generate_task_suite
    except ImportError:  # pragma: no cover
        from ..scenarios.case_generator import generate_task_suite

    suite = generate_task_suite(
        base_seed=seed, easy_count=easy, medium_count=medium, hard_count=hard,
    )
    return [
        {
            "task_id": t.task_id,
            "title": t.title,
            "difficulty": t.difficulty,
            "objective": t.objective,
            "case_count": len(t.cases),
            "max_steps": t.max_steps,
        }
        for t in suite
    ]


@app.get("/grader")
@app.post("/grader")
def grader(episode_id: str | None = None):
    """Return a stored grade for a completed episode."""

    report = get_report(episode_id)
    if report is None:
        raise HTTPException(
            status_code=404,
            detail="No completed episode report found. Finish an episode first or provide a valid episode_id.",
        )
    return report.model_dump()


@app.get("/baseline", response_model=BaselineRunResult)
@app.post("/baseline", response_model=BaselineRunResult)
def baseline(
    provider: str | None = None,
    model_name: str | None = None,
) -> BaselineRunResult:
    """Run the baseline inference policy across all tasks."""

    if provider is None and model_name is None:
        return run_inference()
    return run_baseline(provider=provider, model_name=model_name)


@app.get("/results")
def results():
    """Return all completed episode reports for inspection and replay."""

    reports = list_reports()
    return [report.model_dump() for report in reports]


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Local entry point for uvicorn."""

    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    main()
