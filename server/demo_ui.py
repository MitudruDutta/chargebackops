"""Gradio demo UI for ChargebackOps."""

from __future__ import annotations

import json
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


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
.dashboard-header { text-align: center; padding: 16px 0 8px 0; }
.dashboard-header h1 { margin: 0; font-size: 28px; }
.dashboard-header p { margin: 4px 0 0 0; color: #888; font-size: 14px; }

.score-big { text-align: center; padding: 12px 0; }
.score-big .value { font-size: 56px; font-weight: 800; line-height: 1.1; }
.score-big .label { font-size: 13px; color: #888; margin-top: 4px; }

.bar-row { display: flex; align-items: center; margin: 4px 0; font-size: 13px; }
.bar-row .bar-label { width: 80px; flex-shrink: 0; }
.bar-row .bar-track { flex: 1; background: #2a2a2a; border-radius: 4px; height: 18px; overflow: hidden; margin: 0 8px; }
.bar-row .bar-fill { height: 100%; border-radius: 4px; transition: width 0.3s; }
.bar-row .bar-value { width: 44px; text-align: right; flex-shrink: 0; }

.case-card { border: 1px solid #3a3a3a; border-radius: 8px; padding: 14px; margin: 8px 0; background: #1a1a1a; }
.case-card .case-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
.case-card .case-header .case-id { font-weight: 700; font-size: 15px; }
.case-card .case-header .case-meta { font-size: 12px; color: #999; }
.case-card .case-notes { font-size: 11px; color: #777; margin-top: 8px; }

.queue-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.queue-table th { text-align: left; padding: 8px; border-bottom: 2px solid #444; font-weight: 600; color: #ccc; }
.queue-table td { padding: 8px; border-bottom: 1px solid #2a2a2a; }
.queue-table tr:hover { background: #1e1e1e; }

.urgency-crit { color: #ef4444; font-weight: 700; }
.urgency-warn { color: #eab308; font-weight: 600; }
.urgency-ok { color: #22c55e; }

.status-open { color: #3b82f6; }
.status-done { color: #22c55e; }
.status-fail { color: #ef4444; }

.budget-section { padding: 8px 0; }
.budget-section .budget-label { display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 3px; }

.color-green { color: #22c55e; }
.color-yellow { color: #eab308; }
.color-red { color: #ef4444; }
.color-blue { color: #3b82f6; }
"""


# ---------------------------------------------------------------------------
# HTML builders
# ---------------------------------------------------------------------------

def _bar_html(label: str, value: float, color: str) -> str:
    pct = max(0, min(100, int(value * 100)))
    return (
        f'<div class="bar-row">'
        f'<span class="bar-label">{label}</span>'
        f'<div class="bar-track"><div class="bar-fill" style="width:{pct}%;background:{color};"></div></div>'
        f'<span class="bar-value">{value:.2f}</span>'
        f'</div>'
    )


def _score_color(v: float) -> str:
    if v >= 0.8:
        return "#22c55e"
    if v >= 0.4:
        return "#eab308"
    return "#ef4444"


def _queue_html(observation) -> str:
    if not observation.queue:
        return "<p style='color:#888;'>No cases.</p>"

    rows = ""
    for c in observation.queue:
        sl = c.steps_until_deadline
        if sl <= 1:
            urg_cls, urg_icon = "urgency-crit", "!!"
        elif sl <= 3:
            urg_cls, urg_icon = "urgency-warn", "!"
        else:
            urg_cls, urg_icon = "urgency-ok", ""

        if c.status == "open":
            st_cls = "status-open"
        elif c.status in ("won", "refunded", "accepted_chargeback"):
            st_cls = "status-done"
        else:
            st_cls = "status-fail"

        st_label = c.status.replace("_", " ").title()
        net = f"{c.card_network.upper()} {c.network_reason_code}"

        rows += (
            f"<tr>"
            f"<td><b>{c.case_id}</b></td>"
            f"<td>{net}</td>"
            f"<td>{c.reason_code.replace('_',' ')}</td>"
            f"<td style='text-align:right;'>${c.amount:,.2f}</td>"
            f'<td class="{urg_cls}" style="text-align:center;">{urg_icon} {sl}</td>'
            f'<td class="{st_cls}" style="text-align:center;">{st_label}</td>'
            f"</tr>"
        )

    return (
        f'<table class="queue-table">'
        f"<tr><th>Case</th><th>Network</th><th>Reason</th>"
        f"<th style='text-align:right;'>Amount</th><th style='text-align:center;'>Deadline</th>"
        f"<th style='text-align:center;'>Status</th></tr>"
        f"{rows}</table>"
    )


def _budget_html(steps_used: int, max_steps: int, score: float) -> str:
    steps_pct = min(100, int(100 * steps_used / max(1, max_steps)))
    score_pct = min(100, int(100 * score))
    remaining = max_steps - steps_used

    if steps_pct < 50:
        budget_color = "#22c55e"
    elif steps_pct < 80:
        budget_color = "#eab308"
    else:
        budget_color = "#ef4444"

    return f"""
    <div class="budget-section">
      <div class="budget-label"><span>Steps</span><span>{remaining} left of {max_steps}</span></div>
      <div class="bar-row">
        <div class="bar-track" style="flex:1;margin:0;">
          <div class="bar-fill" style="width:{steps_pct}%;background:{budget_color};"></div>
        </div>
      </div>
      <div class="budget-label" style="margin-top:10px;"><span>Score</span><span>{score:.3f}</span></div>
      <div class="bar-row">
        <div class="bar-track" style="flex:1;margin:0;">
          <div class="bar-fill" style="width:{score_pct}%;background:#3b82f6;"></div>
        </div>
      </div>
    </div>
    """


def _grader_html(report: dict | None) -> str:
    if not report:
        return ""

    score = report.get("normalized_score", 0)
    summary = report.get("summary", "")
    sc = _score_color(score)

    html = (
        f'<div class="score-big">'
        f'<div class="value" style="color:{sc};">{score:.3f}</div>'
        f'<div class="label">{summary}</div>'
        f'</div>'
    )

    dims = [
        ("Strategy", "strategy_correctness", "25%"),
        ("Evidence", "evidence_quality", "20%"),
        ("Packet", "packet_validity", "15%"),
        ("Deadline", "deadline_compliance", "15%"),
        ("Efficiency", "efficiency", "10%"),
        ("Outcome", "outcome_quality", "10%"),
        ("Note", "note_quality", "5%"),
    ]

    for case in report.get("case_reports", []):
        cid = case.get("case_id", "")
        res = case.get("final_resolution", "")
        ws = case.get("weighted_score", 0)

        bars = ""
        for label, key, weight in dims:
            v = case.get(key, 0)
            bars += _bar_html(f"{label} ({weight})", v, _score_color(v))

        notes = case.get("notes", "")
        notes_html = f'<div class="case-notes">{notes}</div>' if notes else ""

        html += (
            f'<div class="case-card">'
            f'<div class="case-header">'
            f'<span class="case-id">{cid}</span>'
            f'<span class="case-meta">{res} &middot; weighted {ws:.3f}</span>'
            f'</div>'
            f'{bars}{notes_html}'
            f'</div>'
        )

    return html


# ---------------------------------------------------------------------------
# Episode runner (generator — yields per step)
# ---------------------------------------------------------------------------

def _resolve_task_id(task_id: str, generated: bool, difficulty: str, seed: int) -> str:
    if generated:
        return f"generated_{difficulty}_s{seed}"
    return task_id


def run_episode(task_id: str, generated: bool, difficulty: str, seed: int):
    tid = _resolve_task_id(task_id, generated, difficulty, int(seed))
    env = ChargebackOpsEnvironment()
    obs = env.reset(task_id=tid, difficulty=difficulty, seed=int(seed))
    max_steps = obs.info.get("current_task_max_steps", 10)
    rows: list[list[Any]] = []

    header = (
        f"### {obs.task_title}\n"
        f"`{obs.task_id}` &mdash; {len(obs.queue)} case(s), "
        f"{max_steps} steps, **{obs.difficulty}**"
    )
    yield (
        header,
        _queue_html(obs),
        _budget_html(0, max_steps, 0.0),
        [row[:] for row in rows],
        "",
        None,
    )

    step = 0
    while not obs.done:
        payload = obs.model_dump()
        cands = candidate_actions(payload)
        if not cands:
            break
        pick = _obvious_next_action(payload, cands) or _heuristic_pick(cands)
        step += 1
        obs = env.step(pick.action)
        rows.append([
            step,
            pick.action.action_type,
            pick.action.case_id or obs.selected_case_id or "",
            pick.action.system_name or "",
            pick.action.strategy or "",
            round(obs.reward or 0.0, 4),
            obs.last_action_result,
        ])

        status_md = (
            f"**Step {step}** &mdash; `{pick.action.action_type}` "
            f"&rarr; reward **{round(obs.reward or 0.0, 4)}**"
        )
        grader = (
            _grader_html(obs.grader_report.model_dump())
            if obs.grader_report else ""
        )
        yield (
            status_md,
            _queue_html(obs),
            _budget_html(step, max_steps, obs.progress_score),
            [row[:] for row in rows],
            grader,
            None,
        )

    report = obs.grader_report.model_dump() if obs.grader_report else None
    sc = f"{obs.grader_report.normalized_score:.3f}" if obs.grader_report else "n/a"
    final_md = f"### Done &mdash; score **{sc}** in **{len(rows)}** steps"
    yield (
        final_md,
        _queue_html(obs),
        _budget_html(step, max_steps, obs.progress_score),
        [row[:] for row in rows],
        _grader_html(report),
        report,
    )


# ---------------------------------------------------------------------------
# Build Gradio app
# ---------------------------------------------------------------------------

def build_demo() -> gr.Blocks:
    tasks = list_tasks()
    task_ids = [t.task_id for t in tasks]
    default = task_ids[0] if task_ids else "goods_not_received_easy"

    with gr.Blocks(title="ChargebackOps") as demo:

        # Inject CSS (Gradio 6 moved css= to launch(); <style> tag works everywhere)
        gr.HTML(f"<style>{_CSS}</style>")

        # Header
        gr.HTML(
            '<div class="dashboard-header">'
            "<h1>ChargebackOps</h1>"
            "<p>Merchant chargeback dispute environment &mdash; OpenEnv benchmark</p>"
            "</div>"
        )

        with gr.Tabs():

            # ── Tab 1: Run Episode ────────────────────────────────
            with gr.Tab("Run Episode"):
                with gr.Row():
                    dd_task = gr.Dropdown(label="Task", choices=task_ids, value=default, scale=3)
                    cb_gen = gr.Checkbox(label="Generated", value=False, scale=1)
                    rd_diff = gr.Radio(
                        ["easy", "medium", "hard", "nightmare"],
                        label="Difficulty", value="easy", scale=2,
                    )
                    nb_seed = gr.Number(label="Seed", value=42, precision=0, scale=1)
                    btn_run = gr.Button("Run", variant="primary", scale=1)

                md_status = gr.Markdown("Pick a task and click **Run**.")

                with gr.Row(equal_height=True):
                    with gr.Column(scale=3):
                        html_queue = gr.HTML(label="Dispute Queue")
                    with gr.Column(scale=1, min_width=200):
                        html_budget = gr.HTML(label="Budget")

                df_trace = gr.Dataframe(
                    headers=["#", "Action", "Case", "System", "Strategy", "Reward", "Result"],
                    datatype=["number", "str", "str", "str", "str", "number", "str"],
                    interactive=False,
                    wrap=True,
                    label="Step Trace",
                )

                html_grader = gr.HTML(label="Grader Report")
                json_raw = gr.JSON(label="Raw JSON", visible=False)

                btn_run.click(
                    fn=run_episode,
                    inputs=[dd_task, cb_gen, rd_diff, nb_seed],
                    outputs=[md_status, html_queue, html_budget, df_trace, html_grader, json_raw],
                )

            # ── Tab 2: Task Catalog ───────────────────────────────
            with gr.Tab("Task Catalog"):
                catalog_rows = []
                for t in tasks:
                    nets = sorted({f"{c.card_network.upper()} {c.network_reason_code}" for c in t.cases})
                    catalog_rows.append([
                        t.task_id, t.title, t.difficulty,
                        len(t.cases), t.max_steps,
                        ", ".join(nets), t.objective,
                    ])
                gr.Dataframe(
                    value=catalog_rows,
                    headers=["Task ID", "Title", "Difficulty", "Cases", "Steps", "Networks", "Objective"],
                    interactive=False,
                    wrap=True,
                    label="10-Task Benchmark Catalog",
                )

            # ── Tab 3: Environment Info ───────────────────────────
            with gr.Tab("Environment"):
                gr.Markdown(
                    "## Action Space (9 typed actions)\n\n"
                    "`select_case` &middot; `inspect_case` &middot; `query_system` &middot; "
                    "`retrieve_policy` &middot; `add_evidence` &middot; `remove_evidence` &middot; "
                    "`set_strategy` &middot; `submit_representment` &middot; `resolve_case`\n\n"
                    "## Merchant Systems (6)\n\n"
                    "`orders` &middot; `payment` &middot; `shipping` &middot; "
                    "`support` &middot; `refunds` &middot; `risk`\n\n"
                    "## Grading (7 dimensions)\n\n"
                    "| Dimension | Weight | Scoring |\n"
                    "|---|---|---|\n"
                    "| Strategy Correctness | 25% | 1.0 optimal, 0.35 acceptable, 0.0 wrong |\n"
                    "| Evidence Quality | 20% | Required + helpful coverage, harmful penalty |\n"
                    "| Packet Validity | 15% | Binary: all required, zero harmful |\n"
                    "| Deadline Compliance | 15% | Binary: resolved before deadline |\n"
                    "| Efficiency | 10% | Penalises waste, rewards early concession |\n"
                    "| Outcome Quality | 10% | 1.0 optimal, 0.4 acceptable, 0.0 wrong |\n"
                    "| Note Quality | 5% | Policy keywords + evidence refs |\n\n"
                    "## Card Networks\n\n"
                    "| Reason Code | Visa | Mastercard |\n"
                    "|---|---|---|\n"
                    "| Goods Not Received | 13.1 (30 days) | 4855 (45 days) |\n"
                    "| Fraud CNP | 10.4 (30 days) | 4837 (45 days) |\n"
                    "| Credit Not Processed | 13.6 (30 days) | 4860 (45 days) |\n"
                    "| Duplicate Processing | 12.4 (30 days) | 4834 (45 days) |\n"
                    "| Product Not As Described | 13.3 (30 days) | 4853 (45 days) |\n"
                    "| Service Not Provided | 13.1 (30 days) | 4855 (45 days) |\n"
                )

    return demo
