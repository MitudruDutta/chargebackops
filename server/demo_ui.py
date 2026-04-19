"""Gradio demo UI for ChargebackOps."""

from __future__ import annotations

import os
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import gradio as gr

try:
    from ..evaluation.agent_brutal_audit import _bad_policy_action
    from ..runners.baseline_runner import (
        _heuristic_pick,
        _obvious_next_action,
        candidate_actions,
    )
    from ..scenarios.simulation import list_tasks
    from .chargeback_ops_environment import ChargebackOpsEnvironment
except ImportError:  # pragma: no cover
    from evaluation.agent_brutal_audit import _bad_policy_action
    from runners.baseline_runner import (
        _heuristic_pick,
        _obvious_next_action,
        candidate_actions,
    )
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

.round-panel { border: 1px solid #3a3a3a; border-radius: 8px; padding: 12px 14px; margin: 8px 0; background: #1a1a1a; }
.round-panel .panel-title { font-weight: 700; font-size: 13px; color: #ccc; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px; }
.round-badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 700; margin-right: 8px; }
.round-1 { background: #1e3a8a; color: #93c5fd; }
.round-2 { background: #78350f; color: #fcd34d; }
.round-3 { background: #7f1d1d; color: #fca5a5; }
.issuer-quote { font-style: italic; color: #d4d4d4; font-size: 13px; padding: 6px 10px; border-left: 3px solid #6366f1; margin: 6px 0; background: #15171f; }
.issuer-decision { font-weight: 700; font-size: 13px; }
.dec-accept { color: #22c55e; }
.dec-request { color: #eab308; }
.dec-escalate { color: #ef4444; }

.arb-panel { border: 1px solid #7f1d1d; border-radius: 8px; padding: 12px 14px; margin: 8px 0; background: #1a0e0e; }
.arb-row { display: flex; justify-content: space-between; padding: 4px 0; font-size: 13px; }
.arb-row .arb-label { color: #999; }
.arb-row .arb-value { font-weight: 700; }
.outcome-merchant { color: #22c55e; }
.outcome-issuer { color: #ef4444; }
.pnl-pos { color: #22c55e; font-weight: 800; }
.pnl-neg { color: #ef4444; font-weight: 800; }
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
        f"</div>"
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
            f"<td>{c.reason_code.replace('_', ' ')}</td>"
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


_DEC_CLASS = {
    "accept": "dec-accept",
    "request_more_evidence": "dec-request",
    "escalate_to_arbitration": "dec-escalate",
    "merchant_wins": "outcome-merchant",
    "issuer_wins": "outcome-issuer",
}


def _round_panel_html(observation) -> str:
    vc = observation.visible_case
    if vc is None:
        return ""

    rnd = vc.round_number or 1
    badge_cls = f"round-{min(rnd, 3)}"
    rnd_label = {1: "Representment", 2: "Pre-Arbitration", 3: "Arbitration"}.get(rnd, f"Round {rnd}")

    body = (
        f'<div class="panel-title">'
        f'<span class="round-badge {badge_cls}">R{rnd}</span>'
        f'{rnd_label} &middot; case <b>{vc.case_id}</b>'
        f'</div>'
    )

    if vc.last_issuer_decision:
        dec = vc.last_issuer_decision
        dec_cls = _DEC_CLASS.get(dec, "")
        dec_pretty = dec.replace("_", " ").title()
        body += f'<div class="issuer-decision {dec_cls}">Issuer: {dec_pretty}</div>'

    if vc.last_issuer_rationale:
        body += f'<div class="issuer-quote">&ldquo;{vc.last_issuer_rationale}&rdquo;</div>'

    if vc.pre_arb_evidence_added:
        ids = ", ".join(vc.pre_arb_evidence_added)
        body += (
            f'<div style="font-size:12px;color:#999;margin-top:4px;">'
            f'Pre-arb evidence added: <code>{ids}</code></div>'
        )

    return f'<div class="round-panel">{body}</div>'


def _arbitration_panel_html(observation) -> str:
    vc = observation.visible_case
    if vc is None or vc.arbitration_outcome is None:
        return ""

    outcome = vc.arbitration_outcome
    outcome_cls = _DEC_CLASS.get(outcome, "")
    outcome_label = outcome.replace("_", " ").title()
    pnl = vc.final_economic_outcome
    pnl_cls = "pnl-pos" if (pnl is not None and pnl >= 0) else "pnl-neg"
    pnl_str = f"${pnl:+,.2f}" if pnl is not None else "n/a"
    fees = vc.arb_fees_paid or 0.0

    return (
        f'<div class="arb-panel">'
        f'<div class="panel-title"><span class="round-badge round-3">ARB</span>Arbitration Outcome</div>'
        f'<div class="arb-row"><span class="arb-label">Ruling</span>'
        f'<span class="arb-value {outcome_cls}">{outcome_label}</span></div>'
        f'<div class="arb-row"><span class="arb-label">Arb fees paid</span>'
        f'<span class="arb-value">${fees:,.2f}</span></div>'
        f'<div class="arb-row"><span class="arb-label">Final P&amp;L</span>'
        f'<span class="arb-value {pnl_cls}">{pnl_str}</span></div>'
        f'</div>'
    )


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
        f"</div>"
    )

    dims = [
        ("Strategy", "strategy_correctness", "20%"),
        ("Evidence", "evidence_quality", "15%"),
        ("Packet", "packet_validity", "10%"),
        ("Deadline", "deadline_compliance", "10%"),
        ("Efficiency", "efficiency", "10%"),
        ("Outcome", "outcome_quality", "10%"),
        ("Note", "note_quality", "5%"),
        ("Esc ROI", "escalation_roi", "20%"),
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
            f"</div>"
            f"{bars}{notes_html}"
            f"</div>"
        )

    return html


# ---------------------------------------------------------------------------
# Episode runner (generator — yields per step)
# ---------------------------------------------------------------------------


def _resolve_task_id(task_id: str, generated: bool, difficulty: str, seed: int) -> str:
    if generated:
        return f"generated_{difficulty}_s{seed}"
    return task_id


def run_episode(
    task_id: str, generated: bool, difficulty: str, seed: int, policy: str = "heuristic"
):
    tid = _resolve_task_id(task_id, generated, difficulty, int(seed))
    env = ChargebackOpsEnvironment()
    obs = env.reset(task_id=tid, difficulty=difficulty, seed=int(seed))
    max_steps = obs.info.get("current_task_max_steps", 10)
    rows: list[list[Any]] = []

    policy_label = (
        "Heuristic" if policy == "heuristic" else "Naive (concede-everything)"
    )
    header = (
        f"### {obs.task_title}\n"
        f"`{obs.task_id}` &mdash; {len(obs.queue)} case(s), "
        f"{max_steps} steps, **{obs.difficulty}** &middot; policy: **{policy_label}**"
    )
    yield (
        header,
        _queue_html(obs),
        _budget_html(0, max_steps, 0.0),
        [row[:] for row in rows],
        _round_panel_html(obs),
        _arbitration_panel_html(obs),
        "",
        None,
    )

    step = 0
    while not obs.done:
        if policy == "bad":
            action = _bad_policy_action(obs)
            summary_action = action
        else:
            payload = obs.model_dump()
            cands = candidate_actions(payload)
            if not cands:
                break
            pick = _obvious_next_action(payload, cands) or _heuristic_pick(cands)
            action = pick.action
            summary_action = pick.action
        step += 1
        obs = env.step(action)
        rows.append(
            [
                step,
                summary_action.action_type,
                summary_action.case_id or obs.selected_case_id or "",
                summary_action.system_name or "",
                summary_action.strategy or "",
                round(obs.reward or 0.0, 4),
                obs.last_action_result,
            ]
        )

        status_md = (
            f"**Step {step}** &mdash; `{summary_action.action_type}` "
            f"&rarr; reward **{round(obs.reward or 0.0, 4)}** &middot; policy: **{policy_label}**"
        )
        grader = (
            _grader_html(obs.grader_report.model_dump()) if obs.grader_report else ""
        )
        yield (
            status_md,
            _queue_html(obs),
            _budget_html(step, max_steps, obs.progress_score),
            [row[:] for row in rows],
            _round_panel_html(obs),
            _arbitration_panel_html(obs),
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
        _round_panel_html(obs),
        _arbitration_panel_html(obs),
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
                    dd_task = gr.Dropdown(
                        label="Task", choices=task_ids, value=default, scale=3
                    )
                    cb_gen = gr.Checkbox(label="Generated", value=False, scale=1)
                    rd_diff = gr.Radio(
                        ["easy", "medium", "hard", "nightmare"],
                        label="Difficulty",
                        value="easy",
                        scale=2,
                    )
                    nb_seed = gr.Number(label="Seed", value=42, precision=0, scale=1)
                with gr.Row():
                    rd_policy = gr.Radio(
                        choices=[
                            ("Heuristic (smart baseline)", "heuristic"),
                            ("Naive (always concede)", "bad"),
                        ],
                        value="heuristic",
                        label="Policy",
                        scale=4,
                    )
                    btn_run = gr.Button("Run Episode", variant="primary", scale=1)

                md_status = gr.Markdown(
                    "Pick a task + policy and click **Run Episode**. Compare **Heuristic** vs "
                    "**Naive** to see how the 8-dimension rubric &mdash; including escalation ROI &mdash; "
                    "separates an EV-rational agent from a lazy one."
                )

                with gr.Row(equal_height=True):
                    with gr.Column(scale=3):
                        html_queue = gr.HTML(label="Dispute Queue")
                    with gr.Column(scale=1, min_width=200):
                        html_budget = gr.HTML(label="Budget")

                df_trace = gr.Dataframe(
                    headers=[
                        "#",
                        "Action",
                        "Case",
                        "System",
                        "Strategy",
                        "Reward",
                        "Result",
                    ],
                    datatype=["number", "str", "str", "str", "str", "number", "str"],
                    interactive=False,
                    wrap=True,
                    label="Step Trace",
                )

                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        html_round = gr.HTML(label="Dispute Round")
                    with gr.Column(scale=1):
                        html_arb = gr.HTML(label="Arbitration")

                html_grader = gr.HTML(label="Grader Report")
                json_raw = gr.JSON(label="Raw JSON", visible=False)

                btn_run.click(
                    fn=run_episode,
                    inputs=[dd_task, cb_gen, rd_diff, nb_seed, rd_policy],
                    outputs=[
                        md_status,
                        html_queue,
                        html_budget,
                        df_trace,
                        html_round,
                        html_arb,
                        html_grader,
                        json_raw,
                    ],
                )

            # ── Tab 2: Task Catalog ───────────────────────────────
            with gr.Tab("Task Catalog"):
                catalog_rows = []
                for t in tasks:
                    nets = sorted(
                        {
                            f"{c.card_network.upper()} {c.network_reason_code}"
                            for c in t.cases
                        }
                    )
                    catalog_rows.append(
                        [
                            t.task_id,
                            t.title,
                            t.difficulty,
                            len(t.cases),
                            t.max_steps,
                            ", ".join(nets),
                            t.objective,
                        ]
                    )
                gr.Dataframe(
                    value=catalog_rows,
                    headers=[
                        "Task ID",
                        "Title",
                        "Difficulty",
                        "Cases",
                        "Steps",
                        "Networks",
                        "Objective",
                    ],
                    interactive=False,
                    wrap=True,
                    label=f"{len(tasks)}-Task Benchmark Catalog",
                )

            # ── Tab 3: Environment Info ───────────────────────────
            with gr.Tab("Environment"):
                gr.Markdown(
                    "## Action Space (12 typed actions)\n\n"
                    "**Round 1 — Representment:** `select_case` &middot; `inspect_case` &middot; "
                    "`query_system` &middot; `retrieve_policy` &middot; `add_evidence` &middot; "
                    "`remove_evidence` &middot; `set_strategy` &middot; `submit_representment` &middot; "
                    "`resolve_case`\n\n"
                    "**Round 2/3 — Pre-arb &amp; Arbitration:** `respond_to_pre_arb` &middot; "
                    "`escalate_to_arbitration` &middot; `accept_arbitration_loss`\n\n"
                    "## Merchant Systems (6)\n\n"
                    "`orders` &middot; `payment` &middot; `shipping` &middot; "
                    "`support` &middot; `refunds` &middot; `risk`\n\n"
                    "## Grading (8 dimensions)\n\n"
                    "| Dimension | Weight | Scoring |\n"
                    "|---|---|---|\n"
                    "| Strategy Correctness | 20% | 1.0 optimal, 0.35 acceptable, 0.0 wrong |\n"
                    "| Evidence Quality | 15% | Required + helpful coverage, harmful penalty |\n"
                    "| Packet Validity | 10% | Binary: all required, zero harmful |\n"
                    "| Deadline Compliance | 10% | Binary: resolved before deadline |\n"
                    "| Efficiency | 10% | Penalises waste, rewards early concession |\n"
                    "| Outcome Quality | 10% | 1.0 optimal, 0.4 acceptable, 0.0 wrong |\n"
                    "| Note Quality | 5% | Policy keywords + evidence refs |\n"
                    "| Escalation ROI | 20% | EV-rational arbitration: P(win)·amount vs $250 fee |\n\n"
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
