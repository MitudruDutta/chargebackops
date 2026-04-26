"""Gradio demo UI for ChargebackOps."""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Callable

# Ensure matplotlib has a writable config dir on locked-down hosts (e.g. HF
# Spaces). Guarded so importing this module from a notebook doesn't pollute
# the user's environment unnecessarily.
if not os.environ.get("MPLCONFIGDIR"):
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

import gradio as gr

try:
    from ..core.models import ChargebackOpsAction
    from ..evaluation.rubrics import (
        CASE_DIMENSION_NAMES,
        CASE_DIMENSION_WEIGHTS,
    )
    from ..runners.baseline_runner import (
        _heuristic_pick,
        _obvious_next_action,
        candidate_actions,
    )
    from ..runners.benchmark_runner import POLICY_REGISTRY
    from ..scenarios.simulation import get_task, list_tasks
    from .chargeback_ops_environment import ChargebackOpsEnvironment
except ImportError:  # pragma: no cover
    from core.models import ChargebackOpsAction
    from evaluation.rubrics import (
        CASE_DIMENSION_NAMES,
        CASE_DIMENSION_WEIGHTS,
    )
    from runners.baseline_runner import (
        _heuristic_pick,
        _obvious_next_action,
        candidate_actions,
    )
    from runners.benchmark_runner import POLICY_REGISTRY
    from scenarios.simulation import get_task, list_tasks
    from server.chargeback_ops_environment import ChargebackOpsEnvironment

# OpenAI-compatible LLM policy is optional — the demo gracefully degrades to
# scripted policies if the openai SDK or runners.inference is unavailable.
try:  # pragma: no cover — exercised only when LLM policy is selected
    from openai import OpenAI  # noqa: F401
    try:
        from ..runners.inference import _pick_with_openai_client
    except ImportError:
        from runners.inference import _pick_with_openai_client
    _LLM_POLICY_AVAILABLE = True
except Exception:  # pragma: no cover
    _pick_with_openai_client = None  # type: ignore[assignment]
    _LLM_POLICY_AVAILABLE = False

# Path to the bundled hero figures (used by the Training Results tab).
_FIGURES_DIR = Path(__file__).resolve().parents[1] / "docs" / "figures"


# ---------------------------------------------------------------------------
# Static metadata
# ---------------------------------------------------------------------------

# Human-readable display labels for the 8 rubric dimensions (in canonical order).
_DIMENSION_LABELS: tuple[str, ...] = (
    "Strategy Correctness",
    "Evidence Quality",
    "Packet Validity",
    "Deadline Compliance",
    "Efficiency",
    "Outcome Quality",
    "Note Quality",
    "Escalation ROI",
)

# Per-dimension scoring summary (kept short so the table fits on one screen).
_DIMENSION_SCORING: tuple[str, ...] = (
    "1.0 optimal · 0.35 acceptable · 0.0 wrong",
    "Required + helpful coverage; harmful evidence penalised",
    "Binary: all required evidence + zero harmful",
    "Binary: case resolved before deadline",
    "Penalises waste; rewards early concession",
    "1.0 optimal · 0.4 acceptable · 0.0 wrong",
    "Policy keywords + evidence references",
    "EV-rational arbitration: P(win)·amount vs $250 fee",
)

# Selectable scripted policies (label shown to user → registry key).
# Order is intentional: best → worst, so radio top-to-bottom reads as a
# discrimination ladder.
_POLICY_CHOICES: tuple[tuple[str, str], ...] = (
    ("Heuristic — EV-rational baseline", "heuristic"),
    ("Escalate-all — contest then always escalate", "escalate_all"),
    ("Concede-all — always accept the chargeback", "concede_all"),
    ("Naive — submit empty packet, no evidence", "naive"),
    ("LLM (OpenAI-compatible API)", "llm"),
)
_POLICY_LABEL_BY_KEY: dict[str, str] = {
    key: label for label, key in _POLICY_CHOICES
}
# Subset used by the Compare tab — scripted-only, deterministic, no API calls.
_COMPARE_POLICIES: tuple[str, ...] = (
    "naive",
    "concede_all",
    "escalate_all",
    "heuristic",
)

# One-click presets for the Run-Episode tab. Each preset is
# (button_label, task_id, generated_flag, difficulty, seed, recommended_policy, blurb).
_PRESETS: tuple[tuple[str, str, bool, str, int, str, str], ...] = (
    (
        "Easy contestable",
        "goods_not_received_easy",
        False,
        "easy",
        42,
        "heuristic",
        "Goods-not-received with strong evidence — heuristic should win round 1.",
    ),
    (
        "Queue optimization (hard)",
        "queue_optimization_hard",
        False,
        "hard",
        42,
        "heuristic",
        "Triage a heterogeneous queue under tight deadlines — exercises EV reasoning.",
    ),
    (
        "Long-horizon backlog",
        "monthly_dispute_backlog_marathon",
        False,
        "medium",
        42,
        "heuristic",
        "12 cases over 60 steps with delayed evidence; tests scheduling + waiting.",
    ),
    (
        "Generated nightmare",
        "generated_nightmare_s31",
        True,
        "nightmare",
        31,
        "heuristic",
        "Adversarial parametric task — even the heuristic struggles.",
    ),
    (
        "Compare all 4 policies",
        "goods_not_received_easy",
        False,
        "easy",
        42,
        "heuristic",
        "Open the Compare tab — same task, all four scripted policies side-by-side.",
    ),
)


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


def _round_panel_html(
    observation, history: list[dict[str, str]] | None = None
) -> str:
    """Render the visible case's round panel, including a chronological
    issuer-message log so multi-round disputes show every R1/R2/R3 message.

    ``history`` is a list of ``{round, decision, rationale}`` dicts the caller
    accumulates across steps.
    """

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

    # Show full issuer-message history if we have it, else fall back to the
    # last-message snapshot from the observation.
    rendered_any = False
    if history:
        for entry in history:
            ent_rnd = entry.get("round", "?")
            ent_dec = entry.get("decision") or ""
            ent_rat = entry.get("rationale") or ""
            ent_badge_cls = f"round-{min(int(ent_rnd) if str(ent_rnd).isdigit() else 1, 3)}"
            dec_cls = _DEC_CLASS.get(ent_dec, "")
            dec_pretty = ent_dec.replace("_", " ").title() if ent_dec else "(no decision)"
            body += (
                f'<div style="margin-top:8px;">'
                f'<span class="round-badge {ent_badge_cls}">R{ent_rnd}</span>'
                f'<span class="issuer-decision {dec_cls}">Issuer: {dec_pretty}</span>'
                f'</div>'
            )
            if ent_rat:
                body += f'<div class="issuer-quote">&ldquo;{ent_rat}&rdquo;</div>'
            rendered_any = True

    if not rendered_any and vc.last_issuer_decision:
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


def _build_llm_policy(
    base_url: str, api_key: str, model_name: str
) -> tuple[Callable[[dict[str, Any]], ChargebackOpsAction | None], str]:
    """Return ``(policy_fn, label)`` calling an OpenAI-compatible chat model.

    The policy mirrors the production inference pipeline in
    :mod:`runners.inference`: candidate generation + obvious-action shortcut +
    LLM pick over the shortlist. On any LLM failure (network, parse, missing
    key) it falls back to the heuristic so the demo never freezes mid-stream.

    UI fields take precedence; blanks fall back to ``HF_TOKEN`` /
    ``API_KEY`` / ``OPENROUTER_API_KEY`` / ``GROQ_API_KEY`` / ``API_BASE_URL``
    / ``MODEL_NAME`` env vars. This lets HF Space operators wire credentials
    via Space Secrets without the public demo asking visitors for keys.
    """

    if not _LLM_POLICY_AVAILABLE or _pick_with_openai_client is None:
        raise RuntimeError(
            "openai SDK is not available — install `openai` to use the LLM policy."
        )

    base_url = (base_url or "").strip()
    api_key = (api_key or "").strip()
    model_name = (model_name or "").strip()

    if not api_key:
        api_key = (
            os.getenv("HF_TOKEN")
            or os.getenv("API_KEY")
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("GROQ_API_KEY")
            or ""
        )
    # Resolve provider from explicit base_url first, then from which key
    # variable was set in the environment. This lets us pick a sensible
    # default model name even when only the key is provided.
    provider: str = ""
    if not base_url:
        base_url = os.getenv("API_BASE_URL", "").strip()
    if base_url:
        lowered = base_url.lower()
        if "groq" in lowered:
            provider = "groq"
        elif "openrouter" in lowered:
            provider = "openrouter"
        elif "huggingface" in lowered or "hf.space" in lowered:
            provider = "hf"
        elif "openai.com" in lowered:
            provider = "openai"
    if not base_url:
        if os.getenv("GROQ_API_KEY"):
            base_url, provider = "https://api.groq.com/openai/v1", "groq"
        elif os.getenv("OPENROUTER_API_KEY"):
            base_url, provider = "https://openrouter.ai/api/v1", "openrouter"
        else:
            base_url, provider = "https://router.huggingface.co/v1", "hf"

    if not model_name:
        model_name = os.getenv("MODEL_NAME", "").strip()
    if not model_name:
        # Provider-appropriate defaults — every option here works without
        # the user having to look up a model card.
        provider_defaults = {
            "groq": "llama-3.3-70b-versatile",
            "openrouter": "meta-llama/llama-3.1-8b-instruct:free",
            "openai": "gpt-4o-mini",
            "hf": "Qwen/Qwen2.5-72B-Instruct",
        }
        model_name = provider_defaults.get(provider, "Qwen/Qwen2.5-72B-Instruct")

    if not api_key:
        raise RuntimeError(
            "No API key — type one in the UI, or set HF_TOKEN / API_KEY / "
            "OPENROUTER_API_KEY / GROQ_API_KEY in the environment (HF Space "
            "Secrets work too)."
        )
    if not model_name:
        raise RuntimeError("Model name is required for the LLM policy.")

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=15.0,
        max_retries=0,
    )

    def policy_fn(observation: dict[str, Any]) -> ChargebackOpsAction | None:
        cands = candidate_actions(observation)
        if not cands:
            return None
        if len(cands) == 1:
            return cands[0].action
        obvious = _obvious_next_action(observation, cands)
        if obvious is not None:
            return obvious.action
        try:
            pick, _ok, _err = _pick_with_openai_client(
                client, model_name, observation, cands
            )
            return pick.action
        except Exception:
            return _heuristic_pick(cands).action

    label = f"LLM ({model_name})"
    return policy_fn, label


def _result_badge(result: str | None) -> str:
    """Prefix a step result string with a status emoji for fast scanning.

    Distinguishes accepted/no-op/rejected so the trace dataframe self-narrates.
    """

    if not result:
        return "· (no result)"
    text = str(result)
    lowered = text.lower()
    if "error" in lowered or "reject" in lowered or "invalid" in lowered or "fail" in lowered:
        return f"✗ {text}"
    if "no-op" in lowered or "noop" in lowered or "ignored" in lowered or "skipped" in lowered:
        return f"⚠ {text}"
    return f"✓ {text}"


def _resolve_max_steps(observation, task_id: str) -> int:
    """Pull the task budget from the observation; fall back to the task definition.

    The legacy implementation defaulted to 10 if the observation field was absent,
    which silently mis-rendered the budget bar. The env always populates
    ``info.current_task_max_steps`` after ``reset``; if it ever doesn't, we read
    the task object directly so the bar still reflects truth.
    """

    cap = observation.info.get("current_task_max_steps")
    if isinstance(cap, int) and cap > 0:
        return cap
    try:
        return int(get_task(task_id).max_steps)
    except Exception:  # pragma: no cover — defensive
        return 60


def run_episode(
    task_id: str,
    generated: bool,
    difficulty: str,
    seed: int,
    policy: str = "heuristic",
    llm_base_url: str = "",
    llm_api_key: str = "",
    llm_model: str = "",
):
    tid = _resolve_task_id(task_id, generated, difficulty, int(seed))
    env = ChargebackOpsEnvironment()
    obs = env.reset(task_id=tid, difficulty=difficulty, seed=int(seed))
    max_steps = _resolve_max_steps(obs, tid)
    rows: list[list[Any]] = []

    policy_fn: Callable[[dict[str, Any]], ChargebackOpsAction | None] | None = None
    if policy == "llm":
        try:
            policy_fn, policy_label = _build_llm_policy(
                llm_base_url, llm_api_key, llm_model
            )
        except Exception as exc:
            err_md = (
                f"### LLM policy unavailable\n"
                f"`{type(exc).__name__}: {exc}`\n\n"
                f"Falling back to **heuristic** for this run."
            )
            policy = "heuristic"
            policy_fn = POLICY_REGISTRY["heuristic"]
            policy_label = _POLICY_LABEL_BY_KEY[policy]
            yield (
                err_md,
                _queue_html(obs),
                _budget_html(0, max_steps, 0.0),
                [],
                "",
                "",
                "",
                None,
            )
    if policy_fn is None:
        policy_fn = POLICY_REGISTRY.get(policy) or POLICY_REGISTRY["heuristic"]
        if policy not in POLICY_REGISTRY:
            policy = "heuristic"
        policy_label = _POLICY_LABEL_BY_KEY.get(policy, policy)

    # Per-case issuer-message log: case_id -> [{"round","decision","rationale"}]
    issuer_log: dict[str, list[dict[str, str]]] = {}

    def _maybe_log_issuer_msg(observation) -> None:
        vc = observation.visible_case
        if vc is None or not vc.last_issuer_decision:
            return
        log = issuer_log.setdefault(vc.case_id, [])
        entry = {
            "round": str(vc.round_number or 1),
            "decision": vc.last_issuer_decision or "",
            "rationale": vc.last_issuer_rationale or "",
        }
        # Avoid duplicating the same message on adjacent steps.
        if not log or log[-1] != entry:
            log.append(entry)

    def _current_history(observation) -> list[dict[str, str]]:
        vc = observation.visible_case
        if vc is None:
            return []
        return issuer_log.get(vc.case_id, [])

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
        _round_panel_html(obs, _current_history(obs)),
        _arbitration_panel_html(obs),
        "",
        None,
    )

    step = 0
    while not obs.done:
        payload = obs.model_dump()
        try:
            action = policy_fn(payload)
        except Exception as exc:  # pragma: no cover — surface in UI
            err_md = (
                f"### Policy error\n"
                f"`{policy}` raised `{type(exc).__name__}: {exc}` on step {step + 1}. "
                f"Halting episode."
            )
            yield (
                err_md,
                _queue_html(obs),
                _budget_html(step, max_steps, obs.progress_score),
                [row[:] for row in rows],
                _round_panel_html(obs, _current_history(obs)),
                _arbitration_panel_html(obs),
                "",
                None,
            )
            return
        if action is None:
            break

        summary_action = action
        step += 1
        try:
            obs = env.step(action)
        except Exception as exc:  # pragma: no cover — surface in UI
            err_md = (
                f"### Environment error\n"
                f"`env.step({summary_action.action_type})` raised "
                f"`{type(exc).__name__}: {exc}` on step {step}. "
                f"Halting episode."
            )
            rows.append(
                [
                    step,
                    summary_action.action_type,
                    summary_action.case_id or "",
                    summary_action.system_name or "",
                    summary_action.strategy or "",
                    0.0,
                    f"✗ error: {type(exc).__name__}",
                ]
            )
            yield (
                err_md,
                _queue_html(obs),
                _budget_html(step, max_steps, obs.progress_score),
                [row[:] for row in rows],
                _round_panel_html(obs, _current_history(obs)),
                _arbitration_panel_html(obs),
                "",
                None,
            )
            return

        _maybe_log_issuer_msg(obs)

        rows.append(
            [
                step,
                summary_action.action_type,
                summary_action.case_id or obs.selected_case_id or "",
                summary_action.system_name or "",
                summary_action.strategy or "",
                round(obs.reward or 0.0, 4),
                _result_badge(obs.last_action_result),
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
            _round_panel_html(obs, _current_history(obs)),
            _arbitration_panel_html(obs),
            grader,
            None,
        )

    report = obs.grader_report.model_dump() if obs.grader_report else None
    sc = f"{obs.grader_report.normalized_score:.3f}" if obs.grader_report else "n/a"
    final_md = (
        f"### Done &mdash; score **{sc}** in **{len(rows)}** steps "
        f"&middot; policy: **{policy_label}**"
    )
    yield (
        final_md,
        _queue_html(obs),
        _budget_html(step, max_steps, obs.progress_score),
        [row[:] for row in rows],
        _round_panel_html(obs, _current_history(obs)),
        _arbitration_panel_html(obs),
        _grader_html(report),
        report,
    )


# ---------------------------------------------------------------------------
# Compare tab — run all four scripted policies on the same task in series and
# render a single side-by-side bar chart of the final scores plus a per-case
# per-dimension breakdown.
# ---------------------------------------------------------------------------


def _run_one_episode_sync(task_id: str, policy_key: str) -> dict[str, Any]:
    """Synchronously run a single scripted-policy episode and return summary.

    Cheap because every policy in :data:`_COMPARE_POLICIES` is pure-Python and
    fully offline (no provider calls).
    """

    env = ChargebackOpsEnvironment()
    obs = env.reset(task_id=task_id)
    policy_fn = POLICY_REGISTRY[policy_key]
    steps = 0
    while not obs.done:
        try:
            action = policy_fn(obs.model_dump())
        except Exception:
            break
        if action is None:
            break
        try:
            obs = env.step(action)
        except Exception:
            break
        steps += 1
    score = obs.grader_report.normalized_score if obs.grader_report else 0.0
    return {
        "policy": policy_key,
        "score": float(score),
        "steps": steps,
        "summary": obs.grader_report.summary if obs.grader_report else "",
    }


def run_compare(task_id: str, generated: bool, difficulty: str, seed: int):
    """Run all four scripted policies on the same task and render a chart."""

    tid = _resolve_task_id(task_id, generated, difficulty, int(seed))
    results = [_run_one_episode_sync(tid, p) for p in _COMPARE_POLICIES]

    # Bar-chart HTML (CSS-only, no extra deps).
    max_score = max((r["score"] for r in results), default=1.0) or 1.0
    bars = ""
    for r in results:
        pct = int(round(100 * r["score"] / max(0.001, max_score)))
        color = _score_color(r["score"])
        bars += (
            f'<div class="bar-row" style="margin:6px 0;">'
            f'<span class="bar-label" style="width:130px;">{r["policy"]}</span>'
            f'<div class="bar-track" style="flex:1;height:22px;">'
            f'<div class="bar-fill" style="width:{pct}%;background:{color};height:100%;"></div>'
            f'</div>'
            f'<span class="bar-value" style="width:120px;">'
            f'{r["score"]:.3f} · {r["steps"]} steps</span>'
            f'</div>'
        )

    # Discrimination delta.
    by_policy = {r["policy"]: r["score"] for r in results}
    delta = by_policy.get("heuristic", 0.0) - by_policy.get("naive", 0.0)
    title = (
        f'<div style="margin:8px 0;font-size:14px;">'
        f'<b>Task</b>: <code>{tid}</code> &middot; '
        f'<b>Discrimination delta</b> (heuristic − naive) = '
        f'<span style="color:{_score_color(delta)};">'
        f'<b>+{delta:.3f}</b></span>'
        f'</div>'
    )

    md = (
        f"### Side-by-side: 4 scripted policies on the same task\n"
        f"Same `task_id`, same `seed`, no provider calls. The discrimination "
        f"gradient (`naive` → `concede_all` → `escalate_all` → `heuristic`) "
        f"is the empirical evidence behind the README's `+0.813` claim."
    )
    table_rows = [
        [r["policy"], f"{r['score']:.3f}", r["steps"], r["summary"]]
        for r in results
    ]
    return md, title + '<div style="padding:8px 0;">' + bars + "</div>", table_rows


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

        # Header + context links
        gr.HTML(
            '<div class="dashboard-header">'
            "<h1>ChargebackOps</h1>"
            "<p>Merchant chargeback dispute environment &mdash; an OpenEnv benchmark for "
            "cost-asymmetric multi-round LLM agents</p>"
            '<div style="margin-top:8px;">'
            '<a href="https://github.com/MitudruDutta/chargebackops" target="_blank" '
            'style="margin:0 6px;color:#3b82f6;text-decoration:none;">📦 GitHub</a> '
            '<a href="https://huggingface.co/spaces/mitudrudutta/ChargeBackOps" target="_blank" '
            'style="margin:0 6px;color:#FFD21E;text-decoration:none;">🤗 HF Space</a> '
            '<a href="https://youtu.be/7dz37JTTMo4" target="_blank" '
            'style="margin:0 6px;color:#FF0000;text-decoration:none;">📺 Walkthrough</a> '
            '<a href="https://colab.research.google.com/drive/1GtLH6_b10oHlAnnGq4hnBkcGJ-pE_za5" target="_blank" '
            'style="margin:0 6px;color:#F9AB00;text-decoration:none;">🧪 Training Colab</a> '
            '<a href="https://github.com/meta-pytorch/OpenEnv" target="_blank" '
            'style="margin:0 6px;color:#0668E1;text-decoration:none;">🦙 Meta OpenEnv</a>'
            "</div>"
            "</div>"
        )

        with gr.Tabs():
            # ── Tab 1: Run Episode ────────────────────────────────
            with gr.Tab("Run Episode"):
                # Preset buttons row — one-click task+policy configuration.
                gr.Markdown("**Quick presets** — click any to load a known-good configuration.")
                with gr.Row():
                    preset_buttons = [
                        gr.Button(p[0], size="sm", scale=1) for p in _PRESETS
                    ]
                preset_blurb = gr.Markdown("")

                with gr.Row():
                    dd_task = gr.Dropdown(
                        label="Task", choices=task_ids, value=default, scale=3
                    )
                    cb_gen = gr.Checkbox(label="Generated", value=False, scale=1)
                    rd_diff = gr.Radio(
                        ["easy", "medium", "hard", "nightmare"],
                        label="Difficulty",
                        value="easy",
                        visible=False,
                        scale=2,
                    )
                    nb_seed = gr.Number(
                        label="Seed", value=42, precision=0, visible=False, scale=1
                    )
                with gr.Row():
                    rd_policy = gr.Radio(
                        choices=list(_POLICY_CHOICES),
                        value="heuristic",
                        label="Policy",
                        scale=4,
                    )
                    btn_run = gr.Button("Run Episode", variant="primary", scale=1)

                # LLM-policy inputs — only visible when "LLM" is selected.
                with gr.Accordion(
                    "LLM policy settings (used when 'LLM' is selected above)",
                    open=False,
                    visible=False,
                ) as llm_accordion:
                    gr.Markdown(
                        "Bring your own OpenAI-compatible endpoint. Defaults match the "
                        "Hugging Face router; OpenRouter, Groq, Together, Fireworks, "
                        "and Anthropic-compatible gateways all work. **Leave fields "
                        "blank** to inherit `HF_TOKEN` / `OPENROUTER_API_KEY` / "
                        "`GROQ_API_KEY` / `API_BASE_URL` / `MODEL_NAME` from the "
                        "environment (set them as Space Secrets when deploying)."
                    )
                    with gr.Row():
                        tb_llm_base = gr.Textbox(
                            label="Base URL",
                            value="https://router.huggingface.co/v1",
                            scale=2,
                        )
                        tb_llm_model = gr.Textbox(
                            label="Model",
                            value="Qwen/Qwen2.5-72B-Instruct",
                            scale=2,
                        )
                        tb_llm_key = gr.Textbox(
                            label="API key",
                            value="",
                            type="password",
                            scale=2,
                        )

                md_status = gr.Markdown(
                    "Pick a task + policy and click **Run Episode**. Run the same task "
                    "under each of the four scripted policies (heuristic, escalate-all, "
                    "concede-all, naive) to reproduce the discrimination gradient — naive "
                    "→ 0.000, concede-all → ~0.44, escalate-all → ~0.77, heuristic → ~0.81. "
                    "Or pick **LLM** and bring your own model. For a side-by-side view, "
                    "open the **Compare policies** tab."
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
                    label="Step Trace (✓ accepted · ⚠ no-op · ✗ rejected)",
                )

                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        html_round = gr.HTML(label="Dispute Round (issuer messages)")
                    with gr.Column(scale=1):
                        html_arb = gr.HTML(label="Arbitration")

                html_grader = gr.HTML(label="Grader Report")
                with gr.Accordion("Raw grader JSON (export-friendly)", open=False):
                    json_raw = gr.JSON(label="Raw JSON", show_label=False)

                btn_run.click(
                    fn=run_episode,
                    inputs=[
                        dd_task, cb_gen, rd_diff, nb_seed, rd_policy,
                        tb_llm_base, tb_llm_key, tb_llm_model,
                    ],
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

                # Generated-checkbox visibility callback.
                def _toggle_generated(generated: bool):
                    return (
                        gr.update(visible=generated),
                        gr.update(visible=generated),
                    )

                cb_gen.change(
                    fn=_toggle_generated,
                    inputs=[cb_gen],
                    outputs=[rd_diff, nb_seed],
                )

                # Show LLM accordion only when 'llm' policy is selected.
                def _toggle_llm(policy: str):
                    return gr.update(visible=(policy == "llm"), open=(policy == "llm"))

                rd_policy.change(
                    fn=_toggle_llm, inputs=[rd_policy], outputs=[llm_accordion]
                )

                # Wire each preset button to populate the inputs atomically.
                def _make_preset_handler(preset):
                    label, t_id, gen, diff, seed_v, pol, blurb = preset

                    def _apply():
                        return (
                            t_id,                              # dd_task
                            gen,                               # cb_gen
                            gr.update(value=diff, visible=gen),  # rd_diff
                            gr.update(value=seed_v, visible=gen),  # nb_seed
                            pol,                               # rd_policy
                            gr.update(visible=(pol == "llm")),  # llm_accordion
                            f"**Preset:** {label} — {blurb}",   # preset_blurb
                        )

                    return _apply

                for btn, preset in zip(preset_buttons, _PRESETS):
                    btn.click(
                        fn=_make_preset_handler(preset),
                        inputs=[],
                        outputs=[
                            dd_task,
                            cb_gen,
                            rd_diff,
                            nb_seed,
                            rd_policy,
                            llm_accordion,
                            preset_blurb,
                        ],
                    )

            # ── Tab 2: Compare policies ──────────────────────────
            with gr.Tab("Compare policies"):
                gr.Markdown(
                    "Run all four scripted policies on the **same task / seed** and see "
                    "the discrimination gradient at a glance. No provider calls, no LLM, "
                    "fully deterministic — this is the empirical evidence behind the "
                    "README's `+0.813` discrimination delta claim."
                )
                with gr.Row():
                    cmp_task = gr.Dropdown(
                        label="Task", choices=task_ids, value=default, scale=3
                    )
                    cmp_gen = gr.Checkbox(label="Generated", value=False, scale=1)
                    cmp_diff = gr.Radio(
                        ["easy", "medium", "hard", "nightmare"],
                        label="Difficulty",
                        value="easy",
                        visible=False,
                        scale=2,
                    )
                    cmp_seed = gr.Number(
                        label="Seed", value=42, precision=0, visible=False, scale=1
                    )
                btn_cmp = gr.Button("Run all 4 policies", variant="primary")
                cmp_md = gr.Markdown("")
                cmp_html = gr.HTML(label="Final-score comparison")
                cmp_table = gr.Dataframe(
                    headers=["Policy", "Score", "Steps", "Summary"],
                    datatype=["str", "str", "number", "str"],
                    interactive=False,
                    wrap=True,
                    label="Per-policy summary",
                )
                btn_cmp.click(
                    fn=run_compare,
                    inputs=[cmp_task, cmp_gen, cmp_diff, cmp_seed],
                    outputs=[cmp_md, cmp_html, cmp_table],
                )
                cmp_gen.change(
                    fn=_toggle_generated,
                    inputs=[cmp_gen],
                    outputs=[cmp_diff, cmp_seed],
                )

            # ── Tab 3: Task Catalog ──────────────────────────────
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
                gr.Markdown(_environment_tab_markdown())

            # ── Tab 5: Rubric Tree ────────────────────────────────
            with gr.Tab("Rubric Tree"):
                gr.Markdown(
                    "Live introspection of `env.rubric.named_rubrics()` — the same composable "
                    "OpenEnv `Rubric` tree that grades every step. Weights and structure below "
                    "are read from the running environment, not hardcoded."
                )
                gr.HTML(_rubric_tree_html())
                gr.Markdown(
                    "See [`docs/METHOD.md`](https://github.com/MitudruDutta/chargebackops/blob/main/docs/METHOD.md) "
                    "and [`docs/SPECIFICATION_GAMING.md`](https://github.com/MitudruDutta/chargebackops/blob/main/docs/SPECIFICATION_GAMING.md) "
                    "for the full design and the GRPO failure-mode write-up."
                )

            # ── Tab 6: Training Results ───────────────────────────
            with gr.Tab("Training Results"):
                gr.Markdown(_training_tab_markdown())
                for caption, fname in (
                    (
                        "**Cross-iteration training curve.** Iter 3 plateaued below the "
                        "heuristic at 0.728. Iter 5 plateaued *bit-exactly* at the heuristic "
                        "at 0.8132 — the signature of the eval-fallback exploit, not "
                        "convergent learning.",
                        "training_curve_cross_iter.png",
                    ),
                    (
                        "**Iter-5 eval-score attribution.** The trained policy contributes "
                        "0.000 (every action is rejected by env validation). The eval rollout "
                        "helper's heuristic-fallback path contributes 0.8132 — i.e. all of it.",
                        "gaming_attribution.png",
                    ),
                    (
                        "**Scripted-policy discrimination gradient.** The 8-dimension "
                        "`WeightedSum` plus the deadline `Gate` defeats every degenerate "
                        "policy: empty-packet zeros out, concede-all caps at 0.44, "
                        "escalate-all caps at 0.77.",
                        "discrimination_gradient.png",
                    ),
                    (
                        "**8-dimension OpenEnv rubric weights**, grouped by category "
                        "(decision / packet / process / terminal). 40% of reward sits on "
                        "decision + terminal — where economically irrational policies "
                        "bleed money fastest.",
                        "rubric_weights.png",
                    ),
                    (
                        "**Iter-5 per-difficulty curves.** Post-step-80 plateau is the "
                        "fallback heuristic across every difficulty band; see "
                        "SPECIFICATION_GAMING.md for the diagnosis.",
                        "training_curve_by_family.png",
                    ),
                ):
                    src = _figure_data_uri(fname)
                    if src is None:
                        gr.Markdown(
                            f"_(figure `{fname}` not bundled — see "
                            f"[`docs/figures/{fname}`](https://github.com/MitudruDutta/chargebackops/blob/main/docs/figures/{fname}))_"
                        )
                        continue
                    gr.Markdown(caption)
                    gr.HTML(
                        f'<img src="{src}" style="width:100%;max-width:1100px;'
                        f'border:1px solid #2a2a2a;border-radius:6px;margin:6px 0;" '
                        f'alt="{fname}" />'
                    )

    return demo


# ---------------------------------------------------------------------------
# Tab content builders (called once at app build; keep cheap)
# ---------------------------------------------------------------------------


def _environment_tab_markdown() -> str:
    """Render the Environment tab content from live constants.

    Reads action types from ``core.models.ActionType`` and the rubric weights
    from ``evaluation.rubrics.CASE_DIMENSION_WEIGHTS`` so this tab can never
    drift from the source of truth.
    """

    try:
        from core.models import ActionType  # type: ignore[attr-defined]
    except ImportError:  # pragma: no cover
        from ..core.models import ActionType  # type: ignore[attr-defined]

    # ``Literal`` exposes its members via ``__args__``.
    actions: tuple[str, ...] = tuple(getattr(ActionType, "__args__", ()))
    n_actions = len(actions)

    r1 = (
        "select_case", "inspect_case", "query_system", "retrieve_policy",
        "add_evidence", "remove_evidence", "set_strategy",
        "submit_representment", "resolve_case",
    )
    r23 = ("respond_to_pre_arb", "escalate_to_arbitration", "accept_arbitration_loss")
    long_horizon = ("wait_for_updates",)

    def _join(items: tuple[str, ...]) -> str:
        return " &middot; ".join(f"`{name}`" for name in items)

    rubric_rows = "\n".join(
        f"| {label} | {int(round(weight * 100))}% | {scoring} |"
        for label, weight, scoring in zip(
            _DIMENSION_LABELS, CASE_DIMENSION_WEIGHTS, _DIMENSION_SCORING
        )
    )

    return (
        f"## Action Space ({n_actions} typed actions)\n\n"
        f"**Round 1 — Representment:** {_join(r1)}\n\n"
        f"**Round 2/3 — Pre-arb &amp; Arbitration:** {_join(r23)}\n\n"
        f"**Long-horizon backlog:** {_join(long_horizon)}\n\n"
        "## Merchant Systems (6)\n\n"
        "`orders` &middot; `payment` &middot; `shipping` &middot; "
        "`support` &middot; `refunds` &middot; `risk`\n\n"
        "## Grading (8 dimensions)\n\n"
        "Weights are read live from `evaluation.rubrics.CASE_DIMENSION_WEIGHTS`.\n\n"
        "| Dimension | Weight | Scoring |\n"
        "|---|---|---|\n"
        f"{rubric_rows}\n\n"
        "## Scripted policies (Run Episode tab)\n\n"
        "| Policy | What it does | Headline avg |\n"
        "|---|---|---|\n"
        "| `naive` | Submit empty packet, no evidence, no policy work | 0.000 |\n"
        "| `concede_all` | Always set strategy `accept_chargeback` and resolve | 0.444 |\n"
        "| `escalate_all` | Contest like the heuristic, then always escalate | 0.767 |\n"
        "| `heuristic` | EV-rational, fully offline | **0.813** |\n\n"
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


def _rubric_tree_html() -> str:
    """Render the live ``env.rubric.named_rubrics()`` tree as nested HTML.

    Also explicitly surfaces the deadline ``Gate(CaseAbandonedRubric)`` that
    sits on top of the per-case ``WeightedSum`` — OpenEnv's default walk
    iterates registered child rubrics only, and the Gate is a sibling of the
    aggregator inside :class:`CaseRubric`.

    Falls back to a static snapshot if introspection fails for any reason
    (e.g. an old OpenEnv build) so the demo never breaks on this tab.
    """

    try:
        env = ChargebackOpsEnvironment()
        named = list(env.rubric.named_rubrics())
    except Exception as exc:  # pragma: no cover — defensive fallback
        return (
            f"<pre style='color:#ef4444;'>Could not introspect rubric tree: "
            f"{type(exc).__name__}: {exc}</pre>"
        )

    # Map weights onto leaf rubrics by name. CASE_DIMENSION_NAMES is the
    # canonical order the WeightedSum was built with; weights align by index.
    weight_by_dim = dict(zip(CASE_DIMENSION_NAMES, CASE_DIMENSION_WEIGHTS))

    rows: list[str] = []
    rows.append(
        "<table class='queue-table' style='font-family:ui-monospace,monospace;'>"
        "<tr><th>Path</th><th>Class</th><th>Weight / Role</th></tr>"
    )

    # Explicitly inject the deadline gate row above the aggregator subtree,
    # since some OpenEnv versions don't yield it via named_rubrics().
    deadline_gate_injected = False
    for path, rubric in named:
        cls_name = type(rubric).__name__
        if (
            not deadline_gate_injected
            and cls_name == "WeightedSum"
            and path.endswith("aggregator")
        ):
            parent = path.rsplit(".", 1)[0]
            rows.append(
                f"<tr><td>{'&nbsp;' * (parent.count('.') * 4 + 4)}"
                f"<code>{parent}.deadline_gate</code></td>"
                f"<td>Gate(CaseAbandonedRubric)</td>"
                f"<td style='text-align:right;color:#eab308;'>hard-zero on miss</td></tr>"
            )
            deadline_gate_injected = True

        weight_str = "—"
        for dim_name, weight in weight_by_dim.items():
            tag = "".join(part.capitalize() for part in dim_name.split("_")) + "Rubric"
            if cls_name == tag:
                weight_str = f"{int(round(weight * 100))}%"
                break
        depth = path.count(".")
        indent = "&nbsp;" * (depth * 4)
        rows.append(
            f"<tr><td>{indent}<code>{path or '(root)'}</code></td>"
            f"<td>{cls_name}</td>"
            f"<td style='text-align:right;'>{weight_str}</td></tr>"
        )
    rows.append("</table>")
    return "".join(rows)


# ---------------------------------------------------------------------------
# Training Results helpers
# ---------------------------------------------------------------------------


def _figure_data_uri(filename: str) -> str | None:
    """Return a base64 ``data:image/png`` URI for a bundled figure, or None.

    Embedding figures inline avoids dependencies on the static-asset routing
    of whatever host serves the demo (HF Spaces, FastAPI sub-mount, etc.).
    """

    path = _FIGURES_DIR / filename
    if not path.is_file():
        return None
    try:
        data = path.read_bytes()
    except OSError:
        return None
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _training_tab_markdown() -> str:
    return (
        "## Real training, end-to-end\n\n"
        "**Pipeline.** Qwen2.5-3B fp16 + LoRA r=16 on a single Colab T4. Phase A is "
        "supervised fine-tuning on heuristic rollouts; Phase B is GRPO with an outcome-"
        "based reward (terminal $-PnL after the model's action plus a heuristic tail-"
        "rollout). The training loop **connects to the live `ChargebackOpsEnvironment`** "
        "— every gradient step is graded by the same rubric and same Issuer adversary "
        "the eval uses. There is no static dataset shortcut.\n\n"
        "**Five iterations, three failure modes.** Iter 1 produced total gradient "
        "collapse (group reward variance ≈ 0). Iter 3 broke through to non-zero gradient "
        "but plateaued at 0.728. **Iter 5 ran 200 GRPO steps and uncovered a reproducible "
        "specification-gaming exploit** where the model emits invalid `accept_case` "
        "actions, triggers the eval rollout helper's heuristic-fallback path, and "
        "scores bit-exactly the heuristic baseline at 0.8132. The full diagnosis is in "
        "[`SPECIFICATION_GAMING.md`](https://github.com/MitudruDutta/chargebackops/blob/main/docs/SPECIFICATION_GAMING.md).\n\n"
        "**Honest trained-vs-untrained delta:** the SFT step at 0.536 — **+0.08 absolute, "
        "+18% relative** over the untrained Qwen2.5-3B base — is the only legitimate "
        "model-attributable improvement on iter 5. We document this honestly because "
        "the failure mode itself is a research artefact future GRPO recipes can target "
        "as a benchmark.\n\n"
        "**Reproduce.** "
        "[Latest training run (Colab — iter 5, 200 GRPO steps)](https://colab.research.google.com/drive/1GtLH6_b10oHlAnnGq4hnBkcGJ-pE_za5?usp=sharing) · "
        "[Previous training run (Colab — iter 3, 62 GRPO steps)](https://colab.research.google.com/drive/1AjG3Sv7FnMeOSls6JMzTunkMzlJi_ySu?usp=sharing) · "
        "[`notebooks/train_merchant_agent.ipynb`](https://github.com/MitudruDutta/chargebackops/blob/main/notebooks/train_merchant_agent.ipynb)\n"
    )
