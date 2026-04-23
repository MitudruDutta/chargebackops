# ChargebackOps — Benchmark Results

Reference numbers for the 12-task headline catalog (5 showcase + 7 seeded
holdout) and the 28-task multi-seed stress grid against the current
multi-round adversarial environment. Reproduce with the commands at the
bottom; scores match to within ±1e-3 (float rounding).

Captured on **2026-04-22** on `main` with the 8-dimension case rubric
(weights `(0.20, 0.15, 0.10, 0.10, 0.10, 0.10, 0.05, 0.20)`,
`escalation_roi` dimension active) and the deterministic Issuer agent
(LLM softening disabled — benchmarks stay fully offline). The
`NoteQualityRubric` is the deterministic scorer; setting
`USE_LLM_NOTE_JUDGE=1` swaps in `LLMNoteJudgeRubric`, which falls back
to the deterministic path on any provider failure so these numbers also
hold with the flag set if no API key is configured.

## TL;DR

| Policy | Headline avg (12 tasks) | Multi-seed avg (28 tasks) | Provider calls |
| --- | --- | --- | --- |
| **naive** (empty packet → submit) | **0.0000** | **0.0000** | 0 |
| **concede_all** (always `accept_chargeback`) | **0.4435** | **0.4454** | 0 |
| **escalate_all** (contest, then always escalate) | **0.7668** | **0.7675** | 0 |
| **heuristic** (EV-rational rule-based pick) | **0.8132** | **0.7628** | 0 |

**Discrimination delta** (heuristic − naive) is **0.8132** on the headline
catalog and **0.7628** on the multi-seed grid — well above the 0.40 target.

The headline catalog now includes `monthly_dispute_backlog_marathon`, a
12-case / 60-step task with wave arrivals, delayed evidence, and delayed
Issuer reviews. It scores lower than the short tasks for every scripted
policy: heuristic 0.679, escalate_all 0.617, concede_all 0.400, naive
0.000. This is intentional: the task is the Theme #2 long-horizon stress
case, while the rest of the catalog keeps the original professional
chargeback mechanics.

## Score Curve by Difficulty (multi-seed grid, 7 seeds / difficulty)

| Difficulty | n | heuristic | escalate_all | concede_all | naive |
| --- | --- | --- | --- | --- | --- |
| easy | 7 | 0.887 | 0.924 | 0.270 | 0.000 |
| medium | 7 | 0.869 | 0.869 | 0.630 | 0.000 |
| hard | 7 | 0.755 | 0.737 | 0.491 | 0.000 |
| nightmare | 7 | 0.540 | 0.540 | 0.390 | 0.000 |

Observations:
- Heuristic score decreases monotonically with generated difficulty
  (0.89 → 0.87 → 0.76 → 0.54). The difficulty gradient is real.
- `escalate_all` beats heuristic on generated easy tasks because those
  generated cases are small and often reward aggressive clean-packet
  escalation. The fixed marathon and pre-arb showcase are what separate
  the EV-rational policy from blanket escalation in the headline catalog.
- `concede_all` collapses on easy (0.270) — small-amount easy cases
  are positive-EV contestable, so the EscalationROI rubric zeros out
  concedes. The gap narrows at nightmare (0.540 vs 0.390) because the
  15-step budget vs. 5-case portfolio forces the heuristic to forfeit
  cases deadline-wise, while conceding is cheap per case.
- `naive` sits flat at 0.000 because an empty packet fails the
  packet-validity gate and every case is scored as unresolved /
  abandoned.

## Headline Per-Task Table (12 tasks, offline)

| Task ID | Difficulty | heuristic | escalate_all | concede_all | naive |
| --- | --- | --- | --- | --- | --- |
| goods_not_received_easy | easy | 0.965 | 0.965 | 0.423 | 0.000 |
| fraud_signal_ambiguity | easy | 0.958 | 0.958 | 0.223 | 0.000 |
| pre_arb_recovery_medium | medium | 0.965 | 0.613 | 0.223 | 0.000 |
| queue_optimization_hard | hard | 0.926 | 0.926 | 0.554 | 0.000 |
| monthly_dispute_backlog_marathon | nightmare | 0.679 | 0.617 | 0.400 | 0.000 |
| generated_easy_s42 | easy | 0.843 | 0.743 | 0.333 | 0.000 |
| generated_medium_s17 | medium | 0.856 | 0.856 | 0.542 | 0.000 |
| generated_medium_s99 | medium | 0.758 | 0.758 | 0.620 | 0.000 |
| generated_hard_s7 | hard | 0.904 | 0.861 | 0.615 | 0.000 |
| generated_hard_s53 | hard | 0.662 | 0.662 | 0.483 | 0.000 |
| generated_nightmare_s31 | nightmare | 0.536 | 0.536 | 0.424 | 0.000 |
| generated_nightmare_s77 | nightmare | 0.708 | 0.708 | 0.484 | 0.000 |
| **Average** | | **0.8132** | **0.7668** | **0.4435** | **0.0000** |

(Per-task numbers from `runners.benchmark_runner.run_policy_sweep()`.)
The rows where heuristic > escalate_all (`pre_arb_recovery_medium`,
`monthly_dispute_backlog_marathon`, and `generated_hard_s7`) are the
cases where the issuer's round-1 rejection, delayed work, or negative-EV
pre-arb branch makes blanket escalation strictly worse.

## Training Curve (GRPO, 200 steps) — legacy first-attempt findings

This section documents the first failed GRPO attempt on the pre-marathon
catalog. It is useful as a failure analysis, not as the current learning
claim. The current notebook has been rewritten to use SFT + GRPO on
`Qwen/Qwen2.5-1.5B-Instruct`; rerun it before making any public claim
about trained-agent improvement.

First end-to-end GRPO run executed **2026-04-20** on a Colab T4 with
`Qwen/Qwen3.5-0.8B`, batch 4 × K=4 generations, 200 steps,
`max_completion_length=128`, `beta=0.0`, `gradient_checkpointing=True`.
Wall time ~52 min, peak VRAM 7.1 GB.

| Step | Mean score (legacy headline 11) | Notes |
| --- | --- | --- |
| 0   | 0.8234 | untrained Qwen3.5-0.8B |
| 50  | 0.8234 | GRPO checkpoint |
| 100 | 0.8234 | GRPO checkpoint |
| 150 | 0.8234 | GRPO checkpoint |
| 200 | 0.8234 | GRPO checkpoint |

**The curve is dead flat at 0.8234 — exactly the heuristic floor (0.8254
± float rounding). This is not noise; it's a complete training failure,
diagnosed below.** Reporting it as-is rather than as a placeholder
because the failure mode is itself a useful artefact.

### Why it failed (and the two fixes already merged)

1. **Truncated JSON ⇒ parse-fail ⇒ no reward variance.** Qwen3.5-0.8B
   chat-tuning makes it write very verbose `strategy` strings.
   `max_completion_length=128` cuts those mid-string. The original
   strict parser required a balanced `}`; truncated JSON returned
   `None`; `run_episode_with_text_policy` fell back to the scripted
   heuristic for **every** action; every K=4 completion in a GRPO group
   produced the same heuristic score; group advantage = 0; gradient = 0.
   Loss collapsed to ~1e-5 after 30 steps and stayed there.

2. **`<think>` blocks burned the rest of the budget.** The eval policy
   used the raw prompt, not `apply_chat_template`. Without
   `enable_thinking=False` Qwen3.5 emits `<think>...</think>` scratchpad
   first, which ate the remaining 64–128 generation tokens before any
   JSON appeared.

Both are now fixed in code (`training/env_adapter.py:101` —
`parse_completion` tolerates code fences, `<think>` blocks, prefix words
naming the action_type, and JSON truncated mid-string by closing at the
last balanced field; `notebooks/train_merchant_agent.ipynb` cell
`fc45953c` raises `max_completion_length` to 512 and the eval cell
applies the chat template with thinking off). Rerun the notebook
end-to-end to overwrite the table above with whatever GRPO actually does
once it has a non-zero learning signal.

### Per-family curve (multi-task RL view)

Section 9 of the notebook re-evaluates each checkpoint grouped by
difficulty (`easy`/`medium`/`hard`/`nightmare`) and overlays per-cohort
heuristic floors from the 28-task multi-seed grid. A healthy run shows
monotone gains in every family; a flat `nightmare` line with rising
`easy` is the overfit-to-cheap-tasks failure mode this view exists to
surface. On the first attempt above all four families collapsed onto
the heuristic line for the same parse-fail reason, so the figure is a
flat fan rather than a curve. Regenerate after the rerun.

(Figures `docs/figures/training_curve.png` and
`docs/figures/training_curve_by_family.png` will land here once the
notebook is re-run with the parser + chat-template fixes.)

## Ablation

| Agent | Mean score (legacy headline 11) | Notes |
| --- | --- | --- |
| **naive** (empty packet → submit) | **0.0000** | PacketValidity gate + EscalationROI vacuous penalty collapse the score |
| **concede_all** (always accept) | **0.4475** | Cheap, but EscalationROIRubric (20%) zeros out concedes on positive-EV contestable cases |
| **escalate_all** (contest, then escalate) | **0.7713** | Strong on cases where the issuer eventually accepts; pays $250 of arb fee on the pre-arb branch |
| **untrained Qwen3.5-0.8B** | **0.8234** | All completions parse-fail → episode driven by heuristic fallback. The 0.0020 gap from heuristic is float-rounding noise across the 11-task aggregate. |
| **heuristic** (EV-rational scripted) | **0.8254** | Strong scripted floor — the bar GRPO has to clear |
| **trained merchant** (GRPO step 200, first attempt) | **0.8234** | Identical to untrained — GRPO learned nothing because reward variance was zero (see Training Curve section for diagnosis). |

The ablation reads top-down: the benchmark gradient from naive → concede_all
→ escalate_all → heuristic is ~0.83 wide, which is the headroom the TRL
GRPO loop has to close. The first GRPO attempt failed to close any of it
— the trained-merchant row matches the untrained row exactly because
parse-fail kicked every action through to the scripted heuristic. The
parser + completion-budget fixes are merged; the next notebook run is
what will actually demonstrate (or refute) learning.

## Rubric Composition (what's wired)

```
ChargebackOpsEpisodeRubric
└── case_rubric: CaseRubric                       # iterates over task.cases, weighted by case.weight
    ├── deadline_gate: Gate(threshold=1.0)        # hard-zero if case abandoned past deadline
    │   └── CaseAbandonedRubric
    └── aggregator: WeightedSum                   # weights sum to 1.0
        ├── rubric_0: StrategyCorrectnessRubric   # 0.20
        ├── rubric_1: EvidenceQualityRubric       # 0.15
        ├── rubric_2: PacketValidityRubric        # 0.10
        ├── rubric_3: DeadlineComplianceRubric    # 0.10
        ├── rubric_4: EfficiencyRubric            # 0.10
        ├── rubric_5: OutcomeQualityRubric        # 0.10
        ├── rubric_6: NoteQualityRubric           # 0.05
        └── rubric_7: EscalationROIRubric         # 0.20
```

Every node is an OpenEnv `Rubric` subclass and every node exposes
`last_score` after forward. `env.rubric.named_rubrics()` walks the tree
and returns the hook-compatible surface for a judge or trainer to
introspect per-dimension scores.

`EscalationROIRubric` encodes the economic rule that escalating to
network arbitration is rational only when
`P(win) × dispute_amount > arb_fee` (fee = $250/side). Scripted policies
that escalate negative-EV cases (or concede positive-EV cases) are
penalised on this axis.

## Reproducing These Numbers

```bash
source ~/python/bin/activate

python - <<'PY'
from runners.benchmark_runner import run_policy_sweep, run_multi_seed

headline = run_policy_sweep()
print("HEADLINE (12 tasks)")
for s in headline.policies:
    print(f"  {s.policy:14s}  mean={s.mean_score:.4f}  stdev={s.stdev:.4f}")
print(f"  delta (heuristic - naive): {headline.discrimination_delta}")

grid = run_multi_seed(
    seeds=[7, 17, 31, 42, 53, 77, 99],
    difficulties=["easy", "medium", "hard", "nightmare"],
)
print("MULTI-SEED (28 tasks)")
for s in grid.policies:
    print(f"  {s.policy:14s}  mean={s.mean_score:.4f}  stdev={s.stdev:.4f}")
print(f"  delta (heuristic - naive): {grid.discrimination_delta}")
PY
```

Optional LLM-assisted baseline (requires `OPENROUTER_API_KEY`):

```bash
python -m runners.baseline_runner | tee /tmp/baseline_run.json
```

## Hardware / Environment

- Python 3.12, pytest 8.x
- `openenv-core`, `pydantic`, `openai` per `pyproject.toml`
- No provider calls for the four scripted policies — all results fully offline
- Full test suite: **107/107 passing** (env, grader, issuer, arbitration, escalation_roi, llm_softening, llm_note_judge, training adapters, marathon mechanics)

## What This Table Does Not Show

- **Per-dimension score dispersion across the full catalog** — the
  headline table aggregates to one scalar per task. Walk
  `env.rubric.named_rubrics()` on any run for the per-dimension
  introspection path.
- **LLM-trained merchant curves** — this environment is the substrate;
  training curves are produced separately by the TRL notebook.
- **Adversarial Issuer with LLM softening enabled** — softening is
  gated on API keys. With keys set, the Issuer can override the
  deterministic midpoint in the ambiguity band; that configuration is
  tested in `tests/test_llm_softening.py` but is not part of the
  offline benchmark numbers above.
