# ChargebackOps — Benchmark Results

Reference numbers for the 11-task headline catalog (4 showcase + 7 seeded
holdout) and the 28-task multi-seed stress grid against the current
multi-round adversarial environment. Reproduce with the commands at the
bottom; scores match to within ±1e-3 (float rounding).

Captured on **2026-04-20** on `main` with the 8-dimension case rubric
(weights `(0.20, 0.15, 0.10, 0.10, 0.10, 0.10, 0.05, 0.20)`,
`escalation_roi` dimension active) and the deterministic Issuer agent
(LLM softening disabled — benchmarks stay fully offline). The
`NoteQualityRubric` is the deterministic scorer; setting
`USE_LLM_NOTE_JUDGE=1` swaps in `LLMNoteJudgeRubric`, which falls back
to the deterministic path on any provider failure so these numbers also
hold with the flag set if no API key is configured.

## TL;DR

| Policy | Headline avg (11 tasks) | Multi-seed avg (28 tasks) | Provider calls |
| --- | --- | --- | --- |
| **naive** (empty packet → submit) | **0.0000** | **0.0000** | 0 |
| **concede_all** (always `accept_chargeback`) | **0.4475** | **0.4454** | 0 |
| **escalate_all** (contest, then always escalate) | **0.7713** | **0.7532** | 0 |
| **heuristic** (EV-rational rule-based pick) | **0.8254** | **0.7628** | 0 |

**Discrimination delta** (heuristic − naive) is **0.8254** on the headline
catalog and **0.7628** on the multi-seed grid — well above the 0.40 target.

The heuristic now beats `escalate_all` by **+0.054** on the headline
catalog because `pre_arb_recovery_medium` deliberately spreads the two
policies apart: heuristic 0.965, escalate_all 0.613, concede_all 0.223.
Outside that case the merchant's round-1 packet is strong enough that
the pre-arb branch never fires and the two scripted policies produce
identical trajectories — that match on the other tasks is a signal, not
a bug. `concede_all` collapses to 0.45 because `EscalationROIRubric`
zeros out concedes on positive-EV contestable cases (`amount > $250`).

## Score Curve by Difficulty (multi-seed grid, 7 seeds / difficulty)

| Difficulty | n | heuristic | escalate_all | concede_all | naive |
| --- | --- | --- | --- | --- | --- |
| easy | 7 | 0.887 | 0.866 | 0.270 | 0.000 |
| medium | 7 | 0.869 | 0.869 | 0.630 | 0.000 |
| hard | 7 | 0.755 | 0.737 | 0.491 | 0.000 |
| nightmare | 7 | 0.540 | 0.540 | 0.390 | 0.000 |

Observations:
- Heuristic score decreases monotonically with difficulty
  (0.89 → 0.87 → 0.76 → 0.54). The difficulty gradient is real.
- Heuristic edges out `escalate_all` on easy (+0.021) and hard (+0.018)
  because the EV-rational policy catches the rare positive-EV pre-arb
  branch where blanket escalation overspends $250 in arb fees.
- `concede_all` collapses on easy (0.270) — small-amount easy cases
  are positive-EV contestable, so the EscalationROI rubric zeros out
  concedes. The gap narrows at nightmare (0.540 vs 0.390) because the
  15-step budget vs. 5-case portfolio forces the heuristic to forfeit
  cases deadline-wise, while conceding is cheap per case.
- `naive` sits flat at 0.000 because an empty packet fails the
  packet-validity gate and every case is scored as unresolved /
  abandoned.

## Headline Per-Task Table (11 tasks, offline)

| Task ID | Difficulty | heuristic | escalate_all | concede_all | naive |
| --- | --- | --- | --- | --- | --- |
| goods_not_received_easy | easy | 0.965 | 0.965 | 0.423 | 0.000 |
| fraud_signal_ambiguity | easy | 0.958 | 0.958 | 0.223 | 0.000 |
| pre_arb_recovery_medium | medium | 0.965 | 0.613 | 0.223 | 0.000 |
| queue_optimization_hard | hard | 0.926 | 0.926 | 0.554 | 0.000 |
| generated_easy_s42 | easy | 0.843 | 0.643 | 0.333 | 0.000 |
| generated_medium_s17 | medium | 0.856 | 0.856 | 0.542 | 0.000 |
| generated_medium_s99 | medium | 0.758 | 0.758 | 0.620 | 0.000 |
| generated_hard_s7 | hard | 0.904 | 0.861 | 0.615 | 0.000 |
| generated_hard_s53 | hard | 0.662 | 0.662 | 0.483 | 0.000 |
| generated_nightmare_s31 | nightmare | 0.536 | 0.536 | 0.424 | 0.000 |
| generated_nightmare_s77 | nightmare | 0.708 | 0.708 | 0.484 | 0.000 |
| **Average** | | **0.8254** | **0.7713** | **0.4475** | **0.0000** |

(Per-task numbers from `runners.benchmark_runner.run_policy_sweep()`.)
The three rows where heuristic > escalate_all (`pre_arb_recovery_medium`,
`generated_easy_s42`, `generated_hard_s7`) are the cases where the
issuer's round-1 rejection plus a negative-EV pre-arb branch would have
made blanket escalation strictly worse. On the other 8 rows the issuer
accepts in round 1 and the two policies produce identical trajectories.

## Training Curve (GRPO, 200 steps) — placeholder

> ⚠️ **The numbers in this section are placeholders.** They are illustrative
> targets, not measured values. The real GRPO run is queued for a Colab T4
> session; until that lands, treat the figure and the table below as the
> shape we expect rather than what we observed. Regenerate both by running
> `notebooks/train_merchant_agent.ipynb` end-to-end and re-rendering this
> table from the printed checkpoint scores.

![Training curve](figures/training_curve.png)

Baselines drawn as dashed lines: `heuristic`, `concede_all`, `naive`.

### Per-family curve (multi-task RL view)

The aggregate curve hides where improvement actually lands. The notebook's
section 9 re-evaluates each checkpoint grouped by difficulty
(`easy`/`medium`/`hard`/`nightmare`) and overlays per-cohort heuristic
floors from the 28-task multi-seed grid. A healthy run shows monotone
gains in every family; a flat `nightmare` line with rising `easy` is the
overfit-to-cheap-tasks failure mode the grouped view exists to surface.

![Training curve by family](figures/training_curve_by_family.png)

| Step | Mean score (headline) | Source |
| --- | --- | --- |
| 0   | _placeholder_ | untrained Qwen3.5-0.8B |
| 50  | _placeholder_ | GRPO checkpoint |
| 100 | _placeholder_ | GRPO checkpoint |
| 150 | _placeholder_ | GRPO checkpoint |
| 200 | _placeholder_ | GRPO checkpoint |

## Ablation

| Agent | Mean score (headline 11) | Notes |
| --- | --- | --- |
| **naive** (empty packet → submit) | **0.0000** | PacketValidity gate + EscalationROI vacuous penalty collapse the score |
| **concede_all** (always accept) | **0.4475** | Cheap, but EscalationROIRubric (20%) zeros out concedes on positive-EV contestable cases |
| **escalate_all** (contest, then escalate) | **0.7713** | Strong on cases where the issuer eventually accepts; pays $250 of arb fee on the pre-arb branch |
| **untrained base model** | _placeholder_ | Curve step 0; not yet measured |
| **heuristic** (EV-rational scripted) | **0.8254** | Strong scripted floor — the bar GRPO has to clear |
| **trained merchant** (step 200) | _placeholder_ | Will overwrite after the Colab T4 run completes |

The ablation reads top-down: the benchmark gradient from naive → concede_all
→ escalate_all → heuristic is ~0.83 wide, which is the headroom the TRL
GRPO loop has to close. The two `_placeholder_` rows are honest holes — they will be
filled in once the notebook run produces real numbers. Until then, do
not cite them as evidence of training performance.

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
print("HEADLINE (10 tasks)")
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
- Full test suite: **86/86 passing** (env, grader, issuer, arbitration, escalation_roi, llm_softening, llm_note_judge, training_curve)

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
