# ChargebackOps — Baseline Results

Reference numbers for the 10-task headline benchmark and the 28-task multi-seed stress grid.
Captured on **2026-04-15** against `main` (Rubric system + `Gate(CaseAbandonedRubric)`
composition, tightened `acceptable_strategies` on contest-optimal templates, expanded
`_obvious_next_action` coverage, improved LLM prompt). Reproduce with the commands at the
bottom; headline scores match to within ±1e-3 (float rounding).

## TL;DR

| Agent | Avg score | Best task | Worst task | Provider calls |
| --- | --- | --- | --- | --- |
| **Bad policy** (concede-everything) | **0.199** | `generated_medium_s99` (0.442) | `generated_nightmare_s77` (0.053) | 0 |
| **Heuristic** (no LLM, rule-based) | **0.724** | `goods_not_received_easy` / `fraud_signal_ambiguity` (0.968) | `generated_hard_s53` (0.440) | 0 |
| **Heuristic + LLM tiebreak** (openrouter gpt-oss-120b) | **0.729** | `goods_not_received_easy` / `fraud_signal_ambiguity` / `generated_easy_s42` (0.958) | `generated_hard_s53` (0.440) | 7 (7 ✓ / 0 ✗) |

**Key signal:** the bad policy vs. heuristic delta is **0.525** (72.4 → 19.9 = 264% spread).
The `Gate(CaseAbandonedRubric)` wrapper around the per-case `WeightedSum` means a case left
unresolved past its deadline hard-zeros — a lazy concede-everything agent cannot game the score,
and a correct agent cannot trivially saturate it on hard tasks. The LLM-assisted run now edges
ahead of the pure heuristic (+0.005) after the v1.1 prompt and `_obvious_next_action` upgrades;
the LLM is invoked only **7 times** across the 10-task run (down from 19 in v1) because
deterministic workflow states are now dispatched without a model call.

## Score Curve by Difficulty

| Difficulty | Task count | Heuristic avg | LLM avg | Bad avg | Target band | Status |
| --- | --- | --- | --- | --- | --- | --- |
| easy | 3 | 0.964 | 0.964 | 0.323 | ≥ 0.90 | ✓ |
| medium | 2 | 0.755 | 0.755 | 0.278 | 0.50 – 0.85 | ✓ |
| hard | 3 | 0.635 | 0.651 | 0.113 | 0.50 – 0.75 | ✓ |
| nightmare | 2 | 0.466 | 0.466 | 0.065 | ≤ 0.55 | ✓ |

Observations:
- The LLM-assisted run now **matches or narrowly beats** the heuristic on every difficulty band
  (overall +0.005). The old v1 regression — where the LLM dropped 0.56 on `fraud_signal_ambiguity`
  and 0.29 on `generated_medium_s99` — was caused by the model picking a concede strategy over
  contest at `set_strategy` time. `_obvious_next_action` now short-circuits all strategy picks
  so the heuristic-derived strategy is used directly, and the prompt explicitly lists the
  reason-code → optimal-strategy mapping for the remaining decision points. Provider call count
  fell from 19 to 7 because deterministic housekeeping (add_evidence, remove_evidence,
  submit_representment, set_strategy, resolve_case) is now bypassed entirely.
- The LLM's remaining upside is on `queue_optimization_hard` (+0.049 over heuristic), where the
  queue-triage branching is genuine and the heuristic's fixed priority order leaves marginal
  value on the table.
- Nightmare tasks cluster around **0.47** for the heuristic because the 15-step budget collides
  with 5-case portfolios that have deadline_step=3–5 per case. Missed deadlines that were
  *attempted* still land in the weighted sum (with 0 on the deadline dimension and ~0.55 from
  the other 85%); truly abandoned cases are zeroed by the `Gate(CaseAbandonedRubric)` wrapper.
  Not a scoring artifact: the bad-policy run shows the same tasks at ~0.065.
- The deadline `Gate` is the v1 upgrade over a flat weighted sum: a case never even attempted by
  the deadline collapses completely, while a case resolved late still earns dimensional credit
  for evidence, strategy, and packet quality. This matches real chargeback operations — a missed
  representment is "case forfeit," while a late one takes a penalty but is still scored on what
  the merchant tried to do.

## Full Per-Task Table

| Task ID | Difficulty | Cases | Heuristic | H steps | LLM | LLM steps | Bad | Bad steps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| goods_not_received_easy | easy | 1 | 0.968 | 6 | 0.968 | 6 | 0.280 | 3 |
| fraud_signal_ambiguity | easy | 1 | 0.968 | 7 | 0.968 | 7 | 0.280 | 3 |
| generated_easy_s42 | easy | 1 | 0.958 | 7 | 0.958 | 7 | 0.408 | 3 |
| generated_medium_s17 | medium | 2 | 0.809 | 10 | 0.809 | 10 | 0.114 | 12 |
| generated_medium_s99 | medium | 2 | 0.701 | 9 | 0.701 | 9 | 0.442 | 12 |
| queue_optimization_hard | hard | 3 | 0.802 | 12 | 0.850 | 11 | 0.129 | 15 |
| generated_hard_s7 | hard | 2 | 0.663 | 5 | 0.663 | 5 | 0.120 | 12 |
| generated_hard_s53 | hard | 3 | 0.440 | 6 | 0.440 | 6 | 0.089 | 15 |
| generated_nightmare_s31 | nightmare | 5 | 0.486 | 15 | 0.486 | 15 | 0.077 | 15 |
| generated_nightmare_s77 | nightmare | 5 | 0.445 | 15 | 0.445 | 15 | 0.053 | 15 |
| **Average** | | | **0.724** | 9.2 | **0.729** | 9.0 | **0.199** | 10.5 |

## Multi-seed Stress Grid (7 seeds × 4 difficulties)

Running the heuristic and bad-policy agents across seven generator seeds per difficulty (seeds
7, 17, 31, 42, 53, 77, 99) gives the statistically defensible version of the headline numbers.
All runs are fully offline — no provider calls involved.

| Difficulty | n | Heuristic mean ± std | Bad mean ± std |
| --- | --- | --- | --- |
| easy | 7 | 0.9696 ± 0.014 | 0.3346 ± 0.068 |
| medium | 7 | 0.8411 ± 0.089 | 0.4369 ± 0.238 |
| hard | 7 | 0.6245 ± 0.151 | 0.1299 ± 0.047 |
| nightmare | 7 | 0.4121 ± 0.079 | 0.0635 ± 0.010 |
| **OVERALL** | **28** | **0.7118 ± 0.235** | **0.2412 ± 0.194** |

Observations:
- Heuristic score decreases cleanly and monotonically with difficulty: 0.97 → 0.84 → 0.62 →
  0.41. The difficulty gradient is real — not a labeling artifact.
- Nightmare std is the tightest (0.079) because every nightmare task is constrained by the
  same step budget vs. case count collision. Hard is the widest (0.151) because case counts
  vary from 2 to 3 across seeds.
- Bad policy shows wide variance on medium (±0.238) because some medium seeds generate
  concede-optimal templates (credit_not_processed, duplicate_processing) where
  concede-everything is trivially correct — exactly the expected behavior of a discriminating
  rubric on a mixed task distribution.
- Overall delta (heuristic − bad) across 28 runs: **0.4706**. The headline 10-task catalog
  delta (0.525) is within 1σ of the multi-seed delta, so the fixed-seed headline is not a
  cherry-picked result.

## Rubric Breakdown (single-case sanity check)

For `goods_not_received_easy` under the heuristic, the 7-dimension breakdown from
`ChargebackOpsEpisodeRubric` (weights sum to 1.0, Gate passes because the case was resolved
before step 8):

| Dimension | Weight | Score | Weighted contribution |
| --- | --- | --- | --- |
| strategy_correctness | 0.25 | 1.00 | 0.2500 |
| evidence_quality | 0.20 | 0.90 | 0.1800 |
| packet_validity | 0.15 | 1.00 | 0.1500 |
| deadline_compliance | 0.15 | 1.00 | 0.1500 |
| efficiency | 0.10 | 0.95 | 0.0950 |
| outcome_quality | 0.10 | 1.00 | 0.1000 |
| note_quality | 0.05 | 0.85 | 0.0425 |
| **Total** | **1.00** | — | **0.9675** |

Per-dimension scores captured by reading `rubric.last_score` on every child in the
`ChargebackOpsEpisodeRubric.case_rubric.aggregator` tree after one forward pass — exactly the
introspection path an RL trainer would use for credit assignment. The small gaps
(`evidence_quality=0.90`, `efficiency=0.95`, `note_quality=0.85`) are the real headroom an
LLM-fine-tuned agent is expected to close.

## Rubric Composition (what's wired)

```
ChargebackOpsEpisodeRubric
└── case_rubric: CaseRubric                       # iterates over task.cases, weighted by case.weight
    ├── deadline_gate: Gate(threshold=1.0)        # hard-zero if case abandoned past deadline
    │   └── CaseAbandonedRubric
    └── aggregator: WeightedSum                   # weights sum to 1.0
        ├── rubric_0: StrategyCorrectnessRubric   # weight 0.25
        ├── rubric_1: EvidenceQualityRubric       # weight 0.20
        ├── rubric_2: PacketValidityRubric        # weight 0.15
        ├── rubric_3: DeadlineComplianceRubric    # weight 0.15 (dimension-level, not gate)
        ├── rubric_4: EfficiencyRubric            # weight 0.10
        ├── rubric_5: OutcomeQualityRubric        # weight 0.10
        └── rubric_6: NoteQualityRubric           # weight 0.05
```

Every node is an OpenEnv `Rubric` subclass and every node exposes `last_score` after forward.
`env.rubric.named_rubrics()` walks the tree and returns 11 named children — the hook-compatible
surface for a judge or trainer to introspect per-dimension scores.

## Reproducing These Numbers

```bash
# Activate the project's venv
source ~/python/bin/activate

# 1. Headline 10-task run (heuristic + bad policy, no network)
python - <<'PY'
from evaluation.agent_brutal_audit import run_episode
from scenarios.simulation import list_tasks
for t in list_tasks():
    h = run_episode(t.task_id, policy='heuristic')
    b = run_episode(t.task_id, policy='bad')
    print(f"{t.task_id:32s}  heur={h['score']:.4f}  bad={b['score']:.4f}")
PY

# 2. Multi-seed stress grid (28 runs across 7 seeds × 4 difficulties, no network)
python - <<'PY'
from statistics import mean, stdev
from evaluation.agent_brutal_audit import run_episode
for d in ("easy","medium","hard","nightmare"):
    hs, bs = [], []
    for s in (7, 17, 31, 42, 53, 77, 99):
        hs.append(run_episode(f"generated_{d}_s{s}", policy='heuristic')['score'])
        bs.append(run_episode(f"generated_{d}_s{s}", policy='bad')['score'])
    print(f"{d:10s} heur={mean(hs):.4f}±{stdev(hs):.4f}  bad={mean(bs):.4f}±{stdev(bs):.4f}")
PY

# 3. LLM tiebreak run (requires OPENROUTER_API_KEY in .env)
python -m runners.baseline_runner | tee /tmp/baseline_run.json
```

## Hardware / Environment

- Python 3.12.13, pytest 7.4.3
- `openenv-core==0.2.3`, `pydantic==2.12.5`, `openai==2.31.0`
- Provider: OpenRouter (model `openai/gpt-oss-120b`), all 7 decision calls succeeded, zero retries
- Average end-to-end episode wall-clock: ~0.8s (heuristic), ~1.8s (with LLM tiebreak — down from
  ~2.5s in v1 because `_obvious_next_action` bypasses most model calls)
- Full test suite: 22/22 passing, `openenv validate .` clean, Docker build clean

## What This Table Does Not Show

- **Per-dimension score dispersion across the full catalog** — the table above shows one task's
  breakdown. An introspection demo command exists for walking `env.rubric.named_rubrics()` on
  any run: see `README.md` → "Rubric introspection".
- **RL training curves** — ChargebackOps is a ready environment, not a trained agent. Anyone
  wiring this into Gym/SB3/CleanRL is expected to produce training curves separately; the
  rubric tree is the machinery they would hook into for credit assignment.
