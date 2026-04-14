# ChargebackOps — Baseline Results

Reference numbers for the 10-task benchmark catalog. Captured on **2026-04-14** against
`main` (Rubric system + deadline `Gate` composition). Reproduce with the commands below; scores
should match to within ±1e-3 (float rounding).

## TL;DR

| Agent | Avg score | Best task | Worst task | Provider calls |
| --- | --- | --- | --- | --- |
| **Bad policy** (concede-everything) | **0.212** | `generated_medium_s99` (0.442) | `generated_nightmare_s77` (0.053) | 0 |
| **Heuristic** (no LLM, rule-based) | **0.738** | `goods_not_received_easy` / `fraud_signal_ambiguity` (0.968) | `generated_nightmare_s77` (0.445) | 0 |
| **Heuristic + LLM tiebreak** (openrouter gpt-oss-120b) | **0.622** | `goods_not_received_easy` (0.968) | `generated_nightmare_s31` (0.134) | 19 (19 ✓ / 0 ✗) |

**Key signal:** the bad policy vs. heuristic delta is **0.526** (73.8 → 21.2 = 248% spread). The
`Gate(CaseAbandonedRubric)` wrapper around the per-case `WeightedSum` means a case left unresolved
past its deadline hard-zeros — a lazy concede-everything agent cannot game the score, and a correct
agent cannot trivially saturate it on hard tasks.

## Score Curve by Difficulty

| Difficulty | Task count | Heuristic avg | LLM avg | Bad avg | Target band | Status |
| --- | --- | --- | --- | --- | --- | --- |
| easy | 3 | 0.964 | 0.778 | 0.365 | ≥ 0.90 | ✓ |
| medium | 2 | 0.755 | 0.608 | 0.278 | 0.50 – 0.85 | ✓ |
| hard | 3 | 0.680 | 0.697 | 0.113 | 0.50 – 0.75 | ✓ |
| nightmare | 2 | 0.466 | 0.289 | 0.065 | ≤ 0.55 | ✓ |

Observations:
- The heuristic **outscores** the LLM-assisted run by 11.6 points on this catalog. The LLM is only
  invoked to break ties between candidate actions; on the current task set the heuristic tiebreak is
  almost always the optimal choice, and the LLM occasionally picks a worse candidate (notably on
  `fraud_signal_ambiguity` and `generated_medium_s99`, where it dropped 0.56 and 0.29 respectively
  against the heuristic). The LLM recovers on `hard` tasks where genuine branching exists
  (`queue_optimization_hard`: +0.048 over heuristic).
- Nightmare tasks cluster around **0.45** for the heuristic because the 15-step budget collides with
  5-case portfolios that have deadline_step=3–5 per case. Missed deadlines that were *attempted* still
  land in the weighted sum (with 0 on the deadline dimension and ~0.55 from the other 85%); truly
  abandoned cases are zeroed by the `Gate(CaseAbandonedRubric)` wrapper. Not a scoring artifact: the
  bad-policy run shows the same tasks at ~0.065.
- The deadline `Gate` is the v1 upgrade over a flat weighted sum: a case never even attempted by
  the deadline collapses completely, while a case resolved late still earns dimensional credit for
  evidence, strategy, and packet quality. This matches real chargeback operations — a missed
  representment is "case forfeit," while a late one takes a penalty but is still scored on what
  the merchant tried to do.

## Full Per-Task Table

| Task ID | Difficulty | Cases | Heuristic | H steps | LLM | LLM steps | Bad | Bad steps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| goods_not_received_easy | easy | 1 | 0.968 | 6 | 0.968 | 6 | 0.280 | 3 |
| fraud_signal_ambiguity | easy | 1 | 0.968 | 7 | 0.408 | 6 | 0.408 | 3 |
| generated_easy_s42 | easy | 1 | 0.958 | 7 | 0.958 | 7 | 0.408 | 3 |
| generated_medium_s17 | medium | 2 | 0.809 | 10 | 0.809 | 10 | 0.114 | 12 |
| generated_medium_s99 | medium | 2 | 0.701 | 9 | 0.406 | 9 | 0.442 | 12 |
| queue_optimization_hard | hard | 3 | 0.802 | 12 | 0.850 | 11 | 0.129 | 15 |
| generated_hard_s7 | hard | 2 | 0.718 | 5 | 0.718 | 5 | 0.120 | 12 |
| generated_hard_s53 | hard | 3 | 0.522 | 6 | 0.522 | 6 | 0.089 | 15 |
| generated_nightmare_s31 | nightmare | 5 | 0.486 | 15 | 0.134 | 15 | 0.077 | 15 |
| generated_nightmare_s77 | nightmare | 5 | 0.445 | 15 | 0.445 | 15 | 0.053 | 15 |
| **Average** | | | **0.738** | 9.2 | **0.622** | 9.0 | **0.212** | 10.5 |

`fraud_signal_ambiguity` has been relabeled from `medium` to `easy` — it scores consistently at 0.968
under the heuristic because the correct action collapses to a single-case contest with attached
policy evidence. The label-difficulty mismatch was flagged in v1 and fixed in this release.

## Rubric Breakdown (single-case sanity check)

For `goods_not_received_easy` under the heuristic, the 7-dimension breakdown from
`ChargebackOpsEpisodeRubric` (weights sum to 1.0, Gate passes because the case was resolved before
step 8):

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

# 1. Run the heuristic + bad-policy comparison (no network)
python - <<'PY'
from evaluation.agent_brutal_audit import run_episode
from scenarios.simulation import list_tasks
for t in list_tasks():
    h = run_episode(t.task_id, policy='heuristic')
    b = run_episode(t.task_id, policy='bad')
    print(f"{t.task_id:32s}  heur={h['score']:.4f}  bad={b['score']:.4f}")
PY

# 2. Run the baseline with a real LLM (requires OPENROUTER_API_KEY in .env)
python -m runners.baseline_runner | tee /tmp/baseline_run.json
```

## Hardware / Environment

- Python 3.12.13, pytest 7.4.3
- `openenv-core==0.2.3`, `pydantic==2.12.5`, `openai==2.31.0`
- Provider: OpenRouter (model `openai/gpt-oss-120b`), all 19 decision calls succeeded, zero retries
- Average end-to-end episode wall-clock: ~0.8s (heuristic), ~2.5s (with LLM tiebreak)
- Full test suite: 22/22 passing, `openenv validate .` clean, Docker build clean

## What This Table Does Not Show

- **Single-seed-per-task** — generated tasks use a fixed seed. A statistically rigorous eval would
  run each task across 10+ seeds. The fixed-seed catalog is intentional for the hackathon (direct
  score comparison between submissions), but is flagged as a scale-up item for v1.1.
- **Per-dimension score dispersion across the full catalog** — the table above shows one task's
  breakdown. An introspection demo command exists for walking `env.rubric.named_rubrics()` on any
  run: see `README.md` → "Rubric introspection".
- **RL training curves** — ChargebackOps is a ready environment, not a trained agent. Anyone
  wiring this into Gym/SB3/CleanRL is expected to produce training curves separately; the rubric
  tree is the machinery they would hook into for credit assignment.
