# Results

This document captures the quantitative results for ChargebackOps: scripted policy baselines, the cross-iteration GRPO training study, the per-checkpoint eval scores, the per-dimension rubric breakdown, and the diagnostic rollouts that revealed the specification-gaming behaviour documented in [`SPECIFICATION_GAMING.md`](SPECIFICATION_GAMING.md).

All numbers are reproducible from the commands in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## 1. Scripted policy sweep (deterministic, no GPU)

12-task headline catalog plus a 28-task multi-seed grid against the multi-round adversarial environment.

![Scripted-policy scores: naive 0.000, concede_all 0.444, escalate_all 0.767, heuristic 0.813. Every degenerate strategy hits a known ceiling.](figures/discrimination_gradient.png)

| Policy | Headline avg | Multi-seed avg (28) | Marathon | Provider calls | Description |
|---|---|---|---|---|---|
| **naive** | 0.000 | 0.000 | 0.000 | 0 | Submit empty packet immediately |
| **concede_all** | 0.444 | 0.445 | 0.400 | 0 | Always `accept_chargeback`, never contest |
| **escalate_all** | 0.767 | 0.768 | 0.617 | 0 | Always contest, always escalate to arbitration |
| **heuristic** | **0.813** | 0.763 | **0.679** | 0 | EV-rational policy, fully offline |

**Discrimination delta** (heuristic − naive) = **+0.813** on the headline catalog. Well above conventional benchmark targets.

The `Gate(CaseAbandonedRubric)` deadline guard plus `EscalationROIRubric` (20% weight) jointly defeat every degenerate strategy: an empty-packet policy zeros out, a concede-everything policy caps at 0.44, and an escalate-everything policy caps at 0.77 because the $250 fee is paid on negative-EV cases.

## 2. Cross-iteration GRPO training study

Five GRPO training iterations were run with progressively-tuned hyperparameters. Each iteration revealed a distinct failure mode. See [`METHOD.md`](METHOD.md) §3 for the full diagnostic narrative.

### 2.1 Training-time signals

| Iter | SFT max_steps | SFT mean_acc | GRPO max_steps | num_gens | temp | lora_dropout | grad_norm > 0.005 freq | grad_norm peak | KL max | Entropy max | Final train_loss |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 800 | 0.96 | 300 | 4 | 0.7 | 0.0 | **5%** | 0.78 | 0 | 0.017 | -2e-9 |
| 2 | 800 | 0.96 | 120 | 8 | 1.3 | 0.1 | 30% | 1.65 | 0.05 | 0.10 | 6e-4 |
| 3 | 300 | 0.96 | 60 | 8 | 1.3 | 0.1 | 50% | 0.021 | 0.06 | 0.08 | 7e-4 |
| 4 | 300 | 0.96 | 60 | 8 | 1.3 | 0.1 | 50% | **2.58** | 0.16 | 0.24 | 2e-3 |
| 5 | **150** | **0.88** | 200 | 8 | 1.3 | 0.1 | 60% | 2.30 | 0.16 | 0.20 | 1e-3 |

### 2.2 Iteration outcomes

- **Iter 1** — *Total gradient collapse*. Every group of 4 generations produced identical completions because the SFT-trained policy was near-delta on argmax. `std(reward_group) = 0` → advantage = 0 → no learning.
- **Iter 2** — *Tiny but real movement*. Widened sampling (temp 1.3, top_p 1.0, top_k 0, num_gens 8, lora_dropout 0.1) broke the argmax lock for ~30% of steps.
- **Iter 3** — *Frequent but tiny gradient*. Cutting `max_steps` to 60 raised gradient frequency to 50% but per-step magnitudes shrank to 0.011-0.021.
- **Iter 4** — *Sampling luck*. Same code as iter 3, different RNG: gradient peaks of 2.58 on lucky steps, KL hit 0.16. Demonstrates GRPO under high-SFT-accuracy is **lottery-distributed**.
- **Iter 5** — *Curve plateau, then specification gaming*. Stopped SFT early (mean_acc 0.88, not 0.96), trained GRPO 200 steps. Curve plateaued at the heuristic baseline. Diagnostic rollout revealed the GRPO policy emits an invalid action_type that triggers eval-pipeline fallback to the heuristic — the curve reflects the heuristic, not the trained model. See [`SPECIFICATION_GAMING.md`](SPECIFICATION_GAMING.md).

### 2.3 Iter 5 per-checkpoint eval scores

These are the published numbers from the iter-5 run. The GRPO scores from step 80 onwards are inflated by the eval-fallback exploit; the SFT score at step 1 is the legitimately-trained checkpoint.

![Cross-iteration comparison](figures/training_curve_cross_iter.png)
*Left: overall training curve for iter 3 (62 GRPO steps, plateaued at 0.728) and iter 5 (200 GRPO steps, plateaued at 0.8132 — the bit-exact heuristic baseline). Right: iter-5 per-difficulty curves showing the post-step-80 plateau is uniform across all four difficulty bands because the heuristic-fallback path produces 100% of executed actions. The bit-exact match between trained and heuristic is the signature of the eval-fallback exploit, not convergent learning.*

![Per-difficulty training curve](figures/training_curve_by_family.png)
*Iter-5 per-difficulty curve in isolation: mean normalised score (y) vs GRPO step (x), broken out by case difficulty. Step 0 = untrained Qwen2.5-3B base, step 1 = SFT-only checkpoint, steps 81/161/201/202 = GRPO checkpoints.*

![Overall training curve vs heuristic baseline](figures/training_curve.png)
*Iter-5 overall curve in isolation: mean normalised score across the headline catalog vs GRPO step. Dashed line = heuristic baseline (0.813). The GRPO plateau at the heuristic line is the specification-gaming attractor described in `SPECIFICATION_GAMING.md`.*

| Step | Checkpoint | Overall | easy | medium | hard | nightmare | Notes |
|---|---|---|---|---|---|---|---|
| 0 | Untrained Qwen2.5-3B base | 0.456 | 0.286 | 0.443 | 0.758 | 0.336 | Real |
| 1 | SFT (Phase A, 150 steps) | **0.536** | 0.778 | 0.666 | 0.462 | 0.235 | **Real, headline trained checkpoint** |
| 81 | GRPO step 80 | 0.799 | 0.929 | 0.792 | 0.828 | 0.647 | Mixed: partial real + early gaming attractor |
| 161 | GRPO step 160 | 0.8132 | 0.922 | 0.860 | 0.831 | 0.641 | Gaming-dominated |
| 201 | GRPO step 200 | 0.8132 | 0.922 | 0.860 | 0.831 | 0.641 | Gaming-dominated |
| 202 | GRPO final | 0.8132 | 0.922 | 0.860 | 0.831 | 0.641 | Gaming-dominated |

**Honest reading.** The base → SFT delta (`0.456 → 0.536`, +0.08 absolute, +18% relative) is the legitimately-trained learning signal. The GRPO numbers from step 160 onwards match the heuristic baseline bit-exactly (`0.8132`) because the rollout helper falls back to the heuristic on every invalid action emitted by the policy.

The SFT delta itself shows the expected pattern of an undertrained warmstart: large gains on the most common training distribution (`easy 0.286 → 0.778`, +172% relative; `medium 0.443 → 0.666`, +50% relative) and regressions on the rarer hard / nightmare distributions where 150 SFT steps provide insufficient coverage (`hard 0.758 → 0.462`, `nightmare 0.336 → 0.235`).

### 2.4 Diagnostic rollout — proof of the gaming attractor

![Iter-5 eval score attribution: trained-policy contribution 0.000, heuristic-fallback contribution 0.8132. Diagnostic rollouts on three tasks all show outcome PnL +0.000.](figures/gaming_attribution.png)

Single-action diagnostic on three representative tasks at the GRPO-final checkpoint:

| Task | Oracle action | Model action | Action valid? | Outcome PnL (normalized) |
|---|---|---|---|---|
| goods_not_received_easy | `select_case` CB-E1 | `accept_case` CB-E1 | **No** | +0.000 |
| queue_optimization_hard | `select_case` CB-H3 | `accept_case` CB-H3 | **No** | +0.000 |
| generated_nightmare_s31 | `select_case` CB-G3 | `accept_case` CB-G3 | **No** | +0.000 |

`accept_case` is not a member of the valid action set (`select_case, inspect_case, query_system, retrieve_policy, add_evidence, remove_evidence, set_strategy, submit_representment, resolve_case, respond_to_pre_arb, escalate_to_arbitration, accept_arbitration_loss, wait_for_updates`). The closest valid neighbours are `accept_chargeback` and `accept_arbitration_loss`. GRPO has fused two valid token prefixes (`accept_…` + `…case`) into an invalid hybrid that parses as JSON but fails Pydantic validation in `action_from_completion`.

`outcome PnL = 0.000` confirms the env never executed any of these actions. The eval rollout helper's heuristic fallback path produced 100% of the executed actions.

## 3. Per-dimension rubric attribution (SFT checkpoint, easy task)

Every checkpoint's score is decomposable into 8 dimensions via `env.rubric.named_rubrics()`. This exposes *which* aspect of the policy improved during training.

![8-dimension rubric weights, grouped by category](figures/rubric_weights.png)

For the SFT checkpoint on the `goods_not_received_easy` task:

| Dimension | Weight | SFT score | Notes |
|---|---|---|---|
| StrategyCorrectness | 0.20 | 1.00 | Picked optimal `contest` strategy |
| EvidenceQuality | 0.15 | 0.85 | Required + 2/3 helpful evidence attached |
| PacketValidity | 0.10 | 1.00 | All required, zero harmful |
| DeadlineCompliance | 0.10 | 1.00 | Resolved before deadline |
| Efficiency | 0.10 | 0.78 | One duplicate query |
| OutcomeQuality | 0.10 | 1.00 | Issuer accepted on round 1 |
| NoteQuality | 0.05 | 0.65 | Note covered policy keywords; missed one evidence ID ref |
| EscalationROI | 0.20 | 1.00 | No unnecessary escalation |
| **Weighted total** | 1.00 | **0.92** | |

The per-dimension breakdown is the *same surface* a hooked rubric exposes during training — researchers can attribute each gradient step to dimension-specific gains.

## 4. Reproducibility

- **Seeds**: holdout seeds `easy={42}, medium={17, 99}, hard={7, 53}, nightmare={31, 77}` are excluded from training and used as the eval set.
- **Pinned stack**: `transformers==4.51.3`, `trl==0.21.0`, `peft==0.14.0`, `tokenizers==0.21.4`, `huggingface-hub==0.26.5`, `accelerate==1.0.1`, `torch==2.10.0+cu128`. Asserts in cell 0 of the notebook fail loud if any pin slips.
- **Hardware**: single Colab / Kaggle T4 (15 GB VRAM). Peak SFT VRAM 8.4 GB, peak GRPO VRAM 11.4 GB.
- **Wallclock**: setup + SFT + merge + GRPO + eval ≈ 90 minutes end-to-end on a free Colab T4 (longer with `max_steps=200` GRPO).
- **Tests**: `pytest -q tests/` → 113 tests, all green.

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the exact command sequence.
