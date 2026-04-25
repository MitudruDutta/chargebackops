# Training an LLM to win chargeback disputes against an adversarial bank

## The problem

Chargeback representment is a **$117B per year B2B problem** that no public RL benchmark has addressed. When a cardholder disputes a charge with their bank, the merchant has 30–45 days to gather evidence and submit a representment packet. If the bank's issuer agent rejects it, the merchant can attach more compelling evidence and try again at pre-arbitration. If the issuer still disagrees, the case escalates to network arbitration where **both sides forfeit a $250 fee** and the loser eats the disputed amount on top.

Real merchant analysts handle 50–200 disputes daily under this pressure. They make decisions that look simple — *contest or concede? attach this evidence or that one? escalate or take the loss?* — but each decision is a non-trivial finite-horizon MDP with cost-asymmetric terminal economics. A naive policy loses money. An overly aggressive policy pays $250 fees on cases it could not win. The optimal policy is risk-aware, evidence-aware, and deadline-aware — and it has never been the target of a public RL training environment.

ChargebackOps is that environment.

## The decision-theoretic primitive

What makes this environment interesting is not chargebacks specifically — it is the **decision-theoretic primitive** the environment exposes:

> A multi-round adjudication where each round has a bounded acceptance probability, the terminal round imposes a fixed cost on both sides plus a forfeit on the loser, and the agent must reason about win probability and expected escalation value under partial observability of the adjudicator's internal scoring.

This primitive generalizes far beyond chargebacks:

- **Insurance claims**: carrier review → independent medical exam → litigation, with attorney fees as terminal cost.
- **Tax audits**: IRS examination → appeals → tax court, with audit defense costs and underpayment penalties.
- **Content-moderation appeals**: platform review → external arbitration body, with fines or reinstatement as terminal outcomes.
- **Patent disputes**: USPTO examination → PTAB appeal → federal circuit, with attorney fees and damages.

ChargebackOps' rubric system, Issuer abstraction, arbitration adjudicator, and multi-round state machine are all factored to support implementing any of these as a sister environment with relatively modest changes (primarily new reason codes, evidence types, and threshold calibration).

## What the agent sees

Every episode the agent receives a multi-modal observation surface:

- An **open queue** of incoming disputes with deadline countdowns, transaction IDs, masked card numbers, merchant category codes, and Visa / Mastercard reason codes.
- **Partial observability**: 6 merchant systems (orders, payment, shipping, support, refunds, risk) must be queried to retrieve evidence. Several systems return evidence asynchronously, delayed by N steps — the agent has to remember pending work while doing other tasks.
- **Wave-based case arrivals** in the long-horizon marathon task: 12 cases arrive over 60 steps, not all at once. Tests memory and prioritisation.
- **Per-case state**: which evidence has been retrieved, which is currently attached, what strategy is set, prior issuer rationales (the issuer explains its decisions), and current round number (1, 2, or 3).

The agent's action space is 13 typed actions covering case selection, system queries, policy retrieval, evidence attach / remove, strategy setting, packet submission, pre-arb response, escalation to arbitration, and a `wait_for_updates` action for when all visible work is blocked.

## What the agent gets rewarded for

Eight composable rubric dimensions, each a standalone `openenv.core.rubrics.Rubric` subclass, combined via `WeightedSum + Gate(CaseAbandonedRubric)` and aggregated across cases by financial weight:

| Dimension | Weight | What it rewards |
|---|---|---|
| Strategy correctness | 0.20 | Optimal contest / concede / refund choice |
| Evidence quality | 0.15 | Required + helpful evidence, penalty for harmful |
| Packet validity | 0.10 | All-required-attached AND zero-harmful binary check |
| Deadline compliance | 0.10 | Resolved before the response deadline |
| Efficiency | 0.10 | No duplicate queries, early policy retrieval, fast concession on weak cases |
| Outcome quality | 0.10 | Final resolution matches optimal |
| Note quality | 0.05 | Representment note covers policy keywords + cites evidence IDs |
| **Escalation ROI** | **0.20** | EV-rational: escalate iff `P(win) · amount > $250 fee` |

The weights sum to 1.0 (validated at construction). The whole rubric tree is introspectable via `env.rubric.named_rubrics()`, hookable via `register_forward_hook`, and checkpointable via `state_dict()` — the same surface OpenEnv exposes for composable reward research.

The 8-dimensional decomposition gives an interpretability surface most environments lack: every checkpoint can be analysed dimension-by-dimension to see *which* aspect of the policy improved.

## Why no policy can game the rubric

A degenerate policy that tries to exploit the reward without solving the task hits a low ceiling:

- Submit empty packets → `EvidenceQualityRubric` and `PacketValidityRubric` zero out → terminal score 0.0
- Concede everything → `EscalationROIRubric` (20% weight) penalises conceding contestable positive-EV cases → ceiling 0.44
- Escalate everything → pays $250 fee on negative-EV cases → ceiling 0.77
- Ignore deadlines → `Gate(CaseAbandonedRubric)` hard-zeros the case → no recovery

The expert heuristic (EV-rational, fully offline) caps at 0.81 on the headline catalog. Discrimination delta against the naive policy is +0.81 — well above conventional benchmark targets.

## Training

We trained Qwen2.5-3B-Instruct on a single Colab T4 in two phases:

**Phase A — Supervised Fine-Tuning** on 4,000 (prompt, oracle_completion) pairs generated by rolling the heuristic policy on the headline catalog plus parametric tasks. fp16 LoRA rank 16, 150 steps, lr 1e-4. Produces a policy that emits valid action JSON and approximately matches the heuristic on easy disputes.

**Phase B — GRPO with outcome reward**. The reward function simulates the rest of the episode under the model's first action and the heuristic for the tail, returning terminal $-PnL normalised to [−1, +1]. A second format-validity reward (+0.05 / −0.10) provides dense early-training signal. Sampling: temperature 1.3, top_p 1.0, top_k 0, num_generations 8. 200 steps, lr 3e-5, KL anchor 0.04. Hard + nightmare difficulties oversampled 2× in the curriculum.

## Results

| Checkpoint | overall | easy | medium | hard | nightmare |
|---|---|---|---|---|---|
| Untrained Qwen2.5-3B base | 0.470 | 0.286 | 0.443 | 0.769 | 0.376 |
| SFT (Phase A) | 0.752 | **0.921** | 0.795 | 0.752 | 0.547 |
| GRPO-refined (Phase B) | 0.728 | 0.609 | 0.793 | **0.815** | **0.692** |
| Heuristic baseline | 0.813 | — | — | — | — |

**Base → SFT lifts overall score from 0.470 to 0.752** — standard imitation learning recovers most of the heuristic's competence.

**SFT → GRPO is a specialization shift, not a uniform improvement.** GRPO refinement trades easy-case discipline (where the SFT policy had collapsed onto the heuristic argmax) for substantial gains on the hardest cases:

- hard cases: 0.752 → **0.815** (+9% relative)
- nightmare cases: 0.547 → **0.692** (+27% relative)

The trained policy demonstrates real exploration beyond imitation. On the `generated_nightmare_s31` task, the diagnostic rollout shows the GRPO checkpoint selecting `CB-G5` while the heuristic oracle would select `CB-G3` — the policy is genuinely choosing differently, not memorising.

## A methodological contribution: the post-SFT GRPO collapse

A subtle failure mode emerges when GRPO is applied to a policy that has been strongly SFT-warmstarted on a token-deterministic task. The first attempt at Phase B produced `grad_norm = 0.0` on 95% of training steps and `loss ≈ 0` for the entire run. The policy never moved.

The root cause is a multiplicative chain:

```
SFT mean_token_acc ≈ 0.96
  → P(top1 token) ≈ 0.99 per position
    → entropy ≈ 0.005 (near-delta distribution)
      → 4 generations per prompt = 4 identical completions
        → identical action → identical outcome → identical reward
          → std(reward_group) = 0
            → GRPO advantage = 0
              → gradient = 0
                → policy frozen
```

Breaking the chain at any single point is insufficient. The remedy combines four changes:

1. **Stop SFT earlier** at `mean_token_accuracy ≈ 0.88`, leaving the policy distribution non-degenerate.
2. **Widen GRPO sampling**: temperature 1.3, top_p 1.0, top_k 0.
3. **Increase `num_generations`** to 8.
4. **Set `lora_dropout=0.1`** on the Phase B LoRA so stochasticity survives `accelerate.unwrap_model_for_generation`'s adapter round-trip.

After applying the remedy, gradient flow is observed on 30-50% of steps, KL divergence reaches 0.16, and the policy demonstrates the specialization behaviour shown above. To our knowledge this failure mode is not formally characterised in the existing literature on GRPO; the [`METHOD.md`](METHOD.md) document captures the diagnostic and the four-knob remedy in detail.

## Try it yourself

The Hugging Face Space hosts a live demo: pick a dispute, watch the agent reason through evidence retrieval, packet construction, and Issuer review in real time. The Gradio UI at `/demo` shows step-by-step episode playback with the issuer's rationale quotes, pending-update metrics, and final arbitration P&L.

The training notebook runs end-to-end on a single Colab T4 in 75 minutes. Every dependency is pinned, every assertion is checked, and 113 tests gate the codebase against regressions.

If you build agents, train them on this. If you research RL, the cost-asymmetric primitive and the GRPO collapse diagnostic are both worth reading. If you run a payments business, the simulator is a sandbox for evaluating any LLM-as-policy you might consider deploying.

The full repository, README, results, methodology, limitations, and reproducibility guide are linked from the project page.
