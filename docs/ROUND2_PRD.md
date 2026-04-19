# ChargebackOps v2 — Round 2 PRD

**Adversarial Multi-Agent Chargeback Disputes with Economic ROI**

10-day upgrade plan for the Meta PyTorch OpenEnv Hackathon Round 2. Same problem statement as Round 1 (merchant chargeback operations), extended along the only two axes that justify a new submission: an adversarial Issuer agent and multi-round dispute lifecycle with arbitration economics. Every other axis is explicitly out of scope.

This document is the contract for the build. If a feature is not listed here, it is not in v2.

---

## 0. The Pitch (the one line judges read)

> Round 1 graded a merchant agent on a static one-shot dispute. Round 2 puts that merchant agent in a 3-round adversarial game against an Issuer agent, with arbitration fees that force economic decisions. We trained the merchant with TRL GRPO; reward improves from baseline 0.42 to 0.71 over 200 steps. Same domain, real game tree, real training story.

Three sentences. Judges absorb in 15 seconds. Everything in this PRD serves these three sentences.

---

## 1. Why This Wins (Theme Alignment)

Mandatory constraint from organizers: must keep Round 1 problem statement. The themes we hit and the honest reason:

| Theme | Hit? | Why |
|---|---|---|
| **#1 Multi-Agent Interactions** (primary) | ✅ Strong | Two agents (Merchant + Issuer) with conflicting objectives in alternating turns. Direct fit for the **Halluminate Multi-Actor** sub-theme — chargeback correspondence is literal multi-actor back-and-forth. |
| **#2 Long-Horizon Planning** | ✅ Natural | Step budget rises from ~15 to ~35-50. 3-round game tree. Branching opponent responses. Decisions in round 1 constrain round 3 escalation economics. |
| **#3.1 World Modeling Professional Tasks** | ✅ Carry-over | Same enterprise workflow as Round 1, now with reactive opponent. Strengthens the existing fit. |
| **#4 Self-Improvement** | ❌ Skip | Would require curriculum / agent-improves-agent loop. Out of scope. |
| **#5 Wild Card** | — | Backup framing only. Not the lead. |

**Sub-themes targeted:** Halluminate Multi-Actor (primary). Scale AI non-code workflows (secondary, via the long-horizon dispute lifecycle).

---

## 2. Goals and Non-Goals

### 2.1 Goals (must-have, all 10 measured at end of Day 10)

1. Merchant agent and Issuer agent run alternating turns inside a single OpenEnv `Environment`. The Issuer never blocks indefinitely; every turn produces a deterministic next observation.
2. A dispute progresses through up to 3 rounds: representment → (issuer accepts | issuer rejects → pre-arb → issuer accepts | escalates) → arbitration ruling. Single-round disputes still terminate cleanly.
3. Arbitration fees ($250 fixed) are deducted from outcome value. Merchant must learn the rule `escalate iff P(win) × dispute_amount > arb_fee`. Rubric directly scores this decision.
4. Issuer agent is a scripted decision module with optional LLM softening for ambiguous reviews. Three deterministic decisions: `accept`, `request_more_evidence`, `escalate_to_arbitration`. No training on the Issuer side.
5. New OpenEnv `Rubric` subclass `EscalationROIRubric` is wired into the existing `WeightedSum` tree. Total weights still sum to 1.0. Composition is introspectable via `env.rubric.named_rubrics()`.
6. Heuristic baseline agent updated to handle multi-round flows. Heuristic vs. naive (concede-everything) discrimination delta stays ≥ 0.40 on the 10-task headline benchmark.
7. TRL GRPO training notebook runs end-to-end on a free Colab T4. Produces a reward curve over ≥200 training steps showing measurable improvement.
8. Reward curve, ablation table (trained / heuristic / naive), and benchmark numbers are reproducible from a single command and documented in `docs/RESULTS_V2.md`.
9. <2 minute demo video shows the 3-round game tree with a real episode: merchant submits → issuer rejects → merchant adds compelling evidence → issuer escalates → arbitration ruling → reward computed. Voiceover in plain English.
10. Existing Round 1 surface (FastAPI, Gradio, Docker, HF Space) still works on the v2 environment. No regression in `pytest -q tests`.

### 2.2 Non-Goals (we will not build these even if time permits)

- **No third agent.** Network arbitrator is a deterministic rule function, not a separate agent. Three agents = confusion.
- **No multi-app split.** The 6 merchant systems (orders, payment, shipping, support, refunds, risk) stay in-process. We are not exposing them as separate OpenEnv envs.
- **No new task sources.** No new datasets, no new connectors. Existing handcrafted + parametric + ISO 20022 + Stripe stay as-is.
- **No new difficulty tier.** The four-tier easy/medium/hard/nightmare grid stays. We extend each tier to multi-round, we do not add a fifth.
- **No `LLMJudge` for note quality.** Round 1 mentioned this as future work. Future-work it stays.
- **No multi-task RL training.** Train on one task family at a time. Single curve. Single ablation.
- **No web demo redesign.** Existing Gradio UI is updated to render multi-round transcripts; visual identity stays.
- **No new auth / API surface.** Existing endpoints take the new env transparently.
- **No cross-currency / FX modeling.** USD only, same as Round 1.

If a teammate proposes adding any of the above mid-build, the answer is no. The discipline of this list is what keeps the project shippable in 10 days.

---

## 3. Architecture

### 3.1 The Game Loop (one paragraph)

The environment runs as a single OpenEnv `Environment`. On each `step(action)`:

1. The Merchant action is applied to the environment state (existing Round 1 logic).
2. If the action is a terminal-round action (`submit_representment`, `respond_to_pre_arb`, `escalate_to_arbitration`, `accept_arbitration_loss`, `resolve_case`), the environment **synchronously invokes the Issuer agent** as part of the same step. The Issuer reads the case state and returns one of three decisions. The environment writes the Issuer's decision into the observation as `last_issuer_decision` and advances the round counter.
3. If the Issuer escalates to arbitration on round 3, the environment invokes the deterministic arbitration ruling function and finalises the case.
4. The observation returned to the Merchant agent contains both its own action result and the Issuer's response. The Merchant's next step is informed by the Issuer's last move.

The Merchant is the only "RL-shaped" agent. The Issuer is a scripted decision module that lives **inside** the environment process and is invoked synchronously. There is no async, no message queue, no separate process. This is the simplest design that delivers genuine multi-agent dynamics.

### 3.2 Why Synchronous Issuer (not separate OpenEnv env)

Rejected alternative: two OpenEnv envs talking via message protocol. Rejected because:

- Doubles the surface area (two envs to maintain, two FastAPI servers, two Dockerfiles).
- Requires inventing an inter-env coordination protocol that is **not** part of OpenEnv core.
- Adds 5+ days of engineering for zero judging upside — judges score the dynamic, not the deployment topology.
- Halluminate Multi-Actor sub-theme description rewards realistic multi-actor interaction, not architectural ceremony.

A scripted-with-optional-LLM Issuer inside the env satisfies "multi-agent interactions" cleanly: two distinct agents, distinct objectives, alternating turns, observable correspondence in the trajectory log. The synchronous invocation is an implementation detail; the *agent-vs-agent dynamic* is what judges see and score.

### 3.3 Round Lifecycle

```
Round 1  Merchant: submit_representment
         Issuer:   review_representment → {accept, request_more, escalate}

Round 2  (only if Issuer requested_more or merchant escalates)
         Merchant: respond_to_pre_arb (add compelling evidence) | accept_arbitration_loss
         Issuer:   review_pre_arb → {accept, escalate}

Round 3  (only if either side escalates to arbitration)
         Network:  arbitration_ruling (deterministic) → {merchant_wins, issuer_wins}
         Both sides pay $250 arb fee. Loser additionally forfeits dispute amount.
```

Single-round outcomes (`accept_chargeback`, `issue_refund` from Round 1) still terminate at round 1. Only `contest` flows can extend to rounds 2-3.

### 3.4 The Two Agents at a Glance

| | Merchant Agent | Issuer Agent |
|---|---|---|
| Role | Maximize dispute recovery − costs | Protect cardholder, recover funds |
| Interface | OpenEnv action space (existing 9 + 3 new) | Internal: `decide_review(case_state) → IssuerDecision` |
| Decision logic | Heuristic + LLM tiebreak (Round 1) → trained via GRPO (v2) | Rule-based scoring of evidence packet + optional LLM softening when score is in 0.4–0.6 ambiguity band |
| Training | Yes — TRL GRPO, ~200 steps | No — scripted, deterministic given seed |
| Lives in | `runners/baseline_runner.py` (heuristic), Colab notebook (trained) | `scenarios/issuer_model.py` (new module) |

### 3.5 Component Diagram

```
┌────────────────────────────────────────────────────────────┐
│  ChargebackOpsEnvironment (OpenEnv)                        │
│                                                            │
│  ┌────────────┐  step(action)   ┌─────────────────────┐   │
│  │  Merchant  │ ──────────────► │  Env State          │   │
│  │  Agent     │                 │  (CaseProgress with │   │
│  │ (external) │ ◄────────────── │   round_number,     │   │
│  └────────────┘  observation    │   issuer_decisions) │   │
│                                  └─────────┬───────────┘   │
│                                            │ on terminal-  │
│                                            │ round action  │
│                                            ▼               │
│                                  ┌─────────────────────┐   │
│                                  │  IssuerAgent         │   │
│                                  │  (scenarios/         │   │
│                                  │   issuer_model.py)   │   │
│                                  │  rule-based +        │   │
│                                  │  optional LLM soften │   │
│                                  └─────────┬───────────┘   │
│                                            │ if escalated  │
│                                            ▼               │
│                                  ┌─────────────────────┐   │
│                                  │ arbitration_ruling() │   │
│                                  │ deterministic        │   │
│                                  └─────────────────────┘   │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │  ChargebackOpsEpisodeRubric                          │ │
│  │  └── case_rubric (existing 7 dims, weights adjusted) │ │
│  │      ├── strategy_correctness 0.20                   │ │
│  │      ├── evidence_quality      0.15                  │ │
│  │      ├── packet_validity       0.10                  │ │
│  │      ├── deadline_compliance   0.10                  │ │
│  │      ├── efficiency            0.10                  │ │
│  │      ├── outcome_quality       0.10                  │ │
│  │      ├── note_quality          0.05                  │ │
│  │      └── escalation_roi        0.20  ← NEW           │ │
│  │                              total: 1.00             │ │
│  │  └── deadline_gate: Gate(CaseAbandonedRubric)        │ │
│  └──────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────┘
```

---

## 4. Component Specs

### 4.1 IssuerAgent (`scenarios/issuer_model.py`, ~250 lines, NEW)

**Inputs:** `InternalCase`, `CaseProgress`, `round_number`, list of attached evidence IDs, network reason code.
**Outputs:** `IssuerDecision` enum: `ACCEPT`, `REQUEST_MORE_EVIDENCE`, `ESCALATE_TO_ARBITRATION`.

**Decision rule (deterministic core):**

1. Compute an `evidence_strength_score` ∈ [0, 1]:
   - +0.4 if all `required_evidence_ids` attached.
   - +0.2 per helpful evidence ID attached (capped at +0.4).
   - −0.3 per harmful evidence ID attached.
   - +0.1 if representment note references ≥2 of `policy_requirements`.
   - Clamped to [0, 1].

2. Map score to decision:
   - Round 1: score ≥ 0.7 → `ACCEPT`. score ≤ 0.4 → `REQUEST_MORE_EVIDENCE`. Otherwise (the 0.4–0.7 ambiguity band) → see step 3.
   - Round 2: score ≥ 0.6 → `ACCEPT`. Otherwise → `ESCALATE_TO_ARBITRATION` (issuer believes they will win at arbitration).

3. **Optional LLM softening** for the round-1 ambiguity band (score ∈ (0.4, 0.7)):
   - If `BASELINE_PROVIDER` is configured, send a compact JSON summary of the case + evidence + note to the issuer LLM with a prompt that asks "as the card-issuing bank's dispute analyst, would you accept this representment, request more evidence, or take it to arbitration? Reply with one of those three tokens."
   - LLM response overrides the rule-based decision only inside the ambiguity band. Outside the band, the rule wins.
   - On LLM failure, fall back to deterministic midpoint rule: `ACCEPT` if score ≥ 0.55, else `REQUEST_MORE_EVIDENCE`.

This means the Issuer is **fully deterministic offline** (reproducible benchmarks) and **slightly adaptive online** (live demo realism). Both modes are valid; the demo uses the LLM path, the benchmark uses the deterministic path.

**What the Issuer does NOT do:**
- Does not see merchant strategy directly (only the submitted packet).
- Does not see merchant's internal "optimal_strategy" label.
- Does not learn or update — fully scripted.
- Does not have its own action space exposed to OpenEnv — it lives inside the env.

### 4.2 Arbitration Ruling Function (`scenarios/arbitration.py`, ~80 lines, NEW)

Deterministic function: `arbitration_ruling(case, progress) → ArbitrationOutcome`.

Outcome is computed from the same `evidence_strength_score` the Issuer uses, with one tiebreaker:
- score ≥ 0.65 → `MERCHANT_WINS`
- score ≤ 0.35 → `ISSUER_WINS`
- otherwise → coin flip seeded by `case_id` (reproducible per-case).

Both parties pay $250 arb fee regardless of winner. Loser additionally forfeits the dispute amount. The function is pure: same inputs, same output, every time.

### 4.3 Multi-Round Action Extensions (`core/models.py`)

Three new entries added to the `ActionType` literal:

```python
ActionType = Literal[
    # existing 9 actions...
    "respond_to_pre_arb",       # round 2: add compelling evidence and re-submit
    "escalate_to_arbitration",  # round 2 or 3: send to network ruling
    "accept_arbitration_loss",  # round 2: concede after issuer rejects round 1
]
```

`ChargebackOpsAction` gains one optional field: `compelling_evidence_ids: list[str]` (used by `respond_to_pre_arb`).

### 4.4 CaseProgress Extensions (`scenarios/simulation.py`)

```python
@dataclass
class CaseProgress:
    # existing fields...
    round_number: int = 1                              # 1, 2, or 3
    issuer_decisions: list[str] = field(default_factory=list)  # log per round
    pre_arb_evidence_added: list[str] = field(default_factory=list)
    arbitration_outcome: str | None = None             # "merchant_wins" / "issuer_wins" / None
    arb_fees_paid: float = 0.0
    final_economic_outcome: float | None = None        # net dollars after fees
```

These fields are **observable** to the Merchant agent via the observation payload, so the agent can learn the game tree.

### 4.5 EscalationROIRubric (`evaluation/rubrics.py`)

```python
class EscalationROIRubric(Rubric):
    """Score the merchant's economic decision-making across rounds."""

    ARB_FEE = 250.0

    def forward(self, action, observation) -> float:
        ctx: GradingContext = action
        progress = ctx.progress
        case = ctx.case

        if progress.round_number == 1 and not progress.arbitration_outcome:
            return 1.0  # never reached round 2 — no escalation decision was made

        # Estimate P(win) from evidence packet at the moment of escalation
        p_win = self._estimate_p_win(case, progress)
        expected_value = p_win * case.amount

        escalated = "escalate_to_arbitration" in (a.action_type for a in progress.action_log)
        accepted_loss = "accept_arbitration_loss" in (a.action_type for a in progress.action_log)

        # The economic rule: escalate iff expected_value > ARB_FEE
        if expected_value > self.ARB_FEE:
            # Should have escalated. Score 1.0 if escalated, 0.0 if conceded.
            return 1.0 if escalated else 0.0
        else:
            # Should have conceded. Score 1.0 if conceded, 0.0 if escalated.
            return 1.0 if accepted_loss else 0.0
```

This rubric **directly scores the economic decision** that defines the new game. It carries 20% of the case score — high enough to dominate trained-agent learning signal, low enough to keep the existing 7 dimensions meaningful.

### 4.6 Environment Wiring (`server/chargeback_ops_environment.py`)

Two structural changes:

1. `_submit_representment` is no longer terminal for `contest` cases. It transitions the case to `awaiting_issuer_round_1`, invokes `IssuerAgent.decide_review(...)`, writes the decision into `progress.issuer_decisions`, and either terminates the case (`ACCEPT`) or sets `progress.round_number = 2`.

2. New private methods:
   - `_invoke_issuer_review(case, progress, round)` — synchronous Issuer call.
   - `_apply_arbitration(case, progress)` — deterministic ruling + fee accounting.
   - `_handle_respond_to_pre_arb(case, action)` — round 2 evidence addition.
   - `_handle_escalate_to_arbitration(case)` — bumps to round 3 and triggers arbitration.

Existing single-round paths (`accept_chargeback`, `issue_refund`) are untouched.

### 4.7 Heuristic Agent Updates (`runners/baseline_runner.py`)

Three new branches added to `candidate_actions(...)`:

1. If `visible_case.round_number == 2` and Issuer requested more evidence: candidate is `respond_to_pre_arb` with the strongest unattached evidence.
2. If `visible_case.round_number == 2` and merchant has weak evidence (P(win) × amount < $250): candidate is `accept_arbitration_loss`.
3. If `visible_case.round_number == 2` and merchant has strong evidence (P(win) × amount > $250): candidate is `escalate_to_arbitration`.

The existing `_obvious_next_action` shortcut handles all three (single-candidate situations).

### 4.8 Naive Baseline Agent (`runners/baseline_runner.py:bad_policy`)

Concede-everything policy stays. We add a second naive baseline `always_escalate` to demonstrate the rubric punishes both extremes. Three baselines total: `heuristic`, `concede_all`, `escalate_all`.

---

## 5. Reward and Rubric Changes

### 5.1 Per-Step Reward Adjustments

New reward signals added to `_apply_action` and the issuer-response handlers:

| Event | Reward |
|---|---|
| Issuer accepts round-1 representment | +0.25 (one-shot win) |
| Issuer requests more evidence | −0.05 (signal: packet was weak) |
| Merchant correctly adds compelling evidence in round 2 | +0.12 |
| Merchant incorrectly escalates a weak case (EV < arb_fee) | −0.20 |
| Merchant correctly escalates a strong case (EV > arb_fee) | +0.10 |
| Merchant correctly accepts loss when EV < arb_fee | +0.08 |
| Merchant wins arbitration | +0.30 |
| Merchant loses arbitration | −0.25 |
| Per-arb-fee (always paid by merchant on escalation) | −0.05 (shaping; final score uses actual $250 in EscalationROIRubric) |

### 5.2 Rubric Weight Reshuffle

| Dimension | v1 weight | v2 weight | Why changed |
|---|---|---|---|
| strategy_correctness | 0.25 | 0.20 | Strategy is now a smaller part of the game (round 1 of 3) |
| evidence_quality | 0.20 | 0.15 | Same reason |
| packet_validity | 0.15 | 0.10 | Same reason |
| deadline_compliance | 0.15 | 0.10 | Step budgets are larger; fewer cases hit the wall |
| efficiency | 0.10 | 0.10 | Unchanged |
| outcome_quality | 0.10 | 0.10 | Unchanged |
| note_quality | 0.05 | 0.05 | Unchanged |
| **escalation_roi** | — | **0.20** | New |
| **Total** | **1.00** | **1.00** | `WeightedSum` invariant preserved |

The `Gate(CaseAbandonedRubric)` wrapper is unchanged.

---

## 6. Training Story (the part that wins 30% of the score)

### 6.1 What We Train

Only the Merchant agent. Issuer is fixed (scripted). This is single-agent RL against a stable opponent — the simplest and most reproducible training setup that still demonstrates "agent learns multi-round game."

### 6.2 Algorithm

**TRL `GRPOTrainer`**. Chosen over Unsloth because GRPO is the natural fit for trajectory-level reward (the rubric returns one normalized score per episode) and TRL is the more general framework. Unsloth is a fine alternative — pick whichever is faster on T4 the day before submission. Either satisfies the mandatory-script criterion.

### 6.3 Model

Base model: `unsloth/Qwen2.5-1.5B-Instruct` (free-tier T4 fits with 4-bit). Single LoRA adapter, rank 16. We are not training a large model; we are demonstrating reward-curve improvement, which a 1.5B model on a constrained env will show clearly.

### 6.4 Training Notebook (`notebooks/train_merchant_agent.ipynb`)

One Colab notebook. Sections:

1. **Setup** — install TRL, openenv-core, project package; clone repo.
2. **Environment harness** — wrap `ChargebackOpsEnvironment` in a thin `gymnasium`-style adapter that exposes `reset()` and `step()` returning `(prompt, completion, reward)` tuples for GRPO consumption.
3. **Reward function** — `lambda episode_actions: env.rubric_score_for_trajectory(episode_actions)`. Pure delegation to the rubric tree.
4. **Training loop** — 200 GRPO steps, batch_size=4, lr=1e-5, 4-bit LoRA. Log `train/reward_mean` per step.
5. **Evaluation** — at steps 0, 50, 100, 150, 200, run the trained adapter on the 10-task benchmark and record average score.
6. **Plot** — single matplotlib chart: training reward curve + 5 evaluation points overlaid. Save to `docs/figures/training_curve.png`.
7. **Export** — adapter weights saved to a Hugging Face dataset/model repo for reproducibility.

### 6.5 Expected Numbers (target, not promise)

| Checkpoint | Target benchmark score |
|---|---|
| Step 0 (untrained) | 0.40–0.45 |
| Step 50 | 0.50–0.55 |
| Step 100 | 0.58–0.65 |
| Step 200 | 0.68–0.75 |

If step-200 score is below 0.55, we have a reward-shaping bug, not a model-capacity bug. Fix shaping, do not blame the model. (See Risk Register §11.)

---

## 7. Day-by-Day Plan with Exit Criteria

Two people: **You** (lead) and **Debanshu** (parallel work). Each day has hard exit criteria — if a day's exit is not met, that day stops and does not roll into the next without explicit re-planning.

### Day 1 — Foundation (Mon)

**You:**
- Add new fields to `CaseProgress` (`round_number`, `issuer_decisions`, `pre_arb_evidence_added`, `arbitration_outcome`, `arb_fees_paid`, `final_economic_outcome`).
- Add 3 new actions to `core/models.py` `ActionType` literal + `compelling_evidence_ids` field.
- Refactor `_submit_representment` so it transitions to `awaiting_issuer_round_1` instead of terminating.
- Stub `_invoke_issuer_review` (returns `ACCEPT` for now, so episodes still terminate).
- All existing tests still pass.

**Debanshu:**
- Sketch `scenarios/issuer_model.py` skeleton: dataclass, enum, scoring function (no LLM yet).
- Write 5 unit tests for the deterministic Issuer score function with hand-picked evidence configurations.

**Exit criteria:**
- `pytest -q tests` green.
- Heuristic agent runs `goods_not_received_easy` end-to-end (Issuer auto-accepts).
- New action types are valid in the schema (Pydantic doesn't reject them).

### Day 2 — Multi-Round Wiring (Tue)

**You:**
- Implement real `_invoke_issuer_review`: calls `IssuerAgent.decide_review`, writes decision, advances round.
- Implement `_handle_respond_to_pre_arb` and `_handle_escalate_to_arbitration` in env.
- Implement `arbitration_ruling` in `scenarios/arbitration.py`.

**Debanshu:**
- Finish `IssuerAgent` deterministic decision logic (no LLM yet).
- Add 5 more unit tests for round-2 and arbitration paths.

**Exit criteria:**
- A test case can complete a full 3-round cycle: submit → issuer requests more → respond_to_pre_arb → issuer escalates → arbitration ruling → final outcome recorded.
- `arb_fees_paid` and `final_economic_outcome` populated correctly.

### Day 3 — Rubric & Heuristic Update (Wed)

**You:**
- Implement `EscalationROIRubric`. Wire into `CaseRubric.aggregator` with new weight tuple. Verify weights sum to 1.0.
- Update `_estimate_p_win` helper inside the rubric module.
- Run rubric introspection (`env.rubric.named_rubrics()`) — verify `escalation_roi` shows up.

**Debanshu:**
- Update `runners/baseline_runner.py` heuristic with the 3 new round-2 branches.
- Update `_obvious_next_action` to short-circuit single-candidate round-2 situations.
- Add `always_escalate` naive baseline.

**Exit criteria:**
- Heuristic runs the full 10-task benchmark on v2 environment without errors.
- Discrimination delta (heuristic vs. concede_all) ≥ 0.40.
- Discrimination delta (heuristic vs. escalate_all) ≥ 0.40.

### Day 4 — LLM-Soft Issuer + Demo UI (Thu)

**You:**
- Add LLM softening to `IssuerAgent` for the 0.4–0.7 ambiguity band. Use existing OpenRouter fallback chain. Defaults to deterministic if no provider configured.
- Bump step budgets in `scenarios/case_generator.py`: easy 25, medium 30, hard 40, nightmare 50.
- Re-balance generated tasks so multi-round tasks are reachable (deadline_step accounts for rounds).

**Debanshu:**
- Update `server/demo_ui.py` (Gradio) to render Issuer decisions and round transitions in the trajectory log. Single new column: "Issuer says".
- Update `server/app.py` `/state` endpoint to include `round_number` and `issuer_decisions` in the response.

**Exit criteria:**
- Live demo at `localhost:8000/demo` shows the full 3-round flow end-to-end on `queue_optimization_hard`.
- API response from `/state` includes round info.

### Day 5 — Test Sweep + Benchmark Numbers (Fri)

**You:**
- Rewrite `tests/test_grader.py` and `tests/test_env.py` for v2 semantics.
- Add `tests/test_issuer.py` (10 unit tests covering deterministic and LLM-fallback paths).
- Add `tests/test_arbitration.py` (5 tests).
- Run full benchmark: heuristic, concede_all, escalate_all, naive, on the 10-task headline.

**Debanshu:**
- Update `evaluation/agent_brutal_audit.py` for v2 episode shape.
- Run multi-seed grid (7 seeds × 4 difficulties = 28 runs) for v2.
- Capture results in a fresh `docs/RESULTS_V2.md` draft (numbers only, narrative comes later).

**Exit criteria:**
- 22+ tests passing (target: ≥30 with new ones).
- Headline 10-task v2 numbers documented. Discrimination delta ≥ 0.40.
- `ruff check .` clean.

### Day 6 — Training Notebook (Sat)

**You:**
- Build `notebooks/train_merchant_agent.ipynb`: setup, env adapter, reward fn, training loop skeleton.
- Get 1 GRPO step running end-to-end (just verifying the wiring; not yet training meaningfully).

**Debanshu:**
- Update Stripe + ISO connectors so they expose multi-round-compatible cases (mostly metadata pass-through).
- Update `README.md` with v2 architecture diagram.

**Exit criteria:**
- Notebook runs cell-by-cell on a fresh Colab T4 without errors.
- One full GRPO step completes (gradient update on a tiny batch).

### Day 7 — Train + Curve (Sun)

**You:**
- Run 200-step GRPO training on Colab T4. Save checkpoints at 0/50/100/150/200.
- Evaluate each checkpoint on the 10-task benchmark.
- Generate `docs/figures/training_curve.png`.
- If reward curve is flat, debug reward shaping (not model). Log step-level rewards, find the dimension that's saturating.

**Debanshu:**
- Update `docs/RESULTS_V2.md` with full per-task table, multi-seed grid, ablation table (untrained / heuristic / trained).
- Draft mini-blog post (~800 words) covering: the game, the agents, the reward, the curve.

**Exit criteria:**
- Reward curve shows monotonic-or-near-monotonic improvement.
- Step-200 evaluation score ≥ 0.55. (If below, halt and re-shape rewards before recording.)
- Blog draft ready for review.

### Day 8 — Demo Video Production (Mon)

**You:**
- Record a screen-capture episode on `queue_optimization_hard` showing all 3 rounds.
- Edit to <2 minutes. Voiceover script: "Round 1: merchant submits evidence. Issuer rejects — wants more proof. Round 2: merchant adds the carrier signature. Issuer escalates to network arbitration. Network rules: merchant wins. Reward 0.78."
- Overlay reward curve at the end.

**Debanshu:**
- Polish blog post.
- Update `AGENT.md` with v2 sections (multi-round, issuer agent, training).
- Final pass on `README.md`.

**Exit criteria:**
- Video uploaded to YouTube (unlisted), link captured.
- Blog post published (Medium / Hashnode / GitHub Pages — whichever is fastest).
- Docs cohere — no contradictions between README, AGENT.md, RESULTS_V2.md.

### Day 9 — Polish, Reproducibility, Submission Prep (Tue)

**You:**
- Run end-to-end repro: fresh clone → install → `pytest` → `openenv validate .` → benchmark → notebook → curve. Time it. Document the command sequence.
- Add a single-command repro script: `scripts/repro_v2.sh`.
- Make sure HF Space rebuild is green.

**Debanshu:**
- Submission packaging: confirm hackathon submission checklist (latest OpenEnv, training script in Colab link, video link, blog link, Github link, HF Space link).
- Tag `v2.0.0-rc1` in git.

**Exit criteria:**
- Single command runs the full repro pipeline in <30 minutes on a fresh machine.
- All submission artefacts (video, blog, repo, space, notebook) reachable from a single root README section "How to evaluate this submission."
- Tag pushed.

### Day 10 — Buffer + Final Checks (Wed)

**Both:**
- Buffer day. Use it for whatever broke in Day 9 dry-run, last-mile bugfixes, narrative tightening.
- Triple-check the submission portal requirements at the end of day.
- Submit.

**Exit criteria:**
- Submitted. Tag `v2.0.0` pushed.

---

## 8. Test Strategy

### 8.1 Test Inventory (target after Day 5)

| File | Tests | Purpose |
|---|---|---|
| `tests/test_env.py` | 12 (existing 7 + 5 new) | Round 1, round 2, round 3 transitions; action validity per round |
| `tests/test_grader.py` | 8 (existing 4 + 4 new) | Each rubric dimension in isolation; new `EscalationROIRubric` |
| `tests/test_issuer.py` | 10 (NEW) | Deterministic decision matrix; LLM fallback; ambiguity band |
| `tests/test_arbitration.py` | 5 (NEW) | Ruling determinism; fee accounting; tiebreak coin-flip stability |
| `tests/test_api.py` | 7 (existing) | Endpoint shapes survive v2 |
| `tests/test_agent_audit.py` | 3 (updated) | Heuristic on v2 hits target scores |
| `tests/test_requirements.py` | 1 (existing) | Smoke |
| **Total target** | **46** | Up from 22 |

### 8.2 Gating Tests (must stay green at every commit)

- `pytest -q tests` — full suite.
- `openenv validate .` — schema sanity.
- `ruff check .` — lint.
- The smoke test from `docs/RUNNING_THE_AGENT.md` §7 (updated for v2 expected score).

### 8.3 What We Are Not Testing (Honest Limits)

- We are not unit-testing the Colab notebook (notebooks are not unit-test fixtures).
- We are not testing the LLM-softening path deterministically (that path is mocked or skipped in CI; it is exercised live during the demo recording).
- We are not load-testing the FastAPI server beyond the existing `/health` smoke.

---

## 9. Demo and Storytelling Plan

### 9.1 The Three Artefacts (mandatory)

1. **Mini-blog post** (~800 words). Sections: (a) "Why chargeback ops needs an opponent" (b) "The 3-round game" (c) "Reward shaping for economic ROI" (d) "Training curve" (e) "What you can run in 5 minutes". Plain English, one diagram (the component diagram from §3.5), one chart (the training curve).
2. **<2 minute video.** Cold open: screen of merchant agent submitting → issuer typing back "I reject this packet, send me carrier signature" → merchant adds signature → issuer escalates → network rules. Final 15 seconds: training curve animation. No music, voiceover only.
3. **Reproducibility story.** One README section titled "How to evaluate this submission" with 4 numbered steps and the time each takes. Judges who try it must succeed in under 30 minutes.

### 9.2 What We Will NOT Do in the Demo

- No leaderboards.
- No comparison to other teams.
- No unsubstantiated claims ("first OpenEnv multi-agent env" — leave that to the judges to notice).
- No more than one chart on screen at a time.
- No jargon without on-screen definition (e.g. "GRPO" appears once with a one-line caption).

### 9.3 The Story Arc (15-second judge attention version)

> Round 1 of this hackathon, we built a polished single-round chargeback simulator. For Round 2, we kept the same domain and added what real chargebacks actually have: an Issuer that reviews your evidence and pushes back, and a network arbitration step with a $250 fee. Now the merchant has to decide whether escalating is worth the fee. We trained the merchant agent on this; reward improves from 0.42 to 0.71 over 200 GRPO steps. Same problem statement, real game.

Memorise this. Every artefact serves it.

---

## 10. Risk Register

Five real risks. Each has a named owner and a concrete mitigation that does not require new scope.

| # | Risk | Probability | Impact | Mitigation | Owner |
|---|---|---|---|---|---|
| 1 | GRPO training reward curve is flat or noisy | Medium | High | Reward-shaping pass on Day 7 morning; if still flat, fall back to a smaller curriculum (start agent on `goods_not_received_easy` only, expand to full benchmark mid-training); accept that the curve story can be told even with modest improvement (0.42 → 0.55 is enough). | You |
| 2 | Issuer LLM softening is unreliable / slow on demo day | Medium | Medium | The deterministic path is the default and the benchmark uses it. Demo can either show LLM live (with a fast provider — Groq) or pre-record the demo offline. Either is acceptable. | Debanshu |
| 3 | Multi-round refactor breaks Round 1 single-round paths | Low | High | Day 1 exit criterion explicitly verifies single-round paths still terminate. Add a regression test on Day 1: `goods_not_received_easy` heuristic still scores ≥ 0.95 on v2. | You |
| 4 | Step budget changes make existing tasks unreachable | Low | Medium | Day 4 budget bump must be tested against all 10 headline tasks before commit. Roll back if any task becomes structurally unsolvable. | Debanshu |
| 5 | One person sick for 1-2 days | Medium | High | Day 10 is a buffer day. Day 5, 6, 8 have natural pause points. Reshuffle by skipping the LLM-softening path (it's a §2.2 nice-to-have) and / or the `always_escalate` baseline (heuristic + concede_all alone is enough discrimination). | Both |

### 10.1 What We Will Cut First If Behind Schedule

In order, lowest pain first:

1. LLM-softening path on the Issuer (keep it deterministic-only).
2. `always_escalate` baseline (keep `concede_all` only).
3. Multi-seed grid (keep the headline 10-task numbers only).
4. Stripe / ISO connector v2 updates (they still work for round-1-only cases).
5. Mini-blog (keep video; expand video voiceover to cover blog content).

We do **not** cut: the multi-round game, the Issuer agent, the new rubric, the training notebook, the reward curve, the video. Those are the win condition.

---

## 11. Definition of Done (the contract)

The submission ships when **all** of the following are true:

- [ ] All 10 goals in §2.1 are met and verifiable.
- [ ] `pytest -q tests` reports ≥ 30 passing tests.
- [ ] `openenv validate .` is clean.
- [ ] `ruff check .` is clean.
- [ ] `scripts/repro_v2.sh` runs end-to-end on a fresh machine in <30 minutes (excluding model download).
- [ ] Demo video is published and link is in the README.
- [ ] Mini-blog is published and link is in the README.
- [ ] Training notebook runs cell-by-cell on a fresh Colab T4.
- [ ] Reward curve PNG is committed at `docs/figures/training_curve.png`.
- [ ] `docs/RESULTS_V2.md` contains: per-task table, multi-seed grid, ablation table, reproduction commands.
- [ ] HF Space rebuilds green from the v2 main branch.
- [ ] Git tag `v2.0.0` is pushed to both `origin` and `hf` remotes.
- [ ] Submission portal entry is filed before the deadline (with a 6-hour buffer).

If any box is unchecked, the project is not done — even if everything else looks impressive.

---

## 12. The One Thing to Remember

A clean 3-round merchant-vs-issuer game with a real training curve will be remembered. Ten half-finished features will not. When in doubt during the next 10 days, ask: *does this serve the §0 pitch?* If no, cut it.
