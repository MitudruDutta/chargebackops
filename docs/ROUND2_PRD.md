# ChargebackOps Theme Alignment PRD

ChargebackOps is a professional OpenEnv environment for merchant-side chargeback dispute operations. The correct hackathon positioning is:

1. **Primary: Theme #3.1 Professional World Modeling**
2. **Secondary: Theme #2 Long-Horizon Planning**
3. **Tertiary: Theme #1 Multi-Agent Interactions**

This is intentionally not pitched as a pure multi-agent arena. The Merchant is the trainable policy. The Issuer is a scripted environment actor with deterministic review behavior and optional LLM softening. That makes the interaction useful and demoable, but not equivalent to self-play between two learned agents.

## One-Line Pitch

ChargebackOps trains an LLM agent to operate a realistic merchant dispute desk: triage chargebacks, query merchant systems, build evidence packets, handle issuer pushback, and manage a month-end backlog with delayed evidence, delayed reviews, deadlines, and arbitration ROI.

## Brutal Positioning

Theme #3.1 is the strongest fit because the environment models a real enterprise workflow with tools, partially observable systems, delayed consequences, and deterministic verification.

Theme #2 is now credible because `monthly_dispute_backlog_marathon` is a 12-case, 60-step backlog with wave arrivals, asynchronous evidence, delayed issuer reviews, deadline pressure, and portfolio optimization. It is long-horizon relative to the original single-case tasks, but it is not yet a 300-step memory-beyond-context benchmark. Do not overclaim it as "super long-horizon"; pitch it as a practical long-horizon professional workflow.

Theme #1 is present through the Merchant-vs-Issuer dispute lifecycle. The Issuer has its own incentives and can accept, request more evidence, or escalate to arbitration. This creates opponent-like feedback and theory-of-mind pressure, but the Issuer is not a separately trained policy. Pitch it as a scripted counterparty, not a full multi-agent RL system.

## Current Implemented Mechanics

- Typed OpenEnv action / observation / state models in `core/models.py`.
- `reset()`, `step()`, and `state` implemented in `server/chargeback_ops_environment.py`.
- 13 typed actions, including `wait_for_updates` for long-horizon blocked states.
- Five showcase tasks plus seven generated holdout tasks in the headline catalog.
- Flagship long-horizon task: `monthly_dispute_backlog_marathon`.
- Deterministic `IssuerAgent` with round-1 and round-2 review logic.
- Network arbitration resolver with a $250 fee and EV-sensitive scoring.
- 8-dimension rubric tree using OpenEnv `Rubric`, `WeightedSum`, and `Gate`.
- Offline benchmark runner with `heuristic`, `escalate_all`, `concede_all`, and `naive` policies.
- SFT + GRPO notebook for a merchant policy, with the critical adapter-loading bug fixed.
- Gradio demo exposed at `/demo`.

## Theme #3.1 Design

The environment is a compact enterprise simulator. The agent must maintain a causal model of:

- which cases are currently visible,
- which systems have already been queried,
- which evidence has been retrieved,
- which evidence is helpful or harmful from visible text,
- which deadlines are close,
- which issuer reviews are pending,
- which cases are worth arbitration fees,
- and which cases should be conceded or refunded.

The task is not a static RAG problem. The state changes after each action. A bad early decision can remove budget, miss a deadline, attach harmful evidence, or create a negative-EV arbitration branch.

## Theme #2 Design

The long-horizon contribution is the marathon backlog:

- 12 disputes in one episode.
- 60-step max horizon.
- Only 4 cases visible at reset.
- 8 future cases arrive in later waves.
- Some merchant systems return evidence after a delay.
- Some issuer reviews return several steps after submission.
- The agent must keep working on other cases while pending work matures.
- Score is portfolio-weighted, so the agent must balance urgency, amount, evidence quality, and arbitration ROI.

This creates long-horizon planning pressure without changing the core chargeback idea.

### Long-Horizon State Variables

- `arrival_step`: hides future cases until their wave arrives.
- `evidence_response_delay_steps`: delays evidence from selected systems.
- `delayed_systems`: marks which merchant systems are asynchronous.
- `issuer_response_delay_steps`: delays issuer decisions after submission.
- `pending_evidence_systems`: tracks delayed evidence requests.
- `pending_issuer_due_step`: tracks delayed issuer review return.
- `merchant_submitted_at_step`: preserves deadline compliance even when issuer response is delayed.

### Long-Horizon Action

`wait_for_updates` advances the clock when visible work is blocked by pending evidence, pending issuer review, or future arrivals.

Waiting while open work exists is penalized. Waiting when the backlog is genuinely blocked is lightly rewarded. This prevents reward hacking by idle looping while still giving agents a legal action when no visible case can progress.

## Theme #1 Design

The Issuer is an environment actor, not the trained policy.

The Merchant submits a representment packet. The Issuer reviews the evidence and returns one of:

- `accept`
- `request_more_evidence`
- `escalate_to_arbitration`

If the Issuer requests more evidence, the Merchant can respond with compelling evidence, escalate, or concede. If arbitration occurs, the environment computes the economic outcome deterministically.

This is enough to demonstrate counterparty modeling: the Merchant must anticipate what evidence the Issuer will accept and whether escalation is worth the fee.

## Grading

Each case is scored with a deterministic rubric:

- Strategy correctness: 20%
- Evidence quality: 15%
- Packet validity: 10%
- Deadline compliance: 10%
- Efficiency: 10%
- Outcome quality: 10%
- Note quality: 5%
- Escalation ROI: 20%

The deadline gate hard-zeros abandoned cases only when the agent never attempted a timely resolution. For long-horizon delayed issuer reviews, deadline compliance is based on merchant submission time, not the delayed issuer response time.

## Current Benchmarks

Headline catalog: 12 tasks.

| Policy | Headline Avg | Multi-Seed Avg | Notes |
| --- | ---: | ---: | --- |
| heuristic | 0.8132 | 0.7628 | best scripted policy |
| escalate_all | 0.7668 | 0.7675 | strong but pays bad arbitration fees |
| concede_all | 0.4435 | 0.4454 | cheap but forfeits positive-EV contests |
| naive | 0.0000 | 0.0000 | empty-packet baseline |

Marathon task only:

| Policy | Score |
| --- | ---: |
| heuristic | 0.6793 |
| escalate_all | 0.6168 |
| concede_all | 0.4004 |
| naive | 0.0000 |

These numbers prove the environment has discrimination and that the marathon is harder than the short tasks. They do not prove the trained LLM has improved yet.

## Training Story

The correct training story is:

1. Use SFT to teach the JSON action schema and per-state action variety.
2. Use GRPO to refine action selection against verifiable reward.
3. Evaluate checkpoints on easy, medium, hard, and nightmare task families.
4. Show reward curves only after the notebook is re-run end to end.

Do not claim a trained reward improvement until the notebook is executed after the current fixes. The previous GRPO attempt was flat and is documented as a failure analysis in `docs/RESULTS.md`.

## Acceptance Criteria

- `pytest -q tests` passes.
- `openenv validate .` passes.
- `/reset`, `/step`, `/state`, `/tasks`, `/grader`, `/baseline`, `/demo`, and `/health` work.
- The marathon task appears in `/tasks`.
- `wait_for_updates` is in the action schema.
- The notebook can be run in Colab and produces a real before/after curve.
- The demo shows the long-horizon backlog, issuer review, and arbitration economics.

## Remaining Risks

- The marathon is long-horizon, but not extreme long-horizon. If the judges expect hundreds of steps or memory beyond context, this only partially satisfies Theme #2.
- The Issuer is deterministic. That is good for reproducibility, but it limits Theme #1 novelty.
- The training reward is currently a per-action oracle reward. It is useful for making GRPO tractable, but it is not yet full trajectory-level RL on portfolio P&L.
- The notebook must be re-run before any public claim of trained-agent improvement.
- Docker and Hugging Face Space deployment should be revalidated after every material change.

## Best Pitch

Lead with professional world modeling:

> ChargebackOps is a realistic enterprise dispute-operations environment. The agent must operate across multiple merchant systems, reason about delayed evidence and issuer pushback, and optimize a portfolio of chargebacks under deadlines and arbitration economics.

Then add Theme #2:

> The flagship marathon task turns this into a 60-step backlog with wave arrivals and asynchronous outcomes, so the agent must remember pending work and plan beyond the next case.

Then add Theme #1 carefully:

> A scripted Issuer acts as the counterparty, forcing the Merchant to anticipate evidence thresholds and escalation economics.

This framing is accurate, defensible, and aligned with the actual code.
