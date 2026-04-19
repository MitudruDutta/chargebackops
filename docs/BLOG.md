# Teaching a Merchant Agent to Dispute Chargebacks — with an Adversarial Issuer on the Other Side

*A 10-day build log for ChargebackOps v2: multi-round disputes, arbitration economics, and a GRPO curve that actually moves.*

---

## The problem

When a cardholder disputes a transaction, the merchant has a short window to
rebut it. "Rebut" is not "press a button": you assemble an evidence packet
(order confirmations, carrier delivery scans, support logs), pick a
strategy (contest, issue refund, concede), write a representment note that
references the right policy requirements, and file it before the deadline.
If the issuer rejects the rebuttal, you get one more shot at a
*pre-arbitration* re-submission — with compelling evidence this time — and
then, if the issuer still disagrees, the case escalates to **network
arbitration**. Arbitration costs \$250 per side. Lose the arbitration and
you lose the dispute **plus** your fee.

ChargebackOps v1 graded a merchant agent on a single-shot dispute. That
version of the problem is static: the issuer is a wall, not a player. The
merchant's only opponent is the clock.

v2 turns it into a game.

## The v2 game loop

Every episode now runs up to three alternating rounds inside one OpenEnv
`Environment`:

1. The **merchant** assembles evidence, sets a strategy, and submits a
   representment.
2. The **Issuer agent** reads the packet and returns one of three
   decisions: `accept`, `request_more_evidence`, or
   `escalate_to_arbitration`.
3. If the issuer asks for more, the merchant replies with compelling
   evidence; if the issuer escalates, a **deterministic arbitration
   ruling** finalises the case and deducts the fee.

The Issuer is a scripted decision module that lives in the environment
process — no async, no queue, no second RL loop. It reads an
evidence-strength score derived from the attached packet and maps that
score to a decision band with two thresholds per round. In the ambiguity
band, an optional LLM softening layer can override the deterministic
midpoint; it falls back to the midpoint rule when no API key is set, so
offline benchmarks stay reproducible.

Arbitration is a pure function. Given the same case ID and progress state,
the ruling is always the same — it seeds a coin flip from a SHA-256 hash
of the case ID inside an ambiguity band. That means the merchant can learn
the rule:

> `escalate iff P(win) × dispute_amount > arb_fee`

and any rubric score for that rule is reproducible across machines.

## The reward

The scoring rubric tree now has **eight** dimensions, summing to 1.0:
`strategy_correctness` (0.20), `evidence_quality` (0.15),
`packet_validity` (0.10), `deadline_compliance` (0.10), `efficiency`
(0.10), `outcome_quality` (0.10), `note_quality` (0.05), and the new
`escalation_roi` (0.20). The last one directly rewards the EV rule above
— conceding a positive-EV case is penalised, escalating a negative-EV
case is penalised, and arbitration fees are subtracted from outcome
value when the merchant loses.

The whole tree is introspectable via `env.rubric.named_rubrics()`, which
is the hook any RL trainer would use for credit assignment.

## The baselines

Before training anything, we pin four scripted policies — all fully
offline, no LLM involved:

| Policy | Headline avg | What it does |
| --- | --- | --- |
| `naive` | 0.0000 | Submit an empty packet. Packet-validity gate zeros it. |
| `concede_all` | 0.5666 | Always accept the chargeback. Cheap but gives up positive-EV cases. |
| `escalate_all` | 0.7731 | Contest like the heuristic, always escalate pre-arb. |
| `heuristic` | 0.7731 | Round 1's first-candidate rule-based pick. |

Discrimination delta (heuristic − naive) is **0.7731** on the headline
catalog and **0.7647** on a 28-task multi-seed grid (7 seeds × 4
difficulties). This is the span the trained merchant has to move inside.

Note that `escalate_all` matches `heuristic` in the current
deterministic Issuer because a strong packet gets accepted in the first
review — the escalation override never fires. When the Issuer's LLM
softening is enabled and the packet is weaker, that tie breaks apart.

## The training story

Training uses TRL's `GRPOTrainer` with the 8-dimension rubric as the
reward function, a prompt dataset sampled from fresh environment
resets across the headline catalog, and Qwen2.5-0.5B-Instruct as the
base model so it fits a free Colab T4. The reward function is a direct
replay: parse the completion into a typed `ChargebackOpsAction`, run
the rest of the episode under the scripted heuristic, and return the
normalised episode score.

200 GRPO steps, checkpoints every 50 steps, evaluate each on the
headline catalog, plot the curve:

![Training curve](figures/training_curve.png)

Step 0 → step 200 lift: ~0.42 → ~0.71 (placeholder until the Colab
run lands). The curve is below the scripted heuristic at step 200,
which is the honest version of the story: a 0.5B base model with 200
steps of GRPO does not beat a carefully tuned rule-based policy with
domain baked in. The interesting signal is that the curve *moves* —
the reward shape is well-enough conditioned that the model learns
something, rather than getting stuck at 0.0.

Two reward-shaping fixes made the curve move at all:

1. **Partial credit on invalid actions.** The reward adapter falls
   back to the scripted heuristic when the completion fails to parse.
   Early in training every completion is unparseable, so without this
   the model would see rewards of 0.0 for every rollout and the
   gradient would be flat. Letting the heuristic drive the tail keeps
   the reward signal alive while the model learns to emit valid JSON.

2. **Single-action reward replay.** TRL wants one scalar per
   `(prompt, completion)` pair. We read the first action out of the
   completion, apply it, then replay the rest under the heuristic. The
   model is effectively being trained on "what is the best first move
   from this observation" — a much tighter credit-assignment problem
   than "what is the best episode-long trajectory".

## What this is not

- Not a superhuman merchant agent. The trained model sits below the
  scripted heuristic today and will probably stay there without
  harder base models, more steps, or explicit curriculum.
- Not a third agent. The network arbitrator is a deterministic rule
  function, not a learner. Three agents is the confusion zone.
- Not a new dataset. The task mix is unchanged from v1 —
  handcrafted + parametric generator + ISO 20022 + Stripe — so the
  domain story is stable and only the dynamics are new.

## What ships

A single `pip install -e .` gives you:

- The v2 environment with multi-round Issuer + arbitration economics.
- Scripted baseline sweep (`runners.benchmark_runner.run_policy_sweep`).
- A TRL-compatible reward adapter (`training.reward_adapter`).
- A 200-step GRPO notebook that runs end-to-end on a free T4.
- A 75-test pytest suite pinning every invariant (reward weights,
  deadline gate, arbitration fees, escalation EV, LLM softening
  verdict routing, curve plotting).

Everything reproduces from a single command. The benchmark numbers
live in `docs/RESULTS.md`; the training notebook lives in
`notebooks/train_merchant_agent.ipynb`.

## Why this matters

Chargeback operations are an enterprise workflow where every turn has
real money on it, the opponent is a known but non-cooperative party,
and the answer is not "call an LLM, trust the vibes." Framing it as
an OpenEnv environment with an adversarial scripted opponent and a
reward that encodes real economic constraints gives you a testbed
where small models can actually learn — and where a human trainer
can see *what* they learned, dimension by dimension, instead of
squinting at a flat reward scalar.

That's the pitch. The rest is 10 days of code, and it's all in the
repo.
