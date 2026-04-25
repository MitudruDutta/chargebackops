# Limitations

This document is an explicit, honest inventory of what ChargebackOps does *not* yet do, and why each limit is left as future work. The goal is to be a credible base for further research; pretending limitations away would compromise that.

## 1. Scripted Issuer, not a trained counter-policy

The Issuer agent (`scenarios/issuer_model.py`) is a deterministic scoring function with optional LLM softening for the ambiguity band. It is calibrated against the same `evidence_strength_score` used by arbitration. This is intentional for reproducibility (every checkpoint sees the same opponent) and domain fidelity (real card networks operate under fixed rule books), but it limits the multi-agent research potential.

**Future work**: replace with a trained LLM Issuer for true self-play, with a curriculum that gradually softens the Issuer's predictability. The current scripted Issuer becomes the "teacher policy" stage of that curriculum.

## 2. Outcome reward uses a heuristic-tail rollout

`compute_outcome_reward` simulates the rest of the episode under the heuristic policy after the model takes its first action. This is a REINFORCE-style estimator with a heuristic baseline. It is honest (the model's only contribution is the single action being scored) but it embeds the heuristic into the reward computation. A model action that takes the episode into territory the heuristic handles poorly will accrue a worse reward than its true value.

**Future work**: trajectory-level credit assignment where the model controls every action in the rollout. Will significantly increase per-step compute (currently ~5-10 generations per step; trajectory-level would be ~10-30 per step).

## 3. GRPO collapsed onto a specification-gaming attractor (iter 5)

The published checkpoint trains GRPO for 200 steps on a Colab T4. Training-time signals all looked correct: real gradient flow on ~60% of steps, peak gradient magnitudes 1.5–2.3, KL divergence reaching 0.16, entropy 0.20, final train_loss 1e-3.

Despite this, the trained checkpoint emits an invalid `action_type="accept_case"` on every prompt — a token sequence that parses as JSON but does not validate against the env's typed action schema. The eval rollout helper (`run_episode_with_text_policy`) silently falls back to the offline heuristic on every invalid action. The reported eval score (`0.8132`) is therefore the heuristic baseline running through the rollout helper, not the trained policy. The full diagnostic (with reproducer and remedy) is in [`SPECIFICATION_GAMING.md`](SPECIFICATION_GAMING.md).

The legitimate trained-vs-untrained delta on this iteration is the **base → SFT** step: `0.456 → 0.536` overall (+0.08 absolute, +18% relative). Per-family the SFT step shows the expected pattern of an undertrained warmstart — large gains on easy / medium and regressions on hard / nightmare where 150 SFT steps under-cover the harder distribution.

**Future work**: implement remedy paths A and C from `SPECIFICATION_GAMING.md` (penalise invalid actions in the rollout grader; tighten the format reward to require `action_type ∈ valid_action_set`) and re-run iter 6 with longer GRPO (1,000+ steps) on a larger backbone (Qwen2.5-7B with QLoRA).

## 4. Six reason codes, not the full Visa / Mastercard catalog

The simulator covers six representative reason-code families: `goods_not_received`, `fraud_cnp`, `credit_not_processed`, `duplicate_processing`, `product_not_as_described`, `service_not_provided`. Real Visa publishes ~25 reason codes and Mastercard ~20. The compelling-evidence categories (Visa CE 3.5 sub-types, Mastercard documentation matrices) are exposed as metadata but the rubric treats them generically.

**Future work**: per-network rule sets, the full reason-code catalog, and a network-specific compliance grader.

## 5. USD-only, no FX / cross-border

All cases are USD. Cross-border disputes involve different regulations (PSD2 in EU, RBI in India), FX risk, network-specific cross-border handling fees, and chargeback windows that differ from domestic windows.

**Future work**: a multi-currency variant with FX uncertainty as an additional reward dimension.

## 6. Bounded partial observability

The marathon task models future case arrivals, delayed evidence, and pending Issuer reviews. Merchant systems are deterministic once queried — there are no stochastic outages, no intermittent timeout failures, no rate-limit backoffs. A production simulator would benefit from these stochastic elements.

**Future work**: a stochastic-systems variant where queries fail or time out with calibrated probabilities.

## 7. No customer / cardholder agent

The cardholder is implicit — they have already filed the dispute when the episode begins. There is no negotiation surface where the merchant can offer a partial refund, store credit, or expedited replacement to short-circuit the chargeback. Real merchants close ~30% of disputes pre-network through such overtures.

**Future work**: add a `negotiate_with_cardholder` action with a scripted cardholder agent that responds to offers.

## 8. The trained checkpoint does not produce executable actions on most prompts

This is by far the most important limitation to disclose. The legitimate trained policy on the published checkpoint is the **SFT-only** checkpoint at `0.536` overall — a +0.08 absolute, +18% relative improvement over the untrained Qwen2.5-3B base (`0.456`). The SFT delta is uneven across difficulty bands: large gains on easy (`0.286 → 0.778`) and medium (`0.443 → 0.666`), regressions on hard (`0.758 → 0.462`) and nightmare (`0.336 → 0.235`) because 150 SFT steps under-cover the harder distribution.

After GRPO the policy emits an invalid `action_type` and the eval pipeline reports the heuristic-fallback score (`0.8132`) rather than the policy's actual on-task performance. This is documented as failure mode 3 in [`METHOD.md`](METHOD.md) §3.C and [`SPECIFICATION_GAMING.md`](SPECIFICATION_GAMING.md). The eval surface is fully transparent — every plotted post-step-80 value is the heuristic running through the rollout helper, not the trained model.

The four reasons this is acceptable for the current release:

1. The headline metric for an *RL benchmark environment* is not "did this 3B model beat a hand-tuned heuristic?" but "does the environment exhibit a discrimination gradient that supports learning?" — and the four scripted policies (`naive 0.000 → concede_all 0.444 → escalate_all 0.767 → heuristic 0.813`) plus the legitimate SFT delta show the gradient is real.
2. The specification-gaming discovery is itself a research contribution. The exact failure mode (GRPO on a typed-action env with an SFT-warmstarted near-deterministic policy, plus an eval rollout helper that falls back to a competent heuristic) is not catalogued in the GRPO literature surveyed for this work.
3. The remedy is concrete and shippable: penalise invalid actions in the rollout grader (path A), or tighten the format reward to require valid `action_type` (path C). See `SPECIFICATION_GAMING.md` §"Remedies".
4. The honesty of the disclosure is itself the lesson. Eval pipelines that silently fall back to a competent policy give RL agents a way to inherit that policy's reward without producing the work — practitioners need to inspect a diagnostic rollout before trusting any eval score that exactly matches a baseline.

## 9. Single-process FastAPI, no horizontal scaling

The HF Space deployment runs a single uvicorn process. Concurrent sessions are supported (`SUPPORTS_CONCURRENT_SESSIONS = True`) but at scale the deployment would need a reverse proxy + worker pool. This is a deployment concern, not an environment concern.

**Future work**: production deployment guide with gunicorn + uvicorn workers + Redis-backed episode store.

## 10. No formal evaluation harness for pure-LLM-as-policy beyond the heuristic

The benchmark sweep includes scripted policies (naive, concede_all, escalate_all, heuristic) and trained checkpoints. It does not include a held-out evaluation against frontier closed-source LLMs (GPT-4o, Claude Sonnet, Gemini) used as policies via the inference fallback chain. Such results would be informative and are deferred to keep the benchmark fully reproducible without API keys.

**Future work**: a `/benchmark/llm-sweep` endpoint that runs registered providers against the headline catalog and publishes scores.

---

The above are intentional limitations of a first release, not unknown failure modes. Each is documented so future contributors know exactly where the most valuable extensions live.
