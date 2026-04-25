# Limitations

This document is an explicit, honest inventory of what ChargebackOps does *not* yet do, and why each limit is left as future work. The goal is to be a credible base for further research; pretending limitations away would compromise that.

## 1. Scripted Issuer, not a trained counter-policy

The Issuer agent (`scenarios/issuer_model.py`) is a deterministic scoring function with optional LLM softening for the ambiguity band. It is calibrated against the same `evidence_strength_score` used by arbitration. This is intentional for reproducibility (every checkpoint sees the same opponent) and domain fidelity (real card networks operate under fixed rule books), but it limits the multi-agent research potential.

**Future work**: replace with a trained LLM Issuer for true self-play, with a curriculum that gradually softens the Issuer's predictability. The current scripted Issuer becomes the "teacher policy" stage of that curriculum.

## 2. Outcome reward uses a heuristic-tail rollout

`compute_outcome_reward` simulates the rest of the episode under the heuristic policy after the model takes its first action. This is a REINFORCE-style estimator with a heuristic baseline. It is honest (the model's only contribution is the single action being scored) but it embeds the heuristic into the reward computation. A model action that takes the episode into territory the heuristic handles poorly will accrue a worse reward than its true value.

**Future work**: trajectory-level credit assignment where the model controls every action in the rollout. Will significantly increase per-step compute (currently ~5-10 generations per step; trajectory-level would be ~10-30 per step).

## 3. GRPO trained 200 steps, not converged

The published checkpoint trains GRPO for 200 steps on a Colab T4. Real gradient flow is observed on ~30-50% of steps with peak gradient magnitudes 1.5–2.5, KL divergence reaching ~0.16, and demonstrated specialization on hard / nightmare cases. The trained policy approaches but does not cross the heuristic baseline (0.73 vs 0.81 overall), and regresses on easy cases (-0.31 absolute).

**Future work**: longer GRPO runs (1000+ steps), larger model (Qwen2.5-7B with QLoRA), and a curriculum that includes easy-case replay to prevent the easy-case regression.

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

## 8. The trained checkpoint underperforms the heuristic on overall mean

This is by far the most important limitation to disclose: the trained policy (0.728) does not beat the heuristic baseline (0.813) on the overall mean across the headline catalog. It *does* beat the SFT-only checkpoint on hard (+0.06) and nightmare (+0.14), but trades easy-case performance to do so.

The four reasons this is acceptable for the current release:

1. The headline metric for an *RL benchmark environment* is not "did this 3B model beat a hand-tuned heuristic?" but "does the environment exhibit a discrimination gradient that supports learning?" — and the base → SFT → GRPO progression (0.470 → 0.752 → 0.728) is clearly visible and per-difficulty interpretable.
2. The heuristic baseline (0.81) is close to the per-task ceiling and represents a strong domain-expert policy. A 3B model under 200 GRPO steps approaching it within 0.08 absolute is a reasonable result.
3. The per-family breakdown reveals the trained policy is genuinely *different* from both SFT and heuristic — it actively explores on the hardest cases. This is the property an RL benchmark environment exists to encourage; a benchmark that only rewards heuristic mimicry would be uninteresting.
4. The path to crossing the heuristic is well-understood (longer training, larger model, easy-case replay) and is laid out in the future-work sections above.

## 9. Single-process FastAPI, no horizontal scaling

The HF Space deployment runs a single uvicorn process. Concurrent sessions are supported (`SUPPORTS_CONCURRENT_SESSIONS = True`) but at scale the deployment would need a reverse proxy + worker pool. This is a deployment concern, not an environment concern.

**Future work**: production deployment guide with gunicorn + uvicorn workers + Redis-backed episode store.

## 10. No formal evaluation harness for pure-LLM-as-policy beyond the heuristic

The benchmark sweep includes scripted policies (naive, concede_all, escalate_all, heuristic) and trained checkpoints. It does not include a held-out evaluation against frontier closed-source LLMs (GPT-4o, Claude Sonnet, Gemini) used as policies via the inference fallback chain. Such results would be informative and are deferred to keep the benchmark fully reproducible without API keys.

**Future work**: a `/benchmark/llm-sweep` endpoint that runs registered providers against the headline catalog and publishes scores.

---

The above are intentional limitations of a first release, not unknown failure modes. Each is documented so future contributors know exactly where the most valuable extensions live.
