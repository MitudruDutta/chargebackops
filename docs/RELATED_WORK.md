# Related Work

ChargebackOps positions at the intersection of four research lines: policy-gradient RL for LLMs, RL with verifiable rewards (RLVR), reward design and specification gaming, and RL environments for agent training.

## 1. Policy-gradient algorithms for LLM post-training

- **PPO**: Schulman et al., *Proximal Policy Optimization Algorithms*, 2017. The originating policy-gradient algorithm with a clipped trust region; provides the conceptual base for most LLM RL trainers.  
  https://arxiv.org/abs/1707.06347
- **GRPO** (Group Relative Policy Optimization): Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*, 2024. Removes the value model from PPO and computes advantages via within-group reward standardisation. ChargebackOps uses GRPO via TRL.  
  https://arxiv.org/abs/2402.03300
- **TRL** library (Hugging Face), the reference implementation for PPO / GRPO / DPO post-training of transformer models.  
  https://huggingface.co/docs/trl

The post-SFT GRPO collapse documented in [`METHOD.md`](METHOD.md) §3 is, to our knowledge, not formally characterised in the existing literature on GRPO. The DeepSeekMath paper's experiments warmstart from base instruct models without the high-token-accuracy SFT phase that triggers the collapse. Practitioners applying GRPO to a strongly imitation-warmstarted policy on a token-deterministic task should be aware of the failure mode.

## 2. RL with verifiable rewards (RLVR)

- Lambert et al., *Tülu 3: Pushing Frontiers in Open Language Model Post-Training*, 2024. Popularised the RLVR framing — replace learned reward models with programmatic verifiers where ground truth is checkable.
- Label Studio, *Reinforcement Learning from Verifiable Rewards*, 2024. Practitioner overview of RLVR vs RLHF tradeoffs.  
  https://labelstud.io/blog/reinforcement-learning-from-verifiable-rewards/
- Hu et al., *Reinforcement Learning with Verifiable Environments*, 2025 (RLVE). Argues that procedurally-generated, adjustable-difficulty environments are a superior reward source vs static-prompt RLVR.  
  https://arxiv.org/html/2511.07317v1

ChargebackOps' outcome reward is RLVR-style: the verifier is the simulated dispute outcome (terminal $-PnL after Issuer review and arbitration), not a learned reward model. The parametric task generator + ISO 20022 adapter make the environment RLVE-style: difficulty is adjustable via reason code and difficulty tier, and the task pool is unbounded.

## 3. Reward design, specification gaming, reward hacking

- Krakovna et al., *Specification Gaming: The Flip Side of AI Ingenuity*, DeepMind, 2020. Catalogue of reward-hacking failures across RL systems; foundational for thinking about what reward functions actually optimise.  
  https://deepmind.google/blog/specification-gaming-the-flip-side-of-ai-ingenuity/
- Weng, *Reward Hacking in Reinforcement Learning*, 2024. Comprehensive survey of how reward hacking arises in modern RL.  
  https://lilianweng.github.io/posts/2024-11-28-reward-hacking/
- Skalse et al., *Defining and Characterizing Reward Hacking*, 2022.  
  https://arxiv.org/abs/2209.13085

ChargebackOps' rubric design is anti-hacking by construction:

- The 8 dimensions impose orthogonal constraints (strategy, evidence, packet, deadline, efficiency, outcome, note, escalation ROI) that no degenerate strategy can simultaneously satisfy.
- The `Gate(CaseAbandonedRubric)` is a hard zero on deadline-violating cases — no recovery.
- The arbitration adjudicator and the Issuer scoring function share a single source of truth (`evidence_strength_score`), so a packet that exploits round-1 review will fare correspondingly worse in round-3 arbitration.
- The four scripted-policy baselines (naive, concede_all, escalate_all, heuristic) cap at 0.0, 0.44, 0.77, and 0.81 respectively — every degenerate strategy hits a low ceiling, validating the rubric's discrimination.

## 4. RL environments for agent training

- **OpenEnv**: Meta-PyTorch's framework for RL environments with composable rubrics, FastAPI-served environments, and Hugging Face Space deployment. ChargebackOps is built directly on `openenv.core.env_server.interfaces.Environment` and `openenv.core.rubrics.{Rubric, WeightedSum, Gate}`.  
  https://github.com/meta-pytorch/OpenEnv
- **BrowserGym**: ServiceNow's browser-task RL environment. Closest in spirit (real-world workflow, partial observability, multi-step) but in a different domain (web navigation vs. financial dispute resolution).  
  https://github.com/ServiceNow/BrowserGym
- **Reasoning Gym**: procedurally-generated reasoning tasks with adjustable difficulty.  
  https://openreview.net/forum?id=GqYSunGmp7

The environment + Rubric system + multi-round adversarial state machine integration in ChargebackOps targets a specific gap in the OpenEnv ecosystem: most existing environments are single-agent puzzle-style or browser-style. A cost-asymmetric multi-round adjudication environment with a programmable Issuer is, to our knowledge, the first of its kind in the OpenEnv catalogue.

## 5. Domain references — chargebacks and dispute resolution

- Visa Compelling Evidence 3.5 (CE 3.5) policy framework. Defines the evidence categories acceptable for representment of fraud-related disputes.
- Mastercard Chargeback Guide. Defines reason codes, response windows, and pre-arbitration thresholds.
- ISO 20022 CASR.003 (Card Issuer-to-Acquirer Chargeback). The standardised message format for cross-network chargeback exchanges; ChargebackOps' [`scenarios/iso_adapter.py`](../scenarios/iso_adapter.py) parses this format directly.
- Stripe Disputes API. Used by [`connectors/stripe_sandbox.py`](../connectors/stripe_sandbox.py) for live or synthetic Stripe-format dispute ingestion.

The domain knowledge encoded in the environment (reason codes, evidence categories, fee schedules, deadline windows) reflects production card-network rules, not stylised abstractions.

## 6. Decision-theoretic foundations

- Howard, *Dynamic Programming and Markov Processes*, 1960. Original framework for optimal policies under uncertainty.
- Puterman, *Markov Decision Processes: Discrete Stochastic Dynamic Programming*, 1994. The cost-asymmetric terminal economics in ChargebackOps (fixed fee + amount forfeit on loss) make each case a non-trivial finite-horizon MDP with risk-sensitive optimal policies.

The "escalate iff `P(win) · amount > $250 fee`" rule encoded in `EscalationROIRubric` is the EV-rational decision criterion under risk neutrality. The rubric does not penalise risk-seeking or risk-averse deviations beyond what their expected-value impact warrants — this is a deliberate choice and a place where extensions could explore CVaR-aware or prospect-theoretic policies.
