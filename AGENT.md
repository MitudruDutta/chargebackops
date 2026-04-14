# ChargebackOps Agent: Complete Technical Reference

This document explains every aspect of the ChargebackOps agent -- what problem it solves, how it thinks, how it is scored, and why every design decision was made.

---

## Table of Contents

- [The Problem](#the-problem)
- [The Use Case](#the-use-case)
- [How the Environment Works](#how-the-environment-works)
- [How the Agent Works](#how-the-agent-works)
- [The Three-Tier Decision Pipeline](#the-three-tier-decision-pipeline)
- [Reason Code Strategies](#reason-code-strategies)
- [Multi-Case Triage](#multi-case-triage)
- [Evidence Handling](#evidence-handling)
- [Representment Notes](#representment-notes)
- [The Grading System](#the-grading-system)
- [LLM Integration](#llm-integration)
- [Key Optimizations](#key-optimizations)
- [File Map](#file-map)

---

## The Problem

When a customer disputes a credit card charge, the card network (Visa, Mastercard) initiates a **chargeback** against the merchant. The merchant loses the transaction amount immediately. To recover the funds, the merchant must build a **representment package** -- a bundle of evidence proving the charge was legitimate -- and submit it before a hard deadline.

This is not a simple yes/no decision. Each dispute has:

- A **reason code** (why the customer disputes: fraud, goods not received, product not as described, etc.)
- A **deadline** (fixed number of steps before the case auto-closes against the merchant)
- **Evidence** scattered across 6 internal merchant systems (orders, payment, shipping, support, refunds, risk)
- Some evidence is **helpful**, some is **required**, and some is **harmful** (weakens the case if included)
- A correct **strategy** that depends on the evidence available (contest, accept the chargeback, or issue a refund)

A human dispute analyst handles 50-200 cases per day. They must triage by urgency, query the right systems, avoid attaching damaging evidence, and submit within deadline. Mistakes are expensive: a lost $500 chargeback plus a $25 network fee per case.

**ChargebackOps turns this into a measurable agent benchmark.** The agent must do exactly what a human analyst does, but programmatically, under step-budget constraints, with deterministic scoring.

---

## The Use Case

ChargebackOps is built for the [OpenEnv](https://meta-pytorch.org/OpenEnv/index.html) evaluation framework. It is a **simulated merchant dispute resolution environment** where an AI agent acts as the dispute analyst.

**What the agent receives:**
- A queue of 1-6 open dispute cases (5-6 at nightmare difficulty)
- A step budget (10-20 actions total, ~2.4 steps/case at nightmare)
- Per-case deadlines (must resolve before step N)

**What the agent must do:**
- Select and focus on one case at a time
- Query internal merchant systems to retrieve evidence
- Decide whether to contest, accept, or refund each case
- Attach the right evidence (and avoid harmful artifacts)
- Write a representment note explaining why the dispute should be reversed
- Submit or resolve each case before its deadline
- Manage step budget across all cases when there are more cases than steps

**What the agent is scored on:**
- Did it choose the correct strategy? (25% of score)
- Did it gather the right evidence? (20%)
- Is the evidence packet complete and clean? (15%)
- Did it meet the deadline? (15%)
- Was it efficient (no wasted steps)? (10%)
- Did the resolution match the strategy? (10%)
- Is the representment note well-written? (5%)

---

## How the Environment Works

The environment follows the OpenEnv `reset()` / `step()` / `state()` contract.

### Lifecycle

```
reset(task_id) → Observation
step(action)   → Observation
state()        → State (includes grader report when done)
```

### Observation

Each observation contains:

| Field | Type | Description |
|---|---|---|
| `queue` | list | All cases with status, reason_code, amount, steps_until_deadline |
| `visible_case` | object or null | The currently selected case with full detail |
| `steps_remaining` | int | Steps left before episode ends |
| `done` | bool | Whether the episode is complete |
| `reward` | float | Immediate reward from the last action |
| `result` | string | Human-readable outcome of the last action |

### The Visible Case

When a case is selected, `visible_case` exposes:

| Field | Description |
|---|---|
| `case_id` | Unique identifier |
| `reason_code` | Why the customer disputed (e.g., `goods_not_received`) |
| `amount` | Transaction amount in dollars |
| `current_strategy` | Currently set strategy (null if not set) |
| `policy` | Policy guidance (null until `retrieve_policy` is called) |
| `systems_revealed` | Which merchant systems have been queried |
| `retrieved_evidence` | Evidence items revealed by queries |
| `attached_evidence` | Evidence currently attached to the representment package |
| `inspection_notes` | Analyst notes (null until `inspect_case` is called) |

### Action Space (9 Actions)

| Action | Arguments | Cost | What It Does |
|---|---|---|---|
| `select_case` | case_id | 1 step | Focus on a case from the queue |
| `inspect_case` | case_id | 1 step | Reveal analyst inspection notes (+0.04 reward) |
| `query_system` | case_id, system_name | 1 step | Pull evidence from orders/payment/shipping/support/refunds/risk |
| `retrieve_policy` | case_id | 1 step | Get reason-code-specific guidance and required evidence list |
| `add_evidence` | case_id, evidence_ids | 1 step | Attach evidence to the representment package |
| `remove_evidence` | case_id, evidence_ids | 1 step | Remove evidence (useful for cleaning harmful attachments) |
| `set_strategy` | case_id, strategy | 1 step | Choose contest / accept_chargeback / issue_refund |
| `submit_representment` | case_id, note | 1 step | Submit the contest package (requires strategy = contest) |
| `resolve_case` | case_id, strategy | 1 step | Close a non-contest case (accept or refund) |

Every action costs exactly 1 step. There is no free action. The agent must be deliberate about every step it takes.

### Reward Signals

The environment returns immediate rewards after each action:

| Event | Reward |
|---|---|
| Select an open case | +0.02 |
| Inspect a case (first time) | +0.04 |
| Query a new system with helpful evidence | +0.06 to +0.08 |
| Query a new system with no useful evidence | -0.01 to +0.01 |
| Query an already-queried system (duplicate) | -0.03 |
| Attach helpful evidence | +0.08 per piece |
| Attach harmful evidence | -0.08 per piece |
| Attach neutral evidence | +0.01 |
| Remove harmful evidence | +0.05 |
| Remove helpful evidence | -0.03 |
| Set optimal strategy | +0.10 |
| Set acceptable strategy | +0.03 |
| Set wrong strategy | -0.08 |
| Submit a strong representment on time | +0.20 |
| Submit after deadline | -0.20 |
| Submit with missing required evidence | -0.18 |
| Submit with harmful evidence attached | -0.15 |
| Contest a case that shouldn't be contested | -0.12 |
| Resolve with optimal strategy | +0.16 |
| Resolve with acceptable strategy | +0.06 |
| Resolve with wrong strategy | -0.12 |
| Invalid action | -0.12 |

These rewards are shaping signals. The final score comes from the deterministic grader, not reward accumulation.

---

## How the Agent Works

The agent is implemented in `baseline_runner.py`. It is a **heuristic-first, LLM-augmented** policy. The heuristic handles ~90% of decisions deterministically. The LLM is only called when the heuristic encounters genuine ambiguity (multiple meaningfully different candidates).

### Why Heuristic-First?

1. **Reliability**: Heuristic decisions never fail, never timeout, never cost money.
2. **Speed**: No network round-trip for obvious moves.
3. **Determinism**: Same input always produces same output (important for reproducibility).
4. **Budget**: LLM calls cost tokens and have rate limits. The agent makes 6-20 decisions per episode.

The LLM acts as a **tiebreaker** when the heuristic produces multiple viable candidates that differ in action type. This hybrid approach gets the best of both worlds.

---

## The Three-Tier Decision Pipeline

Every step, the agent runs this pipeline:

### Tier 1: `candidate_actions(observation)`

Reads the current observation and generates a list of `CandidateAction` objects -- the legal moves the agent considers. This is the core intelligence of the agent.

The function applies these checks in strict priority order:

1. **No case selected?** Generate `select_case` candidates sorted by triage priority.

2. **Current case resolved?** Switch to an open case.

3. **Harmful evidence attached?** Immediately generate `remove_evidence` and return. This fires before anything else because harmful evidence torpedoes the packet_validity score (15% of total).

4. **Deadline <= 1 step?** Emergency submit or resolve. No time for anything else.

5. **Budget too tight to contest?** If there aren't enough steps to run the full contest path (minimum 5 steps: policy + query + attach + strategy + submit), or if this is the lowest-value case in a multi-case triage, fast-concede with `issue_refund`.

6. **Budget pressure (steps <= cases * 2)?** If the inferred strategy is accept/refund, resolve immediately.

7. **Reason code handler**: Dispatches to a reason-code-specific handler that generates the appropriate sequence of queries, evidence attachments, strategy setting, and submission.

### Tier 2: `_obvious_next_action(observation, candidates)`

Before calling any LLM, checks if the choice is trivial:
- Only 1 candidate? Take it.
- All candidates have the same action type? Take the first.
- One candidate targets a case with much tighter deadline? Take it.

If obvious, the LLM is skipped entirely.

### Tier 3: LLM or `_heuristic_pick(candidates)`

When Tier 2 returns None (genuine ambiguity):
- **With LLM**: Sends the observation summary and candidate list as a JSON prompt. The model returns `{"candidate_index": N, "rationale": "..."}`. On failure, walks the fallback chain (OpenRouter -> Gemini -> Groq).
- **Without LLM**: `_heuristic_pick()` returns the first candidate (the heuristic already sorted by priority).

---

## Reason Code Strategies

The agent handles 6 reason code families, each with a different workflow:

### `goods_not_received` (Deterministic: contest)

The customer claims they never received the product. The merchant almost always has delivery proof.

**Steps**: select -> query orders -> query shipping -> attach delivery evidence -> set_strategy contest -> submit representment

**Systems queried**: orders, shipping
**Typical evidence**: Order confirmation, delivery scan, tracking number
**Strategy**: Always contest (delivery proof is definitive)

### `fraud_cnp` (Non-deterministic: contest or accept)

Card-not-present fraud. The customer claims they didn't authorize the transaction. This is the most nuanced reason code -- sometimes the merchant has strong evidence (prior good orders, account match), sometimes they don't.

**Steps**: select -> retrieve_policy -> query risk + support (+ orders if budget allows) -> attach non-harmful evidence -> set_strategy per policy -> submit or resolve

**Systems queried**: risk, support, orders (optional under tight budget)
**Typical evidence**: Risk assessment, prior order linkage, account verification
**Harmful evidence**: AVS mismatch, CVV mismatch (proves the card data didn't fully match)
**Strategy**: Contest if strong evidence exists, accept_chargeback if evidence is weak

### `credit_not_processed` (Deterministic: issue_refund)

The customer claims a refund was promised but never issued. The correct response is to issue the refund.

**Steps**: select -> set_strategy issue_refund -> resolve_case (3 steps total)
**Strategy**: Always issue_refund (cheapest to resolve, no contest needed)

### `duplicate_processing` (Deterministic: issue_refund)

The customer was charged twice. The correct response is to refund the duplicate.

**Steps**: select -> set_strategy issue_refund -> resolve_case (3 steps total)
**Strategy**: Always issue_refund

### `product_not_as_described` (Non-deterministic: contest or accept)

The customer claims the product didn't match its description. Success depends on whether the merchant has listing accuracy proof and whether the customer bypassed the return process.

**Steps**: select -> retrieve_policy -> query orders + support (+ shipping if deadline allows) -> attach listing/return evidence -> contest or accept per guidance

**Systems queried**: orders, support, shipping (optional)
**Strategy**: Contest if listing proof is strong, accept_chargeback if not supportable

### `service_not_provided` (Non-deterministic: contest or accept)

The customer claims a service was never delivered. Success depends on completion records and customer acknowledgment.

**Steps**: select -> retrieve_policy -> query support (+ orders if deadline allows) -> attach completion evidence -> contest or accept per guidance

**Systems queried**: support, orders (optional)
**Strategy**: Contest if service completion proof exists, accept_chargeback otherwise

---

## Multi-Case Triage

When the agent has multiple open cases and the total estimated step cost exceeds the budget, it uses a triage algorithm:

### Step Cost Estimates

| Reason Code | Est. Steps | Notes |
|---|---|---|
| `goods_not_received` | 6 | select + 2 queries + attach + strategy + submit |
| `credit_not_processed` | 3 | select + strategy + resolve |
| `duplicate_processing` | 3 | select + strategy + resolve |
| `fraud_cnp` | 8 | select + policy + 2-3 queries + attach + strategy + submit |
| `product_not_as_described` | 8 | select + policy + 2-3 queries + attach + strategy + submit |
| `service_not_provided` | 7 | select + policy + 2 queries + attach + strategy + submit |

### Triage Algorithm

```
1. If total_estimated_cost > steps_remaining:
     Sort cases: deterministic-strategy codes first, then by amount descending.
     This ensures cheap, guaranteed-outcome cases are handled first,
     and the highest-value non-deterministic cases get remaining budget.

2. When processing each case, check:
   - Is steps_remaining < 5? → Fast-concede (can't even minimally contest).
   - Is this the lowest-value case and total_cost > budget? → Fast-concede.
   - Otherwise → Full contest or policy-guided resolution.

3. Never interrupt a near-complete case:
   - If the current case has evidence attached and is 1-2 steps from
     submission, finish it before switching to another case's deadline.
```

### Why This Ordering Works

- **credit_not_processed/duplicate_processing** cost 3 steps and always get optimal score. Handle them first to free budget.
- **goods_not_received** costs 6 steps and always contests. Handle next.
- **fraud_cnp/product_not_as_described/service_not_provided** cost 7-8 steps and may need to concede. Handle last -- if budget runs out, conceding these with `issue_refund` (an acceptable fallback) still earns 35% strategy correctness.

---

## Evidence Handling

### Harmful Evidence Detection

The agent maintains a set of 15 negative-signal keywords derived from real chargeback dispute patterns:

```
mismatch, failed, declined, suspicious, flagged, fraud risk,
unauthorized, rejected, invalid, expired, violation,
non-compliant, discrepancy, inconsistent, unverified
```

Every evidence item's title and summary are scanned. If any harmful keyword is found, the evidence is:
1. **Never attached** (ranked 999 in the priority sort, excluded from `add_evidence` calls)
2. **Removed if already attached** (a `remove_evidence` action is generated immediately before any other action)

### Evidence Priority Ranking

Non-harmful evidence is ranked by keyword relevance:

| Rank | Keywords | Example |
|---|---|---|
| 0 (highest) | signature, completion, booking, listing | "Delivery signature scan" |
| 1 | duplicate, delivery, prior, account, authenticated | "Prior good order linkage" |
| 2 | return policy, refund, cancel, confirmation, cancellation | "Return policy documentation" |
| 4 (default) | anything else | "Internal memo" |
| 999 (excluded) | mismatch, failed, declined, suspicious, flagged, fraud risk, unauthorized, rejected, invalid, expired, violation, non-compliant, discrepancy, inconsistent, unverified | "AVS mismatch report" |

### Attachment Strategy

The agent attaches **all** non-harmful retrieved evidence in a single `add_evidence` call. This maximizes the evidence_quality score, which rewards `helpful_attached / total_helpful`.

---

## Representment Notes

When the agent submits a contest, it generates a representment note. The grader scores notes on 4 dimensions:

| Dimension | Weight | What Earns Points |
|---|---|---|
| Substance | 20% | Note has >= 5 words |
| Policy claims coverage | 50% | Note mentions keywords from `case.policy_requirements` (e.g., "order confirmation", "carrier delivery") |
| Evidence coherence | 15% | Note references attached evidence IDs (e.g., "E1-ORDER-CONF") |
| Harmful mention penalty | -15% each | Note contains words like "mismatch", "failed", "declined" |

### How the Agent Builds Notes

1. Start with a **reason-code-specific template** that uses policy requirement language:
   - goods_not_received: "Order confirmation and carrier delivery confirmation establish fulfillment..."
   - fraud_cnp: "Prior good order linkage and customer account confirmation..." (never mentions "mismatch")
   - product_not_as_described: "Product listing verification confirms..."
   - service_not_provided: "Service completion record and customer acknowledgment..."

2. If policy was retrieved, append the policy requirements directly:
   - "Evidence covers: order confirmation, carrier delivery confirmation."

3. Append evidence IDs for coherence scoring:
   - "Supporting evidence: E1-ORDER-CONF, E1-DELIVERY-SCAN."

4. Truncate to 500 characters.

---

## The Grading System

After all cases are resolved (or the step budget is exhausted), the grader scores each case across 7 dimensions. Each dimension is an OpenEnv `Rubric` subclass defined in `evaluation/rubrics.py`; they compose into a per-case `WeightedSum` and an episode-level `ChargebackOpsEpisodeRubric` that is wired into `env.rubric`. `evaluation/grading.py` keeps the legacy `score_case` / `grade_episode` API as a thin adapter over the rubric tree.

### Strategy Correctness (25%)

| Outcome | Score |
|---|---|
| Chose the optimal strategy | 1.0 |
| Chose an acceptable fallback | 0.35 |
| Chose the wrong strategy | 0.0 |

"Optimal" and "acceptable" strategies are defined per case by the task scenario. For example, `goods_not_received` optimal is always "contest" with no acceptable fallback. `fraud_cnp` optimal might be "contest" with "accept_chargeback" as acceptable, or vice versa.

### Evidence Quality (20%)

For **contest** cases:
```
quality = 0.7 * (required_attached / required_total)
        + 0.3 * (helpful_attached / helpful_total)
        - 0.25 * harmful_attached_count
```

For **non-contest** cases where optimal strategy is also non-contest:
- 1.0 if no evidence was attached (clean concession)
- 0.7 if evidence was attached (unnecessary work)

For **non-contest** cases where optimal was contest:
- 0.15 (the agent abandoned evidence gathering for a contestable case)

### Packet Validity (15%)

Binary, all-or-nothing:
- **1.0** if ALL required evidence is attached AND zero harmful evidence is attached
- **0.0** otherwise

This is the strictest dimension. Missing one required piece or having one harmful piece zeroes it out.

### Deadline Compliance (15%)

Binary:
- **1.0** if the case was resolved at or before the deadline step
- **0.0** if resolved after the deadline or never resolved

### Efficiency (10%)

```
efficiency = 1.0 - min(0.9, (duplicate_queries + invalid_actions) * 0.1 + submit_attempts * 0.05)
```

The agent loses 0.1 per duplicate system query or invalid action, and 0.05 per submit attempt. Minimum efficiency is 0.0.

Additional penalties for shallow operational behaviour:
- **Over-querying a concedable case**: -0.15 per system queried beyond the 2nd when the agent concedes a case whose optimal strategy is also non-contest. Querying 4+ systems before conceding is wasteful.
- **Late policy retrieval**: -0.08 when policy is retrieved but the case is resolved with a concession that matches the optimal non-contest strategy. The policy step was wasted.
- **Early correct concession bonus**: +0.10 when the agent correctly concedes a case (matching optimal) within 3 steps. Rewards recognising a bad case quickly.

### Outcome Quality (10%)

| Outcome | Score |
|---|---|
| Final resolution matches optimal strategy | 1.0 |
| Final resolution is an acceptable fallback | 0.4 |
| Final resolution is wrong | 0.0 |

### Note Quality (5%)

Only scored for contest cases with a representment note. See [Representment Notes](#representment-notes) for the scoring breakdown.

### Final Score Calculation

```
case_score = 0.25 * strategy_correctness
           + 0.20 * evidence_quality
           + 0.15 * packet_validity
           + 0.15 * deadline_compliance
           + 0.10 * efficiency
           + 0.10 * outcome_quality
           + 0.05 * note_quality

weighted_case_score = case_score * case_weight

episode_score = sum(weighted_case_scores) / sum(case_weights)
```

Case weights are determined by financial impact (amount and difficulty). The episode score normalizes to `[0.0, 1.0]`.

---

## LLM Integration

The agent supports 5 LLM providers through OpenAI-compatible clients:

| Provider | Model | Base URL |
|---|---|---|
| OpenRouter | openai/gpt-oss-120b | openrouter.ai/api/v1 |
| Google Gemini | gemini-2.5-flash | generativelanguage.googleapis.com/v1beta/openai/ |
| Groq | llama-3.3-70b-versatile | api.groq.com/openai/v1 |
| OpenAI | gpt-4.1-mini | api.openai.com/v1 |
| Anthropic | claude-sonnet-4 | (compatible gateway) |

### Fallback Chain

```
Primary (configured in .env) → OpenRouter → Google Gemini → Groq → Heuristic
```

If the primary provider fails (timeout, rate limit, connection error), the agent automatically tries the next provider in the chain. If all providers fail, it falls back to `_heuristic_pick()`.

### What the LLM Sees

```json
{
  "queue_summary": "2 open cases, 8 steps remaining",
  "visible_case": "CB-G1, fraud_cnp, $480, deadline in 3 steps",
  "candidates": [
    {"index": 0, "action": "submit_representment", "summary": "Submit the contest package"},
    {"index": 1, "action": "query_system orders", "summary": "Query orders for more evidence"},
    {"index": 2, "action": "select_case CB-G2", "summary": "Switch to case with deadline in 1 step"}
  ]
}
```

### What the LLM Returns

```json
{
  "candidate_index": 0,
  "rationale": "CB-G1 has sufficient evidence and contesting before deadline takes priority over gathering more evidence."
}
```

### Configuration

| Env Variable | Default | Purpose |
|---|---|---|
| `BASELINE_PROVIDER` | openrouter | Primary LLM provider |
| `BASELINE_MODEL` | openai/gpt-oss-120b | Model to use |
| `BASELINE_REQUEST_TIMEOUT_SECONDS` | 15 | Per-call timeout |
| `PROVIDER_RATE_LIMIT_RETRIES` | 2 | Retry count on rate limits |
| `PROVIDER_RETRY_BACKOFF_SECONDS` | 1.0 | Backoff between retries |
| `MAX_PROVIDER_RESPONSE_TOKENS` | 200 | Max tokens for LLM response |
| `STRICT_LLM_MODE` | false | If true, fail instead of falling back to heuristic |

---

## Key Optimizations

### 1. Deterministic Strategy Inference

For reason codes where the optimal strategy never varies (`goods_not_received` = contest, `credit_not_processed` / `duplicate_processing` = issue_refund), the agent skips `retrieve_policy` entirely. This saves 1 step per case.

### 2. Deadline-Aware Query Limiting

When the remaining steps before deadline can't accommodate all planned queries, the agent reduces the number of systems queried:

- `product_not_as_described`: drops from 3 systems (orders, support, shipping) to 2 (orders, support)
- `fraud_cnp`: drops from 3 systems (risk, support, orders) to 2 (risk, support)
- `service_not_provided`: drops from 2 systems (orders, support) to 1 (support)

### 3. Near-Completion Protection

When the current case has evidence attached and is 1-2 steps from submission, the agent does NOT switch to handle another case's deadline. Finishing the current case is almost always higher-value than interrupting.

### 4. Harmful Evidence Cleanup

Before any submit, the agent checks for harmful evidence in the attached set. If found, it generates a `remove_evidence` action immediately. This prevents the -0.25 per-piece evidence_quality penalty and the packet_validity = 0.0 failure.

### 5. Budget-Aware Note Generation

Representment notes are generated with direct references to policy requirement keywords and evidence IDs, maximizing the note_quality score (policy claims coverage 50% + evidence coherence 15%).

### 6. Adversarial Evidence (Hard/Nightmare)

At hard and nightmare difficulty, the case generator injects **adversarial evidence** — items whose titles sound helpful ("Delivery verification report", "Account verification summary") but whose content is harmful (GPS discrepancies, prior non-receipt claims, failed 3D Secure challenges). This tests whether the agent reads beyond titles and inspects evidence content before attaching.

### 7. Nightmare Difficulty

Nightmare tasks push the step budget to its limit: 5-6 cases with ~2.4 steps per case. The agent must triage aggressively — fast-conceding weak cases, handling deterministic codes first, and accepting that some cases will go unresolved. This tier specifically tests prioritisation under extreme resource pressure.

---

## File Map

| File | Purpose | Lines |
|---|---|---|
| `runners/baseline_runner.py` | The agent: decision pipeline, candidate generation, LLM integration, representment notes | ~1100 |
| `server/chargeback_ops_environment.py` | The environment: step/reset/state, action execution, reward computation | ~500 |
| `evaluation/rubrics.py` | OpenEnv `Rubric` subclasses for all 7 scoring dimensions, composed via `WeightedSum` | ~300 |
| `evaluation/grading.py` | Legacy `score_case` / `grade_episode` adapter that delegates to the rubric tree | ~120 |
| `scenarios/simulation.py` | Task definitions, case progress tracking, evidence metadata | ~600 |
| `core/models.py` | Pydantic models for actions, observations, state, grading | ~600 |
| `runners/inference.py` | OpenEnv-compatible inference entry point with provider fallback | ~200 |
| `inference.py` | Root re-export for submission contract | ~10 |
| `scenarios/case_generator.py` | Parametric task generator with seeded RNG | ~700 |
| `scenarios/iso_adapter.py` | Converts ISO 20022 CASR.003 records to environment cases | ~160 |
| `connectors/stripe_sandbox.py` | Maps Stripe test-mode disputes to environment cases | ~280 |
| `evaluation/agent_brutal_audit.py` | 126-episode evaluation across all data sources | ~300 |
| `server/app.py` | FastAPI routes: /reset, /step, /state, /tasks, /baseline, /grader, /results, /demo | ~200 |
| `server/demo_ui.py` | Gradio live demo UI with step-by-step episode playback | ~150 |
| `core/episode_store.py` | Thread-safe storage with JSONL file persistence | ~60 |
| `core/client.py` | OpenEnv WebSocket client | ~100 |

---

## Performance

Tested across the 10-task benchmark (3 showcase + 7 seeded holdout):

| Difficulty | Tasks | Heuristic | LLM tiebreak | Bad | Key Observations |
|---|---|---|---|---|---|
| Easy | 3 | 0.964 | 0.778 | 0.365 | Near-perfect heuristic; LLM mispicks on `fraud_signal_ambiguity` |
| Medium | 2 | 0.755 | 0.608 | 0.278 | Strategy selection + evidence curation drive the spread |
| Hard | 3 | 0.680 | 0.697 | 0.113 | LLM edges heuristic on `queue_optimization_hard` |
| Nightmare | 2 | 0.466 | 0.289 | 0.065 | 5-case portfolios with deadline_step=3–5; step budget collides |
| **Overall** | **10** | **0.738** | **0.622** | **0.212** | **Delta 0.526 vs bad policy** |

The difficulty curve demonstrates the environment discriminates effectively: easy tasks are near-trivial, nightmare tasks push every agent below 50%. The `Gate(CaseAbandonedRubric)` wrapper hard-zeros cases left unresolved past their deadline, so the heuristic's slowness on nightmare portfolios shows up as a real signal rather than dilution across 7 partial dimensions. See `docs/RESULTS.md` for full per-task numbers.
