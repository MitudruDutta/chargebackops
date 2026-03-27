---
title: ChargebackOps
emoji: 💳
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8000
tags:
  - openenv
---

# ChargebackOps

ChargebackOps is a real-world OpenEnv environment for merchant-side dispute operations. An agent acts like a chargeback analyst: it reviews incoming disputes, decides whether to contest or concede, gathers evidence from synthetic merchant systems, and resolves cases under deadline pressure.

This is not a toy environment and not a retrieval demo. The hard task is a portfolio-optimization problem over a live queue of disputes with different amounts, deadlines, and win profiles.

## Why this environment exists

Chargeback operations are painful, repetitive, and economically important. Real analysts do not just fill forms. They:

- triage cases by deadline and recovery value
- inspect reason-code policy
- gather evidence across internal systems
- avoid harmful or contradictory attachments
- choose whether to contest, accept, or refund
- close cases before network deadlines

That maps cleanly to the standard `reset()` / `step()` / `state()` API and produces deterministic grading.

## Environment design

ChargebackOps simulates a merchant operations stack with fully synthetic data:

- order management
- payment gateway ledger
- shipping and delivery records
- customer support transcripts
- refund ledger
- fraud and device-risk summaries
- dispute policy guidance by reason code

The agent never gets hidden truth directly. It must reveal systems, curate evidence, and resolve cases using typed actions.

## Action space

The action model is [`ChargebackOpsAction`](./models.py) and includes:

- `select_case`: choose which dispute to work on
- `inspect_case`: reveal merchant-side notes for the selected case
- `query_system`: inspect one of `orders`, `payment`, `shipping`, `support`, `refunds`, or `risk`
- `retrieve_policy`: load reason-code guidance and required evidence hints
- `add_evidence`: attach one or more revealed evidence items
- `remove_evidence`: remove attached evidence
- `set_strategy`: set `contest`, `accept_chargeback`, or `issue_refund`
- `submit_representment`: submit the contest package
- `resolve_case`: resolve a case via `accept_chargeback` or `issue_refund`

The full schema is available at `GET /tasks`.

## Observation space

The observation model is [`ChargebackOpsObservation`](./models.py). Each step returns:

- task id, title, objective, and difficulty
- current queue with case amount, reason code, status, and deadline countdown
- selected case workspace
- revealed evidence snippets
- attached evidence
- visible policy guidance
- available actions
- steps remaining
- dense reward in `reward`
- reward breakdown in `metadata.reward_components`
- final grader report when the episode is done

`state()` returns the extended [`ChargebackOpsState`](./models.py) with queue state, action history, and the latest grading report.

## Tasks

ChargebackOps ships with three deterministic tasks.

### 1. Delivered But Disputed

- Difficulty: `easy`
- Goal: contest a `goods_not_received` dispute
- What matters: order confirmation + carrier delivery evidence + submitting before deadline

### 2. Fraud Signal Ambiguity

- Difficulty: `medium`
- Goal: handle a `fraud_cnp` case with both supportive and harmful signals
- What matters: using account-linkage evidence while avoiding AVS/CVV mismatch artifacts

### 3. Dispute Queue Optimization

- Difficulty: `hard`
- Goal: maximize recovery across three simultaneous disputes
- What matters: prioritization, avoiding weak contests, and not missing short deadlines

## Reward shaping

ChargebackOps uses dense trajectory rewards, not a final binary score only.

Positive signals:

- selecting a live case
- revealing a useful system
- attaching helpful evidence
- setting the right strategy
- submitting a valid representment
- resolving a case correctly before deadline

Negative signals:

- duplicate or redundant queries
- invalid actions
- attaching harmful evidence
- late submissions
- contesting unwinnable cases
- leaving cases unresolved when the step budget expires

## Deterministic grading

Each episode ends with a programmatic grader report. Per-case scoring combines:

- strategy correctness
- evidence quality
- packet validity
- deadline compliance
- efficiency
- outcome quality

Scores are normalized to `0.0` to `1.0` and exposed through:

- the final observation
- `state()`
- `GET /grader`

## Baseline providers

The baseline runner now defaults to the more reliable free live path:

- default provider: `groq`
- default model: `llama-3.3-70b-versatile`

The repository also keeps provider integrations for:

- OpenAI
- Anthropic
- Groq
- OpenRouter (`nvidia/nemotron-3-super-120b-a12b:free`)

If no provider key is available, the runner falls back to a deterministic heuristic policy so the project can still be validated locally.
The runner also fast-paths obvious housekeeping actions so live provider calls are spent on genuine branching decisions instead of deterministic retrieval/attach/submit steps.

### Supported environment variables

See [`.env.example`](./.env.example).

Key variables:

- `BASELINE_PROVIDER`
- `BASELINE_MODEL`
- `BASELINE_REQUEST_TIMEOUT_SECONDS`
- `PROVIDER_RATE_LIMIT_RETRIES`
- `PROVIDER_RETRY_BACKOFF_SECONDS`
- `STRICT_LLM_MODE`
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- `INFERENCE_TIMEOUT_SECONDS`
- `OPENROUTER_API_KEY`
- `OPENROUTER_HTTP_REFERER`
- `OPENROUTER_APP_TITLE`
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GROQ_API_KEY`

For `OPENROUTER_HTTP_REFERER`, use the public URL of the deployed app after it exists, such as your Hugging Face Space URL (`https://your-space-name.hf.space`). If nothing is deployed yet, leave it unset. It is optional and only used for OpenRouter app attribution.

`HF_TOKEN` is the generic API key passed to the OpenAI client for the selected `API_BASE_URL`. For OpenRouter, put your OpenRouter key there. For Groq, point `API_BASE_URL` to `https://api.groq.com/openai/v1`, set `MODEL_NAME=llama-3.3-70b-versatile`, and put your Groq key in `HF_TOKEN`.
`PROVIDER_RATE_LIMIT_RETRIES` and `PROVIDER_RETRY_BACKOFF_SECONDS` control bounded retry behavior for transient provider rate limits and timeouts. The default `.env.example` keeps these low on purpose so `inference.py` stays within hackathon runtime expectations.
Set `STRICT_LLM_MODE=1` when you want evaluation to fail immediately on any provider fallback instead of silently dropping to the heuristic policy.

## Baseline scores

Local deterministic fallback baseline:

- `goods_not_received_easy`: `0.7075`
- `fraud_signal_ambiguity`: `0.7075`
- `queue_optimization_hard`: `0.7271`
- average: `0.7140`

When provider credentials are present, the same script and `/baseline` endpoint use the configured LLM provider.
The payload includes `provider_calls_attempted`, `provider_calls_succeeded`, and `provider_errors` so rate-limited free-model runs do not masquerade as successful live inference. If every provider request falls back locally, `mode` is reported as `heuristic_fallback`.

## API surface

OpenEnv endpoints are exposed by the generated server scaffold.

Custom endpoints added by this project:

- `GET /tasks`: list tasks and the action schema
- `GET /grader`: latest grade report, or `?episode_id=<id>` for a specific episode
- `GET /baseline`: run the baseline with optional `provider` and `model_name`

## Local setup

### 1. Install dependencies

```bash
pip install -e .[dev]
```

### 2. Run the server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### 3. Run tests

```bash
pytest -q tests
```

### 3a. Run the problem-statement audit

```bash
python scripts/problem_statement_audit.py
```

This audit checks the environment against the challenge brief:

- easy / medium / hard task coverage
- deterministic grader behavior
- partial-progress reward shaping
- separation between a competent policy and a bad control policy
- `inference.py` contract
- `openenv validate`
- baseline and inference execution

This audit disables live provider keys on purpose so it stays deterministic and fast.

### 3b. Run the live-provider audit

```bash
python scripts/live_provider_audit.py
```

Use this when you want to see whether the configured provider is actually making decisions live, how many provider calls succeeded, and whether fallback was used.
The output also includes `provider_errors` so you can distinguish rate limits from connectivity or response-format failures.

### 4. Run the baseline

```bash
python scripts/run_baseline.py
```

### 5. Run the submission inference script

```bash
python inference.py
```

This script uses the challenge-style environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

To use a provider-backed baseline:

```bash
BASELINE_PROVIDER=groq BASELINE_MODEL=llama-3.3-70b-versatile python scripts/run_baseline.py
```

To force the OpenRouter free path:

```bash
BASELINE_PROVIDER=openrouter BASELINE_MODEL=nvidia/nemotron-3-super-120b-a12b:free python scripts/run_baseline.py
```

## Docker

Build from the project root:

```bash
docker build -t chargebackops .
docker run -p 8000:8000 chargebackops
```

The repository also includes the OpenEnv scaffold Dockerfile at [`server/Dockerfile`](./server/Dockerfile).

## Hugging Face Spaces

This repository is ready for a Docker-based Hugging Face Space.

Typical workflow:

```bash
openenv validate .
openenv push
```

## File layout

```text
chargeback_ops/
├── .env.example
├── README.md
├── baseline_runner.py
├── client.py
├── episode_store.py
├── grading.py
├── models.py
├── openenv.yaml
├── pyproject.toml
├── simulation.py
├── scripts/
│   └── run_baseline.py
├── server/
│   ├── app.py
│   ├── chargeback_ops_environment.py
│   └── Dockerfile
└── tests/
    ├── conftest.py
    ├── test_api.py
    ├── test_env.py
    └── test_grader.py
```

## Notes

- All cases and merchant data are synthetic.
- The environment idea remains fixed: merchant chargeback representment and dispute handling.
- The provider layer is configurable, but the benchmark logic and task design are deterministic.
