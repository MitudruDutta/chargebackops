---
title: ChargebackOps
emoji: "💳"
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
---

# ChargebackOps

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![OpenEnv](https://img.shields.io/badge/OpenEnv-Compatible-111827)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)

ChargebackOps is an OpenEnv environment for merchant-side chargeback and dispute operations. The agent acts as a dispute analyst: it triages open disputes, retrieves evidence from merchant systems, decides whether to contest or concede, and resolves each case under deadline and step-budget pressure.

The repository is designed for agent evaluation rather than generic chat. It exposes a typed action space, deterministic state transitions, dense reward shaping, and programmatic grading so model behavior can be measured as operational performance.

## Why This Environment Matters

Chargeback dispute handling is a real operations workflow. Analysts must:

- interpret reason codes and response deadlines
- gather evidence from the correct internal systems while avoiding harmful artifacts
- decide whether to contest, accept, or refund
- prioritize multiple disputes by urgency, recoverability, and operational cost

That makes ChargebackOps a good benchmark for tool-using agents. It measures retrieval quality, decision quality, prioritization, and operational restraint in a controlled environment with deterministic scoring.

## System Architecture

```mermaid
graph TB
    subgraph Agent["Agent Layer"]
        INF["inference.py\nOpenAI-compatible client\nProvider fallback chain"]
        BL["baseline_runner.py\nThree-tier decision pipeline\nHeuristic + LLM hybrid"]
    end

    subgraph API["API Layer"]
        APP["FastAPI server\nserver/app.py"]
        WS["OpenEnv WebSocket\nclient.py"]
    end

    subgraph Core["Environment Core"]
        ENV["ChargebackOpsEnvironment\nstep() / reset() / state()"]
        SIM["Simulation Engine\nsimulation.py"]
        GRD["Deterministic Grader\n7-dimension scoring"]
        STORE["Episode Store\nepisode_store.py"]
    end

    subgraph Tasks["Task Sources"]
        FIXED["Built-in Tasks\n3 handcrafted scenarios"]
        GEN["Parametric Generator\ncase_generator.py\nSeeded RNG, infinite tasks"]
        ISO["ISO 20022 Adapter\niso_adapter.py\n300 real chargeback records"]
        STRIPE["Stripe Connector\nconnectors/stripe_sandbox.py\nLive API or synthetic disputes"]
    end

    subgraph Systems["Merchant Systems (6)"]
        ORD["Orders"] --- PAY["Payment"]
        SHIP["Shipping"] --- SUP["Support"]
        REF["Refunds"] --- RISK["Risk"]
    end

    INF --> APP
    BL --> ENV
    APP --> ENV
    WS --> APP
    ENV --> SIM
    ENV --> GRD
    GRD --> STORE
    SIM --> FIXED
    SIM --> GEN
    SIM --> ISO
    SIM --> STRIPE
    ENV --> Systems
```

## Agent Decision Pipeline

The agent operates a three-tier decision pipeline on every step. Each observation passes through candidate generation, obvious-move detection, and then either LLM or heuristic resolution.

```mermaid
flowchart TD
    OBS["Observation\n(queue, visible_case, steps_remaining)"] --> CA

    subgraph Tier1["Tier 1: Candidate Generation"]
        CA["candidate_actions()"] --> HARM{"Harmful evidence\nattached?"}
        HARM -->|"Yes"| REM["remove_evidence\n(immediate return)"]
        HARM -->|"No"| DL{"Deadline\n<= 1 step?"}
        DL -->|"Yes"| URG["URGENT: submit or resolve\n(immediate return)"]
        DL -->|"No"| BUD{"Budget too tight\nto contest?"}
        BUD -->|"Yes: steps < 5 or\nlowest-value in triage"| CONC["Fast-concede\nwith issue_refund"]
        BUD -->|"No"| BP{"Budget pressure?\nsteps <= cases * 2"}
        BP -->|"Yes + concedable"| FAST["Fast set_strategy\n+ resolve_case"]
        BP -->|"No"| RC{"Reason code\nhandler"}
    end

    subgraph Handlers["Reason Code Handlers"]
        RC --> GNR["goods_not_received\nquery orders+shipping\nattach delivery proof\ncontest"]
        RC --> FRD["fraud_cnp\ncheck inferred_strategy\nquery risk+support(+orders)\ncontest or accept"]
        RC --> CNP["credit_not_processed\nduplicate_processing\nset issue_refund\nresolve immediately"]
        RC --> PNA["product_not_as_described\nretrieve policy first\ncontest or accept per guidance"]
        RC --> SNP["service_not_provided\nretrieve policy first\ncontest or accept per guidance"]
        RC --> UNK["Unknown reason code\nquery all systems\ndefault to contest"]
    end

    subgraph Tier2["Tier 2: Obvious Move Detection"]
        GNR & FRD & CNP & PNA & SNP & UNK --> OBV["_obvious_next_action()"]
        OBV --> OC{"Only 1 candidate\nor all same type?"}
        OC -->|"Yes"| TAKE["Take it\n(skip LLM)"]
    end

    subgraph Tier3["Tier 3: Ambiguity Resolution"]
        OC -->|"No: genuine\nambiguity"| LLM{"LLM available?"}
        LLM -->|"Yes"| CALL["Provider call\nOpenRouter → Gemini → Groq\nJSON: candidate_index + rationale"]
        LLM -->|"No"| HEUR["_heuristic_pick()\nFirst candidate wins"]
        CALL -->|"Success"| ACT["Execute chosen action"]
        CALL -->|"All providers fail"| HEUR
    end

    TAKE --> ACT
    HEUR --> ACT
    REM --> ACT
    URG --> ACT
    CONC --> ACT
    FAST --> ACT
    ACT --> STEP["env.step(action)\nReturns new observation"]
    STEP --> DONE{"Episode done?"}
    DONE -->|"No"| OBS
    DONE -->|"Yes"| GRADE["Grader scores\neach case"]

    style OBS fill:#1a3a5c,color:#fff
    style GRADE fill:#2d5016,color:#fff
    style URG fill:#8b0000,color:#fff
    style CONC fill:#8b4513,color:#fff
    style REM fill:#800080,color:#fff
```

## Case Triage (Multi-Case Scenarios)

When the agent faces multiple open cases with insufficient budget to contest all, it employs a triage strategy:

```mermaid
flowchart LR
    START["Multiple open cases\ntotal_cost > budget"] --> SORT["Sort by:\n1. Deterministic codes first\n2. Highest amount first"]
    SORT --> LOOP{"Next case"}
    LOOP --> DET{"Deterministic\nstrategy?"}
    DET -->|"Yes: goods_not_received\ncredit_not_processed\nduplicate_processing"| CHEAP["Handle first\n(3-6 steps, no policy needed)"]
    DET -->|"No: fraud_cnp\nproduct_not_as_described\nservice_not_provided"| CHECK{"Steps remaining\n>= 5?"}
    CHECK -->|"Yes"| CONTEST["Retrieve policy\nContest or concede\nper guidance"]
    CHECK -->|"No"| CONCEDE["Fast-concede\nwith issue_refund\n(1 step)"]
    CHEAP --> NEXT["Done, next case"]
    CONTEST --> NEXT
    CONCEDE --> NEXT
    NEXT --> LOOP

    style START fill:#1a3a5c,color:#fff
    style CONCEDE fill:#8b4513,color:#fff
    style CHEAP fill:#2d5016,color:#fff
```

## Episode Workflow

```mermaid
flowchart TD
    A["reset(task_id)"] --> SEL["Select case from queue\n(priority: fast codes first,\nthen by amount desc)"]
    SEL --> DET{"Reason code in\ndeterministic map?"}
    DET -->|"goods_not_received"| SKIP["Infer strategy: contest\nSkip policy retrieval"]
    DET -->|"credit_not_processed\nduplicate_processing"| REFUND["Infer strategy: issue_refund\nResolve immediately"]
    DET -->|"fraud_cnp\nproduct_not_as_described\nservice_not_provided\nor unknown"| POL["Retrieve policy guidance"]
    POL --> PARSE{"Parse guidance"}
    PARSE -->|"'do not contest'\n'concede'\n'not supportable'"| ACCEPT["Strategy: accept_chargeback"]
    PARSE -->|"'refund immediately'"| REFUND2["Strategy: issue_refund"]
    PARSE -->|"Otherwise"| CONT["Strategy: contest"]

    SKIP --> QUERY["Query merchant systems\n(deadline-aware: fewer queries\nwhen deadline is tight)"]
    CONT --> QUERY

    QUERY --> ATTACH["Attach all non-harmful evidence\n(filter by 6 harmful keywords:\nmismatch, failed, declined,\nsuspicious, flagged, fraud risk)"]
    ATTACH --> HARMFUL{"Any harmful\nevidence attached?"}
    HARMFUL -->|"Yes"| REMOVE["remove_evidence\n(clean before submit)"]
    HARMFUL -->|"No"| STRAT["Set strategy"]
    REMOVE --> STRAT

    STRAT --> SUBMIT["Generate representment note\n(policy keywords + evidence IDs)\nSubmit package"]

    ACCEPT --> RESOLVE["Resolve case\n(accept_chargeback / issue_refund)"]
    REFUND --> RESOLVE
    REFUND2 --> RESOLVE

    SUBMIT --> MORE{"More open cases?"}
    RESOLVE --> MORE
    MORE -->|"Yes"| NEAR{"Current case\nnear completion?"}
    NEAR -->|"Yes (evidence attached,\n1-2 steps to finish)"| FINISH["Finish current case\nbefore switching"]
    NEAR -->|"No"| SEL
    FINISH --> MORE
    MORE -->|"No / steps exhausted"| GRADE["Grader scores episode"]

    style A fill:#2d5016,color:#fff
    style GRADE fill:#1a3a5c,color:#fff
    style REMOVE fill:#800080,color:#fff
    style REFUND fill:#8b4513,color:#fff
    style REFUND2 fill:#8b4513,color:#fff
```

## Grading System

```mermaid
pie title Case Score Weights
    "Strategy Correctness (25%)" : 25
    "Evidence Quality (20%)" : 20
    "Packet Validity (15%)" : 15
    "Deadline Compliance (15%)" : 15
    "Efficiency (10%)" : 10
    "Outcome Quality (10%)" : 10
    "Note Quality (5%)" : 5
```

| Dimension | How It's Scored |
|---|---|
| **Strategy Correctness** | 1.0 = optimal strategy, 0.55 = acceptable fallback, 0.0 = wrong |
| **Evidence Quality** | Contest: 0.7 × (required attached / total required) + 0.3 × (helpful / total helpful) − 0.25 per harmful. Non-contest: 1.0 if clean, 0.7 otherwise |
| **Packet Validity** | Binary: 1.0 if all required evidence attached AND zero harmful, else 0.0 |
| **Deadline Compliance** | Binary: 1.0 if resolved before deadline step, else 0.0 |
| **Efficiency** | 1.0 − (duplicate_queries × 0.1 + submit_attempts × 0.05), min 0.1 |
| **Outcome Quality** | 1.0 = optimal resolution, 0.6 = acceptable, 0.0 = wrong |
| **Note Quality** | Contest only: word substance (20%) + policy keyword coverage (50%) + evidence ID refs (15%) − harmful keyword penalty (15%) |

Each case is weighted by financial impact. Episode score normalizes across all cases to `[0.0, 1.0]`.

## LLM Provider Fallback Chain

```mermaid
flowchart LR
    P["Primary provider\n(configured in .env)"] -->|"Fail / timeout"| OR["OpenRouter\nopenai/gpt-oss-120b"]
    OR -->|"Fail"| GEM["Google Gemini\ngemini-2.5-flash"]
    GEM -->|"Fail"| GRQ["Groq\nllama-3.3-70b-versatile"]
    GRQ -->|"All fail"| H["Heuristic fallback\n_heuristic_pick()"]

    style P fill:#2d5016,color:#fff
    style H fill:#8b4513,color:#fff
```

## Agent Performance (63 Episodes)

Results from the heuristic agent across built-in and parametric tasks:

| Source | Avg | >= 0.90 | < 0.50 | Min |
|---|---|---|---|---|
| Built-in tasks (3) | 0.933 | 2/3 | 0/3 | 0.865 |
| Parametric easy (20) | 0.980 | 20/20 | 0/20 | 0.958 |
| Parametric medium (20) | 0.868 | 12/20 | 0/20 | 0.624 |
| Parametric hard (20) | 0.722 | 0/20 | 0/20 | 0.559 |

**Overall: 0.861 avg | 54.0% score >= 0.90 | 0.0% score < 0.50**

## Task Sources

### Built-in Scenarios (3 tasks)

| Task ID | Difficulty | Objective |
|---|---|---|
| `goods_not_received_easy` | Easy | Contest a goods-not-received case with delivery proof |
| `fraud_signal_ambiguity` | Medium | Handle CNP fraud with mixed evidence and harmful artifacts |
| `queue_optimization_hard` | Hard | Maximize recovery across a multi-case queue under deadline pressure |

### Parametric Generator (`case_generator.py`)

Generates infinite reproducible tasks from seeded RNG across 6 reason code families. Usage: `generated_{difficulty}_s{seed}` (e.g., `generated_hard_s42`).

### ISO 20022 Real Data (`iso_adapter.py`)

Converts 300 real chargeback records from ISO 20022 CASR.003 format into environment cases. Covers fraud, goods-not-received, duplicate processing, credit-not-processed, product-not-as-described, and service-not-provided disputes.

### Stripe Sandbox (`connectors/stripe_sandbox.py`)

Maps Stripe test-mode dispute objects into environment cases. Supports live API access with `STRIPE_API_KEY` or falls back to synthetic Stripe-format disputes.

## Action Space

| Action | Purpose |
|---|---|
| `select_case` | Focus a case from the dispute queue |
| `inspect_case` | Reveal analyst inspection notes |
| `query_system` | Pull evidence from a merchant system |
| `retrieve_policy` | Get reason-code guidance and required evidence |
| `add_evidence` | Attach retrieved evidence to the representment package |
| `remove_evidence` | Remove evidence (including harmful attachments) |
| `set_strategy` | Choose `contest`, `accept_chargeback`, or `issue_refund` |
| `submit_representment` | Submit a contest package with an optional rationale note |
| `resolve_case` | Close a non-contest case |

## Quick Start

### Install

```bash
uv sync --extra dev
# or
pip install -e ".[dev]"
```

### Configure

```bash
cp .env.example .env
# Edit .env with your provider keys
```

### Validate

```bash
pytest -q tests
openenv validate .
python -m runners.baseline_runner
python -m evaluation.agent_brutal_audit
```

### Run Server

```bash
uvicorn chargeback_ops.server.app:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t chargebackops .
docker run --rm -p 8000:8000 --env-file .env chargebackops
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Service info |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Interactive OpenAPI docs |
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step` | Apply an action |
| `GET` | `/state` | Current environment state |
| `GET` | `/tasks` | List available tasks |
| `GET` | `/generate` | Generate parametric tasks |
| `GET/POST` | `/grader` | Fetch latest episode grade |
| `GET/POST` | `/baseline` | Run the heuristic baseline |
| `GET` | `/results` | List all completed episode reports |

## Inference Contract

The required entry point [`inference.py`](inference.py) uses the OpenAI-compatible client with:

```bash
API_BASE_URL=https://openrouter.ai/api/v1
MODEL_NAME=openai/gpt-oss-120b
HF_TOKEN=your_key
```

Supported providers: OpenRouter, OpenAI, Google Gemini, Groq, Anthropic-compatible gateways.

## Hugging Face Deployment

1. Create a new HF Space with **Docker** SDK
2. Push this repository
3. Set secrets in Space Settings: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
4. Verify: `/health`, `/tasks`, `/baseline`

## Project Layout

```
.
├── inference.py                 # Root entry point (submission contract)
├── openenv.yaml                 # OpenEnv spec
├── core/
│   ├── models.py                # Pydantic action/observation/state models
│   ├── client.py                # OpenEnv WebSocket client
│   └── episode_store.py         # Thread-safe episode report store
├── evaluation/
│   ├── grading.py               # Deterministic 7-dimension grader
│   └── agent_brutal_audit.py    # Comprehensive agent evaluation
├── runners/
│   ├── baseline_runner.py       # Heuristic agent with LLM fallback
│   └── inference.py             # Challenge-compatible inference logic
├── scenarios/
│   ├── simulation.py            # Task definitions and case progress
│   ├── case_generator.py        # Parametric seeded task generator
│   └── iso_adapter.py           # ISO 20022 real data adapter
├── server/
│   ├── app.py                   # FastAPI application
│   └── chargeback_ops_environment.py  # Core environment
├── connectors/
│   └── stripe_sandbox.py        # Stripe test-mode connector
├── tests/
│   ├── test_env.py              # Environment + generator tests
│   ├── test_grader.py           # Grading logic tests
│   ├── test_api.py              # API endpoint tests
│   ├── test_requirements.py     # Problem statement compliance
│   └── test_agent_audit.py      # Audit validation tests
├── Dockerfile                   # Production container
├── pyproject.toml               # Package config
└── .env.example                 # Environment variable template
```
