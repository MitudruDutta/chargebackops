# Running the ChargebackOps Agent

End-to-end instructions for running the ChargebackOps environment and its baseline agent —
offline (heuristic only), with an LLM tiebreak, against a single task, across the whole
benchmark, or as a live server. If you just want the numbers, see
[`docs/RESULTS.md`](RESULTS.md). If you want to understand the agent internals, see
[`AGENT.md`](../AGENT.md).

---

## 1. Prerequisites

- **Python 3.12** (required — `tomllib` and `Rubric` type hints assume 3.12+)
- **git** (for cloning)
- **Docker** (optional — only if you want the containerized server)

Clone the repo if you haven't already:

```bash
git clone https://github.com/MitudruDutta/chargebackops.git
cd chargebackops
```

Create or reuse a virtual environment, install the project in editable mode with dev extras:

```bash
source ~/python/bin/activate     # or: python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

This installs `openenv-core`, `pydantic`, `openai`, `fastapi`, `uvicorn`, `gradio`, and the test
harness. Nothing else is required for offline runs.

Verify the install:

```bash
pytest -q tests           # expect: 107 passed
openenv validate .        # expect: Ready for multi-mode deployment
```

---

## 2. Configure the environment

Copy the template and edit it:

```bash
cp .env.example .env
```

### 2a. Offline mode (no API keys)

You can run the heuristic and bad-policy agents with **no keys at all**. The runner
automatically falls back to the heuristic when no provider is configured. Skip to section 3.

### 2b. LLM tiebreak mode

Fill in **one** of the provider blocks in `.env`. The runner auto-detects which provider to
use based on `BASELINE_PROVIDER`:

**OpenRouter (recommended — free tier available, used in reference results):**

```env
BASELINE_PROVIDER=openrouter
BASELINE_MODEL=openai/gpt-oss-120b
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_APP_TITLE=ChargebackOps
```

**Groq (fastest, free tier):**

```env
BASELINE_PROVIDER=groq
BASELINE_MODEL=llama-3.3-70b-versatile
GROQ_API_KEY=gsk_...
```

**OpenAI:**

```env
BASELINE_PROVIDER=openai
BASELINE_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
```

**Google Gemini:**

```env
BASELINE_PROVIDER=google
BASELINE_MODEL=gemini-2.5-flash
GOOGLE_API_KEY=AI...
```

The fallback chain is: **primary → OpenRouter → Google → Groq → Heuristic**. If the primary
provider times out or 429s, the runner automatically walks the chain. Set `STRICT_LLM_MODE=1`
if you want failures to surface instead of silently falling back to the heuristic.

---

## 3. Run the agent

### 3a. Against a single task

```bash
source ~/python/bin/activate
python - <<'PY'
from evaluation.agent_brutal_audit import run_episode
result = run_episode("goods_not_received_easy", policy="heuristic")
print(f"score = {result['score']:.4f}")
print(f"steps = {result['steps']}")
print(f"summary: {result['summary']}")
PY
```

Available policies: `"heuristic"` (rule-based, no LLM), `"bad"` (concede-everything baseline).
Any built-in or generated task id works — e.g. `"generated_nightmare_s31"`,
`"fraud_signal_ambiguity"`, `"queue_optimization_hard"`.

### 3b. Across the full 12-task headline benchmark (offline)

```bash
python - <<'PY'
from evaluation.agent_brutal_audit import run_episode
from scenarios.simulation import list_tasks
for t in list_tasks():
    h = run_episode(t.task_id, policy='heuristic')
    b = run_episode(t.task_id, policy='bad')
    print(f"{t.task_id:32s}  heur={h['score']:.4f}  bad={b['score']:.4f}")
PY
```

Expect the heuristic to average **0.8132** and the naive policy to average **0.0000** (±1e-3 for
float rounding). Total wall-clock: a few seconds, zero provider calls.

### 3c. Across the full benchmark with an LLM tiebreak

Make sure a provider is configured (section 2b), then:

```bash
python -m runners.baseline_runner | tee /tmp/baseline_run.json
```

This writes a JSON report with `task_results`, `average_score`, `provider_calls_attempted`,
and `provider_calls_succeeded`. Expect **0.729** average and **7 provider calls** on the
reference setup (OpenRouter + `openai/gpt-oss-120b`).

### 3d. Multi-seed stress grid (28 runs, fully offline)

```bash
python - <<'PY'
from statistics import mean, stdev
from evaluation.agent_brutal_audit import run_episode
for d in ("easy","medium","hard","nightmare"):
    hs, bs = [], []
    for s in (7, 17, 31, 42, 53, 77, 99):
        hs.append(run_episode(f"generated_{d}_s{s}", policy='heuristic')['score'])
        bs.append(run_episode(f"generated_{d}_s{s}", policy='bad')['score'])
    print(f"{d:10s} heur={mean(hs):.4f}±{stdev(hs):.4f}  bad={mean(bs):.4f}±{stdev(bs):.4f}")
PY
```

Expected current grid averages: heuristic **0.7628**, escalate_all **0.7675**, concede_all
**0.4454**, naive **0.0000** across 28 runs.

### 3e. Custom inference contract (challenge submission)

`inference.py` is the submission entry point used by the hackathon harness. It reads
`API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` from the environment and returns decisions via an
OpenAI-compatible client.

```bash
API_BASE_URL=https://openrouter.ai/api/v1 \
MODEL_NAME=openai/gpt-oss-120b \
HF_TOKEN=sk-or-v1-... \
python -m runners.inference
```

---

## 4. Run the server (FastAPI + Gradio demo)

The server exposes the environment via HTTP for OpenEnv-compatible clients and a live demo
at `/demo`.

### 4a. Local

```bash
source ~/python/bin/activate
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints:

| Path | Method | Purpose |
|---|---|---|
| `/reset` | POST | Start an episode (pass `task_id` in JSON body) |
| `/step` | POST | Take one action |
| `/state` | GET | Current observation + progress |
| `/tasks` | GET | Task catalog |
| `/demo` | GET | Gradio live demo — click-through playback |
| `/baseline` | GET/POST | Run the heuristic agent headlessly |
| `/grader` | GET/POST | Score a completed episode |
| `/health` | GET | Health check |
| `/docs` | GET | OpenAPI / Swagger UI |

Example `curl` flow:

```bash
curl -s -X POST localhost:8000/reset -H 'content-type: application/json' \
  -d '{"task_id":"goods_not_received_easy"}' | jq '.task_id, .steps_remaining'

curl -s -X POST localhost:8000/step -H 'content-type: application/json' \
  -d '{"action":{"action_type":"select_case","case_id":"CB-E1"}}' | jq '.reward'
```

### 4b. Docker

```bash
docker build -t chargebackops .
docker run --rm -p 8000:8000 --env-file .env chargebackops
```

The Dockerfile is layered so source edits don't re-run `pip install` — first build takes ~40s,
edits after that rebuild in ~6s.

### 4c. Hugging Face Space

The repo doubles as a Hugging Face Space (see the frontmatter at the top of `README.md`). Push
to the `hf` remote and the space rebuilds automatically:

```bash
git push hf main
```

---

## 5. Inspect the rubric tree

Every scoring dimension is an OpenEnv `Rubric` subclass. Walk the composition tree on any
live environment:

```bash
python - <<'PY'
from server.chargeback_ops_environment import ChargebackOpsEnvironment
env = ChargebackOpsEnvironment()
for name, r in env.rubric.named_rubrics():
    print(f"{name}: {type(r).__name__}")
PY
```

Expected output (11 named children):

```
case_rubric: CaseRubric
case_rubric.aggregator: WeightedSum
case_rubric.aggregator.rubric_0: StrategyCorrectnessRubric
case_rubric.aggregator.rubric_1: EvidenceQualityRubric
case_rubric.aggregator.rubric_2: PacketValidityRubric
case_rubric.aggregator.rubric_3: DeadlineComplianceRubric
case_rubric.aggregator.rubric_4: EfficiencyRubric
case_rubric.aggregator.rubric_5: OutcomeQualityRubric
case_rubric.aggregator.rubric_6: NoteQualityRubric
case_rubric.deadline_gate: Gate
case_rubric.deadline_gate.rubric: CaseAbandonedRubric
```

After a forward pass, each child exposes `last_score` — this is the introspection path an RL
trainer hooks into for credit assignment.

---

## 6. Troubleshooting

**`openenv validate .` fails.** Check `openenv.yaml` is present at repo root and your venv has
`openenv-core>=0.2.3` installed.

**Provider calls all fail / score drops to heuristic.** Run `python -m runners.baseline_runner`
and inspect `provider_errors`. Common causes: expired API key, wrong `BASELINE_MODEL` slug for
the provider, or rate limits (the runner retries twice, then falls back). Set
`BASELINE_REQUEST_TIMEOUT_SECONDS=30` if the provider is slow.

**`ImportError: attempted relative import`.** Always run commands from the repo root with the
venv activated. Use `python -m runners.baseline_runner`, not `python runners/baseline_runner.py`.

**Docker build is slow on every edit.** You probably edited `pyproject.toml` — the deps layer
only caches when that file is unchanged. If you edit source only, rebuilds should be ~6s.

**Scores differ from `docs/RESULTS.md`.** If you pass different seeds or LLM providers you
will get different numbers. The reference numbers are captured on the fixed 12-task catalog
defined by `scenarios.simulation.list_tasks()` plus OpenRouter `openai/gpt-oss-120b`. Anything
else is not directly comparable.

---

## 7. Minimal "does it work?" smoke test

One command to verify everything is wired up correctly:

```bash
source ~/python/bin/activate && \
  pytest -q tests && \
  openenv validate . && \
  python -c "
from evaluation.agent_brutal_audit import run_episode
r = run_episode('goods_not_received_easy', policy='heuristic')
assert 0.0 <= r['score'] <= 1.0, r['score']
assert r['score'] > 0.90, r['score']
print('smoke OK, score =', r['score'])
"
```

If that prints `smoke OK`, the agent runs cleanly and the rubric math is stable.
