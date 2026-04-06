# OpenEnv Overview

OpenEnv is a framework for building and evaluating AI agent environments. Most AI benchmarks test knowledge through Q&A or multiple choice. OpenEnv tests **operational competence** — can an agent actually perform a multi-step job under constraints, make trade-off decisions, and handle real-world workflows?

## The Problem OpenEnv Solves

There is no standard way to measure whether an AI agent can reliably handle operational work. Agents are increasingly deployed for tasks like customer support, code review, compliance checks, and dispute handling — but evaluation is ad hoc, non-reproducible, and rarely involves realistic constraints like deadlines, partial information, or multi-objective trade-offs.

OpenEnv fills this gap the same way ImageNet standardized image classification benchmarks. It provides a common interface so that any environment can be plugged in, any agent can be tested, and results are comparable across teams and models.

## How It Works

Every OpenEnv environment implements three methods:

```python
reset(task_id) -> Observation  # Start a new episode with a specific task
step(action)   -> Observation  # Take one action, get updated observation
state()        -> State        # Full environment state (includes grader report when done)
```

This follows the same pattern as OpenAI Gym / Gymnasium from reinforcement learning, but applied to LLM agents performing operational tasks rather than game-playing or robotics control.

## What Makes a Good OpenEnv Environment

An environment models a real-world workspace. For ChargebackOps, that workspace is a merchant dispute analyst's desk. The key requirements:

1. **Real-world workflow** — the task must reflect something humans actually do, with genuine decision complexity
2. **Typed action space** — discrete, well-defined actions (not free-form text generation)
3. **Deterministic grading** — the same sequence of actions always produces the same score, enabling reproducible evaluation
4. **Baseline agent** — a working agent that demonstrates the environment is solvable and produces a reference score
5. **Deployable** — runs as a Docker container on Hugging Face Spaces with standard HTTP endpoints

## How OpenEnv Helps ChargebackOps

Chargeback dispute handling involves triaging cases by urgency, querying the right merchant systems for evidence, filtering out harmful artifacts, deciding whether to contest or concede, and submitting representment packages before hard deadlines. This is exactly the kind of multi-step, constraint-heavy, tool-using workflow that OpenEnv is designed to evaluate.

Without OpenEnv, testing whether an agent can handle this workflow would require building custom evaluation infrastructure from scratch — a non-standard API, ad hoc scoring, no way to compare results across different agents or models. OpenEnv provides:

- **`openenv-core`** — the `Environment` base class, typed `Action`/`Observation`/`State` contracts, and `create_app()` which gives a FastAPI server with `/reset`, `/step`, `/state`, and `/health` endpoints out of the box
- **`EnvClient`** — a WebSocket client so agents can connect remotely without importing environment internals
- **`openenv validate .`** — a validation tool that checks spec compliance, endpoint availability, and deployment readiness
- **Standardised evaluation** — in Phase 2 of the hackathon, a standard LLM agent (Nemotron 3 Super) runs against all submitted environments, producing directly comparable scores across every team's environment

## The Hackathon

The OpenEnv hackathon asks participants to build an environment, not an agent. The submission is the test, not the test-taker.

Round 1 evaluates environments on real-world utility (30%), task and grader quality (25%), environment design (20%), code quality (15%), and creativity (10%). Round 2 runs a standard agent against all qualifying environments to measure how well each environment discriminates between good and bad agent behaviour.

ChargebackOps is built to perform well on both rounds: the 7-dimension deterministic grader produces a clear difficulty curve (easy 0.96 → nightmare 0.47), and the typed action space with dense reward shaping gives any standard agent enough signal to learn the environment within a single episode.
