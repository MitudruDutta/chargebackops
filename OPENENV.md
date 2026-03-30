# OpenEnv Overview

OpenEnv is a framework for building and evaluating AI agent environments. Think of it as a standardized way to create benchmarks that test how well AI agents perform real-world tasks.

## Core Idea

Most AI benchmarks test knowledge (Q&A, multiple choice). OpenEnv tests **operational competence**: can an agent actually do a job, navigate a workflow, and make multi-step decisions under constraints?

## How It Works

OpenEnv defines a standard interface that every environment must follow:

```python
reset(task_id) -> Observation  # Start a new episode
step(action)   -> Observation  # Take an action and get feedback
state()        -> State        # Current environment state
```

This follows the same pattern as OpenAI Gym/Gymnasium (used in reinforcement learning), but applied to LLM agents doing operational tasks.

## What an Environment Is

An environment is a simulated workspace. For ChargebackOps, that workspace is a merchant dispute desk. Other examples include:

- A customer support queue
- A code review pipeline
- A medical triage system
- An inventory management console

Each environment provides:

- **Tasks**: Specific scenarios to solve
- **Actions**: What the agent can do
- **Observations**: What the agent sees after each action
- **Grading**: Deterministic scoring of performance

## What OpenEnv Provides (`openenv-core`)

- **Environment base class**: Subclass it to build your environment
- **`EnvClient`**: WebSocket client for agents to connect remotely
- **HTTP server scaffolding**: `create_app()` gives you a FastAPI server with `/reset`, `/step`, `/state`, and `/health` endpoints
- **Validation tooling**: `openenv validate .` checks that your environment meets the spec

## Hackathon Context

The OpenEnv hackathon asks participants to build an environment, not an agent. You are creating the test, not the test-taker.

The environment should:

1. Model a real-world workflow
2. Have a typed action space (not free-form text)
3. Have deterministic grading (same actions -> same score)
4. Include a baseline agent that demonstrates the environment works
5. Deploy as a Docker container on Hugging Face Spaces

## Why It Matters

There is still no standard way to measure whether an AI agent can reliably handle operational work. OpenEnv aims to fill that gap, similar to how ImageNet standardized image-classification benchmarks.

ChargebackOps is one such benchmark: it tests whether an agent can handle merchant dispute operations, including triaging cases, gathering evidence, making contest/concede decisions, and respecting deadlines.
