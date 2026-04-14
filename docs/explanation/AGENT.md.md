# AGENT.md

## What this file does
Long-form product and technical specification describing the chargeback operations benchmark, environment contract, and grading philosophy.

## Runtime role
- root documentation

## Key contents
- File size: 28090 bytes
- Approximate line count: 599
- Major headings (12 sampled):
  - # ChargebackOps Agent: Complete Technical Reference
  - ## Table of Contents
  - ## The Problem
  - ## The Use Case
  - ## How the Environment Works
  - ### Lifecycle
  - ### Observation
  - ### The Visible Case
  - ### Action Space (9 Actions)
  - ### Reward Signals
  - ## How the Agent Works
  - ### Why Heuristic-First?

## Connections to other files
### Depends on / references
- .env
- README.md
- connectors/stripe_sandbox.py
- core/client.py
- core/episode_store.py
- core/models.py
- evaluation/agent_brutal_audit.py
- evaluation/grading.py
- evaluation/rubrics.py
- inference.py
- runners/baseline_runner.py
- runners/inference.py
- scenarios/case_generator.py
- scenarios/iso_adapter.py
- scenarios/simulation.py
- server/app.py
- server/chargeback_ops_environment.py
- server/demo_ui.py

### Used by / referenced from
- README.md

## Integration notes
- Keep this file synchronized with the connected files so deployment, packaging, and documentation stay accurate.
