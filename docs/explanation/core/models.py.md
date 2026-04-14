# core/models.py

## What this file does
Canonical Pydantic schemas for actions, observations, environment state, grader output, and baseline result payloads.

## Runtime role
- core library module

## Key contents
- File size: 6296 bytes
- Approximate line count: 244
- Module docstring: Typed models for the ChargebackOps OpenEnv environment.
- Top-level classes (15): CaseQueueItem, EvidenceCard, PolicyView, VisibleCase, TaskSummary, ActionTraceItem, CaseResolutionState, CaseScoreBreakdown, GraderReport, BaselineTaskResult, BaselineRunResult, TasksResponse, ChargebackOpsAction, ChargebackOpsObservation, ChargebackOpsState

## Connections to other files
### Depends on / references
- .env

### Used by / referenced from
- AGENT.md
- __init__.py
- core/client.py
- core/episode_store.py
- evaluation/agent_brutal_audit.py
- evaluation/grading.py
- openenv_chargeback_ops.egg-info/SOURCES.txt
- runners/baseline_runner.py
- runners/inference.py
- server/app.py
- server/chargeback_ops_environment.py
- tests/test_api.py
- tests/test_env.py
- tests/test_requirements.py

## Integration notes
- Update this module together with its direct and reverse dependencies to keep environment behavior and grading contracts consistent.
