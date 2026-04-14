# server/chargeback_ops_environment.py

## What this file does
Main OpenEnv Environment implementation containing reset/step/state logic and action handlers.

## Runtime role
- environment core engine

## Key contents
- File size: 30297 bytes
- Approximate line count: 761
- Module docstring: Core environment implementation for ChargebackOps.
- Top-level classes (1): ChargebackOpsEnvironment

## Connections to other files
### Depends on / references
- .env
- core/episode_store.py
- core/models.py
- evaluation/grading.py
- evaluation/rubrics.py
- scenarios/simulation.py

### Used by / referenced from
- AGENT.md
- docs/RUBRIC_AUDITOR_PRD.md
- episode_logs/episodes.jsonl
- evaluation/agent_brutal_audit.py
- openenv_chargeback_ops.egg-info/SOURCES.txt
- runners/baseline_runner.py
- runners/inference.py
- server/__init__.py
- server/app.py
- server/demo_ui.py
- tests/test_api.py
- tests/test_env.py
- tests/test_grader.py
- tests/test_requirements.py

## Integration notes
- Update this module together with its direct and reverse dependencies to keep environment behavior and grading contracts consistent.
