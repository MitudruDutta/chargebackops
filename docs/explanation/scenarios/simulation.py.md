# scenarios/simulation.py

## What this file does
Primary scenario domain model and task catalog including fixed benchmark tasks and task lookup functions.

## Runtime role
- task/scenario module

## Key contents
- File size: 29654 bytes
- Approximate line count: 747
- Module docstring: Internal task definitions and runtime types for ChargebackOps.
- Top-level classes (5): InternalEvidence, InternalCase, TaskScenario, CaseProgress, ActionRecord
- Top-level functions (4): _ev, get_task, list_tasks, list_iso_tasks

## Connections to other files
### Depends on / references
- connectors/stripe_sandbox.py
- scenarios/case_generator.py
- scenarios/iso_adapter.py

### Used by / referenced from
- AGENT.md
- README.md
- connectors/stripe_sandbox.py
- data/iso20022-card-chargeback-casr-003.csv
- evaluation/agent_brutal_audit.py
- evaluation/grading.py
- evaluation/rubrics.py
- openenv_chargeback_ops.egg-info/SOURCES.txt
- runners/baseline_runner.py
- runners/inference.py
- scenarios/case_generator.py
- scenarios/iso_adapter.py
- server/app.py
- server/chargeback_ops_environment.py
- server/demo_ui.py
- tests/test_grader.py
- tests/test_requirements.py

## Integration notes
- Update this module together with its direct and reverse dependencies to keep environment behavior and grading contracts consistent.
