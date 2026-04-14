# evaluation/grading.py

## What this file does
High-level grading adapters that construct context objects and return typed score breakdown reports.

## Runtime role
- grading/evaluation module

## Key contents
- File size: 3987 bytes
- Approximate line count: 121
- Module docstring: Deterministic grading adapters that delegate to OpenEnv Rubric subclasses.

The real scoring lives in :mod:`evaluation.rubrics`. This module keeps the
legacy call sites (``score_case`` / ``grade_episode`` / ``grade_representment_note``)
stable so the environment, tests, and audit tooling do not need to change.
- Top-level functions (3): _build_case_notes, score_case, grade_episode

## Connections to other files
### Depends on / references
- core/models.py
- evaluation/rubrics.py
- scenarios/simulation.py

### Used by / referenced from
- AGENT.md
- docs/RESULTS.md
- docs/RUBRIC_AUDITOR_PRD.md
- evaluation/__init__.py
- evaluation/agent_brutal_audit.py
- openenv_chargeback_ops.egg-info/SOURCES.txt
- runners/baseline_runner.py
- runners/inference.py
- server/chargeback_ops_environment.py
- tests/test_grader.py
- tests/test_requirements.py

## Integration notes
- Update this module together with its direct and reverse dependencies to keep environment behavior and grading contracts consistent.
