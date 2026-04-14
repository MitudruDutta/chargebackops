# tests/test_env.py

## What this file does
Behavior tests for environment transitions, action effects, and episode lifecycle constraints.

## Runtime role
- test module

## Key contents
- File size: 3705 bytes
- Approximate line count: 109
- Top-level functions (6): test_reset_returns_task_observation, test_reset_accepts_curriculum_difficulty, test_easy_case_can_be_won, test_generated_task_reproducibility, test_generated_task_runs_in_environment, test_generated_task_covers_all_reason_codes

## Connections to other files
### Depends on / references
- core/models.py
- scenarios/case_generator.py
- server/chargeback_ops_environment.py

### Used by / referenced from
- openenv_chargeback_ops.egg-info/SOURCES.txt

## Integration notes
- This file validates behavior from the files listed above; it should evolve with API and rubric changes to prevent regressions.
