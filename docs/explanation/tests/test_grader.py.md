# tests/test_grader.py

## What this file does
Unit tests for rubric logic and scoring consistency guarantees.

## Runtime role
- test module

## Key contents
- File size: 1378 bytes
- Approximate line count: 41
- Top-level functions (2): test_grade_episode_bounds, test_environment_exposes_rubric_tree

## Connections to other files
### Depends on / references
- evaluation/grading.py
- evaluation/rubrics.py
- scenarios/simulation.py
- server/chargeback_ops_environment.py

### Used by / referenced from
- openenv_chargeback_ops.egg-info/SOURCES.txt

## Integration notes
- This file validates behavior from the files listed above; it should evolve with API and rubric changes to prevent regressions.
