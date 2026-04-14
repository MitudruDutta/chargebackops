# tests/test_agent_audit.py

## What this file does
Tests validating the offline audit harness and policy comparison workflow.

## Runtime role
- test module

## Key contents
- File size: 1134 bytes
- Approximate line count: 34
- Top-level functions (2): test_heuristic_beats_bad_on_generated_suite, test_data_directory_is_ignored

## Connections to other files
### Depends on / references
- .dockerignore
- .gitignore
- evaluation/agent_brutal_audit.py

### Used by / referenced from
- openenv_chargeback_ops.egg-info/SOURCES.txt

## Integration notes
- This file validates behavior from the files listed above; it should evolve with API and rubric changes to prevent regressions.
