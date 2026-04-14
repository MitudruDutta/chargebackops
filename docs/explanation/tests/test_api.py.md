# tests/test_api.py

## What this file does
Integration tests for HTTP endpoints and API-level contract validation.

## Runtime role
- test module

## Key contents
- File size: 2841 bytes
- Approximate line count: 87
- Top-level functions (5): test_tasks_endpoint_payload, test_root_endpoint_payload, test_baseline_endpoint_works_without_api_key, test_inference_script_falls_back_without_hf_token, test_grader_endpoint_after_completed_episode

## Connections to other files
### Depends on / references
- core/models.py
- runners/inference.py
- server/app.py
- server/chargeback_ops_environment.py

### Used by / referenced from
- openenv_chargeback_ops.egg-info/SOURCES.txt

## Integration notes
- This file validates behavior from the files listed above; it should evolve with API and rubric changes to prevent regressions.
