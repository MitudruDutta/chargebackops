# runners/inference.py

## What this file does
Challenge-style inference entrypoint that executes baseline policy runs and prints result payloads.

## Runtime role
- inference entrypoint

## Key contents
- File size: 11135 bytes
- Approximate line count: 315
- Module docstring: Challenge-compatible inference entry point for ChargebackOps.
- Top-level functions (8): _inference_timeout_seconds, _provider_label, _default_headers, _build_client, _build_fallback_client, _pick_with_openai_client, run_inference, main

## Connections to other files
### Depends on / references
- core/models.py
- evaluation/grading.py
- runners/baseline_runner.py
- scenarios/simulation.py
- server/chargeback_ops_environment.py

### Used by / referenced from
- .env
- .env.example
- AGENT.md
- README.md
- docs/RESULTS.md
- inference.py
- openenv_chargeback_ops.egg-info/SOURCES.txt
- pyproject.toml
- server/app.py
- tests/test_api.py
- tests/test_requirements.py

## Integration notes
- Update this module together with its direct and reverse dependencies to keep environment behavior and grading contracts consistent.
