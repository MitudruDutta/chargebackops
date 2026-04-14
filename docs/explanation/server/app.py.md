# server/app.py

## What this file does
FastAPI app assembly module wiring environment routes, utility endpoints, baseline execution, and optional demo UI.

## Runtime role
- service entrypoint

## Key contents
- File size: 4912 bytes
- Approximate line count: 178
- Module docstring: FastAPI application for ChargebackOps.
- Top-level functions (7): root, tasks, generate_tasks, grader, baseline, results, main

## Connections to other files
### Depends on / references
- .env
- core/episode_store.py
- core/models.py
- runners/baseline_runner.py
- runners/inference.py
- scenarios/case_generator.py
- scenarios/simulation.py
- server/chargeback_ops_environment.py
- server/demo_ui.py

### Used by / referenced from
- AGENT.md
- Dockerfile
- OPENENV.md
- README.md
- episode_logs/episodes.jsonl
- openenv.yaml
- openenv_chargeback_ops.egg-info/PKG-INFO
- openenv_chargeback_ops.egg-info/SOURCES.txt
- openenv_chargeback_ops.egg-info/entry_points.txt
- pyproject.toml
- tests/test_api.py

## Integration notes
- Update this module together with its direct and reverse dependencies to keep environment behavior and grading contracts consistent.
