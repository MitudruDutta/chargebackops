# inference.py

## What this file does
Root compatibility wrapper that forwards challenge inference execution to runners/inference.py.

## Runtime role
- package entrypoint/helper

## Key contents
- File size: 284 bytes
- Approximate line count: 11
- Module docstring: Challenge-compatible inference entry point (root re-export).

The submission contract requires inference.py at the repository root.
All logic lives in runners/inference.py.

## Connections to other files
### Depends on / references
- runners/inference.py

### Used by / referenced from
- AGENT.md
- README.md
- openenv_chargeback_ops.egg-info/PKG-INFO
- openenv_chargeback_ops.egg-info/SOURCES.txt
- pyproject.toml
- tests/test_requirements.py

## Integration notes
- Update this module together with its direct and reverse dependencies to keep environment behavior and grading contracts consistent.
