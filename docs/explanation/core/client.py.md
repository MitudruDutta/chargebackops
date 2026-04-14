# core/client.py

## What this file does
Typed WebSocket/Env client wrapper that converts generic OpenEnv responses into project-specific models.

## Runtime role
- core library module

## Key contents
- File size: 2819 bytes
- Approximate line count: 92
- Module docstring: WebSocket client for ChargebackOps.
- Top-level classes (1): ChargebackOpsEnv
- Top-level functions (4): _parse_evidence, _parse_policy, _parse_visible_case, _parse_grader

## Connections to other files
### Depends on / references
- core/models.py

### Used by / referenced from
- AGENT.md
- __init__.py
- openenv_chargeback_ops.egg-info/SOURCES.txt

## Integration notes
- Update this module together with its direct and reverse dependencies to keep environment behavior and grading contracts consistent.
