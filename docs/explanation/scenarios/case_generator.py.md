# scenarios/case_generator.py

## What this file does
Synthetic case/task generator that produces deterministic tasks from seeds and difficulty levels.

## Runtime role
- task/scenario module

## Key contents
- File size: 45229 bytes
- Approximate line count: 1278
- Module docstring: Parametric case generator for ChargebackOps.

Generates reproducible chargeback cases from reason-code templates using a
seeded RNG.  Every seed produces the same cases, so benchmarks are replayable
while the scenario space is effectively infinite.
- Top-level classes (2): _EvidenceBlueprint, _CaseTemplate
- Top-level functions (9): _assign_network, _amount, _customer_id, _order_id, _pick_summary, _generate_evidence, generate_case, generate_task, generate_task_suite

## Connections to other files
### Depends on / references
- scenarios/simulation.py

### Used by / referenced from
- AGENT.md
- openenv_chargeback_ops.egg-info/SOURCES.txt
- scenarios/simulation.py
- server/app.py
- tests/test_env.py

## Integration notes
- Update this module together with its direct and reverse dependencies to keep environment behavior and grading contracts consistent.
