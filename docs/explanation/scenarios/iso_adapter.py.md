# scenarios/iso_adapter.py

## What this file does
Adapter that converts ISO 20022 dispute CSV records into internal TaskScenario objects.

## Runtime role
- task/scenario module

## Key contents
- File size: 18026 bytes
- Approximate line count: 516
- Module docstring: Adapter that converts real ISO 20022 chargeback CSV rows into environment cases.

Reads ``data/iso20022-card-chargeback-casr-003.csv`` and produces
``InternalCase`` / ``TaskScenario`` objects so real dispute data flows
through the benchmark.
- Top-level functions (8): _ev, _infer_strategy, _build_evidence, _concedable_guidance, row_to_case, load_iso_rows, build_iso_task, generate_iso_suite

## Connections to other files
### Depends on / references
- data/iso20022-card-chargeback-casr-003.csv
- scenarios/simulation.py

### Used by / referenced from
- AGENT.md
- data/iso20022-card-chargeback-casr-003.csv
- openenv_chargeback_ops.egg-info/SOURCES.txt
- scenarios/simulation.py

## Integration notes
- Update this module together with its direct and reverse dependencies to keep environment behavior and grading contracts consistent.
