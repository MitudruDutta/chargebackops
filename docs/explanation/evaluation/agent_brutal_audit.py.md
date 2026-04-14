# evaluation/agent_brutal_audit.py

## What this file does
Offline stress-test harness used to benchmark baseline and intentionally bad policies across datasets.

## Runtime role
- grading/evaluation module

## Key contents
- File size: 13230 bytes
- Approximate line count: 399
- Module docstring: Brutal local audit for ChargebackOps agent quality.

This script is intentionally harsher than the standard unit tests:

- profiles any datasets placed under ``data/``
- derives deterministic seeds from dataset rows
- runs the heuristic agent across generated easy/medium/hard tasks
- compares it against a deliberately weak control policy
- reports score gaps, failure counts, and difficulty behavior

It does not require external APIs and is safe to run offline.
- Top-level functions (14): _stable_seed, _detect_amount_column, _detect_fraud_column, _quantile, _map_iso_reason, profile_dataset, derive_dataset_seeds, _bad_policy_action, run_episode, aggregate_results, evaluate_generated_suite, evaluate_fixed_tasks, build_report, main

## Connections to other files
### Depends on / references
- core/models.py
- evaluation/grading.py
- runners/baseline_runner.py
- scenarios/simulation.py
- server/chargeback_ops_environment.py

### Used by / referenced from
- AGENT.md
- data/MoMTSim_20240722202413_1000_dataset.csv
- data/credit_card_fraud_transactions.csv
- data/paysim.csv
- data/synthetic_mobile_money_transaction_dataset.csv
- openenv_chargeback_ops.egg-info/SOURCES.txt
- server/demo_ui.py
- tests/test_agent_audit.py

## Integration notes
- Update this module together with its direct and reverse dependencies to keep environment behavior and grading contracts consistent.
