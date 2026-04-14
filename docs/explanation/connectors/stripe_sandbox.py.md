# connectors/stripe_sandbox.py

## What this file does
Stripe sandbox adapter that maps dispute-like records into the project internal scenario format.

## Runtime role
- integration connector module

## Key contents
- File size: 17332 bytes
- Approximate line count: 530
- Module docstring: Stripe sandbox connector for ChargebackOps.

Maps Stripe test-mode dispute objects into ``InternalCase`` / ``TaskScenario``
so real Stripe dispute flows can be processed through the environment.

Usage::

    export STRIPE_API_KEY=sk_test_...
    from connectors.stripe_sandbox import fetch_disputes, build_stripe_task

    disputes = fetch_disputes(limit=10)
    task = build_stripe_task(disputes, difficulty="medium")
- Top-level functions (7): _ev, _infer_strategy, _build_evidence, dispute_to_case, build_stripe_task, fetch_disputes, _synthetic_test_disputes

## Connections to other files
### Depends on / references
- .env
- scenarios/simulation.py

### Used by / referenced from
- AGENT.md
- openenv_chargeback_ops.egg-info/PKG-INFO
- openenv_chargeback_ops.egg-info/SOURCES.txt
- scenarios/simulation.py

## Integration notes
- Update this module together with its direct and reverse dependencies to keep environment behavior and grading contracts consistent.
