# server/demo_ui.py

## What this file does
Gradio-based interactive UI to inspect tasks, run actions, and visualize score components.

## Runtime role
- interactive UI module

## Key contents
- File size: 18267 bytes
- Approximate line count: 483
- Module docstring: Gradio demo UI for ChargebackOps.
- Top-level functions (8): _bar_html, _score_color, _queue_html, _budget_html, _grader_html, _resolve_task_id, run_episode, build_demo

## Connections to other files
### Depends on / references
- .env
- evaluation/agent_brutal_audit.py
- runners/baseline_runner.py
- scenarios/simulation.py
- server/chargeback_ops_environment.py

### Used by / referenced from
- .env
- .env.example
- AGENT.md
- server/app.py

## Integration notes
- Update this module together with its direct and reverse dependencies to keep environment behavior and grading contracts consistent.
