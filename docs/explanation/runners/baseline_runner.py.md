# runners/baseline_runner.py

## What this file does
Reference policy runner combining deterministic heuristics with optional LLM tie-breaking.

## Runtime role
- baseline agent policy module

## Key contents
- File size: 56104 bytes
- Approximate line count: 1477
- Module docstring: Baseline runner for ChargebackOps.
- Top-level classes (3): CandidateChoice, CandidateAction, ProviderConfig
- Top-level functions (25): _provider_timeout_seconds, _provider_retry_attempts, _provider_retry_backoff_seconds, _strict_llm_mode, _should_retry_provider_error, _chat_completion_with_retry, _best_open_case, _build_representment_note, _visible_case_deadline, _is_harmful_evidence, _rank_attachable, _batch_attachable_ids, candidate_actions, _heuristic_pick, _obvious_next_action, _safe_json_loads, _compact_queue_item, _compact_visible_case, _provider_payload, _resolve_provider, _openai_compatible_client, _provider_pick, _provider_pick_with_fallback, run_baseline, main

## Connections to other files
### Depends on / references
- core/models.py
- evaluation/grading.py
- scenarios/simulation.py
- server/chargeback_ops_environment.py

### Used by / referenced from
- .env
- .env.example
- AGENT.md
- README.md
- docs/RESULTS.md
- evaluation/agent_brutal_audit.py
- openenv_chargeback_ops.egg-info/SOURCES.txt
- openenv_chargeback_ops.egg-info/entry_points.txt
- pyproject.toml
- runners/inference.py
- server/app.py
- server/demo_ui.py
- tests/test_requirements.py

## Integration notes
- Update this module together with its direct and reverse dependencies to keep environment behavior and grading contracts consistent.
