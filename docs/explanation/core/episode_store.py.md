# core/episode_store.py

## What this file does
Persistent report store for completed episodes with in-memory index and JSONL append logging.

## Runtime role
- core library module

## Key contents
- File size: 1684 bytes
- Approximate line count: 62
- Module docstring: Thread-safe storage for completed episode grading reports with file persistence.
- Top-level functions (4): _persist, record_report, get_report, list_reports

## Connections to other files
### Depends on / references
- core/models.py

### Used by / referenced from
- AGENT.md
- episode_logs/episodes.jsonl
- openenv_chargeback_ops.egg-info/SOURCES.txt
- server/app.py
- server/chargeback_ops_environment.py

## Integration notes
- Update this module together with its direct and reverse dependencies to keep environment behavior and grading contracts consistent.
