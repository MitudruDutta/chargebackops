# Dockerfile

## What this file does
Container build recipe that installs dependencies and runs the FastAPI service in production mode.

## Runtime role
- build/runtime configuration

## Key contents
- File size: 536 bytes
- Approximate line count: 25

## Connections to other files
### Depends on / references
- openenv.yaml
- pyproject.toml
- server/app.py

### Used by / referenced from
- .dockerignore
- README.md
- openenv.yaml
- openenv_chargeback_ops.egg-info/PKG-INFO

## Integration notes
- Keep this file synchronized with the connected files so deployment, packaging, and documentation stay accurate.
