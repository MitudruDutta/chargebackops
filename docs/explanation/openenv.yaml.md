# openenv.yaml

## What this file does
OpenEnv deployment specification describing runtime type, app entry path, and exposed service port.

## Runtime role
- build/runtime configuration

## Key contents
- File size: 177 bytes
- Approximate line count: 8
- Top-level YAML keys: app, description, name, port, runtime, spec_version, type

## Connections to other files
### Depends on / references
- Dockerfile
- pyproject.toml
- server/app.py

### Used by / referenced from
- Dockerfile
- OPENENV.md
- README.md
- openenv_chargeback_ops.egg-info/PKG-INFO

## Integration notes
- Keep this file synchronized with the connected files so deployment, packaging, and documentation stay accurate.
