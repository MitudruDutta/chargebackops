# data/paysim.csv

## What this file does
Synthetic payment simulation dataset used for baseline stress testing and diagnostics.

## Runtime role
- dataset asset

## Key contents
- File size: 493534783 bytes
- Row count (including header): 6362621
- Columns (11 sampled): step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud

## Connections to other files
### Depends on / references
- evaluation/agent_brutal_audit.py

### Used by / referenced from
- No reverse project-file dependency was detected.

## Integration notes
- This dataset supports scenario generation and/or offline audit scripts; schema changes can affect adapters and audit tooling.
