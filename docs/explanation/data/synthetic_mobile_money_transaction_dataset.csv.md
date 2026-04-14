# data/synthetic_mobile_money_transaction_dataset.csv

## What this file does
Additional synthetic mobile-money dataset used by audit scripts for broader behavior checks.

## Runtime role
- dataset asset

## Key contents
- File size: 156564413 bytes
- Row count (including header): 1720182
- Columns (10 sampled): step, transactionType, amount, initiator, oldBalInitiator, newBalInitiator, recipient, oldBalRecipient, newBalRecipient, isFraud

## Connections to other files
### Depends on / references
- evaluation/agent_brutal_audit.py

### Used by / referenced from
- No reverse project-file dependency was detected.

## Integration notes
- This dataset supports scenario generation and/or offline audit scripts; schema changes can affect adapters and audit tooling.
