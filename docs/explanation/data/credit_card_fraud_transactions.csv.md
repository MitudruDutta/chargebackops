# data/credit_card_fraud_transactions.csv

## What this file does
Auxiliary transaction dataset used for audit/profiling experiments.

## Runtime role
- dataset asset

## Key contents
- File size: 270314728 bytes
- Row count (including header): 1048576
- Columns (23 sampled): , trans_date_trans_time, cc_num, merchant, category, amt, first, last, gender, street, city, state, zip, lat, long, city_pop, job, dob, trans_num, unix_time, merch_lat, merch_long, is_fraud

## Connections to other files
### Depends on / references
- evaluation/agent_brutal_audit.py

### Used by / referenced from
- No reverse project-file dependency was detected.

## Integration notes
- This dataset supports scenario generation and/or offline audit scripts; schema changes can affect adapters and audit tooling.
