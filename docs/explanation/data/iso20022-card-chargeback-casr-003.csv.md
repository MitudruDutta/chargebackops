# data/iso20022-card-chargeback-casr-003.csv

## What this file does
Realistic chargeback sample data used by ISO adapter flows to build benchmark cases.

## Runtime role
- dataset asset

## Key contents
- File size: 90967 bytes
- Row count (including header): 301
- Columns (20 sampled): chargeback_id, original_transaction_id, card_number_masked, cardholder_name, merchant_name, merchant_id, transaction_amount, transaction_currency, transaction_date, chargeback_date, chargeback_reason_code, chargeback_reason_description, investigation_status, investigator_id, representment_deadline, representment_submitted, representment_date, final_decision, final_decision_date, notes

## Connections to other files
### Depends on / references
- scenarios/iso_adapter.py
- scenarios/simulation.py

### Used by / referenced from
- .gitignore
- scenarios/iso_adapter.py

## Integration notes
- This dataset supports scenario generation and/or offline audit scripts; schema changes can affect adapters and audit tooling.
