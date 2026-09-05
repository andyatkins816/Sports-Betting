# SAM Analytics

SAM Analytics is an auditable, provider-neutral sports-market research backend.
It is designed to support lawful analysis, not to promise profits or automate
wager placement. Every live decision must be reproducible from a model version,
timestamped features, a timestamped licensed quote, and a recorded risk policy.

## Current status

The original demo application was removed from the public execution path because
it trained on synthetic data and could write simulated outcomes as performance
records. Its source remains in `project-2/models` and `project-2/services` for
forensic comparison only; it is not imported by the current app. See the
[audit](project-2/docs/AUDIT_2026-09-04.md) before treating any older dashboard
output as meaningful.

The current implementation provides:

- validated American/decimal odds math, de-vig normalization, expected ROI, and
  fractional-Kelly sizing caps;
- provider-neutral, idempotent pregame quote contracts;
- probability calibration, proper scoring rules, and chronological backtesting
  that rejects timestamp look-ahead;
- a fail-closed production configuration, hardened API defaults, container
  runtime, worker separation, immutable PostgreSQL receipt/provenance schema,
  and CI;
- a private, content-addressed raw-payload boundary and transactional odds
  ledger that retain corrections and provenance without exposing raw provider
  data; the provider worker remains deliberately disabled until private object
  storage and an approved provider contract are configured;
- a single analytical endpoint, `POST /api/v1/evaluate`, which never places a
  bet or records a fictional result.

## Quick start

Requires Python 3.11+ in the supported environment.

```bash
cd project-2
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
python -m pytest
flask --app wsgi:app run
```

For a local container stack, copy `.env.example` to `.env`, choose unique local
secrets, set `POSTGRES_PASSWORD`, and run `docker compose up --build`. Do not use
that compose file as the production secret-management plan.

## Documentation

- [Audit and remediation status](project-2/docs/AUDIT_2026-09-04.md)
- [Data, model, backtest, and risk governance](project-2/docs/DATA_AND_MODEL_GOVERNANCE.md)
- [sam.vegas deployment runbook](project-2/docs/DEPLOYMENT_SAM_VEGAS.md)
- [Private raw-evidence object storage](project-2/docs/RAW_EVIDENCE_OBJECT_STORAGE.md)
- [PostgreSQL audit schema](project-2/migrations/001_initial.sql) and
  [provider-receipt lineage migration](project-2/migrations/004_provider_payload_receipts.sql)

## Non-negotiable operating rules

1. Use only data a signed provider agreement permits you to store, derive from,
   display, and redistribute.
2. Keep raw responses, event/market mappings, timestamps, model artifacts,
   feature hashes, and settlement records. Never overwrite them.
3. Do not promote a model from accuracy alone. Require time-split, out-of-sample
   calibration, closing-line comparison where licensed, and shadow monitoring.
4. Never use this system for automated betting without jurisdiction-specific
   legal review, operator/API authorization, responsible-gambling controls, and
   explicit user consent.
