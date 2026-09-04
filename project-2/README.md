# SAM Analytics service

This directory contains the Flask API, research primitives, database migration,
container image, and worker entry point for SAM Analytics.

`sam_analytics/` contains deterministic domain logic and a narrowly scoped
The Odds API v4 pregame adapter. The adapter preserves provider IDs/timestamps,
filters live events, and exposes quota headers; it does not train on samples or
place wagers. Other provider adapters belong in separately reviewed worker
modules after their data contracts are licensed.

The public integration surface is GET /api/v1/integration/status, protected by
a dedicated status-only credential and limited to a sanitized readiness
contract. POST /api/v1/evaluate remains a development/test research harness
only; production disables it because a caller must not supply a probability,
quote, bankroll, policy, or approval label. Production serving will resolve
immutable Python-generated predictions, quotes, portfolio exposure, policy, and
approved artifact identity server-side.

The `models/` and `services/` folders are legacy source retained temporarily for
audit traceability. They are deliberately not wired into `app.py`.

See the repository README and docs/ for the required governance and deployment
process.
