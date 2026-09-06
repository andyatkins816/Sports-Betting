# SAM Analytics service

This directory contains the Flask API, research primitives, database migration,
container image, and worker entry point for SAM Analytics.

`sam_analytics/` contains deterministic domain logic, a narrowly scoped The
Odds API v4 pregame adapter, and an evidence-first persistence boundary. The
adapter preserves exact response bytes, provider IDs/timestamps, event and
bookmaker metadata, and sanitized quota/request scope; it filters live events
and does not train on samples or place wagers. `odds_ledger.py` stores those
bytes privately before a single normalized quote is committed, then records an
immutable receipt, provenance, event identity, snapshot, and lineage link in
one PostgreSQL transaction.

There is intentionally no automatically active provider worker. A concrete
private Cloudflare R2 raw-evidence adapter is wired to a manual, staging-only
synthetic storage probe, and the synthetic fixture cannot create odds, events,
model inputs, or public output. A separate `provider_worker.py` composition root
now admits one operator-dispatched The Odds API shadow request under an exact
provider contract, request scope, quota bound, private evidence prefix, and
append-only run audit. It has no schedule, retry loop, results backend,
settlement task, or public-output path. Do not add its provider key or other
worker-only configuration to the public web service.

See [private worker admission](docs/PRIVATE_WORKER_ADMISSION.md) and
[private raw-evidence object storage](docs/RAW_EVIDENCE_OBJECT_STORAGE.md) for
the exact synthetic staging boundary, provider-specific credential limits, and
compensating retention controls. A real-data staging run has its own separate
[manual provider-shadow admission](docs/PROVIDER_SHADOW_ADMISSION.md).

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
