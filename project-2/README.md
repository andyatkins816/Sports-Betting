# SAM Analytics service

This directory contains the Flask API, research primitives, database migration,
container image, and worker entry point for SAM Analytics.

`sam_analytics/` contains deterministic domain logic, a narrowly scoped The
Odds API v4 adapter, and an evidence-first persistence boundary. The adapter
preserves exact response bytes, provider IDs/timestamps, event and bookmaker
metadata, completed scores, and sanitized quota/request scope; it filters live
odds and does not place wagers. `odds_ledger.py` stores those bytes privately
before normalized quotes or results are committed, then records their immutable
receipt, provenance, event identity, facts, and lineage in PostgreSQL.

`provider_worker.py` supplies the staging-only recurring ingestion runtime:
pregame `h2h` odds every five minutes and completed scores every hour. Its
scheduler runs only when the Render start command includes `--beat`; rollout
must apply migration `008_result_ingestion.sql` first. The separate synthetic
storage probe remains unable to create odds, events, model inputs, or public
output. Do not add provider or R2 credentials to the public web service.

The next code-only foundation is documented in
[`docs/INGESTION_CONTROL_PLANE.md`](docs/INGESTION_CONTROL_PLANE.md). Its pure
planner is disabled by default, and its unwired PostgreSQL admission repository
can only persist a pending dispatch bundle after atomically binding the exact
reviewed provider use and quota receipt used for the decision. The
authorization table is intentionally empty, and there is still no publisher,
consumer, scheduler, credential loader, or provider call. It cannot make
another provider request by itself.

See [private worker admission](docs/PRIVATE_WORKER_ADMISSION.md) and
[private raw-evidence object storage](docs/RAW_EVIDENCE_OBJECT_STORAGE.md) for
the exact synthetic staging boundary, provider-specific credential limits, and
compensating retention controls. A real-data staging run has its own separate
[provider ingestion runtime](docs/PROVIDER_SHADOW_ADMISSION.md).

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
