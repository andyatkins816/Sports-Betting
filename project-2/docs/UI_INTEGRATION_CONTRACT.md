# Trusted UI integration and operational-status contract

The Python service is the source of analytical facts. A browser or an LLM must
not invent a current model, a fresh odds feed, a prediction, an edge, or a
profit claim when the backend has not supplied one.

## Endpoint

`GET /api/v1/integration/status` is an authenticated, versioned status
contract for a trusted UI gateway. Send the separate configured
`SAM_STATUS_API_KEY` only in an `X-API-Key` request header from a server-side
integration function. It returns
`Cache-Control: no-store` and never includes credentials, database URLs, raw
odds, predictions, customer data, or provider payloads.

The unauthenticated `GET /api/healthz` endpoint is a liveness probe only. It
does not establish data or model readiness.

Base44 should use a backend function as the gateway. Store the Python API URL
and the value of `SAM_STATUS_API_KEY` in Base44's server-side secret store under
the name `SAM_BACKEND_STATUS_API_KEY`; do not place either in React code,
browser storage, an AI-agent prompt, or a client-visible entity. Pass the
sanitized status response to the UI or evidence assistant instead.

`SAM_API_KEY` is a separate private-research capability. It must never be
copied to Base44 or used for this status endpoint. Outside development/test,
the research evaluator is disabled.

During the current staging rollout, do not configure Base44 with
`https://api.sam.vegas` until Render has validated that custom domain and its
HTTPS certificate. Until then, leave the production Base44 status gateway
unconfigured rather than weakening host pinning or exposing a temporary URL to
the browser.

## Response behavior

The contract uses only these top-level public-safe JSON keys, so an integration
gateway can whitelist them exactly: `status`, `generated_at`,
`data_freshness`, `model_health`, `risk_status`, and `deployment`.

Their behavior is:

- `generated_at`: the server's UTC timestamp for this status observation.
- `data_freshness`: whether the latest provider observation is fresh, stale,
  unavailable, or invalid relative to `QUOTE_MAX_AGE_SECONDS`.
- `model_health`: whether an identified model is approved, its artifact is
  verified, and an evaluation/monitoring report is present. It does not judge a
  model by accuracy alone or make a profitability claim.
- `status`: `ready` only when the audit database and queue are configured and
  verified healthy, data is fresh, ingestion health is current and healthy,
  and the model is healthy. All other states are `blocked`.
- `deployment`: sanitized infrastructure configuration state, a contract
  version, explicit blockers, whether prediction delivery remains disabled,
  and a nested `ingestion` health projection.
- `risk_status`: research-only mode and the permanent fact that wager
  submission is unsupported.

Contract `v2` adds `deployment.ingestion`. It exposes only finite states,
rounded ages, configured age limits, queue/quota/count bands, and sorted alert
codes. It does not expose exact quota or dead-letter counts, queue or worker
identities, task IDs, request fingerprints, payload digests, private object
URIs, provider endpoints, or exception text. Its `worker_activity` field is
explicitly based on durable transitions; it is not a heartbeat or a process
liveness claim. The internal evaluated value is provider-bound and expires
after at most five seconds, sooner when a known component-age boundary would
be crossed. If it is stale, future-dated, or does not match the validated
provider named by the data-freshness facts, the ingestion projection is
unavailable with `monitoring_invalid` and cannot satisfy readiness.

Until a Postgres-backed health adapter is wired in, the endpoint receives no
trusted ingestion facts and intentionally returns `blocked` with ingestion
health `unavailable`. This is the desired rollout behavior: the interface
should render a transparent unavailable state, not substitute Base44 or
LLM-generated predictions.

## Worker and monitoring integration

The future ingestion adapter should derive health from immutable dispatch
transitions, outstanding reservations, provider quota receipts, and actual
`odds_snapshot.received_at` observations. A successful empty receipt does not
make quote freshness healthy. It must query the same provider used by the
top-level freshness facts. The model-report path should continue to use model
registry approvals, artifact checksums, and validation-report timestamps. The
API process can load these trusted facts into `OperationalSignals` before
creating the response. Do not accept health signals from a browser request.

Alert on any transition to `stale`, `invalid`, `unapproved`,
`artifact_unverified`, or `monitoring_unavailable`, as well as repeated
`blocked` readiness. Include provider, job, event, and model identifiers in
private structured logs/metrics, but keep raw data and credentials out of this
UI-facing contract.

The Base44 evidence assistant may explain these returned statuses in plain
language, summarize a completed model report, or show data-quality warnings.
It must not convert an unavailable status into a pick, estimate missing odds,
or call the result a recommendation.
