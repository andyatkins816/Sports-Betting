# SAM deployment runbook: GoDaddy, Base44, and Render

## Purpose and current safe state

SAM is a lawful, research-only sports-market analytics service. It does not
place wagers, promise profitability, or publish predictions merely because a
website is online.

This is the current staging snapshot, checked 2026-09-05:

| Component | Current state | Important limit |
| --- | --- | --- |
| `sam-api` | Deployed as a Render Docker web service in Oregon; `/api/healthz` and `/api/readyz` passed. | Readiness proves Postgres migrations and Key Value connectivity, not that data or a model is ready. |
| `sam-postgres` | Private Render Postgres in Oregon, connected through `DATABASE_URL`; reviewed migrations run before deploy. | No real provider observations or model facts have been written yet. |
| `sam-key-value` | Private Render Key Value (Valkey) in Oregon, connected through `REDIS_URL`. | It backed the successful synthetic worker proof and remains private. |
| Application environment | `APP_ENV=staging`. | Render's visible project grouping named “Production” is a UI label, not permission to treat SAM as production. |
| `sam.vegas` | Base44 is the public experience and evidence/governance UI. | Do not point the apex domain at the Python API. |
| `api.sam.vegas` | Reserved for the Render API custom domain. | Do not configure DNS or Base44's production status URL until Render domain validation is ready. |
| Raw evidence storage | Private Cloudflare R2 buckets `sam-raw-evidence-staging` and `sam-raw-evidence-prod` are provisioned with Standard storage and public access disabled. | Staging contains the verified content-addressed synthetic and one-request provider proofs under their separate locked prefixes; production remains empty. The temporary provider-proof R2 token was deleted. |
| Workers and ingestion | Both one-request staging proofs succeeded. `sam-synthetic-worker` and `sam-provider-shadow-worker` are suspended; the provider and R2 credentials were removed from the latter. | No scheduled polling, results ingestion, model training, or prediction delivery is authorized. The replacement provider credential was rotated after the proof. |

The desired topology is deliberately simple:

```text
Browser
  │
  └─ GoDaddy DNS for sam.vegas
       ├─ sam.vegas / www.sam.vegas  → Base44 public UI + Evidence Copilot
       └─ api.sam.vegas              → Render sam-api (Flask/Gunicorn)
                                             │
                                             ├─ Render Postgres (private)
                                             ├─ Render Key Value (Valkey/Redis-compatible, private)
                                             ├─ future private worker/scheduler
                                             └─ future private object storage
                                                        ↑
                                          licensed odds and results providers
```

Base44 presents verified evidence and explains safe status in plain language.
The Python service remains the source of analytical facts. Neither surface may
invent an odds feed, prediction, edge, or performance claim.

## What is live versus what is next

### Completed staging foundation

- GitHub `main` passed CI and is the Render deployment source.
- Render runs the Dockerized `sam-api` service in Oregon.
- Render has generated and stores the web-service secrets; do not export,
  reveal, or paste their values into chat, source code, Base44 client code, or
  GitHub.
- The unauthenticated liveness endpoint responds at `/api/healthz`.
- Render's private Postgres and Key Value services are in the same Oregon
  region and the API's dependency-readiness check has passed.

### Explicitly not ready

- The immutable odds repository passed one bounded provider-shadow proof, but
  it is not activated for recurring ingestion.
- Both proof workers are suspended. The provider-shadow worker has no provider
  key or storage token, and no scheduler exists.
- No licensed odds/results pipeline is polling, no model artifact is approved,
  and no public prediction delivery is enabled.
- Base44 is not yet connected to the Python status endpoint with its separate
  status capability.
- `api.sam.vegas` is not yet the confirmed public API hostname.

The status endpoint must therefore remain `blocked` when it is eventually
connected to Base44. A green `/api/healthz` response is intentionally different
from a data/model-readiness response.

## Domain, DNS, and HTTPS

GoDaddy remains the registrar and DNS provider. Cloudflare is not a required
part of this deployment. It can be evaluated later as an optional DNS/WAF
layer, but do not move nameservers or introduce a proxy during the initial
cutover.

### Guardrails before touching DNS

1. Inventory the existing GoDaddy records. Preserve MX, SPF, DKIM, DMARC, and
   any other email or verification records.
2. `docs/CNAME` is a legacy GitHub Pages-style artifact containing
   `sam.vegas`. Do not rely on it for the live site. Confirm GitHub Pages is not
   competing for the custom domain before changing records; remove or disable
   that legacy configuration only after the Base44 domain is verified.
3. Copy DNS values exactly from the Base44 and Render custom-domain screens.
   Never substitute an address or CNAME target from an old guide or another
   service.
4. Change only the specific hostname being configured. In particular, a CNAME
   for `api` must not overwrite the apex (`@`) or unrelated mail records.

### Configure the Base44 public site

1. In Base44, add or confirm `sam.vegas` as the public custom domain.
2. In GoDaddy, create the exact apex and/or `www` records Base44 supplies.
3. Pick one public canonical host (`sam.vegas` is preferred) and configure the
   other host to redirect as Base44 instructs.
4. Wait for Base44 domain verification and HTTPS before declaring the public UI
   live.

### Configure the Render API hostname

Do this only after the database/queue rollout plan is ready; it is not needed
to prove the temporary Render service works.

1. In Render's `sam-api` settings, add the custom domain `api.sam.vegas`.
2. Render will display a domain-validation target. In GoDaddy, add a CNAME with
   host/name `api` and the exact target supplied by Render.
3. Wait for Render to validate the record and provision its managed TLS
   certificate. Do not bypass a certificate warning or force HTTPS before
   validation completes.
4. Verify `https://api.sam.vegas/api/healthz` returns the expected liveness
   response over HTTPS.
5. Keep `ALLOWED_ORIGINS` limited to `https://sam.vegas` and
   `https://www.sam.vegas`. The Base44 status call is server-to-server and does
   not need browser CORS access.

Only after step 4 is successful should Base44's status gateway move to
`https://api.sam.vegas`. Update the URL and pinned host together; see
`BASE44_HANDOFF.md`.

## Secrets and trust boundaries

Use Render's encrypted environment-variable store and Base44's server-secret
store. A secret belongs in exactly the service that needs it.

| Secret/configuration | Where it belongs now | Purpose and rule |
| --- | --- | --- |
| `APP_ENV=staging` | Render `sam-api` | Keeps this rollout in staging even if Render's project UI says “Production.” |
| `SESSION_SECRET` | Render `sam-api` | Generated random value; never copied to Base44 or GitHub. |
| `SAM_STATUS_API_KEY` | Render `sam-api`, then Base44 server secret named `SAM_BACKEND_STATUS_API_KEY` | The only credential that authorizes `GET /api/v1/integration/status`. Send it only as a server-side `X-API-Key` header. |
| `SAM_API_KEY` | Render only, if retained | Reserved for the separate private research capability. Never send this key to Base44; that capability is disabled outside development/test. |
| `ALLOWED_ORIGINS` | Render `sam-api` | Exact public Base44 origins only; never use `*` for an authenticated API. |
| `DATABASE_URL` | Render `sam-api` and future worker | Use Render's internal Postgres connection string, not an external URL. The current readiness endpoint checks it without revealing it. |
| `REDIS_URL` | Render `sam-api` and future worker | Use only the private/internal Key Value connection string. The current readiness endpoint checks it without revealing it. |
| `ODDS_PROVIDER_API_KEY` | Future private worker only | Rotate the previously exposed key first. Never add it to the web API, Base44, browser, GitHub, or logs. Staging/production `sam-api` refuses to start if this or a results-provider key is present. |
| object-storage credentials | Future private worker only | Use the smallest provider-supported identity. For an R2 probe, scope the standard Object Read & Write token only to `sam-raw-evidence-staging`, choose a time-limited TTL, add the required prefix lock, and remove its values from the worker immediately afterward; revoke it when that control is available. Staging/production `sam-api` refuses to start if worker-only settings or credentials are present. |

Never use a URL query parameter for a secret. Never expose secrets through an
API response, Base44 entity, client-side JavaScript, logs, screenshots, or an
AI-agent prompt.

## Staged infrastructure rollout

### Stage 1 — maintain and validate Postgres

1. Keep `sam-postgres` in Oregon with `sam-api`.
2. Keep the Render internal database URL in `DATABASE_URL`; never replace it
   with a public endpoint.
3. The numbered migration runner already executes as the reviewed pre-deploy
   command. For each future migration, record the version and verify a restore
   procedure before loading real provider data.
4. Restrict database access to Render services that require it. Use a
   least-privilege application role once the repository is implemented.

### Stage 2 — keep the private queue/cache ready

1. Keep `sam-key-value` in Oregon in the same project.
2. Keep its internal endpoint in `REDIS_URL`; do not expose the service
   publicly.
3. Verify queue connectivity independently. A configured URL is not evidence
   that a worker is healthy.
4. Do not begin a worker solely because a broker exists. The worker needs
   reviewed database persistence, idempotency, retry/dead-letter handling, and
   alerting first.

### Stage 3 — add private immutable object storage

Use an S3-compatible provider selected for the applicable data-license and
retention requirements. Store only the data the provider contract permits SAM
to retain.

The repository includes a concrete AWS S3 / Cloudflare R2 adapter. The
synthetic worker uses it only for its fixed proof. A separate, manual-only
provider-shadow worker can pass one licensed response through `OddsLedger`
under a different exact prefix and admission guard. The dedicated staging and
production R2 buckets are provisioned. Follow
[private raw-evidence object storage](RAW_EVIDENCE_OBJECT_STORAGE.md) and the
[manual provider-shadow admission guide](PROVIDER_SHADOW_ADMISSION.md) before
the provider worker constructs the adapter.

- Keep the bucket private; block anonymous listing and public object reads.
- Leave `sam-raw-evidence-prod` empty and credential-free while the application
  is staging. The fixed synthetic proof exists under
  `sam-raw-evidence-staging/raw/synthetic/`, and the single provider proof
  exists under `raw/the_odds_api/`; both prefixes retain their reviewed
  seven-day lock.
- The provider-proof R2 token was removed from the worker and permanently
  deleted after the three-system verification. Create a new least-privilege,
  staging-bucket-only identity only after a later activation review.
- Encrypt data at rest and in transit; enable versioning or write-once evidence
  controls where available.
- Separate raw provider payloads, model artifacts, and reports by restricted
  prefixes/buckets with lifecycle policies.
- Store provider responses, feature snapshots, model artifacts, calibration
  reports, and backups only after their contracts and retention rules are
  documented.
- Give the public API no object-storage credentials. Each private worker
  receives its own smallest provider-supported scope, with separately enforced
  retention when the provider bundles authority the adapter does not use.

### Stage 4 — verify each private worker manually

The synthetic storage proof completed successfully across Render, R2, and
PostgreSQL. Its worker is suspended, its temporary R2 credentials were removed,
and the short-lived token is expiring. Keep it that way; it never receives a
provider credential.

The separate provider-shadow checkpoint also completed successfully. The
single task `29ab597f-df4f-41cc-9a95-40b08732899c` produced the audited run
`f3cd3650-568a-4f36-89b8-acde937c23a1`, a matching receipt/provenance record,
and the private R2 object with digest
`309b6d1cdd2b999c6830bd4cd4492d17e919c65c62e6e5385c1f703c9d0a898b`.
The latest audit state was `succeeded` with one attempt. The worker was then
suspended, its temporary provider/R2 values were removed, the provider key was
rotated, and the R2 token was deleted.

That proof is still not a general provider dispatcher. The inactive
[ingestion control-plane safety boundary](INGESTION_CONTROL_PLANE.md) now
defines deterministic idempotency, quota/spacing/batch admission, bounded
retry/dead-letter decisions, append-only transactional dispatch facts, and a
sanitized monitoring projection. It is deliberately unwired and disabled by
default; it made no additional provider request and creates no schedule.

Before starting scheduled or production ingestion, finish and test all of the
following:

- transactional Postgres persistence and source-received timestamps;
- a transactional repository/outbox publisher that uses the new idempotency,
  quota-reservation, and dead-letter records;
- a results-provider contract and settlement reconciliation;
- a trusted database adapter that supplies the new ingestion-health contract;
- production-reviewed rate limits that respect the provider agreement and
  current subscription quota; and
- delivery of finite alerts for failures, queue depth, stalled jobs, stale
  data, and quota loss to a private monitoring service.

Neither manual worker can settle, train, publish predictions, or place wagers.
This is a safety boundary, not an outage.

### Stage 5 — provider onboarding and model governance

1. Rotate the Odds API credential that was shared outside a secret manager.
2. Review the provider's current storage, derivation, display, and
   redistribution rights before enabling any feed.
3. Store the replacement credential only in the private worker configuration.
4. Start with a paper/shadow ingest and verify lineage, freshness, and
   settlement before a model is eligible for review.
5. Require time-split backtests, calibration, approved immutable artifacts,
   monitoring, and an explicit approval record. No model is “ready” from a
   neural-network or gradient-boosting score alone.

## Base44 integration after API DNS is verified

Base44 is a server-side gateway and evidence experience, not a second prediction
engine. After `https://api.sam.vegas/api/healthz` verifies:

1. Set Base44 server secret `SAM_BACKEND_URL` to `https://api.sam.vegas`.
2. Set Base44 server secret `SAM_BACKEND_HOST` to `api.sam.vegas`.
3. Set Base44 server secret `SAM_BACKEND_STATUS_API_KEY` to the value of
   Render's separate `SAM_STATUS_API_KEY`.
4. Have the Base44 backend function call only
   `GET /api/v1/integration/status` with `X-API-Key`.
5. Confirm Base44 renders a transparent `blocked` state until independently
   verified operational evidence exists.

Do not put the status key in a React component, browser storage, a Base44 AI
prompt, or a client-visible entity. Do not use `SAM_API_KEY` in Base44.

Python-to-Base44 evidence publishing is a later, separate capability. It needs
a new random webhook secret and the host-pinned, signed webhook configuration
described in `BASE44_HANDOFF.md`. It must not reuse the status or provider key.

## Monitoring, security, and deployment operations

### Immediate controls

- Keep Render's health check at `/api/healthz`; treat it as liveness only.
- Turn on Render service/database notifications for deploy failures and service
  availability.
- Keep GitHub pull-request checks required before merging to `main`; enable MFA
  for GitHub, Render, GoDaddy, Base44, and any provider account.
- Review Render deploy logs after every environment-variable or database change.
- Keep the public API rate-limited at the application/edge layer before broad
  use. A future WAF/DDoS layer can be added after the basic domain cutover.

### Before operational data is accepted

- Add structured logs with request, job, provider, event, and model IDs while
  redacting Authorization headers and all secret values.
- Add error reporting and metrics/traces (for example, Sentry plus an
  OpenTelemetry-compatible backend).
- Alert on API availability, p95 latency, 5xx rate, database errors/storage,
  queue/worker failures, feed freshness, provider quota, ingest/settlement lag,
  dead letters, backup failures, and calibration/model drift.
- Test a Postgres backup/restore and access revocation procedure.
- Use a reviewed deployment path: CI → staging deploy → smoke check → human
  approval → production release and rollback plan.

## Budget and purchase gates

The verified initial Render baseline is approximately:

| Resource | Selected plan | Monthly cost before tax/usage |
| --- | --- | --- |
| `sam-api` | small always-on web service | $7 |
| `sam-postgres` | small managed Postgres | $6 |
| `sam-key-value` | small persistent Valkey/Key Value | $10 |
| Current baseline | API + database + Key Value | **$23** |

Do not infer future prices from this baseline. Before enabling more components,
check Render's current checkout price for a Key Value instance and Background
Worker, then add object storage, monitoring, bandwidth, backups, and provider
costs. The data feed is likely the largest variable cost; a retail odds plan is
not equivalent to official or low-latency syndicate-grade data rights.

Suggested purchasing order:

1. Finish database integration and restore verification.
2. Add Key Value and a private worker only after persistence/operations work is
   ready.
3. Add private object storage before retaining licensed raw data or model
   artifacts.
4. Purchase/enable a provider only after its terms, quotas, and permitted uses
   are documented.
5. Upgrade capacity only from measured CPU, memory, query, queue, and latency
   evidence.

## Legal and product-release gate

Before publishing a prediction/edge page or allowing users to act on model
output, complete signed data licenses, a data inventory, privacy/terms,
age/jurisdiction rules, responsible-gambling disclosures, model approvals,
monitoring, and jurisdiction-specific legal review. Do not enable wager
submission without the necessary regulatory and operator authorization.

## Primary references

- Render service types: <https://render.com/docs/service-types>
- Render environment variables: <https://render.com/docs/configure-environment-variables>
- Render health checks: <https://render.com/docs/health-checks>
- Render custom domains: <https://render.com/docs/custom-domains>
- Render Postgres backups/recovery: <https://render.com/docs/postgresql-backups>
- The Odds API terms: <https://the-odds-api.com/terms-and-conditions.html>
