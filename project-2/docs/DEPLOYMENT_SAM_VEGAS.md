# sam.vegas deployment runbook

## Recommended initial topology

Use `sam.vegas` for a marketing/static landing page and `api.sam.vegas` for the
authenticated API. Keep worker/admin endpoints private; they must not share the
public web service route.

```text
Browser → Cloudflare DNS/WAF/TLS → static site (sam.vegas)
                                 └→ API load balancer (api.sam.vegas)
                                       ├→ Flask/Gunicorn API (stateless)
                                       ├→ worker + scheduler (private)
                                       ├→ managed PostgreSQL
                                       ├→ managed Redis/Valkey
                                       └→ object storage for raw payloads/artifacts
                                                  ↑
                                   licensed data-provider adapters
```

Render is a sensible first managed host: it supports Docker web services,
background workers, cron/workflows, managed Postgres, and Redis-compatible Key
Value. Keep production Postgres and Redis managed rather than using the local
`compose.yaml` containers. A cloud VM is a viable lower-level alternative only
if you are ready to own patching, backups, network segmentation, and recovery.

## Pre-production gate

Do not publish a prediction/edge page yet. Before a public launch, complete:

1. Signed data licences and a data inventory; provider adapter contract tests;
   no public raw-data API unless the contract expressly permits it.
2. Migration runner plus a tested backup and point-in-time restore procedure.
3. Private service identity for workers, managed secret store, least-privilege
   database roles, restricted database/cache ingress, and no secrets in GitHub
   Actions logs or Docker images.
4. Authentication/authorization: a real identity provider for dashboards and
   admin actions, rate limits by identity/IP, and an API gateway/WAF rule set.
5. A locked paper/shadow evaluation with model approval records and monitoring.
6. Terms of service, privacy policy, age/jurisdiction access rules, responsible
   gambling disclosures, and advice from qualified counsel. Do not present a
   model as profitable or enable wager submission without the appropriate
   regulatory and operator authorization.

## Domain, DNS, and HTTPS

The existing `docs/CNAME` contains `sam.vegas`, which is appropriate only for a
static GitHub Pages-style site. It is not an API deployment. Decide whether to
keep that static page or replace it; avoid having two systems compete for the
same apex records.

1. Add `sam.vegas` to Cloudflare and change the registrar nameservers to the two
   values Cloudflare supplies. Verify zone activation before changing traffic.
2. Add the static-site host's custom-domain records for `sam.vegas` and `www`
   exactly as that host instructs. Redirect one canonical host to the other.
3. Add `api.sam.vegas` as a CNAME to the API host's target. At the host, register
   that custom domain and enforce its domain validation before proxying traffic.
4. Enable Cloudflare proxying only after the origin certificate is active. Set
   SSL/TLS to **Full (strict)**, redirect HTTP to HTTPS, enable HSTS after testing
   subdomains, and keep origin access restricted to Cloudflare/host networks.
5. Configure a restrictive CORS allow-list (`https://sam.vegas` and
   `https://www.sam.vegas` only), CSP, WAF/rate-limit rules, and bot/DDoS
   controls. Never use a wildcard CORS origin for authenticated APIs.

Cloudflare's Universal SSL covers the apex and first-level names such as
`api.sam.vegas`; it does not remove the need for valid origin TLS and strict
origin verification.

## Service configuration and scheduled jobs

Build from `project-2/Dockerfile`. The web command is Gunicorn; it has no direct
provider credentials beyond what it needs to validate requests. Run two separate
private processes from the same image:

- `celery -A worker.celery_app worker --loglevel=INFO`
- `celery -A worker.celery_app beat --loglevel=INFO`

The included schedule intentionally does nothing when providers are unconfigured.
The Odds API v4 pregame adapter is implemented but deliberately not polled until
its PostgreSQL repository and data-quality incident writer are connected. Then
schedule polling at the agreed rate and settlement reconciliation every five
minutes. Add a nightly model/data-quality report and a daily encrypted backup
verification. Use a queue dead-letter strategy and idempotency keys, not blind
retries.

## Observability, security, and delivery

- Emit structured JSON logs with request/job/event/provider/model IDs; redact
  Authorization and provider keys. Send errors to Sentry (or equivalent) and
  metrics/traces to an OpenTelemetry-compatible backend.
- Define alerts for availability, p95 latency, 5xx rate, feed freshness, provider
  quota, ingest/settlement lag, worker failures, dead letters, DB CPU/storage,
  backup failures, and model/calibration drift.
- Store secrets in the host's secret manager, rotate them, use separate
  development/staging/production credentials, and protect production deployment
  with MFA and branch protection. Run dependency/image scanning and secret
  scanning in CI.
- CI should run formatting/linting, unit/contract tests, migrations against a
  disposable Postgres, and an image scan. CD should build an immutable image,
  deploy staging, run smoke checks, require approval, then use a rolling or
  blue/green production release with automatic rollback.

## Monthly operating budget (September 2026 planning estimate)

| Stage | Platform components | Estimate, excluding data and tax |
| --- | --- | --- |
| Development | local containers + free DNS/TLS | $0–15 |
| Private beta | 1 small API, 1 small worker/scheduler, managed Postgres, small Redis, object storage, error monitoring | $40–120 |
| Reliable public service | redundant API/worker capacity, managed DB backups, Redis, WAF/observability, object storage | $200–600+ |

For an inexpensive managed proof of concept, Render reports an always-on small
web service plus small Postgres at about $13/month before growth; this system
also needs at least a worker and persistent queue/cache, so use the higher beta
range as the actual planning floor. Render bills services independently and
usage/bandwidth can change the number.

Data is the variable cost that matters most. The Odds API currently lists $29/mo
for 20,000 requests and $99/mo for 200,000 requests with additional historical
and market coverage; its enterprise/official-feed alternatives require quotes.
Do not regard a retail API plan as equivalent to professional low-latency or
official data rights.

## Hosting decision

Start with: Cloudflare + Render + managed Postgres/Key Value + S3-compatible
object storage + licensed pregame odds/results provider. Revisit a cloud account
(AWS/GCP/Azure) when you need private networking across more services, audited
IAM, multi-region recovery, or vendor-compliance requirements. The data contract
and evidence pipeline—not a more complicated neural network—are the next
critical investment.

## Primary references (checked 2026-09-04)

- Render service types, managed Postgres/Key Value, and free-tier limitations:
  <https://render.com/docs/service-types>
- Render current cost guidance and billing variables:
  <https://render.com/articles/how-much-does-cloud-application-hosting-cost-for-small-businesses>
- Cloudflare Universal SSL setup and coverage:
  <https://developers.cloudflare.com/ssl/edge-certificates/universal-ssl/enable-universal-ssl/>
- The Odds API published plans/coverage:
  <https://theoddsapi.com/pricing>
- The Odds API terms, especially limits on reselling/repackaging raw data:
  <https://the-odds-api.com/terms-and-conditions.html>
