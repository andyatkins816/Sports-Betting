# Private worker admission boundary

This document covers only `worker.py`, the synthetic proof worker. The later
one-request real-provider path is a separate service and boundary documented in
[manual provider-shadow admission](PROVIDER_SHADOW_ADMISSION.md); neither
worker's credentials or queue may be reused by the other.

`worker.py` is deliberately **not** a general-purpose Celery entry point. This
release admits exactly one staging-only synthetic storage probe. It does not
authorize a provider request, odds ingestion, event settlement, model training,
or scheduled work.

## Exact admitted configuration

The worker refuses to start unless its private environment group contains this
exact non-secret boundary:

| Setting | Required value |
| --- | --- |
| `APP_ENV` | `staging` |
| `SAM_WORKER_ROLE` | `private_ingestion` |
| `SAM_WORKER_MODE` | `synthetic_storage_probe` |
| `DATABASE_URL` | Render's internal PostgreSQL URL |
| `REDIS_URL` | Render's internal Key Value URL |
| `SAM_INGESTION_ENABLED` | `false` or absent |
| `SAM_RAW_EVIDENCE_STORE_BACKEND` | `cloudflare_r2` |
| `SAM_RAW_EVIDENCE_STORE_URI` | `s3://sam-raw-evidence-staging/raw/synthetic` |
| `SAM_RAW_EVIDENCE_S3_REGION` | `auto` |
| `SAM_RAW_EVIDENCE_S3_ENDPOINT_URL` | The account's official HTTPS R2 S3 endpoint |
| `SAM_RAW_EVIDENCE_MAX_BYTES` | `1048576` |

`DATABASE_URL` and `REDIS_URL` are admitted only with Render's single-label
internal host forms (`dpg-...` and `red-...`). Explicit loopback and Compose
service aliases are also allowed for isolated CI/local verification. Public
Render hosts and arbitrary internet hostnames fail worker admission.

A non-empty R2 S3 access-key pair is also required, but the settings object
does not retain it. The pair must come from an **Object Read & Write** token
restricted to `sam-raw-evidence-staging`. Cloudflare's standard permission
bundles object listing, and its write category should be assumed to allow copy
and deletion, even though this worker never calls those operations. Before
creating the token, add a seven-day bucket-lock rule for `raw/synthetic/`;
choose a time-limited TTL if the dashboard offers one, and revoke it immediately
after the one probe when that control is available. Otherwise, remove both
credential values from the worker immediately and let only the shortest approved
expiration remain. Bucket-level token scope plus the prefix retention rule are
the security boundary because the standard R2 token cannot restrict access to
`raw/synthetic` alone. Never include `sam-raw-evidence-prod`.

The worker rejects provider credentials, public API/session credentials,
Base44 settings, ambient AWS credentials, and Cloudflare account-administration
credentials. In addition to known names, any configured environment variable
whose name looks like an API key, token, secret, password, authorization value,
credential, cookie, or bearer value is rejected unless it is one of the exact
two R2 credential names above. It also rejects production mode, the production
bucket, a generic S3 backend, a larger payload limit, or any live-ingestion
switch. Error text never includes the rejected name or value.

Use `.env.worker.example` as the shape for local review. Never copy the public
API's environment group into this worker, never add worker settings to
`sam-api`, and never commit a populated `.env.worker` file.

## What the probe does

The sole admitted task is zero-argument and manual-only. At execution time it
revalidates the complete boundary, uses Celery's generated task ID as a safe
job identity, and, while each dependency remains available:

1. commits an append-only `queued` audit fact to PostgreSQL;
2. commits `running` before any object-store write;
3. stores one fixed, versioned synthetic JSON fixture under the staging
   content-addressed prefix;
4. verifies the object with `HeadObject` and a bounded `GetObject` SHA-256
   read; and
5. attempts to commit `succeeded`, or an enumerated safe failure.

The approved fixture is exactly 61 bytes and has SHA-256
`5dc961d33ef2a18a1e47b6ffc52475bf0442bf7ba3959787a4718e5fd5015aa1`.
That pins its private content-addressed key under
`raw/synthetic/sha256/<digest>` without disclosing any credential or endpoint.

It returns no payload, object URL, digest, credential, or provider content.
The storage adapter never lists or deletes objects, changes an ACL, or creates
a public/presigned URL. Repeating the probe verifies the same content-addressed
object rather than accumulating synthetic files.

`ingest_quotes` and `settle_events` remain visibly inert. The worker imports no
provider client and cannot write synthetic odds, events, or snapshots through
`OddsLedger`.

## Manual-only delivery controls

The probe uses the dedicated `sam_manual_shadow` queue with concurrency one.
Celery result storage, automatic retries, late acknowledgement, worker-lost
redelivery, missing-queue creation, and Beat scheduling are disabled. The
checked-in Compose worker remains behind the opt-in `private-worker` profile
and reads a separate `.env.worker` file.

Start the reviewed worker with:

```text
celery -A worker:celery_app worker --loglevel=INFO --concurrency=1 --queues=sam_manual_shadow --without-gossip --without-mingle
```

From a separate process with the same reviewed private worker environment,
publish exactly one zero-argument probe with:

```text
celery -A worker:celery_app call sam_analytics.verify_staging_raw_evidence
```

Do not supply arguments, a custom task ID, a different queue, or a repeat loop.
"Manual-only" is an operational boundary, not Celery authentication: keep the
internal Redis endpoint private and restrict its credential to the approved
Render services plus the operator process used for this one publish. Anyone
who can publish to that broker could enqueue work.

Early acknowledgement makes this staging probe at-most-once, not guaranteed
delivery. A process loss before the first database transaction can leave no
audit row. A process or database failure later can leave the latest durable
state at `queued` or `running`; in particular, the fixed object can exist even
if the final `succeeded` append fails. Treat every such outcome as
**inconclusive**, inspect the private audit and deterministic object state, and
then deliberately publish a new task with a new Celery-generated ID if needed.
Never infer success from the object's presence alone.

These controls bound one deliberate synthetic staging dispatch; they are not a
complete production provider dispatcher, outbox, reconciliation process, or
dead-letter system.

## Current completed proof and what remains unauthorized

The private `sam-raw-evidence-staging` and `sam-raw-evidence-prod` R2 buckets
exist with public access disabled. Staging contains the verified synthetic
object under `raw/synthetic/`; production remains empty. The synthetic worker is
suspended, its R2 credential values were removed, and its short-lived token is
expiring. Do not resume or repurpose that worker.

Creating the separate provider-shadow Render Background Worker is another paid
infrastructure action. Confirm the displayed price and merge its reviewed code
before creating it. Do not create a Cron Job or start Celery Beat.

This synthetic worker remains permanently blocked from real provider
ingestion. A separate, one-request provider-shadow path now has its own exact
admission boundary, retention prefix, queue, settings, and operator runbook in
[manual provider-shadow admission](PROVIDER_SHADOW_ADMISSION.md). That narrow
shadow proof does not authorize scheduling, production ingestion, settlement,
model training, or public output.
