# Manual provider-shadow admission

This milestone permits one bounded The Odds API request into private staging.
It must run in a **separate Render Background Worker** that is suspended by
default. Do not reuse `sam-api` or `sam-synthetic-worker`.

The admitted request selects exactly one sport from `baseball_mlb`,
`basketball_nba`, or `americanfootball_nfl` (the template defaults to MLB),
one region (`us`), and one market (`h2h`). That is one provider request and at
most one provider credit under the reviewed v4 quota calculation. Within the
same retained append-only staging database, a fixed, code-reviewed audit ID
also blocks every later task before it can contact the provider. Replacing or
resetting that audit database is outside this admission and requires a new
review. There is no scheduler, retry loop, public output, result backend,
results ingestion, settlement, or model use.

## Prerequisites

Do not create or resume the worker until all of these are true:

- the provider-shadow pull request has passed checks and is merged to `main`;
- Cloudflare R2 bucket `sam-raw-evidence-staging` has an enabled seven-day
  bucket-lock rule scoped to the exact prefix `raw/the_odds_api/`;
- a new, separate R2 Object Read & Write token is restricted to only
  `sam-raw-evidence-staging` and has not been placed in another service;
- a newly rotated replacement The Odds API key is available for this one
  shadow request; do not reuse the previously exposed key; and
- Render's **Internal Database URL** and **Internal Key Value URL** are ready.

Keep the provider key, R2 credentials, database URL, and Redis URL only in the
worker's Render environment. Never paste them into source control, logs, a pull
request, or a public service.

## Render worker configuration

Create `sam-provider-shadow-worker` as a Docker Background Worker from `main`.
Set Auto-Deploy to **Off**. When the Render service's Root Directory is
`project-2`, use:

| Setting | Value |
| --- | --- |
| Root Directory | `project-2` |
| Docker Build Context Directory | `.` |
| Dockerfile Path | `Dockerfile` |
| Start command | `celery -A provider_worker:celery_app worker --loglevel=INFO --concurrency=1 --queues=sam_provider_shadow --without-gossip --without-mingle` |

Enter the exact environment shape from `.env.provider-shadow.example`,
replacing only its blank secret/private values and the `ACCOUNT_ID` endpoint
placeholder. Do not add environment variables copied from `sam-api`, Base44, a
results worker, a scheduler, or an administrator account.

Let the first deployment reach `Live` and confirm the log says the Celery
worker is ready. Then suspend the worker until the operator is ready to perform
the single approved test.

## Send exactly one task

Resume the worker and wait for the Celery `ready` line. In that worker's Render
Web Shell, run this command exactly once, with no arguments or custom task ID:

```text
celery -A provider_worker:celery_app call --queue sam_provider_shadow sam_analytics.ingest_the_odds_api_shadow
```

Record the generated task ID. Do not run the command again, place it in a loop,
create a Cron Job, start Celery Beat, or enable Auto-Deploy.

## Verify, then shut down

Treat the run as complete only when all three private records agree:

1. **Render logs:** the recorded task ID is `received` and then `succeeded`,
   with no configuration, provider, evidence, or database error.
2. **Cloudflare R2:** an object exists under
   `raw/the_odds_api/sha256/<digest>` in `sam-raw-evidence-staging`; the bucket
   remains private and the seven-day prefix lock remains enabled.
3. **PostgreSQL:** the matching `ingestion_run` has provider `the_odds_api`,
   admission ID `f3cd3650-568a-4f36-89b8-acde937c23a1`, job identity
   `celery:<task-id>`, and latest state `succeeded`; its provider payload
   receipt and raw-data provenance rows exist. A valid empty admitted `h2h`
   response can create no odds snapshots and still be a successful receipt.
   If an exchange adds `h2h_lay`, the exact response remains in raw evidence,
   but that added market is not normalized into the h2h-only ledger.

An object without a matching successful audit, or an audit without its object
and receipt, is inconclusive. Do not retry automatically or infer success from
the Render deployment badge alone.

After verification, suspend `sam-provider-shadow-worker` immediately. While it
is suspended, remove the provider key and both R2 credential values from the
worker, and rotate or revoke the provider key. Revoke the separate R2 token when
that control is available; otherwise make sure it has the shortest approved
expiration and remains absent from the worker. Leave Auto-Deploy off, do not
resume the worker, and keep the seven-day R2 lock in place. Any additional
provider request requires a new deliberate approval and a reviewed admission-ID
change.
