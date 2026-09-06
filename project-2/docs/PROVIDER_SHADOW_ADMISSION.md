# Provider ingestion runtime

SAM's licensed odds feed runs in the separate Render Background Worker
`sam-provider-shadow-worker`. It ingests pregame `h2h` odds every five minutes
and completed scores every hour for the configured sport. It does not belong in
the public `sam-api` service.

## Required rollout order

1. Merge the reviewed code and let `sam-api` apply every numbered PostgreSQL
   migration through `008_result_ingestion.sql`.
2. Confirm the migration completed before resuming this worker. The old schema
   cannot retain provider score corrections.
3. Confirm the provider plan can sustain the configured maximum: 288 one-credit
   odds requests plus 24 two-credit score requests per day.
4. Confirm the private R2 bucket and `raw/the_odds_api/` retention rule, then add
   the least-privilege provider, R2, PostgreSQL, and Key Value credentials to
   this worker only.
5. Deploy or resume the worker with the command below. Keep Auto-Deploy off for
   the initial rollout.

Never activate the scheduler before the migration is current. Never place the
provider key, R2 credentials, database URL, or Key Value URL in source control,
logs, the public web service, or the frontend.

## Render configuration

When the service Root Directory is `project-2`, use:

| Setting | Value |
| --- | --- |
| Root Directory | `project-2` |
| Docker Build Context Directory | `.` |
| Dockerfile Path | `Dockerfile` |
| Start command | `celery -A provider_worker:celery_app worker --beat --loglevel=INFO --concurrency=1 --queues=sam_provider_shadow --without-gossip --without-mingle` |

Use the exact environment-variable names from `.env.provider-shadow.example`.
The worker remains fail-closed unless its staging boundary, one approved sport,
`us` region, `h2h` market, provider license, private Render connections, and R2
evidence prefix all match the reviewed settings.

## Verification

After startup, Render logs must show both the Celery worker and embedded
scheduler running. Do not send a duplicate manual task while waiting. A healthy
runtime then shows:

- an odds task received and succeeded within five minutes;
- a score task received and succeeded within one hour;
- matching immutable receipts and successful ingestion runs in PostgreSQL; and
- retained objects under `raw/the_odds_api/sha256/` in the private R2 bucket.

Any task error, missing evidence object, missing receipt, or mismatched audit
record is a failed rollout. Suspend the worker before changing configuration or
retrying manually.

## Earlier one-request proof

The original bounded proof succeeded on 2026-09-05 and was then shut down. That
historical task does not activate this recurring runtime and must not be
replayed.
