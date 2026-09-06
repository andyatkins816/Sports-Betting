# Ingestion control-plane safety boundary

This milestone defines the inactive control plane required before SAM can
consider scheduled provider ingestion. It does **not** create a scheduler,
publish a task, call a provider, read a credential, resume either shadow
worker, or authorize production data use.

## What this milestone adds

`sam_analytics.ingestion_dispatch` is a pure planning layer. A
`DispatchPolicy` is disabled by default and, even when explicitly enabled in a
test, produces only an in-memory plan. Admission requires an explicitly
allowed provider/source type, a fresh trusted quota observation, sufficient
quota above the reviewed floor, a current durable provider-activity snapshot,
request spacing, a bounded batch, and a new logical request identity. The
activity snapshot includes terminal attempts, so completing or dead-lettering
work cannot erase the last provider-call time and permit an early request.

The idempotency scheme is `sam-ingestion-dispatch-v1`. Its canonical preimage
contains only the provider, source type, credential-free request fingerprint,
and UTC window bounds. Policy revisions and queue task IDs are deliberately
excluded, so neither can turn a replay into a new provider request.

Every reservation supplied to the planner represents an outstanding credit
reservation and is subtracted from the observed quota. A newer quota receipt
does not silently release an inconclusive reservation. Reconciliation must
establish that a dispatch is terminal before an adapter omits its reservation
from the outstanding set.

`sam_analytics.ingestion_dispatch.plan_retry` permits only finite retryable
failure codes, at most five total attempts, reviewed delays, and a bounded
provider `Retry-After`. A disabled policy, non-retryable failure, exhausted
attempt budget, or excessive delay becomes a terminal dead-letter plan for
human review.

Migration `006_ingestion_control_plane.sql` adds four append-only records:

- `ingestion_dispatch`: one immutable logical request and its globally unique
  idempotency key;
- `ingestion_quota_reservation`: one immutable estimated credit reservation
  for each permitted attempt;
- `ingestion_dispatch_outbox`: one credential-free publication intent per
  permitted attempt; and
- `ingestion_dispatch_transition`: an ordered, bounded state history from
  `pending` through queue/worker activity to `succeeded`, `dead_lettered`, or
  `cancelled`.

Deferred database checks require the initial dispatch, reservation, outbox
intent, and `pending` transition to commit together. A retry-wait transition
and its next outbox intent must also commit together. Database triggers reject
out-of-order states, attempt overflow, backward timestamps, invalid retry
classes, mutation, and incomplete bundles.

The schema's latest-state and worker-activity views are read models over those
immutable facts. They are not proof of current process liveness.

## Monitoring contract

`sam_analytics.ingestion_health` turns trusted, sanitized database facts into
finite status bands for the authenticated integration-status endpoint. It
uses the latest real `odds_snapshot.received_at` for feed freshness; an empty
provider receipt does not manufacture a fresh quote. Each evaluated health
value is bound internally to one validated provider, and the status contract
rejects a value that does not match the provider named by the data-freshness
facts. It also rejects evaluated health dated in the future, more than five
seconds old, or past the next known component-age boundary, so a cached
healthy result cannot be replayed across a freshness threshold. The provider
binding, evaluation timestamp, and validity deadline are not duplicated in the
public ingestion object.
It reports:

- durable worker activity as `active`, `stalled`, `idle_unverified`,
  `unavailable`, or `invalid`;
- queue depth and oldest-work bands;
- effective quota after outstanding reservations, without returning the exact
  remaining count;
- retry-wait and dead-letter presence without task IDs or exact counts; and
- a fixed, sorted set of alert codes.

The public projection contains no credential, provider URL, private object
URI, payload digest, request fingerprint, queue name, worker identity, job ID,
or raw exception. Missing, malformed, future-dated, or stale inputs fail
closed. `/api/healthz` and `/api/readyz` remain dependency probes and do not
claim ingestion health.

## Still required before activation

This migration is intentionally unwired. A later reviewed change must:

1. implement a narrow PostgreSQL repository that locks admission per provider,
   reads a licensed provider's latest quota receipt and latest durable attempt,
   records a foreign-key binding from every reservation to the exact quota
   receipt used for its decision, and commits each admitted bundle
   transactionally; migration 006 deliberately does not yet persist that quota
   observation binding, so its records alone are not sufficient to activate;
2. implement an idempotent outbox publisher and worker consumer that append
   transitions before any provider request can occur;
3. load the sanitized health facts into the existing status contract and send
   alert codes to a private monitoring service;
4. add the separately contracted results-provider and settlement path;
5. document current provider rights, quota rules, polling cadence, retention,
   incident response, and a monthly cost ceiling;
6. pass CI, staging migration/restore testing, replay/concurrency tests, and a
   new explicit human activation review; and
7. issue fresh least-privilege staging credentials only after those checks.

Until all seven items are complete, keep provider workers suspended,
Auto-Deploy off, scheduler/Beat/Cron absent, the production evidence bucket
empty, and prediction delivery blocked.
