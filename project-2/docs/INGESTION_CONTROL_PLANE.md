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

Migration `007_ingestion_admission_repository.sql` closes the persistence gap
without activating the dispatcher. It adds an empty, append-only table whose
rows identify one exact reviewed private provider use and the digest of its
sanitized authorization manifest. New dispatches must reference one of those
rows, and every quota reservation must reference the exact immutable provider
receipt used for admission. Database triggers enforce the provider, source,
license, validity-window, quota, and timestamp relationships and serialize
receipts, admissions, and transition appends per provider. Authorization is
checked against database time as well as recorded event time, preventing a new
row from backdating itself into an expired window. Historic records remain
queryable but are never silently treated as approved. A renewal or re-review
must receive a new `license_version`; an existing exact use is immutable.
`authorization_manifest_sha256` is the SHA-256 of the exact bytes of a retained,
credential-free canonical manifest in the controlled governance system. That
manifest must repeat the provider/use tuple and validity window and identify
the immutable reviewed contract artifact and its version. It must also record
the reviewed sports/markets, pregame or in-play scope, geography, quota rules,
retention/deletion, display/attribution, derived/model-training rights,
redistribution, and termination obligations. The digest proves which bytes
were reviewed; it does not itself create or prove legal authority.

`sam_analytics.ingestion_admission_repository` is the corresponding narrow
PostgreSQL boundary. A disabled policy returns before opening a database
connection. An enabled evaluation acquires the provider lock, reads one
database-owned decision time, resolves the exact reviewed use, then reads the
matching latest quota receipt, all conservatively outstanding reservations,
candidate duplicates, and durable provider activity. It passes those facts to
the unchanged pure planner and writes each admitted dispatch, first quota
reservation, outbox intent, and `pending` transition in the same transaction.
It neither publishes the outbox nor imports a worker, provider client,
credential source, or scheduler. The authorization table is deliberately
unseeded, so this milestone cannot authorize production work.

Quota-receipt selection is source-type independent only under the explicit
assumption that the reviewed provider reports one shared account-wide quota
pool. Before activating more than one source type, the retained authorization
manifest must establish that shared-pool rule. Otherwise add and enforce a
durable `quota_pool` binding, or fail closed by requiring the receipt's exact
source type.

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

1. implement an idempotent outbox publisher and worker consumer that append and
   commit their `running` transition under the provider lock before any
   provider request can occur, then bind each resulting payload receipt to the
   exact dispatch and attempt (and therefore its immutable authorization) with
   provider/source/license/window validation and replay tests;
2. add an append-only authorization revocation/supersession fact and require a
   fresh validity check before every retry; the current finite authorization
   window does not model early termination;
3. add an append-only quota-reconciliation fact before any terminal reservation
   may be released; until then every historical reservation remains outstanding;
4. load the sanitized health facts into the existing status contract and send
   alert codes to a private monitoring service;
5. add the separately contracted results-provider and settlement path;
6. document current provider rights, quota rules, polling cadence, retention,
   incident response, and a monthly cost ceiling;
7. pass CI, staging migration/restore testing, replay/concurrency tests, and a
   new explicit human activation review; and
8. issue fresh least-privilege staging credentials only after those checks.

Until all eight items are complete, keep provider workers suspended,
Auto-Deploy off, scheduler/Beat/Cron absent, the production evidence bucket
empty, and prediction delivery blocked. The eventual runtime database role
must be separate from the migration/admin role, deny authorization inserts and
direct table writes, and expose only the reviewed repository operations; the
schema guards are defense in depth, not a substitute for least privilege.
