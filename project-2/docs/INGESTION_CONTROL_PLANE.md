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
human review. The durable outbox runtime is narrower: it schedules a retry only
when the provider adapter proves that no request was sent. A provider response,
including `Retry-After`, is not itself replay-safety proof and cannot authorize
another request in this milestone.

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

Migration `008_ingestion_outbox_runtime.sql` and the broker-neutral
`sam_analytics.ingestion_outbox_runtime` close the next durability gap without
activating ingestion. The publisher claims one outbox intent under a
database-owned lease, sends only `dispatch_id` and `attempt_number`, and then
records delivery. A crash after broker acceptance may therefore redeliver the
same small message, but cannot lose the durable intent. The consumer claims an
exact attempt and commits its `running` transition before it can invoke a
provider callback. A duplicate or expired in-flight attempt is treated as
inconclusive and cannot automatically make another provider request; recovery
requires explicit reconciliation. The dispatch snapshots the rate/quota inputs
and a canonical SHA-256 of the ordered retry delays and maximum retry delay, so
reusing a policy-version label with changed retry behavior fails closed. Only
reviewed, finite failure classes with `request_not_sent` proof may schedule a
bounded retry, and success is bound to the exact payload receipt, provider use,
authorization, quota receipt, request fingerprint, and attempt.

`sam_analytics.ingestion_outbox_repository` exposes only those narrow stored
functions, validates every returned shape inside the transaction, and commits
only a result the Python boundary can decode. The migration adds append-only
publication-claim, publication-delivery, attempt-claim,
attempt-receipt, and attempt-completion facts, revokes public execution of its
functions, and leaves runtime-role grants for a later least-privilege rollout.
There is intentionally no broker adapter, provider callback, credential
loader, scheduler, task registration, or active composition root.

Quota-receipt selection and reservations are provider-wide only under the
explicit assumption that every authorized credential/account for that provider
reports one shared account-wide quota pool. Source type is not a quota-pool
identity. Before activating even one source type, the retained authorization
manifest and staging evidence must establish that shared-pool rule. Otherwise
persist an opaque `quota_pool_id` on receipts, authorizations, dispatches, and
reservations and serialize/account within that exact pool.

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

This runtime is intentionally unwired. A later reviewed change must:

1. add reviewed broker-specific publisher and consumer composition around the
   broker-neutral runtime, a function-only runtime database role, and a trusted
   receipt-insert routine; direct table writes remain forbidden. The deployment
   must also verify that the application schema is migration-role-owned and
   untrusted roles, including `PUBLIC`, have no schema `CREATE` privilege;
2. add an execution authorization/fence immediately before every socket write.
   It must recheck fresh quota, spacing, authorization, and revocation, pass a
   database-derived absolute deadline into the provider adapter, and enforce
   timeout/cancellation so a paused process cannot resume outside its lease;
3. add append-only authorization revocation/supersession facts and revalidate
   them before every attempt, including the first. Cap authorization review
   horizons or require periodic reauthorization; a finite timestamp alone is
   not sufficient;
4. prove the provider has one account-wide quota pool or add and enforce the
   exact opaque `quota_pool_id` binding described above;
5. design a fenced recovery procedure for a crash after `running` commits and
   test it. Until then an unresolved attempt intentionally blocks its provider
   lane rather than risking a duplicate licensed call;
6. make retry completion atomically convert any unsafe or unreservable retry to
   a durable dead letter so a failed reservation cannot leave the claim stuck;
7. add an append-only quota-reconciliation fact before any terminal reservation
   may be released; until then every historical reservation remains outstanding;
8. load the sanitized health facts into the existing status contract and send
   alert codes to a private monitoring service;
9. add the separately contracted results-provider and settlement path; and
10. document current provider rights, quota rules, polling cadence, retention,
    incident response, and cost ceiling; then pass CI, staging migration/restore,
    replay/concurrency/crash tests, and a new explicit human activation review
    before issuing fresh least-privilege credentials.

Until all ten items are complete, keep provider workers suspended,
Auto-Deploy off, scheduler/Beat/Cron absent, the production evidence bucket
empty, and prediction delivery blocked. The eventual runtime database role
must be separate from the migration/admin role, deny authorization inserts and
direct table writes, and expose only the reviewed repository operations; the
schema guards are defense in depth, not a substitute for least privilege.
