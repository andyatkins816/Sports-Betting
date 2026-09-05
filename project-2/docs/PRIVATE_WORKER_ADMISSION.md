# Private worker admission boundary

`worker.py` is deliberately **not** a general-purpose Celery entry point.
It refuses to start unless all of the following are present in the worker's
own secret/configuration group:

| Setting | Required value | Why it is required |
| --- | --- | --- |
| `SAM_WORKER_ROLE` | `private_ingestion` | Prevents an accidental generic or public-worker deployment. |
| `DATABASE_URL` | Render's internal PostgreSQL URL | The evidence ledger cannot exist without durable PostgreSQL persistence. |
| `REDIS_URL` | Render's internal Key Value URL | Provides the private task broker. |
| `SAM_RAW_EVIDENCE_STORE_URI` | A secret-free `s3://bucket/prefix` URI | Declares the dedicated private raw-evidence location before a worker can run. |
| `SAM_INGESTION_ENABLED` | `false` or absent | This release has no provider polling implementation. `true` is refused. |

The URI must have a bucket and non-empty prefix, with no credentials, port,
query string, fragment, or `.`/`..` path component. It is a stable object-store
reference—not a public or signed download URL. Use lowercase S3-compatible
bucket/prefix names; reserve the `sha256` path segment for the immutable
content-addressed object key that SAM creates.

The current worker also refuses to start if a web/API, provider, or
object-store credential is present. That is intentional: credentials in an
inert worker create an unnecessary chance of accidental consumption or
disclosure. The worker has no provider import, no provider request path, no
periodic schedule, and no Celery Beat schedule. A manually dispatched ingest
or settlement task also fails visibly rather than reporting a misleading
successful no-op.

## What this does and does not prove

This is an **admission guard**, not a live ingestion path. It verifies that
the worker was explicitly configured for a private object-store prefix; it
does not prove the provider has blocked public access, enabled encryption, or
given the worker least-privilege access. The deployment must establish those
controls before the existing storage adapter is wired into an ingestion task.

Do not create a paid Background Worker just to run this inert release. Wait
for the reviewed ledger integration and bounded dispatcher. At that point, the
service must be a Render **Background Worker**, not a Web Service, and it must
not receive a public URL.

## Next infrastructure checkpoint

Before a future ingestion release, create private object storage with a
dedicated prefix for raw odds evidence. Confirm all of the following outside
this codebase:

1. Anonymous listing and object reads are blocked.
2. Encryption at rest and HTTPS-only access are enabled.
3. Versioning or a write-once/retention control is enabled where the provider
   agreement and storage provider support it.
4. A worker-only service identity is limited to the exact evidence prefix;
   `sam-api` receives no broad object-store credential.
5. The provider's current retention, derived-data, display, and
   redistribution rights have been recorded in the approved contract.

The repository includes a concrete S3-compatible `RawPayloadStore` boundary,
but it is deliberately not constructed by `worker.py` or wired to
`OddsLedger` yet. After the storage checkpoint, the next reviewed change must
exercise that adapter against the selected provider, wire it into the ledger,
and introduce a single bounded dispatcher. Only then should a replacement
provider credential be placed in the private worker's secret store.
