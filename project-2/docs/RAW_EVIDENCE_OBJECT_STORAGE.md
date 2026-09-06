# Private raw-evidence object storage

This document describes the storage boundary that must exist **before** SAM
accepts licensed provider responses. The public `sam-api` web service never
constructs this adapter and must not receive object-store or odds-provider
credentials. The private worker can construct it only in the exact
staging-only boundaries described below. The synthetic proof uses
`raw/synthetic/`; the separate one-request provider shadow uses
`raw/the_odds_api/` and the additional controls in
[manual provider-shadow admission](PROVIDER_SHADOW_ADMISSION.md).

The private R2 buckets `sam-raw-evidence-staging` and
`sam-raw-evidence-prod` are provisioned with Standard storage and public
access disabled. Staging now contains the verified synthetic proof under
`raw/synthetic/`; its worker is suspended and no longer holds its temporary R2
credentials. Production remains empty and credential-free. That completed
infrastructure proof is not approval to make a provider request. The separate
provider shadow requires its reviewed code, a new staging-bucket-only token,
and a seven-day lock on `raw/the_odds_api/` before it is created or resumed.

`sam_analytics.s3_payload_store.S3CompatibleRawPayloadStore` is a concrete
S3-compatible implementation of the `RawPayloadStore` contract. It supports
AWS S3 and Cloudflare R2, but it has no public URL, polling loop, Flask route,
or scheduler. The fixed synthetic worker remains isolated from provider code.
The separate provider-shadow worker can wire a single manually dispatched,
licensed response into `OddsLedger` only after its independent settings and
operational admission checks pass.

## What the adapter guarantees

For every raw provider response, the adapter:

1. Computes SHA-256 over the original bytes.
2. Writes only to `<prefix>/sha256/<digest>` with an S3 conditional
   `If-None-Match: *` request, so it cannot overwrite an earlier object.
3. Uses `Content-MD5` for transport integrity and writes only metadata that is
   intrinsic to the shared byte object: the adapter format, SHA-256, and byte
   count. Its object `Content-Type` is always `application/octet-stream`.
   Receipt-specific provenance (provider receipt ID, source type, timestamps,
   schema/license versions, and the provider's claimed content type) remains
   in the immutable PostgreSQL receipt/provenance ledger. This lets two lawful
   receipts for the same bytes safely refer to one content-addressed object.
4. Reads the object's metadata immediately after the write, then performs a
   bounded `GetObject` SHA-256 re-read of the retained bytes. It fails closed
   if the content type, length, encryption expectation, intrinsic metadata, or
   retained bytes do not match exactly.
5. Returns only a stable `s3://bucket/prefix/sha256/<digest>` identifier. It
   never generates browser, custom-domain, public, or presigned URLs.

The adapter only calls `PutObject`, `HeadObject`, and `GetObject`; it never
lists buckets, deletes data, changes ACLs, or enables public access.
`GetObject` streams the complete retained object in bounded chunks to verify
the SHA-256 digest, so its read permission is confined to the exact private
evidence prefix. Any SDK/network error is converted to a generic error without
retaining an SDK exception chain that could contain a request URL or
credential-adjacent material.

For AWS S3, SAM sends and verifies SSE-S3 (`AES256`). Cloudflare R2 documents
that its S3 `PutObject` API does not implement the server-side-encryption
header, while R2 encrypts objects at rest automatically; SAM therefore omits
that unsupported header for R2. See [R2 S3 API compatibility](https://developers.cloudflare.com/r2/api/s3/api/)
and [R2 data security](https://developers.cloudflare.com/r2/reference/data-security/).

## Configuration contract for the synthetic staging probe

Real deployment values for these settings belong only to the future Render
Background Worker. The public `sam-api`, Base44, browser code, and GoDaddy must
never receive them. GitHub Actions may use deliberate non-secret placeholders
only to prove that the worker container starts; CI must never receive real R2,
database, or Redis credentials. The worker admits R2 credentials only when
every other value matches this exact synthetic staging boundary.

The worker also checks that database and broker URLs use Render's documented
single-label private host forms. Public Render URLs and arbitrary internet
hosts are rejected; explicit loopback and Compose aliases remain available for
isolated CI/local verification.

| Setting | Exact value for this release |
| --- | --- |
| `APP_ENV` | `staging` |
| `SAM_WORKER_ROLE` / `SAM_WORKER_MODE` | `private_ingestion` / `synthetic_storage_probe` |
| `SAM_RAW_EVIDENCE_STORE_URI` | `s3://sam-raw-evidence-staging/raw/synthetic` |
| `SAM_RAW_EVIDENCE_STORE_BACKEND` | `cloudflare_r2` |
| `SAM_RAW_EVIDENCE_S3_REGION` | `auto` |
| `SAM_RAW_EVIDENCE_S3_ENDPOINT_URL` | Required: `https://<ACCOUNT_ID>.r2.cloudflarestorage.com`; supported jurisdiction forms are `.eu`, `.us`, and `.fedramp` |
| `SAM_RAW_EVIDENCE_MAX_BYTES` | `1048576` (1 MiB) |
| `SAM_RAW_EVIDENCE_S3_ACCESS_KEY_ID` / `SAM_RAW_EVIDENCE_S3_SECRET_ACCESS_KEY` | Required Object Read & Write R2 token pair scoped only to `sam-raw-evidence-staging`; choose a time-limited TTL if offered, then revoke it when possible or remove it from the worker immediately and let only the shortest approved expiration remain |

The URI is a stable identifier, not a connection URL. It must be lowercase
`s3://`, contain a valid bucket and non-empty safe prefix, and cannot contain a
credential, port, query string, fragment, encoded query/fragment delimiter,
or path traversal. It must exactly match the worker's
`SAM_RAW_EVIDENCE_STORE_URI` admission guard.

R2's official S3 endpoint and scoped-token setup are documented in
[Use R2 with S3](https://developers.cloudflare.com/r2/get-started/s3/) and
[R2 authentication](https://developers.cloudflare.com/r2/api/tokens/).

## Provisioning requirements outside the codebase

Before any future worker deploy, an operator must verify all of the following
in the selected storage provider:

1. Create a dedicated bucket and a dedicated `raw/odds`-style prefix for this
   licensed evidence. Do not reuse the bucket for public website files,
   Base44 exports, model artifacts, or backups.
2. Block anonymous list/read access and do not attach a public/custom domain.
   For AWS, enable Block Public Access. For R2, do not use a public bucket or
   custom-domain route for this evidence bucket.
3. Require HTTPS to the provider's API endpoint. Do not use a URL with an
   embedded access key, secret, query parameter, or presigned signature.
4. Enable AWS S3 bucket versioning and, where allowed by the provider contract,
   an appropriate retention/write-once control. For this R2 staging probe, add
   a seven-day bucket-lock rule limited to `raw/synthetic/` before creating the
   token. Cloudflare documents that bucket locks prevent deletion and overwrite
   for their retention period. Revisit the duration and provider agreement
   before any licensed payload is stored.
5. Set a lifecycle/retention policy approved by the provider agreement. Raw
   licensed data may not be retained or redistributed indefinitely merely
   because the storage platform permits it.
6. Store a restore and access-revocation procedure with the deployment record.

The code cannot safely prove bucket privacy or retention configuration without
granting bucket-administration permissions, so both are intentionally operator
preconditions rather than runtime "checks." This keeps administration authority
out of the application credential.

### Application verification is not provider-level WORM retention

The adapter-level conditional create plus `HeadObject` and full-byte
`GetObject` verification makes SAM reject a pre-existing object whose length
and metadata appear valid but whose bytes do not match the SHA-256-addressed
key. It is not a substitute for provider-administered retention or WORM
controls: a sufficiently privileged administrator could delete an object,
change bucket policy, or alter retention/versioning controls outside this
process. Bucket versioning, Object Lock or an equivalent retention policy, and
explicit denial of delete/overwrite operations remain operator responsibilities
subject to the data-provider agreement.

## Provider-specific credential boundary

The application code needs only write and full-object read access to its own
prefix. The read is necessary because verification streams the retained raw
bytes and SHA-256 hashes them; it is not merely a metadata lookup. The adapter
does not call bucket listing, deletion, ACL, bucket-policy, or public-URL APIs.
The identity that a provider can issue may nevertheless bundle broader
permissions, so granted authority must be assessed separately from calls the
code makes.

For a conventional AWS S3 bucket, the object portion of an IAM policy should
be restricted to the exact prefix, conceptually:

```json
{
  "Effect": "Allow",
  "Action": ["s3:PutObject", "s3:GetObject"],
  "Resource": "arn:aws:s3:::YOUR_PRIVATE_BUCKET/raw/odds/*"
}
```

`HeadObject` and the full raw-data `GetObject` verification read are both
authorized by `s3:GetObject`. Keep that permission scoped to the exact
evidence prefix shown above. Do not add `s3:ListBucket`, `s3:DeleteObject`,
`s3:PutObjectAcl`, or `s3:PutBucketPolicy` merely to make SAM work. The future
Render worker must receive a dedicated AWS access-key pair in its secret group;
the adapter intentionally does not fall back to a developer machine's ambient
AWS credential chain.

For R2, the dashboard's standard **Object Read & Write** token grants read,
write, and list access to selected buckets. That is broader than this adapter,
which never lists, copies, or deletes. Cloudflare's S3 operation mapping places
copy and deletion in the write category, so assume the issued identity can do
all three. Restrict it to `sam-raw-evidence-staging` only, never include
`sam-raw-evidence-prod`, choose a time-limited TTL if the dashboard offers one,
place it only in the private worker, and revoke it immediately after the single
manual probe when possible. If the dashboard offers no revocation control,
remove both values from the worker immediately and require the shortest
approved token expiration. The seven-day `raw/synthetic/` bucket-lock rule is the
provider-enforced safeguard against overwrite or deletion during the review
window. These controls are acceptable only for the fixed synthetic staging
fixture; they do not approve this credential shape for licensed provider data.

Cloudflare documents the standard token's bundled object permissions in
[R2 authentication](https://developers.cloudflare.com/r2/api/tokens/), the
write-operation mapping in [R2 temporary credentials](https://developers.cloudflare.com/r2/api/s3/temporary-credentials/),
and the retention behavior in [R2 bucket locks](https://developers.cloudflare.com/r2/buckets/bucket-locks/).

The first staging probe uses only the pinned 61-byte synthetic fixture whose
SHA-256 is
`5dc961d33ef2a18a1e47b6ffc52475bf0442bf7ba3959787a4718e5fd5015aa1`.
Its object write and terminal PostgreSQL audit append are separate operations.
If the object exists while the latest audit fact is still `running`, the probe
is inconclusive and must be reviewed manually; object presence alone is not a
successful audit receipt. A direct run-to-receipt link or outbox and automated
crash reconciliation remain mandatory before broader or scheduled provider
ingestion; the one manual provider-shadow milestone instead requires the
three-system reconciliation in its separate admission guide.

## Not yet authorized

Do not create the separate paid provider-shadow Render Background Worker until
its displayed price is confirmed and its reviewed release is merged. A new
staging-only R2 token may then be created for its one manual request, but only
after the `raw/the_odds_api/` seven-day bucket-lock rule is visible in R2. It
must never be placed in `sam-api`, the synthetic worker, Base44, GitHub, a URL,
or a screenshot. Remove it from the worker immediately after verification and
revoke it when possible; otherwise its shortest approved expiration is the
remaining backstop.

Do not add a provider credential to the synthetic worker or public API. One
real-data staging request is admitted only through the separate worker and the
preconditions in
[manual provider-shadow admission](PROVIDER_SHADOW_ADMISSION.md). Any broader
provider ingestion checkpoint must still review:

- the storage provider's current retention/security settings;
- the licensed odds provider's storage, derivation, display, and
  redistribution rights;
- a bounded dispatcher, retry/dead-letter plan, alerts, and a worker health
  signal; and
- a successful one-request provider-shadow integration test against the
  non-production private bucket with its matching raw receipt and audit.

The replacement Odds API credential may be stored only in the separately
admitted provider-shadow worker for that deliberate run. It must remain absent
from the synthetic worker, `sam-api`, Base44 secrets, URL query strings, source
control, logs, and screenshots.
