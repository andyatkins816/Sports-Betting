# Private raw-evidence object storage

This document describes the storage boundary that must exist **before** SAM
accepts licensed provider responses. It is intentionally not a deployment
instruction for today: the current `sam-api` web service and inert worker do
not construct this adapter, and neither should receive object-store or odds
provider credentials.

`sam_analytics.s3_payload_store.S3CompatibleRawPayloadStore` is a concrete
S3-compatible implementation of the `RawPayloadStore` contract. It supports
AWS S3 and Cloudflare R2, but it has no public URL, polling loop, Flask route,
or scheduler. A later reviewed change must wire it into `OddsLedger` inside a
private background worker.

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

## Configuration contract for a future private worker

All of these settings belong only to the future Render Background Worker. The
current worker rejects the credentials below because provider ingestion is not
implemented yet. The public `sam-api`, Base44, browser code, GitHub Actions,
and GoDaddy must never receive them.

| Setting | AWS S3 | Cloudflare R2 |
| --- | --- | --- |
| `SAM_RAW_EVIDENCE_STORE_URI` | `s3://bucket/raw/odds` | Same stable, secret-free form |
| `SAM_RAW_EVIDENCE_STORE_BACKEND` | `aws_s3` | `cloudflare_r2` |
| `SAM_RAW_EVIDENCE_S3_REGION` | Required AWS region such as `us-west-2` | Exactly `auto` |
| `SAM_RAW_EVIDENCE_S3_ENDPOINT_URL` | Leave blank for AWS's SDK endpoint, or use the matching AWS regional endpoint | Required: `https://<ACCOUNT_ID>.r2.cloudflarestorage.com`; supported jurisdiction forms are `.eu`, `.us`, and `.fedramp` |
| `SAM_RAW_EVIDENCE_MAX_BYTES` | Max one raw response; defaults to `20971520` (20 MiB), hard limit 100 MiB | Same |
| `SAM_RAW_EVIDENCE_S3_ACCESS_KEY_ID` / `SAM_RAW_EVIDENCE_S3_SECRET_ACCESS_KEY` | Required narrowly scoped AWS key pair, stored only in the Render worker secret group | Required bucket-scoped R2 S3 API token pair |

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
   an appropriate retention/write-once control. R2 has different lifecycle and
   object-lock capabilities, so confirm the current R2 controls and the data
   license before promising immutability beyond SAM's conditional-write rule.
5. Set a lifecycle/retention policy approved by the provider agreement. Raw
   licensed data may not be retained or redistributed indefinitely merely
   because the storage platform permits it.
6. Store a restore and access-revocation procedure with the deployment record.

The code cannot safely prove bucket privacy without granting broad
bucket-administration permissions, so privacy is intentionally an operator
precondition rather than a runtime "check." This preserves least privilege.

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

## Least-privilege identity

The worker needs only write and full-object read access to its own prefix. The
read is necessary because verification streams the retained raw bytes and
SHA-256 hashes them; it is not merely a metadata lookup. It does **not** need
bucket listing, deletion, ACL changes, bucket-policy changes, or public URL
permissions.

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

For R2, create an S3 API token with **Object Read & Write** limited to the
single evidence bucket. The read permission supports the full-byte SHA-256
verification above. R2's token interface scopes to buckets; keep the bucket
dedicated to SAM raw evidence so that its full raw-data read scope remains
narrow.

## Not yet authorized

Do not create a Render Background Worker, add a provider credential, or put
object-store credentials in Render today solely because this code exists. The
next implementation checkpoint must first review:

- the storage provider's current retention/security settings;
- the licensed odds provider's storage, derivation, display, and
  redistribution rights;
- a bounded dispatcher, retry/dead-letter plan, alerts, and a worker health
  signal; and
- an integration test against a non-production private bucket with synthetic
  bytes only.

Only after that review should the replacement Odds API credential be stored in
the private worker's secret group. It must remain absent from `sam-api`, Base44
secrets, URL query strings, source control, logs, and screenshots.
