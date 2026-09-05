import base64
import hashlib
import io
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from sam_analytics.raw_payload_store import RawPayloadMetadata, RawPayloadStoreViolation
from sam_analytics.s3_payload_store import (
    RawPayloadStoreConfigurationError,
    RawPayloadStoreUnavailable,
    S3CompatibleRawPayloadStore,
    S3RawPayloadStoreConfig,
)


class _S3Error(Exception):
    def __init__(self, code: str, status: int, message: str = ""):
        super().__init__(message)
        self.response = {
            "Error": {"Code": code},
            "ResponseMetadata": {"HTTPStatusCode": status},
        }


class _ReadErrorBody:
    def __init__(self, error: Exception):
        self._error = error

    def read(self, _size: int) -> bytes:
        raise self._error

    def close(self) -> None:
        return None


class _CloseFailingBody:
    """Readable stream whose cleanup failure must not hide verified bytes."""

    def __init__(self, payload: bytes):
        self._stream = io.BytesIO(payload)

    def read(self, size: int) -> bytes:
        return self._stream.read(size)

    def close(self) -> None:
        raise RuntimeError("simulated transport cleanup failure")


class _FakeS3Client:
    def __init__(self):
        self.objects = {}
        self.put_calls = []
        self.head_calls = []
        self.get_calls = []
        self.next_put_error = None
        self.next_head_error = None
        self.next_get_error = None
        self.next_get_body = None

    def put_object(self, **kwargs):
        self.put_calls.append(kwargs)
        if self.next_put_error is not None:
            error = self.next_put_error
            self.next_put_error = None
            raise error
        identity = (kwargs["Bucket"], kwargs["Key"])
        if identity in self.objects and kwargs.get("IfNoneMatch") == "*":
            raise _S3Error("PreconditionFailed", 412)
        object_metadata = {
            "Body": kwargs["Body"],
            "ContentLength": kwargs["ContentLength"],
            "ContentType": kwargs["ContentType"],
            "Metadata": dict(kwargs["Metadata"]),
        }
        if "ServerSideEncryption" in kwargs:
            object_metadata["ServerSideEncryption"] = kwargs["ServerSideEncryption"]
        self.objects[identity] = object_metadata
        return {"ETag": "safe-test-etag"}

    def head_object(self, **kwargs):
        self.head_calls.append(kwargs)
        if self.next_head_error is not None:
            error = self.next_head_error
            self.next_head_error = None
            raise error
        try:
            stored = self.objects[(kwargs["Bucket"], kwargs["Key"])]
            return {key: value for key, value in stored.items() if key != "Body"}
        except KeyError as error:
            raise _S3Error("NoSuchKey", 404, str(error)) from error

    def get_object(self, **kwargs):
        self.get_calls.append(kwargs)
        if self.next_get_error is not None:
            error = self.next_get_error
            self.next_get_error = None
            raise error
        try:
            stored = self.objects[(kwargs["Bucket"], kwargs["Key"])]
            if self.next_get_body is not None:
                body = self.next_get_body
                self.next_get_body = None
                return {"Body": body}
            return {"Body": io.BytesIO(stored["Body"])}
        except KeyError as error:
            raise _S3Error("NoSuchKey", 404, str(error)) from error


class S3RawPayloadStoreTests(unittest.TestCase):
    def setUp(self):
        self.received_at = datetime(2026, 9, 4, 18, tzinfo=timezone.utc)
        self.metadata = RawPayloadMetadata(
            provider="the_odds_api",
            provider_record_id="receipt-20260904-001",
            source_type="odds",
            captured_at=self.received_at - timedelta(seconds=10),
            received_at=self.received_at,
            schema_version="v4",
            license_scope="internal_analytics_only",
            license_version="terms-2026-08-31",
        )

    def _aws_config(self, **changes):
        settings = {
            "backend": "aws_s3",
            "bucket": "sam-evidence-private",
            "prefix": "raw/odds",
            "region": "us-west-2",
            "max_payload_bytes": 1024,
        }
        settings.update(changes)
        return S3RawPayloadStoreConfig(**settings)

    def _r2_config(self, **changes):
        settings = {
            "backend": "cloudflare_r2",
            "bucket": "sam-evidence-private",
            "prefix": "raw/odds",
            "region": "auto",
            "endpoint_url": "https://0123456789abcdef0123456789abcdef.r2.cloudflarestorage.com",
            "max_payload_bytes": 1024,
        }
        settings.update(changes)
        return S3RawPayloadStoreConfig(**settings)

    def test_aws_write_uses_exact_content_address_and_private_metadata(self):
        client = _FakeS3Client()
        store = S3CompatibleRawPayloadStore(self._aws_config(), client=client)
        payload = b'{"opaque":"provider-data"}'

        receipt = store.store(payload, metadata=self.metadata)

        digest = hashlib.sha256(payload).hexdigest()
        self.assertEqual(receipt.payload_uri, f"s3://sam-evidence-private/raw/odds/sha256/{digest}")
        self.assertEqual(receipt.payload_sha256, digest)
        self.assertEqual(receipt.byte_count, len(payload))
        self.assertNotIn("provider-data", repr(receipt))
        self.assertEqual(len(client.put_calls), 1)
        put = client.put_calls[0]
        self.assertEqual(put["Bucket"], "sam-evidence-private")
        self.assertEqual(put["Key"], f"raw/odds/sha256/{digest}")
        self.assertEqual(put["IfNoneMatch"], "*")
        self.assertEqual(put["CacheControl"], "private, no-store")
        self.assertEqual(put["ServerSideEncryption"], "AES256")
        self.assertEqual(
            put["ContentMD5"],
            base64.b64encode(hashlib.md5(payload, usedforsecurity=False).digest()).decode("ascii"),
        )
        self.assertNotIn("ACL", put)
        self.assertNotIn("WebsiteRedirectLocation", put)
        self.assertEqual(put["ContentType"], "application/octet-stream")
        self.assertEqual(
            put["Metadata"],
            {
                "sam-evidence-format": "raw-provider-payload-v1",
                "sam-sha256": digest,
                "sam-byte-count": str(len(payload)),
            },
        )
        self.assertNotIn("sam-provider", put["Metadata"])
        self.assertNotIn("sam-provider-record-id", put["Metadata"])
        self.assertNotIn("sam-license-version", put["Metadata"])

    def test_identical_payload_with_distinct_receipt_metadata_is_verified_not_overwritten(self):
        client = _FakeS3Client()
        store = S3CompatibleRawPayloadStore(self._aws_config(), client=client)
        payload = b"[]"
        later_metadata = replace(
            self.metadata,
            provider_record_id="receipt-20260904-002",
            received_at=self.received_at + timedelta(minutes=1),
            license_version="terms-2026-09-01",
            content_type="application/json; charset=utf-8",
        )

        first = store.store(payload, metadata=self.metadata)
        second = store.store(payload, metadata=later_metadata)

        self.assertEqual(first.payload_uri, second.payload_uri)
        self.assertEqual(first.payload_sha256, second.payload_sha256)
        self.assertNotEqual(first.metadata, second.metadata)
        self.assertEqual(second.metadata, later_metadata)
        self.assertEqual(len(client.objects), 1)
        self.assertEqual(len(client.put_calls), 2)
        self.assertGreaterEqual(len(client.head_calls), 2)
        self.assertGreaterEqual(len(client.get_calls), 2)

    def test_existing_key_with_mismatched_intrinsic_metadata_fails_closed(self):
        client = _FakeS3Client()
        store = S3CompatibleRawPayloadStore(self._aws_config(), client=client)
        payload = b"[]"
        store.store(payload, metadata=self.metadata)
        identity = next(iter(client.objects))
        client.objects[identity]["Metadata"]["sam-byte-count"] = "999"

        with self.assertRaisesRegex(RawPayloadStoreViolation, "verification failed"):
            store.store(payload, metadata=self.metadata)

    def test_forged_preexisting_same_length_object_with_spoofed_metadata_is_rejected(self):
        client = _FakeS3Client()
        store = S3CompatibleRawPayloadStore(self._aws_config(), client=client)
        payload = b"[]"
        store.store(payload, metadata=self.metadata)
        identity = next(iter(client.objects))
        # This preserves the length and all expected S3 metadata, so only the
        # bounded GetObject SHA-256 verification can catch the forged body.
        client.objects[identity]["Body"] = b"{}"

        with self.assertRaisesRegex(RawPayloadStoreViolation, "verification failed"):
            store.store(payload, metadata=self.metadata)

    def test_transport_error_is_redacted_and_has_no_exception_chain(self):
        client = _FakeS3Client()
        provider_secret = "do-not-log-provider-secret"
        client.next_put_error = _S3Error("AccessDenied", 403, provider_secret)
        store = S3CompatibleRawPayloadStore(self._aws_config(), client=client)

        with self.assertRaises(RawPayloadStoreUnavailable) as caught:
            store.store(b"[]", metadata=self.metadata)

        self.assertEqual(str(caught.exception), "private object-store write failed")
        self.assertNotIn(provider_secret, str(caught.exception))
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)

    def test_verification_error_is_redacted_and_has_no_exception_chain(self):
        client = _FakeS3Client()
        provider_secret = "never-log-head-object-details"
        client.next_head_error = _S3Error("AccessDenied", 403, provider_secret)
        store = S3CompatibleRawPayloadStore(self._aws_config(), client=client)

        with self.assertRaises(RawPayloadStoreUnavailable) as caught:
            store.store(b"[]", metadata=self.metadata)

        self.assertEqual(str(caught.exception), "private object-store verification failed")
        self.assertNotIn(provider_secret, str(caught.exception))
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)

    def test_get_object_error_is_redacted_and_has_no_exception_chain(self):
        client = _FakeS3Client()
        provider_secret = "never-log-get-object-details"
        client.next_get_error = _S3Error("AccessDenied", 403, provider_secret)
        store = S3CompatibleRawPayloadStore(self._aws_config(), client=client)

        with self.assertRaises(RawPayloadStoreUnavailable) as caught:
            store.store(b"[]", metadata=self.metadata)

        self.assertEqual(str(caught.exception), "private object-store verification failed")
        self.assertNotIn(provider_secret, str(caught.exception))
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)

    def test_get_object_body_error_is_redacted_and_has_no_exception_chain(self):
        client = _FakeS3Client()
        provider_secret = "never-log-get-object-body-details"
        client.next_get_body = _ReadErrorBody(RuntimeError(provider_secret))
        store = S3CompatibleRawPayloadStore(self._aws_config(), client=client)

        with self.assertRaises(RawPayloadStoreUnavailable) as caught:
            store.store(b"[]", metadata=self.metadata)

        self.assertEqual(str(caught.exception), "private object-store verification failed")
        self.assertNotIn(provider_secret, str(caught.exception))
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)

    def test_close_error_does_not_hide_already_verified_object_bytes(self):
        client = _FakeS3Client()
        client.next_get_body = _CloseFailingBody(b"[]")
        store = S3CompatibleRawPayloadStore(self._aws_config(), client=client)

        receipt = store.store(b"[]", metadata=self.metadata)

        self.assertEqual(receipt.byte_count, 2)

    def test_r2_uses_approved_endpoint_and_never_sends_unsupported_sse_header(self):
        client = _FakeS3Client()
        store = S3CompatibleRawPayloadStore(self._r2_config(), client=client)

        receipt = store.store(b"[]", metadata=self.metadata)

        self.assertTrue(receipt.payload_uri.startswith("s3://sam-evidence-private/raw/odds/sha256/"))
        self.assertNotIn("ServerSideEncryption", client.put_calls[0])

    def test_rejects_unsafe_bucket_prefix_endpoint_and_size_configuration(self):
        invalid_configs = (
            {"bucket": "SAM_EVIDENCE"},
            {"bucket": "192.168.0.1"},
            {"prefix": "raw/../odds"},
            {"prefix": "raw/sha256"},
            {"endpoint_url": "http://s3.us-west-2.amazonaws.com"},
            {"endpoint_url": "https://credential@example.invalid"},
            {"endpoint_url": "https://127.0.0.1"},
            {"max_payload_bytes": 0},
        )
        for changes in invalid_configs:
            with self.subTest(changes=changes):
                with self.assertRaises(RawPayloadStoreConfigurationError):
                    self._aws_config(**changes)

        with self.assertRaises(RawPayloadStoreConfigurationError):
            self._r2_config(
                endpoint_url="https://0123456789abcdef0123456789abcdef.r2.cloudflarestorage.com?token=no"
            )
        with self.assertRaises(RawPayloadStoreConfigurationError):
            self._r2_config(
                endpoint_url="https://0123456789abcdef0123456789abcdef.invalid.example"
            )

    def test_environment_uses_same_canonical_uri_as_worker_admission_guard(self):
        client = _FakeS3Client()
        captured = {}

        def factory(config, access_key_id, secret_access_key):
            captured["config"] = config
            captured["access_key_id"] = access_key_id
            captured["secret_access_key"] = secret_access_key
            return client

        store = S3CompatibleRawPayloadStore.from_environment(
            {
                "SAM_RAW_EVIDENCE_STORE_BACKEND": "cloudflare_r2",
                "SAM_RAW_EVIDENCE_STORE_URI": "s3://sam-evidence-private/raw/odds",
                "SAM_RAW_EVIDENCE_S3_REGION": "auto",
                "SAM_RAW_EVIDENCE_S3_ENDPOINT_URL": (
                    "https://0123456789abcdef0123456789abcdef.us.r2.cloudflarestorage.com"
                ),
                "SAM_RAW_EVIDENCE_S3_ACCESS_KEY_ID": "r2-access-key-id",
                "SAM_RAW_EVIDENCE_S3_SECRET_ACCESS_KEY": "r2-secret-key",
                "SAM_RAW_EVIDENCE_MAX_BYTES": "1024",
            },
            client_factory=factory,
        )

        self.assertEqual(store.config.bucket, "sam-evidence-private")
        self.assertEqual(store.config.prefix, "raw/odds")
        self.assertEqual(captured["access_key_id"], "r2-access-key-id")
        self.assertEqual(captured["secret_access_key"], "r2-secret-key")

    def test_environment_fails_without_r2_credentials_or_with_signed_store_uri(self):
        values = {
            "SAM_RAW_EVIDENCE_STORE_BACKEND": "cloudflare_r2",
            "SAM_RAW_EVIDENCE_STORE_URI": "s3://sam-evidence-private/raw/odds",
            "SAM_RAW_EVIDENCE_S3_REGION": "auto",
            "SAM_RAW_EVIDENCE_S3_ENDPOINT_URL": (
                "https://0123456789abcdef0123456789abcdef.r2.cloudflarestorage.com"
            ),
        }
        with self.assertRaisesRegex(RawPayloadStoreConfigurationError, "credentials are required"):
            S3CompatibleRawPayloadStore.from_environment(values, client_factory=lambda *_: _FakeS3Client())

        values["SAM_RAW_EVIDENCE_S3_ACCESS_KEY_ID"] = "r2-id"
        values["SAM_RAW_EVIDENCE_S3_SECRET_ACCESS_KEY"] = "r2-secret"
        values["SAM_RAW_EVIDENCE_STORE_URI"] = "s3://sam-evidence-private/raw/odds?signature=no"
        with self.assertRaises(RawPayloadStoreConfigurationError):
            S3CompatibleRawPayloadStore.from_environment(values, client_factory=lambda *_: _FakeS3Client())

    def test_environment_requires_scoped_aws_credentials_without_ambient_fallback(self):
        values = {
            "SAM_RAW_EVIDENCE_STORE_BACKEND": "aws_s3",
            "SAM_RAW_EVIDENCE_STORE_URI": "s3://sam-evidence-private/raw/odds",
            "SAM_RAW_EVIDENCE_S3_REGION": "us-west-2",
        }
        with self.assertRaisesRegex(RawPayloadStoreConfigurationError, "AWS object-store credentials"):
            S3CompatibleRawPayloadStore.from_environment(values, client_factory=lambda *_: _FakeS3Client())

        values["SAM_RAW_EVIDENCE_S3_ACCESS_KEY_ID"] = "aws-scoped-access-key"
        values["SAM_RAW_EVIDENCE_S3_SECRET_ACCESS_KEY"] = "aws-scoped-secret-key"
        store = S3CompatibleRawPayloadStore.from_environment(
            values, client_factory=lambda *_: _FakeS3Client()
        )
        self.assertEqual(store.config.backend, "aws_s3")

    def test_client_factory_failure_is_redacted_without_a_secret_exception_context(self):
        factory_secret = "never-include-this-in-an-exception"
        values = {
            "SAM_RAW_EVIDENCE_STORE_BACKEND": "aws_s3",
            "SAM_RAW_EVIDENCE_STORE_URI": "s3://sam-evidence-private/raw/odds",
            "SAM_RAW_EVIDENCE_S3_REGION": "us-west-2",
            "SAM_RAW_EVIDENCE_S3_ACCESS_KEY_ID": "aws-scoped-access-key",
            "SAM_RAW_EVIDENCE_S3_SECRET_ACCESS_KEY": "aws-scoped-secret-key",
        }

        def leaking_factory(*_):
            raise RuntimeError(factory_secret)

        with self.assertRaises(RawPayloadStoreConfigurationError) as caught:
            S3CompatibleRawPayloadStore.from_environment(values, client_factory=leaking_factory)

        self.assertEqual(str(caught.exception), "S3-compatible client could not be configured")
        self.assertNotIn(factory_secret, str(caught.exception))
        self.assertIsNone(caught.exception.__context__)

    def test_store_enforces_the_configured_payload_limit_before_sdk_io(self):
        client = _FakeS3Client()
        store = S3CompatibleRawPayloadStore(self._aws_config(max_payload_bytes=2), client=client)

        with self.assertRaisesRegex(RawPayloadStoreViolation, "byte limit"):
            store.store(b"123", metadata=self.metadata)
        self.assertEqual(client.put_calls, [])


if __name__ == "__main__":
    unittest.main()
