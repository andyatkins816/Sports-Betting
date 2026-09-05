import hashlib
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from sam_analytics.raw_payload_store import (
    InMemoryRawPayloadStore,
    RawPayloadMetadata,
    RawPayloadStoreViolation,
    validate_private_payload_uri,
)


class RawPayloadStoreTests(unittest.TestCase):
    def setUp(self):
        self.received_at = datetime(2026, 9, 4, 18, tzinfo=timezone.utc)
        self.metadata = RawPayloadMetadata(
            provider="the_odds_api",
            provider_record_id="request-20260904-001",
            source_type="odds",
            captured_at=self.received_at - timedelta(seconds=15),
            received_at=self.received_at,
            schema_version="v4",
            license_scope="internal_analytics_only",
            license_version="terms-2026-08-31",
        )

    def test_store_is_deterministic_content_addressed_and_does_not_expose_body(self):
        store = InMemoryRawPayloadStore(namespace="sam-private-test")
        payload = b'{"opaque":"provider-response-not-for-public-output"}'
        receipt = store.store(payload, metadata=self.metadata)
        duplicate = store.store(payload, metadata=self.metadata)

        digest = hashlib.sha256(payload).hexdigest()
        self.assertEqual(receipt.payload_sha256, digest)
        self.assertEqual(receipt.payload_uri, f"memory://sam-private-test/sha256/{digest}")
        self.assertEqual(receipt.byte_count, len(payload))
        self.assertEqual(receipt, duplicate)
        self.assertEqual(store.stored_count, 1)
        self.assertEqual(store.read_bytes(receipt.payload_uri), payload)
        self.assertNotIn("provider-response-not-for-public-output", repr(receipt))

    def test_identical_bytes_can_have_distinct_provenance_metadata_without_duplication(self):
        store = InMemoryRawPayloadStore()
        payload = b"[]"
        first = store.store(payload, metadata=self.metadata)
        corrected_metadata = replace(
            self.metadata,
            provider_record_id="request-20260904-002",
            received_at=self.received_at + timedelta(minutes=1),
        )
        second = store.store(payload, metadata=corrected_metadata)

        self.assertEqual(first.payload_uri, second.payload_uri)
        self.assertEqual(first.payload_sha256, second.payload_sha256)
        self.assertNotEqual(first.metadata.provider_record_id, second.metadata.provider_record_id)
        self.assertEqual(store.stored_count, 1)

    def test_rejects_public_or_unstable_or_non_content_addressed_uris(self):
        digest = hashlib.sha256(b"payload").hexdigest()
        unsafe_uris = (
            f"https://bucket.example/private/sha256/{digest}",
            f"s3://sam-raw/private/sha256/{digest}?signature=not-allowed",
            f"s3://sam-raw/private/sha256/{digest}#fragment",
            "file:///tmp/raw-payload",
            f"s3://sam-raw/private/sha256/{digest.upper()}",
            f"s3://sam-raw/private/../sha256/{digest}",
        )
        for payload_uri in unsafe_uris:
            with self.subTest(payload_uri=payload_uri):
                with self.assertRaises(RawPayloadStoreViolation):
                    validate_private_payload_uri(payload_uri, payload_sha256=digest)

        with self.assertRaises(RawPayloadStoreViolation):
            validate_private_payload_uri(
                f"s3://sam-raw/private/sha256/{digest}",
                payload_sha256=hashlib.sha256(b"different").hexdigest(),
            )

    def test_rejects_unsafe_metadata_and_invalid_write_times_or_payloads(self):
        store = InMemoryRawPayloadStore(max_payload_bytes=3)
        unsafe = replace(self.metadata, license_scope="not\na-license")
        with self.assertRaises(RawPayloadStoreViolation):
            store.store(b"[]", metadata=unsafe)
        implausible_clock = replace(self.metadata, captured_at=self.received_at + timedelta(minutes=6))
        with self.assertRaises(RawPayloadStoreViolation):
            store.store(b"[]", metadata=implausible_clock)
        with self.assertRaises(RawPayloadStoreViolation):
            store.store(b"", metadata=self.metadata)
        with self.assertRaises(RawPayloadStoreViolation):
            store.store(bytearray(b"[]"), metadata=self.metadata)  # type: ignore[arg-type]
        with self.assertRaises(RawPayloadStoreViolation):
            store.store(b"1234", metadata=self.metadata)
        with self.assertRaises(RawPayloadStoreViolation):
            store.store(b"[]", metadata=self.metadata, stored_at=self.received_at - timedelta(seconds=1))

    def test_unknown_private_test_object_cannot_be_read(self):
        store = InMemoryRawPayloadStore()
        digest = hashlib.sha256(b"missing").hexdigest()
        with self.assertRaises(RawPayloadStoreViolation):
            store.read_bytes(f"memory://sam-private-test/sha256/{digest}")


if __name__ == "__main__":
    unittest.main()
