import json
import hashlib
import hmac
import os
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from urllib.error import HTTPError

from sam_analytics.base44_sync import (
    Base44EvidencePublisher,
    EvidencePublishError,
    EvidenceRecord,
)


class _Response:
    def __init__(self, status_code=201):
        self.status_code = status_code
        self.closed = False

    def getcode(self):
        return self.status_code

    def close(self):
        self.closed = True


class _Opener:
    def __init__(self, response=None, error=None):
        self.response = response or _Response()
        self.error = error
        self.request = None
        self.timeout = None

    def open(self, request, timeout):
        self.request = request
        self.timeout = timeout
        if self.error:
            raise self.error
        return self.response


class Base44SyncTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
        self.publisher = Base44EvidencePublisher(
            webhook_url="https://sam.example.base44.app/api/functions/python-evidence-ingest",
            token="x" * 32,
        )
        self.record = EvidenceRecord(
            evidence_type="calibration",
            severity="info",
            observed_at=self.now,
            source="sam-model-worker",
            provenance_id="calibration:abc123",
            decision_state="hold",
            summary="Calibration report passed the configured gate.",
            sport="NBA",
            model_version="nba-moneyline-v1",
            metrics={"brier_score": 0.19, "ece": 0.02},
            expires_at=self.now + timedelta(hours=24),
        )

    def test_publisher_sends_only_validated_evidence_with_bearer_auth(self):
        opener = _Opener()
        with patch("sam_analytics.base44_sync.build_opener", return_value=opener):
            receipt = self.publisher.publish(self.record)

        self.assertEqual(receipt.status_code, 201)
        self.assertTrue(opener.response.closed)
        self.assertEqual(opener.timeout, 5.0)
        self.assertEqual(opener.request.get_method(), "POST")
        self.assertEqual(opener.request.get_header("Authorization"), "Bearer " + ("x" * 32))
        timestamp = opener.request.get_header("X-sam-evidence-timestamp")
        signature = opener.request.get_header("X-sam-evidence-signature")
        self.assertIsNotNone(timestamp)
        self.assertEqual(
            signature,
            "sha256="
            + hmac.new(
                ("x" * 32).encode("utf-8"),
                timestamp.encode("utf-8") + b"." + opener.request.data,
                hashlib.sha256,
            ).hexdigest(),
        )
        payload = json.loads(opener.request.data.decode("utf-8"))
        self.assertEqual(payload["sport"], "NBA")
        self.assertEqual(payload["metrics"], {"brier_score": 0.19, "ece": 0.02})
        self.assertNotIn("token", payload)

    def test_publisher_rejects_non_https_or_token_bearing_url(self):
        with self.assertRaisesRegex(EvidencePublishError, "HTTPS"):
            Base44EvidencePublisher(webhook_url="http://example.test/ingest", token="x" * 32)
        with self.assertRaisesRegex(EvidencePublishError, "query"):
            Base44EvidencePublisher(
                webhook_url="https://example.test/ingest?token=wrong-place",
                token="x" * 32,
            )

    def test_invalid_evidence_does_not_make_a_network_request(self):
        bad_record = EvidenceRecord(
            evidence_type="calibration",
            severity="info",
            observed_at=self.now,
            source="sam-worker",
            provenance_id="id",
            decision_state="hold",
            summary="A valid enough summary.",
            metrics={"ece": float("nan")},
        )
        with patch("sam_analytics.base44_sync.build_opener") as build_opener:
            with self.assertRaisesRegex(EvidencePublishError, "finite"):
                self.publisher.publish(bad_record)
        build_opener.assert_not_called()

    def test_remote_error_is_sanitized_and_does_not_include_the_token(self):
        error = HTTPError(
            self.publisher.webhook_url,
            401,
            "Unauthorized",
            hdrs=None,
            fp=None,
        )
        opener = _Opener(error=error)
        with patch("sam_analytics.base44_sync.build_opener", return_value=opener):
            with self.assertRaises(EvidencePublishError) as raised:
                self.publisher.publish(self.record)
        self.assertIn("HTTP 401", str(raised.exception))
        self.assertNotIn("x" * 32, str(raised.exception))

    def test_environment_requires_both_values_or_neither(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(Base44EvidencePublisher.from_environment())
        with patch.dict(
            os.environ,
            {"BASE44_EVIDENCE_WEBHOOK_URL": "https://example.test/ingest"},
            clear=True,
        ):
            with self.assertRaisesRegex(EvidencePublishError, "set together"):
                Base44EvidencePublisher.from_environment()

    def test_environment_requires_an_exact_webhook_host_pin(self):
        with patch.dict(
            os.environ,
            {
                "BASE44_EVIDENCE_WEBHOOK_URL": "https://example.test/ingest",
                "BASE44_EVIDENCE_WEBHOOK_TOKEN": "x" * 32,
            },
            clear=True,
        ):
            with self.assertRaisesRegex(EvidencePublishError, "HOST"):
                Base44EvidencePublisher.from_environment()
        with self.assertRaisesRegex(EvidencePublishError, "host does not match"):
            Base44EvidencePublisher(
                webhook_url="https://example.test/ingest",
                token="x" * 32,
                expected_host="different.test",
            )


if __name__ == "__main__":
    unittest.main()
