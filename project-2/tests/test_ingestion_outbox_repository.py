"""Focused unit tests for the unwired PostgreSQL outbox adapter."""

from __future__ import annotations

import unittest
from contextlib import AbstractContextManager
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sam_analytics.ingestion_dispatch import (
    DeadLetterReason,
    DispatchPolicy,
    RetryDisposition,
    RetryPlan,
    retry_schedule_sha256,
)
from sam_analytics.ingestion_outbox_repository import (
    IngestionOutboxRepositoryUnavailable,
    PostgresIngestionOutboxRepository,
)
from sam_analytics.ingestion_outbox_runtime import (
    AttemptClaimDisposition,
    AttemptCompletion,
    AttemptCompletionCommit,
    AttemptCompletionOutcome,
    AttemptRetrySafety,
    ClaimedDispatchAttempt,
    IngestionOutboxConfigurationError,
    OutboxMessage,
    OutboxPublicationClaim,
    PublicationCommit,
)
from sam_analytics.ingestion_runs import IngestionFailureCode
from sam_analytics.provider_contracts import ProviderUse


class _FakeTransaction(AbstractContextManager[None]):
    def __init__(self, database: _FakeDatabase) -> None:
        self.database = database

    def __enter__(self) -> None:
        self.database.events.append("transaction_enter")
        return None

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.database.events.append(
            "transaction_commit" if exc_type is None else "transaction_rollback"
        )
        if self.database.fail_commit and exc_type is None:
            raise RuntimeError("postgresql://sam:secret@private.invalid/sam")
        return False


class _FakeCursor:
    def __init__(self, database: _FakeDatabase) -> None:
        self.database = database

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return False

    def execute(self, query, params=None) -> None:
        self.database.executions.append((" ".join(query.split()), params))

    def fetchone(self):
        return self.database.rows.pop(0) if self.database.rows else None


class _FakeConnection:
    def __init__(self, database: _FakeDatabase) -> None:
        self.database = database

    def transaction(self) -> _FakeTransaction:
        return _FakeTransaction(self.database)

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self.database)

    def close(self) -> None:
        self.database.events.append("close")


class _FakeDatabase:
    def __init__(self) -> None:
        self.rows = []
        self.executions = []
        self.events = []
        self.urls = []
        self.fail_commit = False

    def connect(self, database_url: str) -> _FakeConnection:
        self.urls.append(database_url)
        return _FakeConnection(self)


class PostgresIngestionOutboxRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 9, 6, 21, tzinfo=UTC)
        self.database = _FakeDatabase()
        self.repository = PostgresIngestionOutboxRepository(
            "postgresql://sam:never-print@private.invalid/sam",
            connection_factory=self.database.connect,
        )
        self.message = OutboxMessage(dispatch_id=uuid4(), attempt_number=1)
        self.provider_use = ProviderUse(
            provider="the_odds_api",
            source_type="odds",
            license_scope="internal_analytics_only",
            license_version="terms-2026-09-06",
            exposure="private_raw",
        )
        self.policy = DispatchPolicy(
            provider=self.provider_use.provider,
            policy_version="admission-v1",
            enabled=True,
            allowed_source_types=frozenset({self.provider_use.source_type}),
            max_attempts=3,
            quota_floor=2,
        )

    def _started_attempt(self) -> ClaimedDispatchAttempt:
        return ClaimedDispatchAttempt(
            attempt_claim_id=uuid4(),
            message=self.message,
            worker_identity="private-worker-1",
            lease_token=uuid4(),
            running_transition_id=uuid4(),
            provider_use_authorization_id=uuid4(),
            quota_receipt_id=uuid4(),
            provider_use=self.provider_use,
            request_fingerprint_sha256="a" * 64,
            estimated_cost=1,
            policy_version=self.policy.policy_version,
            max_attempts=self.policy.max_attempts,
            min_request_interval=self.policy.min_request_interval,
            quota_floor=self.policy.quota_floor,
            quota_max_age=self.policy.quota_max_age,
            retry_schedule_sha256=retry_schedule_sha256(self.policy),
            window_start=self.now - timedelta(hours=1),
            window_end=self.now + timedelta(hours=1),
            authorization_effective_until=self.now + timedelta(days=1),
            claimed_at=self.now,
            lease_expires_at=self.now + timedelta(minutes=5),
        )

    def test_valid_publication_claim_commits_after_decode(self) -> None:
        claim_id = uuid4()
        outbox_id = uuid4()
        lease_token = uuid4()
        self.database.rows = [
            (
                "publishable",
                claim_id,
                outbox_id,
                self.message.dispatch_id,
                1,
                self.now,
                self.now + timedelta(minutes=2),
            )
        ]

        claim = self.repository.claim_ingestion_outbox_publication(
            publisher_identity="publisher-1",
            lease_token=lease_token,
        )

        self.assertEqual(claim.publication_claim_id, claim_id)
        self.assertEqual(claim.message, self.message)
        self.assertEqual(
            self.database.events,
            ["transaction_enter", "transaction_commit", "close"],
        )
        self.assertIn("claim_ingestion_outbox_publication", self.database.executions[0][0])

    def test_no_publishable_row_returns_none(self) -> None:
        result = self.repository.claim_ingestion_outbox_publication(
            publisher_identity="publisher-1",
            lease_token=uuid4(),
        )
        self.assertIsNone(result)

    def test_secret_shaped_identity_is_rejected_before_database_access(self) -> None:
        with self.assertRaises(IngestionOutboxConfigurationError):
            self.repository.claim_ingestion_outbox_publication(
                publisher_identity="Bearer-secret",
                lease_token=uuid4(),
            )
        with self.assertRaises(IngestionOutboxConfigurationError):
            self.repository.claim_ingestion_dispatch_attempt(
                self.message,
                worker_identity="api-key-worker",
                lease_token=uuid4(),
            )

        self.assertEqual(self.database.urls, [])

    def test_publication_record_maps_exact_idempotent_disposition(self) -> None:
        claim_id = uuid4()
        claim = self._publication_claim(claim_id)
        for stored, expected in (
            ("recorded", PublicationCommit.RECORDED),
            ("already_recorded", PublicationCommit.ALREADY_RECORDED),
        ):
            with self.subTest(stored=stored):
                self.database.rows = [(stored, uuid4())]
                result = self.repository.record_ingestion_outbox_publication(claim)
                self.assertEqual(result, expected)

    def _publication_claim(self, claim_id):
        return OutboxPublicationClaim(
            publication_claim_id=claim_id,
            message=self.message,
            publisher_identity="publisher-1",
            lease_token=uuid4(),
            claimed_at=self.now,
            lease_expires_at=self.now + timedelta(minutes=2),
        )

    def test_started_attempt_decodes_exact_authorization_and_permission(self) -> None:
        attempt = self._started_attempt()
        self.database.rows = [
            (
                "started",
                True,
                attempt.attempt_claim_id,
                attempt.running_transition_id,
                attempt.provider_use_authorization_id,
                attempt.quota_receipt_id,
                self.provider_use.provider,
                self.provider_use.source_type,
                attempt.request_fingerprint_sha256,
                attempt.window_start,
                attempt.window_end,
                1,
                attempt.policy_version,
                attempt.max_attempts,
                self.provider_use.license_scope,
                self.provider_use.license_version,
                self.provider_use.exposure,
                attempt.claimed_at,
                attempt.lease_expires_at,
                attempt.min_request_interval,
                attempt.quota_floor,
                attempt.quota_max_age,
                attempt.retry_schedule_sha256,
                attempt.authorization_effective_until,
            )
        ]

        result = self.repository.claim_ingestion_dispatch_attempt(
            self.message,
            worker_identity=attempt.worker_identity,
            lease_token=attempt.lease_token,
        )

        self.assertEqual(result.disposition, AttemptClaimDisposition.STARTED)
        self.assertEqual(result.started, attempt)
        self.assertIn("claim_ingestion_dispatch_attempt", self.database.executions[0][0])

    def test_malformed_runtime_policy_or_authorization_claim_fails_closed(self) -> None:
        attempt = self._started_attempt()
        valid_row = [
            "started",
            True,
            attempt.attempt_claim_id,
            attempt.running_transition_id,
            attempt.provider_use_authorization_id,
            attempt.quota_receipt_id,
            self.provider_use.provider,
            self.provider_use.source_type,
            attempt.request_fingerprint_sha256,
            attempt.window_start,
            attempt.window_end,
            attempt.estimated_cost,
            attempt.policy_version,
            attempt.max_attempts,
            self.provider_use.license_scope,
            self.provider_use.license_version,
            self.provider_use.exposure,
            attempt.claimed_at,
            attempt.lease_expires_at,
            attempt.min_request_interval,
            attempt.quota_floor,
            attempt.quota_max_age,
            attempt.retry_schedule_sha256,
            attempt.authorization_effective_until,
        ]
        invalid_values = (
            (19, "30 seconds"),
            (20, True),
            (21, timedelta(0)),
            (22, "not-a-digest"),
            (23, attempt.lease_expires_at - timedelta(seconds=1)),
        )
        for index, invalid in invalid_values:
            with self.subTest(index=index, invalid=invalid):
                self.database.events.clear()
                row = valid_row.copy()
                row[index] = invalid
                self.database.rows = [tuple(row)]
                with self.assertRaises(IngestionOutboxRepositoryUnavailable):
                    self.repository.claim_ingestion_dispatch_attempt(
                        self.message,
                        worker_identity=attempt.worker_identity,
                        lease_token=attempt.lease_token,
                    )
                self.assertEqual(
                    self.database.events,
                    ["transaction_enter", "transaction_rollback", "close"],
                )

    def test_nonstarted_attempt_never_returns_provider_permission(self) -> None:
        self.database.rows = [
            ("inconclusive", False) + (None,) * 22,
        ]

        result = self.repository.claim_ingestion_dispatch_attempt(
            self.message,
            worker_identity="private-worker-1",
            lease_token=uuid4(),
        )

        self.assertEqual(result.disposition, AttemptClaimDisposition.INCONCLUSIVE)
        self.assertIsNone(result.started)

    def test_database_commit_failure_is_sanitized(self) -> None:
        self.database.fail_commit = True
        self.database.rows = [(self.now,)]
        attempt = self._started_attempt()

        with self.assertRaises(IngestionOutboxRepositoryUnavailable) as caught:
            self.repository.read_ingestion_dispatch_attempt_time(attempt)

        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)
        self.assertNotIn("secret", str(caught.exception).lower())
        self.assertNotIn("never-print", repr(self.repository))

    def test_retry_completion_maps_runtime_names_to_database_state(self) -> None:
        attempt = self._started_attempt()
        retry_at = self.now + timedelta(minutes=2)
        completion = AttemptCompletion(
            attempt_claim_id=attempt.attempt_claim_id,
            message=attempt.message,
            provider_use_authorization_id=attempt.provider_use_authorization_id,
            outcome=AttemptCompletionOutcome.RETRY_WAIT,
            retry_plan=RetryPlan(
                disposition=RetryDisposition.RETRY,
                failure_code=IngestionFailureCode.PROVIDER_RATE_LIMITED,
                completed_attempts=1,
                next_attempt_at=retry_at,
            ),
            retry_safety=AttemptRetrySafety.REQUEST_NOT_SENT,
        )
        completion_id = uuid4()
        self.database.rows = [("committed", completion_id)]

        result = self.repository.complete_ingestion_dispatch_attempt(
            attempt,
            completion,
        )

        self.assertEqual(result, AttemptCompletionCommit.COMMITTED)
        params = self.database.executions[0][1]
        self.assertEqual(params[3:9], (
            "retry_wait",
            IngestionFailureCode.PROVIDER_RATE_LIMITED.value,
            None,
            retry_at,
            None,
            AttemptRetrySafety.REQUEST_NOT_SENT.value,
        ))

    def test_success_and_dead_letter_parameters_are_exact(self) -> None:
        attempt = self._started_attempt()
        success = AttemptCompletion(
            attempt_claim_id=attempt.attempt_claim_id,
            message=attempt.message,
            provider_use_authorization_id=attempt.provider_use_authorization_id,
            outcome=AttemptCompletionOutcome.SUCCEEDED,
            provider_payload_receipt_id=uuid4(),
        )
        dead_letter = AttemptCompletion(
            attempt_claim_id=attempt.attempt_claim_id,
            message=attempt.message,
            provider_use_authorization_id=attempt.provider_use_authorization_id,
            outcome=AttemptCompletionOutcome.DEAD_LETTERED,
            retry_plan=RetryPlan(
                disposition=RetryDisposition.DEAD_LETTER,
                failure_code=IngestionFailureCode.PROVIDER_RESPONSE_INVALID,
                completed_attempts=1,
                dead_letter_reason=DeadLetterReason.NON_RETRYABLE,
            ),
        )
        for completion, expected in (
            (
                success,
                (
                    "succeeded",
                    None,
                    None,
                    None,
                    success.provider_payload_receipt_id,
                    None,
                ),
            ),
            (
                dead_letter,
                (
                    "dead_lettered",
                    IngestionFailureCode.PROVIDER_RESPONSE_INVALID.value,
                    DeadLetterReason.NON_RETRYABLE.value,
                    None,
                    None,
                    None,
                ),
            ),
        ):
            with self.subTest(outcome=completion.outcome):
                self.database.rows = [("already_committed", uuid4())]
                result = self.repository.complete_ingestion_dispatch_attempt(
                    attempt,
                    completion,
                )
                self.assertEqual(result, AttemptCompletionCommit.ALREADY_COMMITTED)
                params = self.database.executions[-1][1]
                self.assertEqual(params[3:9], expected)

    def test_invalid_started_permission_fails_closed(self) -> None:
        attempt = self._started_attempt()
        self.database.rows = [
            (
                "started",
                False,
                attempt.attempt_claim_id,
                attempt.running_transition_id,
                attempt.provider_use_authorization_id,
                attempt.quota_receipt_id,
                self.provider_use.provider,
                self.provider_use.source_type,
                attempt.request_fingerprint_sha256,
                attempt.window_start,
                attempt.window_end,
                1,
                attempt.policy_version,
                attempt.max_attempts,
                self.provider_use.license_scope,
                self.provider_use.license_version,
                self.provider_use.exposure,
                attempt.claimed_at,
                attempt.lease_expires_at,
                attempt.min_request_interval,
                attempt.quota_floor,
                attempt.quota_max_age,
                attempt.retry_schedule_sha256,
                attempt.authorization_effective_until,
            )
        ]
        with self.assertRaises(IngestionOutboxRepositoryUnavailable):
            self.repository.claim_ingestion_dispatch_attempt(
                self.message,
                worker_identity=attempt.worker_identity,
                lease_token=attempt.lease_token,
            )


if __name__ == "__main__":
    unittest.main()
