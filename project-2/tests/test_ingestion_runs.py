"""Tests for credential-free, append-only future ingestion-run audit facts."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from sam_analytics.ingestion_runs import (
    IngestionFailureClass,
    IngestionFailureCode,
    IngestionRunState,
    IngestionRunTransitionError,
    IngestionRunValidationError,
    mark_blocked,
    mark_failed,
    mark_succeeded,
    new_manual_shadow_run,
    start_next_attempt,
)
from sam_analytics.migrate import discover_migrations


class IngestionRunAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 9, 4, 18, tzinfo=timezone.utc)
        self.run, self.queued = new_manual_shadow_run(
            provider="the_odds_api",
            job_identity="manual-shadow-20260904-001",
            source_type="odds",
            max_attempts=2,
            created_at=self.now,
            run_id=uuid4(),
        )

    def test_retryable_failure_can_be_manually_retried_but_non_retryable_cannot(self) -> None:
        running = start_next_attempt(self.run, self.queued, occurred_at=self.now + timedelta(seconds=1))
        retryable_failure = mark_failed(
            self.run,
            running,
            failure_code=IngestionFailureCode.NETWORK_TIMEOUT,
            occurred_at=self.now + timedelta(seconds=2),
        )

        self.assertEqual(retryable_failure.state, IngestionRunState.FAILED)
        self.assertEqual(retryable_failure.failure.classification, IngestionFailureClass.RETRYABLE)
        self.assertTrue(retryable_failure.can_retry)

        second_attempt = start_next_attempt(
            self.run,
            retryable_failure,
            occurred_at=self.now + timedelta(seconds=3),
        )
        self.assertEqual(second_attempt.state, IngestionRunState.RUNNING)
        self.assertEqual(second_attempt.attempt_count, 2)
        self.assertEqual(second_attempt.state_sequence, 4)
        self.assertEqual(
            mark_succeeded(self.run, second_attempt, occurred_at=self.now + timedelta(seconds=4)).state,
            IngestionRunState.SUCCEEDED,
        )

        new_run, queued = new_manual_shadow_run(
            provider="the_odds_api",
            job_identity="manual-shadow-20260904-002",
            source_type="odds",
            created_at=self.now,
        )
        non_retryable_failure = mark_failed(
            new_run,
            start_next_attempt(new_run, queued, occurred_at=self.now + timedelta(seconds=1)),
            failure_code=IngestionFailureCode.LICENSE_NOT_PERMITTED,
            occurred_at=self.now + timedelta(seconds=2),
        )
        self.assertEqual(
            non_retryable_failure.failure.classification,
            IngestionFailureClass.NON_RETRYABLE,
        )
        self.assertFalse(non_retryable_failure.can_retry)
        with self.assertRaisesRegex(IngestionRunTransitionError, "retryable failed"):
            start_next_attempt(
                new_run,
                non_retryable_failure,
                occurred_at=self.now + timedelta(seconds=3),
            )

    def test_blocked_runs_are_non_retryable_and_can_stop_before_an_attempt(self) -> None:
        blocked = mark_blocked(
            self.run,
            self.queued,
            failure_code=IngestionFailureCode.PROVIDER_CONTRACT_UNAPPROVED,
            occurred_at=self.now + timedelta(seconds=1),
        )

        self.assertEqual(blocked.state, IngestionRunState.BLOCKED)
        self.assertEqual(blocked.attempt_count, 0)
        self.assertEqual(blocked.failure.classification, IngestionFailureClass.NON_RETRYABLE)
        with self.assertRaisesRegex(IngestionRunTransitionError, "non-retryable"):
            mark_blocked(
                self.run,
                self.queued,
                failure_code=IngestionFailureCode.NETWORK_TIMEOUT,
                occurred_at=self.now + timedelta(seconds=1),
            )

    def test_attempt_limit_and_state_ownership_fail_closed(self) -> None:
        one_attempt_run, one_attempt_queued = new_manual_shadow_run(
            provider="the_odds_api",
            job_identity="manual-shadow-20260904-003",
            source_type="odds",
            created_at=self.now,
        )
        running = start_next_attempt(
            one_attempt_run,
            one_attempt_queued,
            occurred_at=self.now + timedelta(seconds=1),
        )
        failed = mark_failed(
            one_attempt_run,
            running,
            failure_code=IngestionFailureCode.DATABASE_UNAVAILABLE,
            occurred_at=self.now + timedelta(seconds=2),
        )
        with self.assertRaisesRegex(IngestionRunTransitionError, "attempt limit"):
            start_next_attempt(
                one_attempt_run,
                failed,
                occurred_at=self.now + timedelta(seconds=3),
            )

        other_run, _ = new_manual_shadow_run(
            provider="the_odds_api",
            job_identity="manual-shadow-20260904-004",
            source_type="odds",
            created_at=self.now,
        )
        with self.assertRaisesRegex(IngestionRunTransitionError, "another ingestion run"):
            start_next_attempt(other_run, self.queued, occurred_at=self.now + timedelta(seconds=1))

    def test_identifiers_and_failure_codes_cannot_carry_secret_or_free_text(self) -> None:
        for unsafe_job_identity in (
            "odds-api-key",
            "provider-token",
            "manual shadow",
            "https://example.invalid/job",
        ):
            with self.subTest(unsafe_job_identity=unsafe_job_identity), self.assertRaises(
                IngestionRunValidationError
            ):
                new_manual_shadow_run(
                    provider="the_odds_api",
                    job_identity=unsafe_job_identity,
                    source_type="odds",
                    created_at=self.now,
                )
        with self.assertRaisesRegex(IngestionRunValidationError, "approved safe code"):
            mark_failed(
                self.run,
                start_next_attempt(self.run, self.queued, occurred_at=self.now + timedelta(seconds=1)),
                failure_code="untrusted provider exception text",  # type: ignore[arg-type]
                occurred_at=self.now + timedelta(seconds=2),
            )

    def test_checked_in_migration_has_append_only_sanitized_run_state_contract(self) -> None:
        migrations_dir = Path(__file__).resolve().parents[1] / "migrations"
        migration = next(
            item
            for item in discover_migrations(migrations_dir)
            if item.filename == "005_ingestion_run_audit.sql"
        )

        self.assertIn("CREATE TABLE ingestion_run", migration.sql)
        self.assertIn("CREATE TABLE ingestion_run_state_transition", migration.sql)
        self.assertIn("failure_class IN ('retryable', 'non_retryable')", migration.sql)
        self.assertIn("enforce_ingestion_run_state_transition", migration.sql)
        self.assertIn("ingestion_run_append_only", migration.sql)
        self.assertIn("ingestion_run_state_transition_append_only", migration.sql)
        self.assertNotIn("api_key", migration.sql.lower())
        self.assertNotIn("request_url", migration.sql.lower())
        self.assertNotIn("response_body", migration.sql.lower())


if __name__ == "__main__":
    unittest.main()
