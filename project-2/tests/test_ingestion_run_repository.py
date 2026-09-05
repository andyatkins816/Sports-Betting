"""Unit tests for the credential-safe PostgreSQL ingestion-run adapter."""

from __future__ import annotations

import copy
import unittest
from contextlib import AbstractContextManager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from sam_analytics.ingestion_run_repository import (
    IngestionRunRepositoryConflict,
    IngestionRunRepositoryError,
    IngestionRunRepositoryNotFound,
    IngestionRunRepositoryUnavailable,
    PostgresIngestionRunRepository,
)
from sam_analytics.ingestion_runs import (
    IngestionFailureClass,
    IngestionFailureCode,
    IngestionRunStateTransition,
    IngestionRunTransitionError,
    mark_cancelled,
    mark_failed,
    new_manual_shadow_run,
    start_next_attempt,
)


class _FakeDatabaseError(RuntimeError):
    def __init__(self, message: str, *, sqlstate: str | None = None) -> None:
        super().__init__(message)
        self.sqlstate = sqlstate


class _FakeTransaction(AbstractContextManager[None]):
    def __init__(self, database: "_FakeDatabase") -> None:
        self.database = database
        self.snapshot = None

    def __enter__(self) -> None:
        self.snapshot = copy.deepcopy(self.database.state)
        return None

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_type is not None:
            self.database.state = self.snapshot
        return False


class _FakeCursor:
    def __init__(self, database: "_FakeDatabase") -> None:
        self.database = database
        self.row = None

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return False

    def execute(self, query, params=None) -> None:
        normalized = " ".join(query.split())
        self.database.executions.append((normalized, params))
        self.row = None
        if self.database.fail_prefix and normalized.startswith(self.database.fail_prefix):
            failure = self.database.failure
            self.database.fail_prefix = None
            raise failure

        if normalized.startswith("INSERT INTO ingestion_run ("):
            run_id = params[0]
            if run_id in self.database.state["runs"]:
                raise _FakeDatabaseError("duplicate identity", sqlstate="23505")
            self.database.state["runs"][run_id] = (
                params[1],
                params[2],
                params[3],
                params[4],
                params[5],
                params[6],
            )
            return

        if normalized.startswith("INSERT INTO ingestion_run_state_transition"):
            run_id = params[0]
            row = (
                params[1],
                params[2],
                params[3],
                params[6],
                params[4],
                params[5],
            )
            transitions = self.database.state["transitions"].setdefault(run_id, [])
            if any(existing[0] == row[0] for existing in transitions):
                raise _FakeDatabaseError("duplicate transition", sqlstate="23505")
            transitions.append(row)
            return

        if normalized.startswith("SELECT provider, job_identity"):
            self.row = self.database.state["runs"].get(params[0])
            return

        if normalized.startswith("SELECT state_sequence, state"):
            transitions = self.database.state["transitions"].get(params[0], [])
            self.row = max(transitions, key=lambda item: item[0]) if transitions else None
            return

        raise AssertionError(f"unexpected repository SQL: {normalized}")

    def fetchone(self):
        return self.row


class _FakeConnection:
    def __init__(self, database: "_FakeDatabase") -> None:
        self.database = database

    def transaction(self) -> _FakeTransaction:
        return _FakeTransaction(self.database)

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self.database)

    def close(self) -> None:
        self.database.closed_connections += 1


class _FakeDatabase:
    def __init__(self) -> None:
        self.state = {"runs": {}, "transitions": {}}
        self.executions = []
        self.supplied_urls = []
        self.closed_connections = 0
        self.fail_prefix = None
        self.failure = _FakeDatabaseError("simulated database failure")

    def connect(self, database_url: str) -> _FakeConnection:
        self.supplied_urls.append(database_url)
        return _FakeConnection(self)

    def fail_once(
        self,
        prefix: str,
        *,
        message: str,
        sqlstate: str | None = None,
    ) -> None:
        self.fail_prefix = prefix
        self.failure = _FakeDatabaseError(message, sqlstate=sqlstate)


class IngestionRunRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 9, 4, 18, tzinfo=timezone.utc)
        self.run, self.queued = new_manual_shadow_run(
            provider="sam_synthetic",
            job_identity="staging-probe-20260904-001",
            source_type="storage_probe",
            max_attempts=2,
            created_at=self.now,
            run_id=uuid4(),
        )
        self.database_url = "postgresql://sam:never-show-this@db.example/sam"
        self.database = _FakeDatabase()
        self.repository = PostgresIngestionRunRepository(
            self.database_url,
            connection_factory=self.database.connect,
        )

    def test_create_is_atomic_parameterized_and_latest_round_trips(self) -> None:
        created = self.repository.create_run(self.run, self.queued)
        latest = self.repository.latest_transition(self.run.id)

        self.assertEqual(created, self.queued)
        self.assertEqual(latest, self.queued)
        self.assertEqual(self.database.supplied_urls, [self.database_url, self.database_url])
        self.assertEqual(self.database.closed_connections, 2)
        self.assertEqual(len(self.database.state["runs"]), 1)
        self.assertEqual(len(self.database.state["transitions"][self.run.id]), 1)
        for sql, _ in self.database.executions:
            self.assertNotIn(self.run.job_identity, sql)
            self.assertNotIn(self.database_url, sql)
        self.assertNotIn(self.database_url, repr(self.repository))

    def test_append_round_trips_enumerated_failure_without_free_text(self) -> None:
        self.repository.create_run(self.run, self.queued)
        running = start_next_attempt(
            self.run,
            self.queued,
            occurred_at=self.now + timedelta(seconds=1),
        )
        self.repository.append_transition(self.run, self.queued, running)
        failed = mark_failed(
            self.run,
            running,
            failure_code=IngestionFailureCode.STORAGE_UNAVAILABLE,
            occurred_at=self.now + timedelta(seconds=2),
        )

        appended = self.repository.append_transition(self.run, running, failed)
        latest = self.repository.latest_transition(self.run.id)

        self.assertEqual(appended, failed)
        self.assertEqual(latest, failed)
        self.assertEqual(latest.failure.classification, IngestionFailureClass.RETRYABLE)
        persisted = self.database.state["transitions"][self.run.id][-1]
        self.assertEqual(persisted[-2:], ("retryable", "storage_unavailable"))

    def test_stale_previous_state_is_rejected_without_an_insert(self) -> None:
        self.repository.create_run(self.run, self.queued)
        running = start_next_attempt(
            self.run,
            self.queued,
            occurred_at=self.now + timedelta(seconds=1),
        )
        self.repository.append_transition(self.run, self.queued, running)
        stale_cancellation = mark_cancelled(
            self.run,
            self.queued,
            occurred_at=self.now + timedelta(seconds=2),
        )
        before = len(self.database.state["transitions"][self.run.id])

        with self.assertRaisesRegex(IngestionRunRepositoryConflict, "changed"):
            self.repository.append_transition(
                self.run, self.queued, stale_cancellation
            )

        self.assertEqual(len(self.database.state["transitions"][self.run.id]), before)
        self.assertEqual(self.repository.latest_transition(self.run.id), running)

    def test_stored_identity_mismatch_and_missing_run_fail_closed(self) -> None:
        self.repository.create_run(self.run, self.queued)
        altered_run = replace(self.run, provider="another_provider")
        running = start_next_attempt(
            altered_run,
            self.queued,
            occurred_at=self.now + timedelta(seconds=1),
        )
        with self.assertRaisesRegex(IngestionRunRepositoryConflict, "identity"):
            self.repository.append_transition(altered_run, self.queued, running)

        with self.assertRaises(IngestionRunRepositoryNotFound):
            self.repository.latest_transition(uuid4())

    def test_invalid_initial_state_never_opens_a_database_connection(self) -> None:
        wrong_time = IngestionRunStateTransition(
            ingestion_run_id=self.run.id,
            state_sequence=1,
            state=self.queued.state,
            attempt_count=0,
            occurred_at=self.now + timedelta(seconds=1),
        )

        with self.assertRaises(IngestionRunTransitionError):
            self.repository.create_run(self.run, wrong_time)

        self.assertEqual(self.database.supplied_urls, [])

    def test_database_failure_rolls_back_and_redacts_driver_details(self) -> None:
        leaked_detail = f"driver exposed {self.database_url} and a signed request"
        self.database.fail_once(
            "INSERT INTO ingestion_run_state_transition",
            message=leaked_detail,
        )

        with self.assertRaises(IngestionRunRepositoryUnavailable) as caught:
            self.repository.create_run(self.run, self.queued)

        self.assertEqual(str(caught.exception), "ingestion run audit creation failed")
        self.assertNotIn(self.database_url, str(caught.exception))
        self.assertNotIn("signed request", str(caught.exception))
        self.assertIsNone(caught.exception.__context__)
        self.assertEqual(self.database.state, {"runs": {}, "transitions": {}})

    def test_integrity_failure_is_reported_as_a_safe_conflict(self) -> None:
        self.database.fail_once(
            "INSERT INTO ingestion_run (",
            message=f"duplicate leaked {self.database_url}",
            sqlstate="23505",
        )

        with self.assertRaises(IngestionRunRepositoryConflict) as caught:
            self.repository.create_run(self.run, self.queued)

        self.assertNotIn(self.database_url, str(caught.exception))
        self.assertIsNone(caught.exception.__context__)

    def test_corrupted_failure_classification_is_not_returned(self) -> None:
        self.repository.create_run(self.run, self.queued)
        run_id: UUID = self.run.id
        self.database.state["transitions"][run_id][0] = (
            1,
            "failed",
            1,
            self.now,
            "non_retryable",
            "storage_unavailable",
        )

        with self.assertRaisesRegex(IngestionRunRepositoryError, "state is invalid"):
            self.repository.latest_transition(run_id)


if __name__ == "__main__":
    unittest.main()
