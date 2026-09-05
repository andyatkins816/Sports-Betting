"""PostgreSQL integration tests for append-only ingestion-run state facts."""

from __future__ import annotations

import importlib.util
import os
import unittest
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from sam_analytics.ingestion_run_repository import (
    IngestionRunRepositoryConflict,
    PostgresIngestionRunRepository,
)
from sam_analytics.ingestion_runs import (
    IngestionRunState,
    IngestionFailureCode,
    mark_failed,
    mark_succeeded,
    new_manual_shadow_run,
    start_next_attempt,
)
from sam_analytics.raw_payload_store import InMemoryRawPayloadStore
from sam_analytics.synthetic_evidence_probe import SyntheticEvidenceProbe


_DATABASE_URL = os.getenv("DATABASE_URL")
_PSYCOPG_AVAILABLE = importlib.util.find_spec("psycopg") is not None


@unittest.skipUnless(_DATABASE_URL and _PSYCOPG_AVAILABLE, "requires disposable PostgreSQL")
class IngestionRunRepositoryPostgresTests(unittest.TestCase):
    def test_synthetic_probe_crosses_the_real_append_only_repository_seam(self) -> None:
        repository = PostgresIngestionRunRepository(_DATABASE_URL)
        store = InMemoryRawPayloadStore(namespace="sam-postgres-probe-integration")
        task_id = uuid4()

        result = SyntheticEvidenceProbe(
            raw_payload_store=store,
            ingestion_run_repository=repository,
        ).run(job_identity=f"celery:{task_id}")

        self.assertEqual(store.stored_count, 1)
        self.assertEqual(
            repository.latest_transition(result.ingestion_run_id).state,
            IngestionRunState.SUCCEEDED,
        )

        import psycopg

        with psycopg.connect(_DATABASE_URL) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT state, attempt_count, failure_class, failure_code
                    FROM ingestion_run_state_transition
                    WHERE ingestion_run_id = %s
                    ORDER BY state_sequence
                    """,
                    (result.ingestion_run_id,),
                )
                rows = cursor.fetchall()

        self.assertEqual(
            rows,
            [
                ("queued", 0, None, None),
                ("running", 1, None, None),
                ("succeeded", 1, None, None),
            ],
        )

    def test_retryable_manual_run_round_trips_through_database_trigger(self) -> None:
        now = datetime.now(timezone.utc)
        run, queued = new_manual_shadow_run(
            provider="sam_synthetic",
            job_identity=f"staging-probe-{uuid4().hex}",
            source_type="storage_probe",
            max_attempts=2,
            created_at=now,
        )
        repository = PostgresIngestionRunRepository(_DATABASE_URL)

        repository.create_run(run, queued)
        first_attempt = start_next_attempt(
            run,
            queued,
            occurred_at=now + timedelta(milliseconds=1),
        )
        repository.append_transition(run, queued, first_attempt)
        failed = mark_failed(
            run,
            first_attempt,
            failure_code=IngestionFailureCode.STORAGE_UNAVAILABLE,
            occurred_at=now + timedelta(milliseconds=2),
        )
        repository.append_transition(run, first_attempt, failed)
        second_attempt = start_next_attempt(
            run,
            failed,
            occurred_at=now + timedelta(milliseconds=3),
        )
        repository.append_transition(run, failed, second_attempt)
        succeeded = mark_succeeded(
            run,
            second_attempt,
            occurred_at=now + timedelta(milliseconds=4),
        )
        repository.append_transition(run, second_attempt, succeeded)

        self.assertEqual(repository.latest_transition(run.id), succeeded)

        import psycopg

        with psycopg.connect(_DATABASE_URL) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT state_sequence, state, attempt_count,
                           failure_class, failure_code
                    FROM ingestion_run_state_transition
                    WHERE ingestion_run_id = %s
                    ORDER BY state_sequence
                    """,
                    (run.id,),
                )
                rows = cursor.fetchall()

        self.assertEqual(
            rows,
            [
                (1, "queued", 0, None, None),
                (2, "running", 1, None, None),
                (3, "failed", 1, "retryable", "storage_unavailable"),
                (4, "running", 2, None, None),
                (5, "succeeded", 2, None, None),
            ],
        )

        with self.assertRaises(IngestionRunRepositoryConflict):
            repository.append_transition(run, failed, second_attempt)

        with self.assertRaises(IngestionRunRepositoryConflict):
            repository.create_run(run, queued)


if __name__ == "__main__":
    unittest.main()
