"""Contract tests for the unwired append-only ingestion control plane."""

from __future__ import annotations

import importlib.util
import os
import re
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from sam_analytics.migrate import discover_migrations

_DATABASE_URL = os.getenv("DATABASE_URL")
_PSYCOPG_AVAILABLE = importlib.util.find_spec("psycopg") is not None


def _is_disposable_database_url(database_url: str | None) -> bool:
    """Limit mutation tests to the loopback CI database used by this project."""

    if not database_url:
        return False
    try:
        parsed = urlsplit(database_url)
    except ValueError:
        return False
    return (
        parsed.scheme in {"postgres", "postgresql"}
        and parsed.hostname in {"127.0.0.1", "localhost"}
        and parsed.path == "/sam"
    )


class IngestionControlPlaneMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        migrations_dir = Path(__file__).resolve().parents[1] / "migrations"
        cls.migration = next(
            migration
            for migration in discover_migrations(migrations_dir)
            if migration.filename == "006_ingestion_control_plane.sql"
        )
        cls.sql = cls.migration.sql
        cls.normalized = " ".join(cls.sql.split())

    def test_defines_only_append_only_control_plane_records(self) -> None:
        for table in (
            "ingestion_dispatch",
            "ingestion_quota_reservation",
            "ingestion_dispatch_outbox",
            "ingestion_dispatch_transition",
        ):
            with self.subTest(table=table):
                self.assertIn(f"CREATE TABLE {table}", self.sql)
                self.assertIn(
                    f"BEFORE UPDATE OR DELETE ON {table}",
                    self.normalized,
                )
                self.assertIn(
                    f"BEFORE TRUNCATE ON {table}",
                    self.normalized,
                )
        self.assertEqual(self.sql.upper().count("CREATE TABLE "), 4)
        self.assertNotIn("INSERT INTO", self.sql.upper())

    def test_idempotency_key_and_logical_identity_are_both_unique(self) -> None:
        self.assertIn(
            "idempotency_key CHAR(64) NOT NULL CONSTRAINT "
            "ingestion_dispatch_idempotency_key_unique UNIQUE",
            self.normalized,
        )
        self.assertIn(
            "CHECK (idempotency_key ~ '^[0-9a-f]{64}$')",
            self.normalized,
        )
        self.assertIn(
            "CONSTRAINT ingestion_dispatch_identity_unique UNIQUE ( provider, "
            "request_fingerprint_sha256, source_type, window_start, window_end )",
            self.normalized,
        )
        self.assertNotIn("digest(", self.sql.lower())

    def test_initial_dispatch_and_each_retry_require_transactional_records(self) -> None:
        self.assertIn(
            "DEFERRABLE INITIALLY DEFERRED",
            self.normalized,
        )
        self.assertIn("enforce_ingestion_dispatch_initial_bundle", self.sql)
        self.assertIn("reservation_credits <> NEW.estimated_cost", self.sql)
        self.assertIn("attempt_number = 1", self.sql)
        self.assertIn("initial_outbox_time <> NEW.admitted_at", self.sql)
        self.assertIn("reservation_time > initial_outbox_time", self.sql)
        self.assertIn("enforce_ingestion_dispatch_outbox_transition_pair", self.sql)
        self.assertIn("enforce_ingestion_dispatch_retry_outbox_pair", self.sql)
        self.assertIn("enforce_ingestion_quota_reservation_outbox_pair", self.sql)
        self.assertIn(
            "UNIQUE (ingestion_dispatch_id, attempt_number)",
            self.normalized,
        )
        self.assertIn(
            "attempt_number = NEW.attempt_count + 1",
            self.normalized,
        )
        self.assertIn(
            "paired_reservation_time < paired_retry_occurred_at",
            self.sql,
        )

    def test_state_attempt_and_time_invariants_are_database_enforced(self) -> None:
        for state in (
            "pending",
            "queued",
            "running",
            "retry_wait",
            "succeeded",
            "dead_lettered",
            "cancelled",
        ):
            with self.subTest(state=state):
                self.assertRegex(self.sql, rf"'{state}'")
        self.assertIn("UNIQUE (ingestion_dispatch_id, state_sequence)", self.sql)
        self.assertIn("state_sequence <> previous_sequence + 1", self.sql)
        self.assertIn("NEW.attempt_count > dispatch_max_attempts", self.sql)
        self.assertIn("NEW.occurred_at < previous_occurred_at", self.sql)
        self.assertIn("window_end <= window_start + interval '7 days'", self.sql)
        self.assertIn("retry_not_before_at > occurred_at", self.normalized)
        self.assertIn(
            "retry_not_before_at <= occurred_at + interval '7 days'",
            self.normalized,
        )
        self.assertIn("NEW.occurred_at >= previous_retry_not_before_at", self.sql)
        self.assertIn("clock_timestamp() >= previous_retry_not_before_at", self.sql)
        self.assertIn("paired_available_at > clock_timestamp()", self.sql)
        self.assertIn("NEW.created_at := clock_timestamp()", self.sql)
        self.assertIn(
            "dead_letter_reason = 'policy_disabled' AND failure_code IS NOT NULL",
            self.normalized,
        )
        self.assertIn("attempts-exhausted dead letter requires the final attempt", self.sql)

    def test_activity_and_backlog_reads_are_derived_from_transitions(self) -> None:
        self.assertIn("CREATE VIEW ingestion_dispatch_latest_state", self.sql)
        self.assertIn("CREATE VIEW ingestion_worker_activity", self.sql)
        self.assertIn("max(transition.occurred_at) AS latest_worker_activity_at", self.sql)
        self.assertNotRegex(
            self.sql,
            re.compile(r"CREATE\s+TABLE\s+\w*heartbeat", re.IGNORECASE),
        )
        self.assertIn(
            "CREATE INDEX IF NOT EXISTS odds_snapshot_provider_received_idx",
            self.sql,
        )
        for index in (
            "ingestion_dispatch_provider_admitted_idx",
            "ingestion_dispatch_transition_latest_idx",
            "ingestion_dispatch_outbox_backlog_idx",
            "ingestion_dispatch_transition_backlog_idx",
            "ingestion_dispatch_transition_worker_activity_idx",
        ):
            with self.subTest(index=index):
                self.assertIn(f"CREATE INDEX {index}", self.sql)

        self.assertIn(
            "ON odds_snapshot (provider, received_at DESC)",
            self.normalized,
        )

    def test_schema_has_no_runtime_or_mutable_delivery_surface(self) -> None:
        lowered = self.sql.lower()
        for prohibited_column in (
            "api_key",
            "access_token",
            "password",
            "request_url",
            "response_body",
            "published_at",
            "delivered_at",
            "processed_at",
            "claimed_at",
            "heartbeat_at",
            "status",
            "updated_at",
        ):
            with self.subTest(prohibited_column=prohibited_column):
                self.assertNotRegex(
                    lowered,
                    re.compile(
                        rf"^\s*{prohibited_column}\s+",
                        re.IGNORECASE | re.MULTILINE,
                    ),
                )
        for prohibited_text in (
            "http://",
            "https://",
        ):
            with self.subTest(prohibited_text=prohibited_text):
                self.assertNotIn(prohibited_text, lowered)


@unittest.skipUnless(
    _PSYCOPG_AVAILABLE and _is_disposable_database_url(_DATABASE_URL),
    "requires the disposable loopback PostgreSQL test database",
)
class IngestionControlPlanePostgresTests(unittest.TestCase):
    @staticmethod
    def _new_dispatch_values() -> tuple[UUID, str, str, datetime]:
        return (
            uuid4(),
            uuid4().hex + uuid4().hex,
            uuid4().hex + uuid4().hex,
            datetime.now(UTC),
        )

    @staticmethod
    def _insert_dispatch(
        connection,
        *,
        dispatch_id: UUID,
        idempotency_key: str,
        request_fingerprint: str,
        admitted_at: datetime,
    ) -> None:
        connection.execute(
            """
            INSERT INTO ingestion_dispatch (
                id, provider, source_type, request_fingerprint_sha256,
                window_start, window_end, estimated_cost, policy_version,
                max_attempts, admitted_at, idempotency_key
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                dispatch_id,
                "sam_synthetic",
                "storage_probe",
                request_fingerprint,
                admitted_at - timedelta(minutes=1),
                admitted_at,
                1,
                "ci-control-plane-v1",
                2,
                admitted_at,
                idempotency_key,
            ),
        )

    @classmethod
    def _insert_valid_initial_bundle(
        cls,
        connection,
        *,
        admitted_at: datetime | None = None,
    ) -> UUID:
        dispatch_id, idempotency_key, request_fingerprint, generated_admitted_at = (
            cls._new_dispatch_values()
        )
        admitted_at = admitted_at or generated_admitted_at
        cls._insert_dispatch(
            connection,
            dispatch_id=dispatch_id,
            idempotency_key=idempotency_key,
            request_fingerprint=request_fingerprint,
            admitted_at=admitted_at,
        )
        connection.execute(
            """
            INSERT INTO ingestion_quota_reservation (
                ingestion_dispatch_id, attempt_number, reserved_credits, reserved_at
            ) VALUES (%s, 1, 1, %s)
            """,
            (dispatch_id, admitted_at),
        )
        connection.execute(
            """
            INSERT INTO ingestion_dispatch_outbox (
                ingestion_dispatch_id, attempt_number, available_at
            ) VALUES (%s, 1, %s)
            """,
            (dispatch_id, admitted_at),
        )
        connection.execute(
            """
            INSERT INTO ingestion_dispatch_transition (
                ingestion_dispatch_id, state_sequence, state,
                attempt_count, occurred_at
            ) VALUES (%s, 1, 'pending', 0, %s)
            """,
            (dispatch_id, admitted_at),
        )
        return dispatch_id

    @staticmethod
    def _append_transition(
        connection,
        *,
        dispatch_id: UUID,
        state_sequence: int,
        state: str,
        attempt_count: int,
        occurred_at: datetime,
        worker_identity: str | None = None,
        failure_code: str | None = None,
        dead_letter_reason: str | None = None,
        retry_not_before_at: datetime | None = None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO ingestion_dispatch_transition (
                ingestion_dispatch_id, state_sequence, state, attempt_count,
                worker_identity, failure_code, dead_letter_reason,
                retry_not_before_at, occurred_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                dispatch_id,
                state_sequence,
                state,
                attempt_count,
                worker_identity,
                failure_code,
                dead_letter_reason,
                retry_not_before_at,
                occurred_at,
            ),
        )

    @classmethod
    def _advance_to_first_running(
        cls,
        connection,
        *,
        dispatch_id: UUID,
        admitted_at: datetime,
    ) -> None:
        cls._append_transition(
            connection,
            dispatch_id=dispatch_id,
            state_sequence=2,
            state="queued",
            attempt_count=0,
            occurred_at=admitted_at + timedelta(seconds=1),
        )
        cls._append_transition(
            connection,
            dispatch_id=dispatch_id,
            state_sequence=3,
            state="running",
            attempt_count=1,
            worker_identity="ci-worker-one",
            occurred_at=admitted_at + timedelta(seconds=2),
        )

    @classmethod
    def _append_second_attempt_retry_bundle(
        cls,
        connection,
        *,
        dispatch_id: UUID,
        failure_at: datetime,
        retry_at: datetime,
    ) -> None:
        cls._append_transition(
            connection,
            dispatch_id=dispatch_id,
            state_sequence=4,
            state="retry_wait",
            attempt_count=1,
            worker_identity="ci-worker-one",
            failure_code="network_timeout",
            retry_not_before_at=retry_at,
            occurred_at=failure_at,
        )
        connection.execute(
            """
            INSERT INTO ingestion_quota_reservation (
                ingestion_dispatch_id, attempt_number, reserved_credits, reserved_at
            ) VALUES (%s, 2, 1, %s)
            """,
            (dispatch_id, failure_at),
        )
        connection.execute(
            """
            INSERT INTO ingestion_dispatch_outbox (
                ingestion_dispatch_id, attempt_number, available_at
            ) VALUES (%s, 2, %s)
            """,
            (dispatch_id, retry_at),
        )

    def test_valid_initial_bundle_commits_and_projects_latest_state(self) -> None:
        import psycopg

        with psycopg.connect(_DATABASE_URL) as connection:
            with connection.transaction():
                dispatch_id = self._insert_valid_initial_bundle(connection)

            row = connection.execute(
                """
                SELECT provider, state_sequence, state, attempt_count
                FROM ingestion_dispatch_latest_state
                WHERE ingestion_dispatch_id = %s
                """,
                (dispatch_id,),
            ).fetchone()

        self.assertEqual(row, ("sam_synthetic", 1, "pending", 0))

    def test_full_retry_flow_reaches_succeeded_with_two_reserved_attempts(self) -> None:
        import psycopg

        admitted_at = datetime.now(UTC) - timedelta(seconds=30)
        with psycopg.connect(_DATABASE_URL) as connection:
            with connection.transaction():
                dispatch_id = self._insert_valid_initial_bundle(
                    connection,
                    admitted_at=admitted_at,
                )
            with connection.transaction():
                self._advance_to_first_running(
                    connection,
                    dispatch_id=dispatch_id,
                    admitted_at=admitted_at,
                )
            with connection.transaction():
                self._append_second_attempt_retry_bundle(
                    connection,
                    dispatch_id=dispatch_id,
                    failure_at=admitted_at + timedelta(seconds=3),
                    retry_at=admitted_at + timedelta(seconds=4),
                )
            with connection.transaction():
                self._append_transition(
                    connection,
                    dispatch_id=dispatch_id,
                    state_sequence=5,
                    state="queued",
                    attempt_count=1,
                    occurred_at=admitted_at + timedelta(seconds=5),
                )
                self._append_transition(
                    connection,
                    dispatch_id=dispatch_id,
                    state_sequence=6,
                    state="running",
                    attempt_count=2,
                    worker_identity="ci-worker-two",
                    occurred_at=admitted_at + timedelta(seconds=6),
                )
                self._append_transition(
                    connection,
                    dispatch_id=dispatch_id,
                    state_sequence=7,
                    state="succeeded",
                    attempt_count=2,
                    worker_identity="ci-worker-two",
                    occurred_at=admitted_at + timedelta(seconds=7),
                )

            transitions = connection.execute(
                """
                SELECT state_sequence, state, attempt_count, worker_identity, failure_code
                FROM ingestion_dispatch_transition
                WHERE ingestion_dispatch_id = %s
                ORDER BY state_sequence
                """,
                (dispatch_id,),
            ).fetchall()
            attempts = connection.execute(
                """
                SELECT reservation.attempt_number, outbox.attempt_number
                FROM ingestion_quota_reservation AS reservation
                JOIN ingestion_dispatch_outbox AS outbox
                  ON outbox.ingestion_dispatch_id = reservation.ingestion_dispatch_id
                 AND outbox.attempt_number = reservation.attempt_number
                WHERE reservation.ingestion_dispatch_id = %s
                ORDER BY reservation.attempt_number
                """,
                (dispatch_id,),
            ).fetchall()
            latest = connection.execute(
                """
                SELECT state_sequence, state, attempt_count
                FROM ingestion_dispatch_latest_state
                WHERE ingestion_dispatch_id = %s
                """,
                (dispatch_id,),
            ).fetchone()

        self.assertEqual(
            transitions,
            [
                (1, "pending", 0, None, None),
                (2, "queued", 0, None, None),
                (3, "running", 1, "ci-worker-one", None),
                (4, "retry_wait", 1, "ci-worker-one", "network_timeout"),
                (5, "queued", 1, None, None),
                (6, "running", 2, "ci-worker-two", None),
                (7, "succeeded", 2, "ci-worker-two", None),
            ],
        )
        self.assertEqual(attempts, [(1, 1), (2, 2)])
        self.assertEqual(latest, (7, "succeeded", 2))

    def test_retry_wait_requires_next_attempt_reservation_and_outbox(self) -> None:
        import psycopg

        admitted_at = datetime.now(UTC) - timedelta(seconds=30)
        failure_at = admitted_at + timedelta(seconds=3)
        retry_at = admitted_at + timedelta(seconds=4)
        with psycopg.connect(_DATABASE_URL) as connection:
            with connection.transaction():
                dispatch_id = self._insert_valid_initial_bundle(
                    connection,
                    admitted_at=admitted_at,
                )
            with connection.transaction():
                self._advance_to_first_running(
                    connection,
                    dispatch_id=dispatch_id,
                    admitted_at=admitted_at,
                )

            with self.assertRaisesRegex(psycopg.Error, "transactional outbox"):
                with connection.transaction():
                    self._append_transition(
                        connection,
                        dispatch_id=dispatch_id,
                        state_sequence=4,
                        state="retry_wait",
                        attempt_count=1,
                        worker_identity="ci-worker-one",
                        failure_code="network_timeout",
                        retry_not_before_at=retry_at,
                        occurred_at=failure_at,
                    )

            with self.assertRaisesRegex(psycopg.Error, "matching quota reservation"):
                with connection.transaction():
                    self._append_transition(
                        connection,
                        dispatch_id=dispatch_id,
                        state_sequence=4,
                        state="retry_wait",
                        attempt_count=1,
                        worker_identity="ci-worker-one",
                        failure_code="network_timeout",
                        retry_not_before_at=retry_at,
                        occurred_at=failure_at,
                    )
                    connection.execute(
                        """
                        INSERT INTO ingestion_dispatch_outbox (
                            ingestion_dispatch_id, attempt_number, available_at
                        ) VALUES (%s, 2, %s)
                        """,
                        (dispatch_id, retry_at),
                    )

            latest = connection.execute(
                """
                SELECT state_sequence, state
                FROM ingestion_dispatch_latest_state
                WHERE ingestion_dispatch_id = %s
                """,
                (dispatch_id,),
            ).fetchone()

        self.assertEqual(latest, (3, "running"))

    def test_retry_cannot_be_queued_before_server_due_time(self) -> None:
        import psycopg

        admitted_at = datetime.now(UTC) - timedelta(seconds=10)
        failure_at = datetime.now(UTC)
        retry_at = failure_at + timedelta(minutes=4)
        with psycopg.connect(_DATABASE_URL) as connection:
            with connection.transaction():
                dispatch_id = self._insert_valid_initial_bundle(
                    connection,
                    admitted_at=admitted_at,
                )
            with connection.transaction():
                self._advance_to_first_running(
                    connection,
                    dispatch_id=dispatch_id,
                    admitted_at=admitted_at,
                )
            with connection.transaction():
                self._append_second_attempt_retry_bundle(
                    connection,
                    dispatch_id=dispatch_id,
                    failure_at=failure_at,
                    retry_at=retry_at,
                )

            with self.assertRaisesRegex(psycopg.Error, "invalid ingestion dispatch transition"):
                with connection.transaction():
                    self._append_transition(
                        connection,
                        dispatch_id=dispatch_id,
                        state_sequence=5,
                        state="queued",
                        attempt_count=1,
                        occurred_at=retry_at,
                    )

            latest = connection.execute(
                """
                SELECT state_sequence, state
                FROM ingestion_dispatch_latest_state
                WHERE ingestion_dispatch_id = %s
                """,
                (dispatch_id,),
            ).fetchone()

        self.assertEqual(latest, (4, "retry_wait"))

    def test_wrong_reservation_cost_is_rejected(self) -> None:
        import psycopg

        dispatch_id, idempotency_key, request_fingerprint, admitted_at = (
            self._new_dispatch_values()
        )
        with psycopg.connect(_DATABASE_URL) as connection:
            with self.assertRaisesRegex(psycopg.Error, "equal the estimated attempt cost"):
                with connection.transaction():
                    self._insert_dispatch(
                        connection,
                        dispatch_id=dispatch_id,
                        idempotency_key=idempotency_key,
                        request_fingerprint=request_fingerprint,
                        admitted_at=admitted_at,
                    )
                    connection.execute(
                        """
                        INSERT INTO ingestion_quota_reservation (
                            ingestion_dispatch_id, attempt_number,
                            reserved_credits, reserved_at
                        ) VALUES (%s, 1, 2, %s)
                        """,
                        (dispatch_id, admitted_at),
                    )

    def test_attempt_count_cannot_exceed_dispatch_limit(self) -> None:
        import psycopg

        admitted_at = datetime.now(UTC) - timedelta(seconds=10)
        with psycopg.connect(_DATABASE_URL) as connection:
            with connection.transaction():
                dispatch_id = self._insert_valid_initial_bundle(
                    connection,
                    admitted_at=admitted_at,
                )
            with connection.transaction():
                self._append_transition(
                    connection,
                    dispatch_id=dispatch_id,
                    state_sequence=2,
                    state="queued",
                    attempt_count=0,
                    occurred_at=admitted_at + timedelta(seconds=1),
                )

            with self.assertRaisesRegex(psycopg.Error, "attempt exceeds its reviewed limit"):
                with connection.transaction():
                    self._append_transition(
                        connection,
                        dispatch_id=dispatch_id,
                        state_sequence=3,
                        state="running",
                        attempt_count=3,
                        worker_identity="ci-worker-overflow",
                        occurred_at=admitted_at + timedelta(seconds=2),
                    )

    def test_incomplete_initial_bundle_fails_at_deferred_commit(self) -> None:
        import psycopg

        dispatch_id, idempotency_key, request_fingerprint, admitted_at = (
            self._new_dispatch_values()
        )
        with psycopg.connect(_DATABASE_URL) as connection:
            with self.assertRaisesRegex(psycopg.Error, "atomic reservation"):
                with connection.transaction():
                    self._insert_dispatch(
                        connection,
                        dispatch_id=dispatch_id,
                        idempotency_key=idempotency_key,
                        request_fingerprint=request_fingerprint,
                        admitted_at=admitted_at,
                    )

            row = connection.execute(
                "SELECT count(*) FROM ingestion_dispatch WHERE id = %s",
                (dispatch_id,),
            ).fetchone()

        self.assertEqual(row, (0,))

    def test_update_delete_and_truncate_are_rejected(self) -> None:
        import psycopg

        with psycopg.connect(_DATABASE_URL) as connection:
            with connection.transaction():
                dispatch_id = self._insert_valid_initial_bundle(connection)

            statements = (
                (
                    """
                    UPDATE ingestion_dispatch_transition
                    SET occurred_at = occurred_at
                    WHERE ingestion_dispatch_id = %s
                    """,
                    (dispatch_id,),
                ),
                (
                    """
                    DELETE FROM ingestion_dispatch_transition
                    WHERE ingestion_dispatch_id = %s
                    """,
                    (dispatch_id,),
                ),
                ("TRUNCATE ingestion_dispatch_transition", None),
            )
            for statement, params in statements:
                with self.subTest(statement=statement.strip().split()[0]):
                    with self.assertRaisesRegex(psycopg.Error, "append-only"):
                        with connection.transaction():
                            connection.execute(statement, params)


if __name__ == "__main__":
    unittest.main()
