"""Contract tests for exact, unwired ingestion-admission evidence bindings."""

from __future__ import annotations

import importlib.util
import os
import re
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from sam_analytics.ingestion_admission_repository import (
    IngestionAdmissionRepositoryConflict,
    PostgresIngestionAdmissionRepository,
)
from sam_analytics.ingestion_dispatch import (
    DispatchCandidate,
    DispatchPolicy,
)
from sam_analytics.migrate import discover_migrations
from sam_analytics.provider_contracts import ProviderUse

_DATABASE_URL = os.getenv("DATABASE_URL")
_PSYCOPG_AVAILABLE = importlib.util.find_spec("psycopg") is not None


def _is_disposable_database_url(database_url: str | None) -> bool:
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


class IngestionAdmissionRepositoryMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        migrations_dir = Path(__file__).resolve().parents[1] / "migrations"
        cls.migration = next(
            migration
            for migration in discover_migrations(migrations_dir)
            if migration.filename == "007_ingestion_admission_repository.sql"
        )
        cls.sql = cls.migration.sql
        cls.normalized = " ".join(cls.sql.split())

    def test_authorization_is_an_empty_exact_append_only_contract(self) -> None:
        self.assertIn("CREATE TABLE provider_use_authorization", self.sql)
        self.assertEqual(self.sql.upper().count("CREATE TABLE "), 1)
        self.assertNotIn("INSERT INTO", self.sql.upper())
        for field in (
            "provider TEXT NOT NULL",
            "license_scope TEXT NOT NULL",
            "license_version TEXT NOT NULL",
            "source_type TEXT NOT NULL",
            "exposure TEXT NOT NULL",
            "authorization_manifest_sha256 CHAR(64) NOT NULL UNIQUE",
            "reviewed_at TIMESTAMPTZ NOT NULL",
            "effective_from TIMESTAMPTZ NOT NULL",
            "effective_until TIMESTAMPTZ NOT NULL",
        ):
            with self.subTest(field=field):
                self.assertIn(field, self.normalized)
        self.assertIn(
            "UNIQUE ( provider, license_scope, license_version, source_type, exposure )",
            self.normalized,
        )
        self.assertIn("exposure IN ('private_raw', 'derived')", self.normalized)
        self.assertIn(
            "provider_use_authorization_finite_times CHECK ( "
            "isfinite(reviewed_at) AND isfinite(effective_from) AND "
            "isfinite(effective_until) )",
            self.normalized,
        )
        self.assertIn("effective_from < effective_until", self.normalized)

    def test_new_records_require_exact_authorization_and_quota_receipt_ids(self) -> None:
        self.assertIn(
            "ADD COLUMN provider_use_authorization_id UUID REFERENCES "
            "provider_use_authorization(id)",
            self.normalized,
        )
        self.assertIn(
            "ingestion_dispatch_authorization_required_for_new_records CHECK "
            "(provider_use_authorization_id IS NOT NULL) NOT VALID",
            self.normalized,
        )
        self.assertIn(
            "ADD COLUMN provider_payload_receipt_id UUID REFERENCES "
            "provider_payload_receipt(id)",
            self.normalized,
        )
        self.assertIn(
            "ingestion_quota_receipt_required_for_new_records CHECK "
            "(provider_payload_receipt_id IS NOT NULL) NOT VALID",
            self.normalized,
        )
        self.assertIn("ingestion_dispatch_authorization_idx", self.sql)
        self.assertIn("ingestion_quota_reservation_receipt_idx", self.sql)

    def test_provider_admission_uses_one_transaction_scoped_lock_namespace(self) -> None:
        self.assertIn("lock_ingestion_provider", self.sql)
        self.assertIn("pg_advisory_xact_lock", self.sql)
        self.assertIn("sam-ingestion-admission:", self.sql)
        self.assertIn("hashtextextended", self.sql)
        self.assertIn("provider_payload_receipt_admission_lock", self.sql)
        self.assertGreaterEqual(
            self.sql.count("PERFORM lock_ingestion_provider"),
            3,
        )
        self.assertNotIn("pg_advisory_lock(", self.sql)

    def test_worker_activity_transition_takes_provider_lock_before_integrity(self) -> None:
        self.assertIn(
            "CREATE TRIGGER ingestion_dispatch_transition_admission_lock "
            "BEFORE INSERT ON ingestion_dispatch_transition",
            self.normalized,
        )
        self.assertIn(
            "lock_ingestion_dispatch_transition_provider",
            self.sql,
        )
        self.assertLess(
            "ingestion_dispatch_transition_admission_lock",
            "ingestion_dispatch_transition_integrity",
        )

    def test_retry_outbox_takes_provider_lock_before_integrity(self) -> None:
        self.assertIn(
            "CREATE TRIGGER ingestion_dispatch_outbox_admission_lock "
            "BEFORE INSERT ON ingestion_dispatch_outbox",
            self.normalized,
        )
        self.assertIn(
            "lock_ingestion_dispatch_outbox_provider",
            self.sql,
        )
        self.assertLess(
            "ingestion_dispatch_outbox_admission_lock",
            "ingestion_dispatch_outbox_integrity",
        )

    def test_dispatch_authorization_is_exact_private_and_time_bounded(self) -> None:
        self.assertIn("enforce_ingestion_dispatch_authorization", self.sql)
        self.assertIn(
            "authorization_provider IS DISTINCT FROM NEW.provider",
            self.sql,
        )
        self.assertIn(
            "authorization_source_type IS DISTINCT FROM NEW.source_type",
            self.sql,
        )
        self.assertIn(
            "authorization_exposure IS DISTINCT FROM 'private_raw'",
            self.sql,
        )
        self.assertIn(
            "NEW.admitted_at < greatest( authorization_reviewed_at, "
            "authorization_effective_from )",
            self.normalized,
        )
        self.assertIn(
            "NEW.admitted_at >= authorization_effective_until",
            self.normalized,
        )
        self.assertEqual(
            self.sql.count("authorization_checked_at := clock_timestamp()"),
            2,
        )
        self.assertIn(
            "authorization_checked_at < greatest( authorization_reviewed_at, "
            "authorization_effective_from )",
            self.normalized,
        )
        self.assertIn(
            "authorization_checked_at >= authorization_effective_until",
            self.sql,
        )

    def test_reservation_binds_provider_license_time_and_quota_conservatively(self) -> None:
        for clause in (
            "receipt_provider IS DISTINCT FROM dispatch_provider",
            "receipt_license_scope IS DISTINCT FROM authorization_license_scope",
            "receipt_license_version IS DISTINCT FROM authorization_license_version",
            "receipt_quota_remaining IS NULL",
            "greatest(receipt_received_at, receipt_created_at) > NEW.reserved_at",
            "receipt_created_at < greatest(",
            "license_scope IS NOT DISTINCT FROM authorization_license_scope",
            "license_version IS NOT DISTINCT FROM authorization_license_version",
            "ORDER BY received_at DESC, created_at DESC, "
            "provider_quota_remaining ASC, id DESC",
            "latest_quota_receipt_id IS DISTINCT FROM NEW.provider_payload_receipt_id",
            "provider_reserved_credits + NEW.reserved_credits",
        ):
            with self.subTest(clause=clause):
                self.assertIn(clause, self.normalized)
        self.assertIn(
            "FROM ingestion_quota_reservation AS reservation JOIN "
            "ingestion_dispatch AS dispatch",
            self.normalized,
        )
        self.assertNotRegex(
            self.sql.lower(),
            re.compile(r"\b(delete|update)\s+ingestion_quota_reservation\b"),
        )
        self.assertIn(
            "CREATE INDEX provider_payload_receipt_quota_latest_idx ON "
            "provider_payload_receipt ( provider, license_scope, license_version, "
            "received_at DESC, created_at DESC, provider_quota_remaining ASC, "
            "id DESC ) WHERE provider_quota_remaining IS NOT NULL",
            self.normalized,
        )

    def test_new_and_legacy_receipt_records_cannot_be_mutated_or_truncated(self) -> None:
        for trigger in (
            "provider_use_authorization_append_only",
            "provider_use_authorization_append_only_truncate",
            "provider_payload_receipt_append_only_truncate",
        ):
            with self.subTest(trigger=trigger):
                self.assertIn(trigger, self.sql)
        self.assertIn(
            "BEFORE UPDATE OR DELETE ON provider_use_authorization",
            self.normalized,
        )
        self.assertIn(
            "BEFORE TRUNCATE ON provider_use_authorization",
            self.normalized,
        )
        self.assertIn(
            "BEFORE TRUNCATE ON provider_payload_receipt",
            self.normalized,
        )

    def test_migration_adds_no_runtime_or_secret_surface(self) -> None:
        lowered = self.sql.lower()
        for prohibited in (
            "api_key",
            "access_token",
            "password",
            "authorization_header",
            "request_url",
            "response_body",
            "http://",
            "https://",
            "create extension",
            "notify ",
        ):
            with self.subTest(prohibited=prohibited):
                self.assertNotIn(prohibited, lowered)


@unittest.skipUnless(
    _PSYCOPG_AVAILABLE and _is_disposable_database_url(_DATABASE_URL),
    "requires the disposable loopback PostgreSQL test database",
)
class IngestionAdmissionRepositoryPostgresTests(unittest.TestCase):
    @staticmethod
    def _provider() -> str:
        return f"ci_{uuid4().hex[:20]}"

    @staticmethod
    def _insert_authorization(
        connection,
        *,
        provider: str,
        source_type: str = "odds",
        license_scope: str = "internal_analytics_only",
        license_version: str = "ci-v1",
        exposure: str = "private_raw",
        admitted_at: datetime,
        effective_from: datetime | None = None,
        effective_until: datetime | None = None,
    ) -> UUID:
        authorization_id = uuid4()
        connection.execute(
            """
            INSERT INTO provider_use_authorization (
                id, provider, license_scope, license_version, source_type,
                exposure, authorization_manifest_sha256, reviewed_at,
                effective_from, effective_until
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                authorization_id,
                provider,
                license_scope,
                license_version,
                source_type,
                exposure,
                uuid4().hex + uuid4().hex,
                admitted_at - timedelta(days=2),
                effective_from or admitted_at - timedelta(days=1),
                effective_until or admitted_at + timedelta(days=1),
            ),
        )
        return authorization_id

    @staticmethod
    def _insert_quota_receipt(
        connection,
        *,
        provider: str,
        received_at: datetime,
        receipt_id: UUID | None = None,
        created_at: datetime | None = None,
        quota_remaining: int | None = 100,
        license_scope: str = "internal_analytics_only",
        license_version: str = "ci-v1",
        source_type: str = "odds",
    ) -> UUID:
        receipt_id = receipt_id or uuid4()
        payload_sha256 = uuid4().hex + uuid4().hex
        connection.execute(
            """
            INSERT INTO provider_payload_receipt (
                id, provider, source_type, request_fingerprint_sha256,
                payload_sha256, payload_uri, captured_at, received_at,
                provider_response_status, payload_bytes,
                provider_quota_remaining, schema_version, license_scope,
                license_version, receipt_sha256, created_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s,
                200, 0, %s, 'ci-v1', %s, %s, %s, %s
            )
            """,
            (
                receipt_id,
                provider,
                source_type,
                uuid4().hex + uuid4().hex,
                payload_sha256,
                f"s3://ci-admission/raw/{provider}/sha256/{payload_sha256}",
                received_at,
                received_at,
                quota_remaining,
                license_scope,
                license_version,
                uuid4().hex + uuid4().hex,
                created_at or received_at,
            ),
        )
        return receipt_id

    @staticmethod
    def _insert_dispatch(
        connection,
        *,
        provider: str,
        authorization_id: UUID | None,
        admitted_at: datetime,
        estimated_cost: int = 1,
    ) -> UUID:
        dispatch_id = uuid4()
        connection.execute(
            """
            INSERT INTO ingestion_dispatch (
                id, provider, source_type, request_fingerprint_sha256,
                window_start, window_end, estimated_cost, policy_version,
                max_attempts, admitted_at, idempotency_key,
                provider_use_authorization_id, min_request_interval,
                quota_floor, quota_max_age, retry_schedule_sha256
            ) VALUES (
                %s, %s, 'odds', %s, %s, %s, %s,
                'ci-admission-v1', 2, %s, %s, %s,
                interval '0 seconds', 0, interval '5 minutes', repeat('a', 64)
            )
            """,
            (
                dispatch_id,
                provider,
                uuid4().hex + uuid4().hex,
                admitted_at - timedelta(minutes=1),
                admitted_at,
                estimated_cost,
                admitted_at,
                uuid4().hex + uuid4().hex,
                authorization_id,
            ),
        )
        return dispatch_id

    @staticmethod
    def _complete_initial_bundle(
        connection,
        *,
        dispatch_id: UUID,
        receipt_id: UUID | None,
        admitted_at: datetime,
        estimated_cost: int = 1,
    ) -> None:
        connection.execute(
            """
            INSERT INTO ingestion_quota_reservation (
                ingestion_dispatch_id, attempt_number, reserved_credits,
                reserved_at, provider_payload_receipt_id
            ) VALUES (%s, 1, %s, %s, %s)
            """,
            (dispatch_id, estimated_cost, admitted_at, receipt_id),
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

    @staticmethod
    def _repository_policy(provider: str) -> DispatchPolicy:
        return DispatchPolicy(
            provider=provider,
            policy_version="ci-admission-v1",
            enabled=True,
            allowed_source_types=frozenset({"odds"}),
            max_batch_size=1,
            max_attempts=2,
            min_request_interval=timedelta(0),
            quota_floor=1,
            quota_max_age=timedelta(minutes=5),
        )

    @staticmethod
    def _repository_use(provider: str) -> ProviderUse:
        return ProviderUse(
            provider=provider,
            license_scope="internal_analytics_only",
            license_version="ci-v1",
            source_type="odds",
            exposure="private_raw",
        )

    @staticmethod
    def _repository_candidate(provider: str, observed_at: datetime) -> DispatchCandidate:
        return DispatchCandidate(
            provider=provider,
            source_type="odds",
            request_fingerprint_sha256=uuid4().hex + uuid4().hex,
            window_start=observed_at,
            window_end=observed_at + timedelta(hours=1),
            estimated_cost=1,
        )

    def test_real_repository_commits_exact_four_row_bundle(self) -> None:
        import psycopg

        provider = self._provider()
        observed_at = datetime.now(UTC) - timedelta(seconds=1)
        with psycopg.connect(_DATABASE_URL) as connection:
            with connection.transaction():
                authorization_id = self._insert_authorization(
                    connection,
                    provider=provider,
                    admitted_at=observed_at,
                )
                quota_receipt_id = self._insert_quota_receipt(
                    connection,
                    provider=provider,
                    received_at=observed_at,
                )

        repository = PostgresIngestionAdmissionRepository(_DATABASE_URL or "")
        result = repository.admit(
            [self._repository_candidate(provider, observed_at)],
            policy=self._repository_policy(provider),
            provider_use=self._repository_use(provider),
        )

        self.assertEqual(len(result.decision.admitted), 1)
        self.assertEqual(len(result.persisted), 1)
        dispatch_id = result.persisted[0].ingestion_dispatch_id
        with psycopg.connect(_DATABASE_URL) as connection:
            row = connection.execute(
                """
                SELECT dispatch.provider_use_authorization_id,
                       reservation.provider_payload_receipt_id,
                       outbox.attempt_number,
                       transition.state,
                       transition.attempt_count,
                       (SELECT count(*)
                          FROM ingestion_dispatch
                         WHERE id = dispatch.id),
                       (SELECT count(*)
                          FROM ingestion_quota_reservation
                         WHERE ingestion_dispatch_id = dispatch.id),
                       (SELECT count(*)
                          FROM ingestion_dispatch_outbox
                         WHERE ingestion_dispatch_id = dispatch.id),
                       (SELECT count(*)
                          FROM ingestion_dispatch_transition
                         WHERE ingestion_dispatch_id = dispatch.id)
                FROM ingestion_dispatch AS dispatch
                JOIN ingestion_quota_reservation AS reservation
                  ON reservation.ingestion_dispatch_id = dispatch.id
                 AND reservation.attempt_number = 1
                JOIN ingestion_dispatch_outbox AS outbox
                  ON outbox.ingestion_dispatch_id = dispatch.id
                 AND outbox.attempt_number = 1
                JOIN ingestion_dispatch_transition AS transition
                  ON transition.ingestion_dispatch_id = dispatch.id
                 AND transition.state_sequence = 1
                WHERE dispatch.id = %s
                """,
                (dispatch_id,),
            ).fetchone()

        self.assertEqual(
            row,
            (
                authorization_id,
                quota_receipt_id,
                1,
                "pending",
                0,
                1,
                1,
                1,
                1,
            ),
        )

    def test_real_repository_denies_missing_or_expired_authorization_without_writes(
        self,
    ) -> None:
        import psycopg

        cases = ("missing", "expired")
        for authorization_state in cases:
            provider = self._provider()
            observed_at = datetime.now(UTC) - timedelta(seconds=1)
            with self.subTest(authorization_state=authorization_state):
                with psycopg.connect(_DATABASE_URL) as connection:
                    with connection.transaction():
                        if authorization_state == "expired":
                            expired_at = observed_at - timedelta(hours=1)
                            self._insert_authorization(
                                connection,
                                provider=provider,
                                admitted_at=expired_at,
                                effective_from=expired_at - timedelta(days=1),
                                effective_until=expired_at + timedelta(minutes=30),
                            )
                        self._insert_quota_receipt(
                            connection,
                            provider=provider,
                            received_at=observed_at,
                        )

                repository = PostgresIngestionAdmissionRepository(_DATABASE_URL or "")
                with self.assertRaises(IngestionAdmissionRepositoryConflict):
                    repository.admit(
                        [self._repository_candidate(provider, observed_at)],
                        policy=self._repository_policy(provider),
                        provider_use=self._repository_use(provider),
                    )

                with psycopg.connect(_DATABASE_URL) as connection:
                    row = connection.execute(
                        """
                        SELECT
                            (SELECT count(*)
                               FROM ingestion_dispatch
                              WHERE provider = %s),
                            (SELECT count(*)
                               FROM ingestion_quota_reservation AS reservation
                               JOIN ingestion_dispatch AS dispatch
                                 ON dispatch.id = reservation.ingestion_dispatch_id
                              WHERE dispatch.provider = %s),
                            (SELECT count(*)
                               FROM ingestion_dispatch_outbox AS outbox
                               JOIN ingestion_dispatch AS dispatch
                                 ON dispatch.id = outbox.ingestion_dispatch_id
                              WHERE dispatch.provider = %s),
                            (SELECT count(*)
                               FROM ingestion_dispatch_transition AS transition
                               JOIN ingestion_dispatch AS dispatch
                                 ON dispatch.id = transition.ingestion_dispatch_id
                              WHERE dispatch.provider = %s)
                        """,
                        (provider, provider, provider, provider),
                    ).fetchone()
                self.assertEqual(row, (0, 0, 0, 0))

    def test_authorization_rejects_infinite_time_bounds(self) -> None:
        import psycopg

        finite_past = "2026-01-01T00:00:00+00:00"
        finite_future = "2099-01-01T00:00:00+00:00"
        cases = (
            ("-infinity", finite_past, finite_future),
            (finite_past, "-infinity", finite_future),
            (finite_past, finite_past, "infinity"),
        )
        with psycopg.connect(_DATABASE_URL) as connection:
            for reviewed_at, effective_from, effective_until in cases:
                with self.subTest(
                    reviewed_at=reviewed_at,
                    effective_from=effective_from,
                    effective_until=effective_until,
                ):
                    with self.assertRaisesRegex(psycopg.Error, "finite_times"):
                        with connection.transaction():
                            connection.execute(
                                """
                                INSERT INTO provider_use_authorization (
                                    provider, license_scope, license_version,
                                    source_type, exposure,
                                    authorization_manifest_sha256, reviewed_at,
                                    effective_from, effective_until
                                ) VALUES (
                                    %s, 'internal_analytics_only', 'ci-v1',
                                    'odds', 'private_raw', %s,
                                    %s::timestamptz, %s::timestamptz,
                                    %s::timestamptz
                                )
                                """,
                                (
                                    self._provider(),
                                    uuid4().hex + uuid4().hex,
                                    reviewed_at,
                                    effective_from,
                                    effective_until,
                                ),
                            )

    def test_authorization_exact_use_is_unique_but_new_version_is_distinct(self) -> None:
        import psycopg

        provider = self._provider()
        admitted_at = datetime.now(UTC) - timedelta(seconds=1)
        with psycopg.connect(_DATABASE_URL) as connection:
            with connection.transaction():
                self._insert_authorization(
                    connection,
                    provider=provider,
                    admitted_at=admitted_at,
                )

            with self.assertRaises(psycopg.errors.UniqueViolation):
                with connection.transaction():
                    self._insert_authorization(
                        connection,
                        provider=provider,
                        admitted_at=admitted_at,
                    )

            with connection.transaction():
                self._insert_authorization(
                    connection,
                    provider=provider,
                    admitted_at=admitted_at,
                    license_version="ci-v2",
                )

            row = connection.execute(
                """
                SELECT count(*)
                FROM provider_use_authorization
                WHERE provider = %s
                  AND license_scope = 'internal_analytics_only'
                  AND source_type = 'odds'
                  AND exposure = 'private_raw'
                """,
                (provider,),
            ).fetchone()

        self.assertEqual(row, (2,))

    def test_exact_authorization_and_quota_receipt_bundle_commits(self) -> None:
        import psycopg

        provider = self._provider()
        admitted_at = datetime.now(UTC) - timedelta(seconds=1)
        with psycopg.connect(_DATABASE_URL) as connection:
            with connection.transaction():
                authorization_id = self._insert_authorization(
                    connection,
                    provider=provider,
                    admitted_at=admitted_at,
                )
                receipt_id = self._insert_quota_receipt(
                    connection,
                    provider=provider,
                    received_at=admitted_at,
                )
                dispatch_id = self._insert_dispatch(
                    connection,
                    provider=provider,
                    authorization_id=authorization_id,
                    admitted_at=admitted_at,
                )
                self._complete_initial_bundle(
                    connection,
                    dispatch_id=dispatch_id,
                    receipt_id=receipt_id,
                    admitted_at=admitted_at,
                )

            row = connection.execute(
                """
                SELECT dispatch.provider_use_authorization_id,
                       reservation.provider_payload_receipt_id
                FROM ingestion_dispatch AS dispatch
                JOIN ingestion_quota_reservation AS reservation
                  ON reservation.ingestion_dispatch_id = dispatch.id
                WHERE dispatch.id = %s
                """,
                (dispatch_id,),
            ).fetchone()

        self.assertEqual(row, (authorization_id, receipt_id))

    def test_dispatch_rejects_wrong_provider_derived_or_expired_authorization(self) -> None:
        import psycopg

        admitted_at = datetime.now(UTC) - timedelta(seconds=1)
        cases = (
            ("wrong_provider", "private_raw", None),
            ("same_provider", "derived", None),
            (
                "same_provider",
                "private_raw",
                admitted_at - timedelta(seconds=1),
            ),
        )
        with psycopg.connect(_DATABASE_URL) as connection:
            for provider_case, exposure, effective_until in cases:
                provider = self._provider()
                authorization_provider = (
                    self._provider() if provider_case == "wrong_provider" else provider
                )
                with self.subTest(
                    provider_case=provider_case,
                    exposure=exposure,
                    effective_until=effective_until,
                ):
                    with self.assertRaisesRegex(psycopg.Error, "provider authorization"):
                        with connection.transaction():
                            authorization_id = self._insert_authorization(
                                connection,
                                provider=authorization_provider,
                                exposure=exposure,
                                admitted_at=admitted_at,
                                effective_from=admitted_at - timedelta(days=2),
                                effective_until=effective_until,
                            )
                            self._insert_dispatch(
                                connection,
                                provider=provider,
                                authorization_id=authorization_id,
                                admitted_at=admitted_at,
                            )

    def test_dispatch_rejects_backdated_or_not_yet_effective_authorization(self) -> None:
        import psycopg

        database_now = datetime.now(UTC)
        cases = (
            (
                database_now - timedelta(hours=1),
                database_now - timedelta(days=1),
                database_now - timedelta(minutes=30),
            ),
            (
                database_now + timedelta(minutes=4),
                database_now + timedelta(minutes=3),
                database_now + timedelta(days=1),
            ),
        )
        with psycopg.connect(_DATABASE_URL) as connection:
            for admitted_at, effective_from, effective_until in cases:
                provider = self._provider()
                with self.subTest(
                    admitted_at=admitted_at,
                    effective_from=effective_from,
                    effective_until=effective_until,
                ):
                    with self.assertRaisesRegex(psycopg.Error, "authorization window"):
                        with connection.transaction():
                            authorization_id = self._insert_authorization(
                                connection,
                                provider=provider,
                                admitted_at=admitted_at,
                                effective_from=effective_from,
                                effective_until=effective_until,
                            )
                            self._insert_dispatch(
                                connection,
                                provider=provider,
                                authorization_id=authorization_id,
                                admitted_at=admitted_at,
                            )

    def test_reservation_rejects_wrong_license_null_quota_and_future_receipt(self) -> None:
        import psycopg

        admitted_at = datetime.now(UTC) - timedelta(seconds=2)
        cases = (
            (
                "different_scope",
                100,
                admitted_at,
                admitted_at,
                "license authorization",
            ),
            (
                "internal_analytics_only",
                None,
                admitted_at,
                admitted_at,
                "no trusted remaining",
            ),
            (
                "internal_analytics_only",
                100,
                admitted_at + timedelta(seconds=1),
                admitted_at + timedelta(seconds=1),
                "cannot follow",
            ),
            (
                "internal_analytics_only",
                100,
                admitted_at,
                admitted_at + timedelta(seconds=1),
                "cannot follow",
            ),
        )
        with psycopg.connect(_DATABASE_URL) as connection:
            for (
                license_scope,
                quota_remaining,
                received_at,
                created_at,
                message,
            ) in cases:
                provider = self._provider()
                with self.subTest(
                    license_scope=license_scope,
                    quota_remaining=quota_remaining,
                    received_at=received_at,
                    created_at=created_at,
                ):
                    with self.assertRaisesRegex(psycopg.Error, message):
                        with connection.transaction():
                            authorization_id = self._insert_authorization(
                                connection,
                                provider=provider,
                                admitted_at=admitted_at,
                            )
                            receipt_id = self._insert_quota_receipt(
                                connection,
                                provider=provider,
                                license_scope=license_scope,
                                quota_remaining=quota_remaining,
                                received_at=received_at,
                                created_at=created_at,
                            )
                            dispatch_id = self._insert_dispatch(
                                connection,
                                provider=provider,
                                authorization_id=authorization_id,
                                admitted_at=admitted_at,
                            )
                            self._complete_initial_bundle(
                                connection,
                                dispatch_id=dispatch_id,
                                receipt_id=receipt_id,
                                admitted_at=admitted_at,
                            )

    def test_reservation_rejects_receipt_outside_authorization_window(self) -> None:
        import psycopg

        provider = self._provider()
        admitted_at = datetime.now(UTC) - timedelta(seconds=1)
        authorization_start = admitted_at - timedelta(minutes=5)
        receipt_at = authorization_start - timedelta(seconds=1)
        with psycopg.connect(_DATABASE_URL) as connection:
            with self.assertRaisesRegex(psycopg.Error, "authorization window"):
                with connection.transaction():
                    authorization_id = self._insert_authorization(
                        connection,
                        provider=provider,
                        admitted_at=admitted_at,
                        effective_from=authorization_start,
                    )
                    receipt_id = self._insert_quota_receipt(
                        connection,
                        provider=provider,
                        received_at=receipt_at,
                        created_at=receipt_at,
                    )
                    dispatch_id = self._insert_dispatch(
                        connection,
                        provider=provider,
                        authorization_id=authorization_id,
                        admitted_at=admitted_at,
                    )
                    self._complete_initial_bundle(
                        connection,
                        dispatch_id=dispatch_id,
                        receipt_id=receipt_id,
                        admitted_at=admitted_at,
                    )

    def test_reservation_requires_latest_quota_receipt_for_license(self) -> None:
        import psycopg

        provider = self._provider()
        admitted_at = datetime.now(UTC) - timedelta(seconds=1)
        with psycopg.connect(_DATABASE_URL) as connection:
            with self.assertRaisesRegex(psycopg.Error, "latest provider quota receipt"):
                with connection.transaction():
                    authorization_id = self._insert_authorization(
                        connection,
                        provider=provider,
                        admitted_at=admitted_at,
                    )
                    older_receipt_id = self._insert_quota_receipt(
                        connection,
                        provider=provider,
                        received_at=admitted_at - timedelta(seconds=2),
                        quota_remaining=100,
                    )
                    self._insert_quota_receipt(
                        connection,
                        provider=provider,
                        received_at=admitted_at - timedelta(seconds=1),
                        quota_remaining=1,
                    )
                    dispatch_id = self._insert_dispatch(
                        connection,
                        provider=provider,
                        authorization_id=authorization_id,
                        admitted_at=admitted_at,
                    )
                    self._complete_initial_bundle(
                        connection,
                        dispatch_id=dispatch_id,
                        receipt_id=older_receipt_id,
                        admitted_at=admitted_at,
                    )

    def test_tied_quota_receipts_select_lowest_remaining_before_id(self) -> None:
        import psycopg

        provider = self._provider()
        observed_at = datetime.now(UTC) - timedelta(seconds=1)
        receipt_at = observed_at - timedelta(seconds=1)
        id_tail = uuid4().hex[1:]
        higher_quota_id = UUID(hex=f"f{id_tail}")
        lower_quota_id = UUID(hex=f"0{id_tail}")
        with psycopg.connect(_DATABASE_URL) as connection:
            with connection.transaction():
                authorization_id = self._insert_authorization(
                    connection,
                    provider=provider,
                    admitted_at=observed_at,
                )
                self._insert_quota_receipt(
                    connection,
                    provider=provider,
                    receipt_id=higher_quota_id,
                    received_at=receipt_at,
                    created_at=receipt_at,
                    quota_remaining=100,
                )
                self._insert_quota_receipt(
                    connection,
                    provider=provider,
                    receipt_id=lower_quota_id,
                    received_at=receipt_at,
                    created_at=receipt_at,
                    quota_remaining=2,
                )

        repository = PostgresIngestionAdmissionRepository(_DATABASE_URL or "")
        result = repository.admit(
            [self._repository_candidate(provider, observed_at)],
            policy=self._repository_policy(provider),
            provider_use=self._repository_use(provider),
        )
        dispatch_id = result.persisted[0].ingestion_dispatch_id

        with psycopg.connect(_DATABASE_URL) as connection:
            selected_receipt_id = connection.execute(
                """
                SELECT provider_payload_receipt_id
                FROM ingestion_quota_reservation
                WHERE ingestion_dispatch_id = %s
                  AND attempt_number = 1
                """,
                (dispatch_id,),
            ).fetchone()[0]
            self.assertEqual(selected_receipt_id, lower_quota_id)

            with self.assertRaisesRegex(psycopg.Error, "latest provider quota receipt"):
                with connection.transaction():
                    bypass_dispatch_id = self._insert_dispatch(
                        connection,
                        provider=provider,
                        authorization_id=authorization_id,
                        admitted_at=observed_at,
                    )
                    self._complete_initial_bundle(
                        connection,
                        dispatch_id=bypass_dispatch_id,
                        receipt_id=higher_quota_id,
                        admitted_at=observed_at,
                    )

    def test_latest_quota_receipt_is_scoped_to_exact_license(self) -> None:
        import psycopg

        provider = self._provider()
        admitted_at = datetime.now(UTC) - timedelta(seconds=1)
        with psycopg.connect(_DATABASE_URL) as connection:
            with connection.transaction():
                authorization_id = self._insert_authorization(
                    connection,
                    provider=provider,
                    admitted_at=admitted_at,
                )
                matching_receipt_id = self._insert_quota_receipt(
                    connection,
                    provider=provider,
                    received_at=admitted_at - timedelta(seconds=2),
                )
                self._insert_quota_receipt(
                    connection,
                    provider=provider,
                    received_at=admitted_at - timedelta(seconds=1),
                    license_scope="separate_license_scope",
                )
                dispatch_id = self._insert_dispatch(
                    connection,
                    provider=provider,
                    authorization_id=authorization_id,
                    admitted_at=admitted_at,
                )
                self._complete_initial_bundle(
                    connection,
                    dispatch_id=dispatch_id,
                    receipt_id=matching_receipt_id,
                    admitted_at=admitted_at,
                )

    def test_quota_receipt_source_type_is_provider_wide(self) -> None:
        import psycopg

        provider = self._provider()
        admitted_at = datetime.now(UTC) - timedelta(seconds=1)
        with psycopg.connect(_DATABASE_URL) as connection:
            with connection.transaction():
                authorization_id = self._insert_authorization(
                    connection,
                    provider=provider,
                    admitted_at=admitted_at,
                )
                receipt_id = self._insert_quota_receipt(
                    connection,
                    provider=provider,
                    received_at=admitted_at,
                    source_type="provider_quota_status",
                )
                dispatch_id = self._insert_dispatch(
                    connection,
                    provider=provider,
                    authorization_id=authorization_id,
                    admitted_at=admitted_at,
                )
                self._complete_initial_bundle(
                    connection,
                    dispatch_id=dispatch_id,
                    receipt_id=receipt_id,
                    admitted_at=admitted_at,
                )

    def test_all_prior_reservations_remain_conservatively_outstanding(self) -> None:
        import psycopg

        provider = self._provider()
        first_at = datetime.now(UTC) - timedelta(seconds=3)
        with psycopg.connect(_DATABASE_URL) as connection:
            with connection.transaction():
                authorization_id = self._insert_authorization(
                    connection,
                    provider=provider,
                    admitted_at=first_at,
                )
                first_receipt_id = self._insert_quota_receipt(
                    connection,
                    provider=provider,
                    received_at=first_at,
                    quota_remaining=1,
                )
                first_dispatch_id = self._insert_dispatch(
                    connection,
                    provider=provider,
                    authorization_id=authorization_id,
                    admitted_at=first_at,
                )
                self._complete_initial_bundle(
                    connection,
                    dispatch_id=first_dispatch_id,
                    receipt_id=first_receipt_id,
                    admitted_at=first_at,
                )

            second_at = datetime.now(UTC) - timedelta(seconds=1)
            with self.assertRaisesRegex(psycopg.Error, "conservatively available"):
                with connection.transaction():
                    second_receipt_id = self._insert_quota_receipt(
                        connection,
                        provider=provider,
                        received_at=second_at,
                        quota_remaining=1,
                    )
                    second_dispatch_id = self._insert_dispatch(
                        connection,
                        provider=provider,
                        authorization_id=authorization_id,
                        admitted_at=second_at,
                    )
                    self._complete_initial_bundle(
                        connection,
                        dispatch_id=second_dispatch_id,
                        receipt_id=second_receipt_id,
                        admitted_at=second_at,
                    )

    def test_same_provider_advisory_lock_blocks_a_second_transaction(self) -> None:
        import psycopg

        provider = self._provider()
        with (
            psycopg.connect(_DATABASE_URL) as first,
            psycopg.connect(_DATABASE_URL) as second,
        ):
            with first.transaction():
                first.execute(
                    "SELECT lock_ingestion_provider(%s)",
                    (provider,),
                )
                with self.assertRaises(psycopg.errors.LockNotAvailable):
                    with second.transaction():
                        second.execute("SET LOCAL lock_timeout = '100ms'")
                        second.execute(
                            "SELECT lock_ingestion_provider(%s)",
                            (provider,),
                        )

    def test_transition_insert_waits_for_same_provider_admission_lock(self) -> None:
        import psycopg

        provider = self._provider()
        admitted_at = datetime.now(UTC) - timedelta(seconds=1)
        with psycopg.connect(_DATABASE_URL) as setup:
            with setup.transaction():
                authorization_id = self._insert_authorization(
                    setup,
                    provider=provider,
                    admitted_at=admitted_at,
                )
                receipt_id = self._insert_quota_receipt(
                    setup,
                    provider=provider,
                    received_at=admitted_at,
                )
                dispatch_id = self._insert_dispatch(
                    setup,
                    provider=provider,
                    authorization_id=authorization_id,
                    admitted_at=admitted_at,
                )
                self._complete_initial_bundle(
                    setup,
                    dispatch_id=dispatch_id,
                    receipt_id=receipt_id,
                    admitted_at=admitted_at,
                )

        with (
            psycopg.connect(_DATABASE_URL) as first,
            psycopg.connect(_DATABASE_URL) as second,
        ):
            with first.transaction():
                first.execute("SELECT lock_ingestion_provider(%s)", (provider,))
                with self.assertRaises(psycopg.errors.LockNotAvailable):
                    with second.transaction():
                        second.execute("SET LOCAL lock_timeout = '100ms'")
                        second.execute(
                            """
                            INSERT INTO ingestion_dispatch_transition (
                                ingestion_dispatch_id, state_sequence, state,
                                attempt_count, occurred_at
                            ) VALUES (%s, 2, 'queued', 0, %s)
                            """,
                            (dispatch_id, admitted_at + timedelta(seconds=1)),
                        )

    def test_outbox_insert_waits_for_same_provider_admission_lock(self) -> None:
        import psycopg

        provider = self._provider()
        admitted_at = datetime.now(UTC) - timedelta(seconds=1)
        with psycopg.connect(_DATABASE_URL) as setup:
            with setup.transaction():
                authorization_id = self._insert_authorization(
                    setup,
                    provider=provider,
                    admitted_at=admitted_at,
                )
                receipt_id = self._insert_quota_receipt(
                    setup,
                    provider=provider,
                    received_at=admitted_at,
                )
                dispatch_id = self._insert_dispatch(
                    setup,
                    provider=provider,
                    authorization_id=authorization_id,
                    admitted_at=admitted_at,
                )
                self._complete_initial_bundle(
                    setup,
                    dispatch_id=dispatch_id,
                    receipt_id=receipt_id,
                    admitted_at=admitted_at,
                )

        with (
            psycopg.connect(_DATABASE_URL) as first,
            psycopg.connect(_DATABASE_URL) as second,
        ):
            with first.transaction():
                first.execute("SELECT lock_ingestion_provider(%s)", (provider,))
                with self.assertRaises(psycopg.errors.LockNotAvailable):
                    with second.transaction():
                        second.execute("SET LOCAL lock_timeout = '100ms'")
                        second.execute(
                            """
                            INSERT INTO ingestion_dispatch_outbox (
                                ingestion_dispatch_id, attempt_number,
                                available_at
                            ) VALUES (%s, 2, %s)
                            """,
                            (dispatch_id, admitted_at + timedelta(seconds=1)),
                        )

    def test_authorization_update_delete_and_truncate_are_rejected(self) -> None:
        import psycopg

        provider = self._provider()
        admitted_at = datetime.now(UTC) - timedelta(seconds=1)
        with psycopg.connect(_DATABASE_URL) as connection:
            with connection.transaction():
                authorization_id = self._insert_authorization(
                    connection,
                    provider=provider,
                    admitted_at=admitted_at,
                )

            statements = (
                (
                    "UPDATE provider_use_authorization SET provider = provider "
                    "WHERE id = %s",
                    (authorization_id,),
                ),
                (
                    "DELETE FROM provider_use_authorization WHERE id = %s",
                    (authorization_id,),
                ),
                ("TRUNCATE provider_use_authorization CASCADE", None),
            )
            for statement, params in statements:
                with self.subTest(statement=statement.split()[0]):
                    with self.assertRaisesRegex(psycopg.Error, "append-only"):
                        with connection.transaction():
                            connection.execute(statement, params)


if __name__ == "__main__":
    unittest.main()
