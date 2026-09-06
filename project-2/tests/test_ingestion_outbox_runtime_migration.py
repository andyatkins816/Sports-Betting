"""Contract tests for the append-only outbox and provider-attempt runtime."""

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


class IngestionOutboxRuntimeMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        migrations_dir = Path(__file__).resolve().parents[1] / "migrations"
        cls.migration = next(
            migration
            for migration in discover_migrations(migrations_dir)
            if migration.filename == "008_ingestion_outbox_runtime.sql"
        )
        cls.sql = cls.migration.sql
        cls.normalized = " ".join(cls.sql.split())

    def test_defines_append_only_publication_and_attempt_lineage(self) -> None:
        tables = (
            "ingestion_outbox_publication_claim",
            "ingestion_outbox_publication_delivery",
            "ingestion_dispatch_attempt_claim",
            "ingestion_dispatch_attempt_receipt",
            "ingestion_dispatch_attempt_completion",
        )
        self.assertEqual(self.sql.upper().count("CREATE TABLE "), len(tables))
        for table in tables:
            with self.subTest(table=table):
                self.assertIn(f"CREATE TABLE {table}", self.sql)
                self.assertIn(
                    f"BEFORE UPDATE OR DELETE ON {table}", self.normalized
                )
                self.assertIn(f"BEFORE TRUNCATE ON {table}", self.normalized)
                self.assertIn(f"REVOKE ALL ON TABLE {table} FROM PUBLIC", self.sql)

    def test_publisher_never_records_delivery_before_broker_acceptance(self) -> None:
        self.assertIn("claim_ingestion_outbox_publication", self.sql)
        self.assertIn("record_ingestion_outbox_publication", self.sql)
        self.assertIn("interval '2 minutes'", self.sql)
        self.assertIn("outbox.available_at <= clock_timestamp()", self.sql)
        self.assertIn("NOT EXISTS ( SELECT 1 FROM ingestion_outbox_publication_delivery", self.normalized)
        self.assertIn("'publishable'::TEXT", self.sql)
        self.assertIn("'expired'", self.sql)
        self.assertIn("'delivered'", self.sql)
        self.assertIn("'recorded'", self.sql)
        self.assertIn("'already_recorded'", self.sql)
        self.assertNotIn("published_at", self.sql.lower())
        claim_start = self.sql.index("CREATE OR REPLACE FUNCTION claim_ingestion_outbox_publication")
        claim_end = self.sql.index("\n$$;", claim_start)
        claim_body = self.sql[claim_start:claim_end]
        self.assertNotIn("INSERT INTO ingestion_dispatch_transition", claim_body)

    def test_provider_attempt_claim_is_one_shot_and_finite(self) -> None:
        self.assertIn("claim_ingestion_dispatch_attempt", self.sql)
        self.assertIn("interval '5 minutes'", self.sql)
        self.assertIn(
            "UNIQUE (ingestion_dispatch_id, attempt_number)", self.normalized
        )
        for disposition in (
            "started",
            "not_ready",
            "inconclusive",
            "terminal",
            "rejected",
        ):
            with self.subTest(disposition=disposition):
                self.assertIn(f"'{disposition}'", self.sql)
        self.assertIn("provider_call_permitted := TRUE", self.sql)
        self.assertEqual(self.sql.count("provider_call_permitted := TRUE"), 1)
        self.assertIn("v_existing_completion_id IS NULL THEN 'inconclusive'", self.normalized)
        self.assertNotRegex(
            self.sql,
            re.compile(
                r"UPDATE\s+ingestion_dispatch_attempt_claim",
                re.IGNORECASE,
            ),
        )

    def test_provider_receipt_runtime_times_cannot_be_future_dated(self) -> None:
        self.assertIn(
            "CREATE TRIGGER provider_payload_receipt_runtime_times BEFORE INSERT "
            "ON provider_payload_receipt",
            self.normalized,
        )
        self.assertIn("database_now := clock_timestamp()", self.sql)
        self.assertIn(
            "ALTER TABLE provider_payload_receipt ALTER COLUMN created_at "
            "SET DEFAULT clock_timestamp()",
            self.normalized,
        )
        self.assertIn("NEW.received_at > NEW.created_at", self.sql)
        self.assertIn("NEW.captured_at > NEW.created_at", self.sql)
        for field in ("created_at", "received_at", "captured_at"):
            with self.subTest(field=field):
                self.assertIn(f"NEW.{field} > database_now", self.sql)
                self.assertIn(f"NOT isfinite(NEW.{field})", self.sql)
        self.assertNotIn("NEW.created_at := clock_timestamp()", self.sql)

    def test_runtime_policy_and_capability_ids_are_snapshotted_safely(self) -> None:
        self.assertIn("ADD COLUMN retry_schedule_sha256 CHAR(64)", self.sql)
        self.assertIn("retry_schedule_sha256 IS NOT NULL", self.sql)
        self.assertIn("retry_schedule_sha256 := v_retry_schedule_sha256", self.sql)
        for constraint in (
            "provider_use_authorization_nonzero_id",
            "provider_payload_receipt_nonzero_id",
        ):
            with self.subTest(constraint=constraint):
                self.assertIn(constraint, self.sql)

    def test_only_provably_unsent_requests_can_be_retried(self) -> None:
        self.assertIn("retry_safety TEXT CHECK (retry_safety = 'request_not_sent')", self.normalized)
        self.assertIn(
            "p_retry_safety IS DISTINCT FROM 'request_not_sent'",
            self.sql,
        )
        self.assertNotIn("provider_confirmed_not_accepted", self.sql)

    def test_attempt_claim_carries_exact_credential_free_use_lineage(self) -> None:
        claim_table = self.sql[
            self.sql.index("CREATE TABLE ingestion_dispatch_attempt_claim") :
            self.sql.index("CREATE INDEX ingestion_dispatch_attempt_claim_provider_idx")
        ]
        for field in (
            "publication_delivery_id UUID NOT NULL UNIQUE",
            "provider_use_authorization_id UUID NOT NULL",
            "quota_reservation_id UUID NOT NULL UNIQUE",
            "quota_receipt_id UUID NOT NULL",
            "running_transition_id UUID NOT NULL UNIQUE",
            "request_fingerprint_sha256 CHAR(64) NOT NULL",
            "window_start TIMESTAMPTZ NOT NULL",
            "window_end TIMESTAMPTZ NOT NULL",
            "license_scope TEXT NOT NULL",
            "license_version TEXT NOT NULL",
            "exposure TEXT NOT NULL CHECK (exposure = 'private_raw')",
        ):
            with self.subTest(field=field):
                self.assertIn(field, " ".join(claim_table.split()))
        for credential_field in (
            "api_key TEXT",
            "access_token TEXT",
            "password TEXT",
            "authorization_header TEXT",
            "cookie TEXT",
            "request_url TEXT",
            "response_body TEXT",
        ):
            with self.subTest(credential_field=credential_field):
                self.assertNotIn(credential_field, claim_table.lower())

    def test_running_and_terminal_facts_are_deferred_atomic_bundles(self) -> None:
        for trigger in (
            "ingestion_publication_delivery_transition_integrity",
            "ingestion_attempt_claim_transition_integrity",
            "ingestion_attempt_receipt_completion_integrity",
            "ingestion_attempt_completion_transition_integrity",
            "ingestion_runtime_transition_reverse_integrity",
        ):
            with self.subTest(trigger=trigger):
                self.assertIn(f"CREATE CONSTRAINT TRIGGER {trigger}", self.sql)
        self.assertGreaterEqual(
            self.normalized.count("DEFERRABLE INITIALLY DEFERRED"), 7
        )
        self.assertIn("exact committed running transition", self.sql)
        self.assertIn("receipt.provider_response_status BETWEEN 200 AND 299", self.sql)
        self.assertIn("attempt_receipt_id IS NOT NULL", self.sql)

    def test_completion_is_exact_idempotent_and_retry_is_transactional(self) -> None:
        self.assertIn("complete_ingestion_dispatch_attempt", self.sql)
        self.assertIn("'committed'", self.sql)
        self.assertIn("'already_committed'", self.sql)
        self.assertIn("conflicting immutable completion", self.sql)
        self.assertIn(
            "ORDER BY receipt.received_at DESC, receipt.created_at DESC, "
            "receipt.provider_quota_remaining ASC, receipt.id DESC",
            self.normalized,
        )
        completion_start = self.sql.index(
            "CREATE OR REPLACE FUNCTION complete_ingestion_dispatch_attempt"
        )
        completion_end = self.sql.index(
            "CREATE TRIGGER ingestion_outbox_publication_claim_append_only"
        )
        completion_body = self.sql[completion_start:completion_end]
        for insert in (
            "INSERT INTO ingestion_dispatch_attempt_completion",
            "INSERT INTO ingestion_quota_reservation",
            "INSERT INTO ingestion_dispatch_outbox",
            "INSERT INTO ingestion_dispatch_transition",
        ):
            with self.subTest(insert=insert):
                self.assertIn(insert, completion_body)

    def test_expired_attempts_have_no_unsafe_reclaim_or_resolution_path(self) -> None:
        self.assertNotIn("reconcile_expired_ingestion_dispatch_attempt", self.sql)
        self.assertNotIn("'lease_reconciliation'", self.sql)
        self.assertIn("There is intentionally no reclaim path", self.sql)
        self.assertIn("provider lane until a later", self.normalized.lower())
        self.assertIn(
            "UNIQUE (ingestion_dispatch_id, attempt_number)",
            self.normalized,
        )

    def test_runtime_functions_are_invoker_only_and_public_execute_is_revoked(self) -> None:
        public_functions = (
            "claim_ingestion_outbox_publication(TEXT, UUID)",
            "record_ingestion_outbox_publication(UUID, TEXT, UUID)",
            "claim_ingestion_dispatch_attempt(UUID, SMALLINT, TEXT, UUID)",
            "read_ingestion_dispatch_attempt_time(UUID, TEXT, UUID)",
        )
        self.assertNotIn("SECURITY DEFINER", self.sql.upper())
        self.assertGreaterEqual(self.sql.upper().count("SECURITY INVOKER"), 16)
        for signature in public_functions:
            with self.subTest(signature=signature):
                self.assertIn(
                    f"REVOKE EXECUTE ON FUNCTION {signature} FROM PUBLIC",
                    self.normalized,
                )
        self.assertIn(
            "REVOKE EXECUTE ON FUNCTION complete_ingestion_dispatch_attempt( "
            "UUID, TEXT, UUID, TEXT, TEXT, TEXT, TIMESTAMPTZ, UUID, TEXT ) FROM PUBLIC",
            self.normalized,
        )
        self.assertIn(
            "REVOKE EXECUTE ON FUNCTION lock_ingestion_provider(TEXT) FROM PUBLIC",
            self.normalized,
        )
        self.assertIn("deployment grants must be constrained", self.sql)

    def test_migration_pins_ddl_name_resolution(self) -> None:
        self.assertIn(
            "SET LOCAL search_path = public, pg_temp",
            self.normalized,
        )
        first_statement = self.sql.index("SET LOCAL search_path")
        first_ddl = self.sql.index("ALTER TABLE ingestion_dispatch")
        self.assertLess(first_statement, first_ddl)

    def test_migration_contains_no_service_endpoint_or_seeded_runtime_data(self) -> None:
        lowered = self.sql.lower()
        self.assertNotIn("http://", lowered)
        self.assertNotIn("https://", lowered)
        self.assertNotIn("insert into provider_use_authorization", lowered)
        self.assertNotIn("security definer", lowered)


@unittest.skipUnless(
    _PSYCOPG_AVAILABLE and _is_disposable_database_url(_DATABASE_URL),
    "requires the disposable loopback PostgreSQL test database",
)
class IngestionOutboxRuntimePostgresTests(unittest.TestCase):
    @staticmethod
    def _drain_publishable_outbox(connection) -> None:
        """Remove ordering dependence on outboxes left by earlier CI tests."""

        while True:
            publisher_identity = f"ci-drain-{uuid4().hex[:10]}"
            lease_token = uuid4()
            claim = connection.execute(
                "SELECT * FROM claim_ingestion_outbox_publication(%s, %s)",
                (publisher_identity, lease_token),
            ).fetchone()
            if claim is None:
                return
            if claim[0] != "publishable":
                raise AssertionError("unexpected publication drain disposition")
            connection.execute(
                "SELECT * FROM record_ingestion_outbox_publication(%s, %s, %s)",
                (claim[1], publisher_identity, lease_token),
            ).fetchone()

    @classmethod
    def _create_initial_bundle(cls, connection) -> dict[str, object]:
        cls._drain_publishable_outbox(connection)
        provider = f"rt_{uuid4().hex[:20]}"
        source_type = "odds"
        license_scope = "internal_analytics_only"
        license_version = "ci-v1"
        admitted_at = datetime.now(UTC) - timedelta(seconds=2)
        authorization_id = uuid4()
        quota_receipt_id = uuid4()
        dispatch_id = uuid4()
        outbox_id = uuid4()
        request_fingerprint = uuid4().hex + uuid4().hex
        quota_payload_sha = uuid4().hex + uuid4().hex
        connection.execute(
            """
            INSERT INTO provider_use_authorization (
                id, provider, license_scope, license_version, source_type,
                exposure, authorization_manifest_sha256, reviewed_at,
                effective_from, effective_until
            ) VALUES (%s, %s, %s, %s, %s, 'private_raw', %s, %s, %s, %s)
            """,
            (
                authorization_id,
                provider,
                license_scope,
                license_version,
                source_type,
                uuid4().hex + uuid4().hex,
                admitted_at - timedelta(days=2),
                admitted_at - timedelta(days=1),
                admitted_at + timedelta(days=1),
            ),
        )
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
                200, 0, 10000, 'ci-v1', %s, %s, %s, %s
            )
            """,
            (
                quota_receipt_id,
                provider,
                source_type,
                uuid4().hex + uuid4().hex,
                quota_payload_sha,
                f"s3://ci-runtime/raw/{provider}/sha256/{quota_payload_sha}",
                admitted_at,
                admitted_at,
                license_scope,
                license_version,
                uuid4().hex + uuid4().hex,
                admitted_at,
            ),
        )
        connection.execute(
            """
            INSERT INTO ingestion_dispatch (
                id, provider, source_type, request_fingerprint_sha256,
                window_start, window_end, estimated_cost, policy_version,
                max_attempts, admitted_at, idempotency_key,
                provider_use_authorization_id, min_request_interval,
                quota_floor, quota_max_age, retry_schedule_sha256
            ) VALUES (
                %s, %s, %s, %s, %s, %s, 1, 'ci-runtime-v1', 2,
                %s, %s, %s, interval '0 seconds', 0, interval '5 minutes',
                repeat('a', 64)
            )
            """,
            (
                dispatch_id,
                provider,
                source_type,
                request_fingerprint,
                admitted_at - timedelta(hours=1),
                admitted_at,
                admitted_at,
                uuid4().hex + uuid4().hex,
                authorization_id,
            ),
        )
        connection.execute(
            """
            INSERT INTO ingestion_quota_reservation (
                ingestion_dispatch_id, attempt_number, reserved_credits,
                reserved_at, provider_payload_receipt_id
            ) VALUES (%s, 1, 1, %s, %s)
            """,
            (dispatch_id, admitted_at, quota_receipt_id),
        )
        connection.execute(
            """
            INSERT INTO ingestion_dispatch_outbox (
                id, ingestion_dispatch_id, attempt_number, available_at
            ) VALUES (%s, %s, 1, %s)
            """,
            (outbox_id, dispatch_id, admitted_at),
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
        return {
            "provider": provider,
            "source_type": source_type,
            "license_scope": license_scope,
            "license_version": license_version,
            "authorization_id": authorization_id,
            "quota_receipt_id": quota_receipt_id,
            "dispatch_id": dispatch_id,
            "outbox_id": outbox_id,
            "request_fingerprint": request_fingerprint,
        }

    @staticmethod
    def _publish(connection, bundle: dict[str, object]) -> dict[str, object]:
        publisher_identity = f"publisher-{uuid4().hex[:10]}"
        lease_token = uuid4()
        claim = connection.execute(
            "SELECT * FROM claim_ingestion_outbox_publication(%s, %s)",
            (publisher_identity, lease_token),
        ).fetchone()
        if claim is None:
            raise AssertionError("expected one publishable outbox record")
        if claim[3] != bundle["dispatch_id"]:
            raise AssertionError("publisher claimed another test's outbox record")
        recorded = connection.execute(
            "SELECT * FROM record_ingestion_outbox_publication(%s, %s, %s)",
            (claim[1], publisher_identity, lease_token),
        ).fetchone()
        return {
            "publisher_identity": publisher_identity,
            "lease_token": lease_token,
            "claim": claim,
            "recorded": recorded,
        }

    @staticmethod
    def _claim_attempt(connection, bundle: dict[str, object]) -> dict[str, object]:
        worker_identity = f"worker-{uuid4().hex[:10]}"
        lease_token = uuid4()
        row = connection.execute(
            "SELECT * FROM claim_ingestion_dispatch_attempt(%s, %s, %s, %s)",
            (bundle["dispatch_id"], 1, worker_identity, lease_token),
        ).fetchone()
        if row is None:
            raise AssertionError("expected one attempt-claim disposition")
        return {
            "row": row,
            "worker_identity": worker_identity,
            "lease_token": lease_token,
            "claim_id": row[2],
        }

    @staticmethod
    def _insert_provider_response(
        connection,
        bundle: dict[str, object],
        *,
        response_status: int = 200,
    ) -> UUID:
        receipt_id = uuid4()
        observed_at = datetime.now(UTC)
        payload_sha = uuid4().hex + uuid4().hex
        connection.execute(
            """
            INSERT INTO provider_payload_receipt (
                id, provider, source_type, request_fingerprint_sha256,
                payload_sha256, payload_uri, captured_at, received_at,
                provider_response_status, payload_bytes,
                provider_quota_remaining, schema_version, license_scope,
                license_version, receipt_sha256
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s,
                %s, 2, 9999, 'ci-v1', %s, %s, %s
            )
            """,
            (
                receipt_id,
                bundle["provider"],
                bundle["source_type"],
                bundle["request_fingerprint"],
                payload_sha,
                f"s3://ci-runtime/raw/{bundle['provider']}/sha256/{payload_sha}",
                observed_at,
                observed_at,
                response_status,
                bundle["license_scope"],
                bundle["license_version"],
                uuid4().hex + uuid4().hex,
            ),
        )
        return receipt_id

    def test_publish_claim_does_not_queue_until_delivery_is_recorded(self) -> None:
        import psycopg

        with psycopg.connect(_DATABASE_URL, autocommit=True) as connection:
            with connection.transaction():
                bundle = self._create_initial_bundle(connection)
            publisher_identity = f"publisher-{uuid4().hex[:10]}"
            lease_token = uuid4()
            with connection.transaction():
                claim = connection.execute(
                    "SELECT * FROM claim_ingestion_outbox_publication(%s, %s)",
                    (publisher_identity, lease_token),
                ).fetchone()
            self.assertIsNotNone(claim)
            self.assertEqual(claim[0], "publishable")
            self.assertEqual(claim[2], bundle["outbox_id"])
            self.assertEqual(claim[3], bundle["dispatch_id"])
            self.assertEqual(claim[4], 1)
            self.assertEqual(claim[6] - claim[5], timedelta(minutes=2))
            state = connection.execute(
                "SELECT state FROM ingestion_dispatch_latest_state WHERE ingestion_dispatch_id = %s",
                (bundle["dispatch_id"],),
            ).fetchone()
            self.assertEqual(state, ("pending",))

            with connection.transaction():
                first = connection.execute(
                    "SELECT * FROM record_ingestion_outbox_publication(%s, %s, %s)",
                    (claim[1], publisher_identity, lease_token),
                ).fetchone()
            with connection.transaction():
                second = connection.execute(
                    "SELECT * FROM record_ingestion_outbox_publication(%s, %s, %s)",
                    (claim[1], publisher_identity, lease_token),
                ).fetchone()
            self.assertEqual(first[0], "recorded")
            self.assertEqual(second, ("already_recorded", first[1]))
            state = connection.execute(
                "SELECT state, attempt_count FROM ingestion_dispatch_latest_state WHERE ingestion_dispatch_id = %s",
                (bundle["dispatch_id"],),
            ).fetchone()
            self.assertEqual(state, ("queued", 0))

    def test_broker_redelivery_never_reclaims_a_provider_call(self) -> None:
        import psycopg

        with psycopg.connect(_DATABASE_URL, autocommit=True) as connection:
            with connection.transaction():
                bundle = self._create_initial_bundle(connection)
            with connection.transaction():
                self._publish(connection, bundle)
            with connection.transaction():
                attempt = self._claim_attempt(connection, bundle)
            row = attempt["row"]
            self.assertEqual(row[0:2], ("started", True))
            self.assertEqual(row[3], connection.execute(
                "SELECT id FROM ingestion_dispatch_transition WHERE ingestion_dispatch_id = %s AND state = 'running'",
                (bundle["dispatch_id"],),
            ).fetchone()[0])
            self.assertEqual(row[18] - row[17], timedelta(minutes=5))

            with connection.transaction():
                redelivery = connection.execute(
                    "SELECT * FROM claim_ingestion_dispatch_attempt(%s, 1, %s, %s)",
                    (bundle["dispatch_id"], "worker-redelivery", uuid4()),
                ).fetchone()
            self.assertEqual(redelivery[0], "inconclusive")
            self.assertFalse(redelivery[1])
            self.assertEqual(redelivery[2], attempt["claim_id"])
            count = connection.execute(
                "SELECT count(*) FROM ingestion_dispatch_attempt_claim WHERE ingestion_dispatch_id = %s",
                (bundle["dispatch_id"],),
            ).fetchone()
            self.assertEqual(count, (1,))

    def test_success_completion_is_receipt_bound_and_exact_idempotent(self) -> None:
        import psycopg

        with psycopg.connect(_DATABASE_URL, autocommit=True) as connection:
            with connection.transaction():
                bundle = self._create_initial_bundle(connection)
            with connection.transaction():
                self._publish(connection, bundle)
            with connection.transaction():
                attempt = self._claim_attempt(connection, bundle)
            with connection.transaction():
                response_receipt_id = self._insert_provider_response(connection, bundle)
            arguments = (
                attempt["claim_id"],
                attempt["worker_identity"],
                attempt["lease_token"],
                "succeeded",
                None,
                None,
                None,
                response_receipt_id,
                None,
            )
            with connection.transaction():
                first = connection.execute(
                    "SELECT * FROM complete_ingestion_dispatch_attempt(%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    arguments,
                ).fetchone()
            with connection.transaction():
                second = connection.execute(
                    "SELECT * FROM complete_ingestion_dispatch_attempt(%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    arguments,
                ).fetchone()
            self.assertEqual(first[0], "committed")
            self.assertEqual(second, ("already_committed", first[1]))
            lineage = connection.execute(
                """
                SELECT completion.outcome, transition.state,
                       receipt.provider_payload_receipt_id
                FROM ingestion_dispatch_attempt_completion AS completion
                JOIN ingestion_dispatch_transition AS transition
                  ON transition.id = completion.completion_transition_id
                JOIN ingestion_dispatch_attempt_receipt AS receipt
                  ON receipt.id = completion.attempt_receipt_id
                WHERE completion.id = %s
                """,
                (first[1],),
            ).fetchone()
            self.assertEqual(lineage, ("succeeded", "succeeded", response_receipt_id))

    def test_non_success_provider_receipt_cannot_commit_as_success(self) -> None:
        import psycopg

        with psycopg.connect(_DATABASE_URL, autocommit=True) as connection:
            with connection.transaction():
                bundle = self._create_initial_bundle(connection)
            with connection.transaction():
                self._publish(connection, bundle)
            with connection.transaction():
                attempt = self._claim_attempt(connection, bundle)
            with connection.transaction():
                response_receipt_id = self._insert_provider_response(
                    connection, bundle, response_status=500
                )
            with self.assertRaises(psycopg.Error):
                with connection.transaction():
                    connection.execute(
                        "SELECT * FROM complete_ingestion_dispatch_attempt(%s, %s, %s, 'succeeded', NULL, NULL, NULL, %s, NULL)",
                        (
                            attempt["claim_id"],
                            attempt["worker_identity"],
                            attempt["lease_token"],
                            response_receipt_id,
                        ),
                    ).fetchone()

    def test_retry_completion_appends_the_next_atomic_bundle(self) -> None:
        import psycopg

        with psycopg.connect(_DATABASE_URL, autocommit=True) as connection:
            with connection.transaction():
                bundle = self._create_initial_bundle(connection)
            with connection.transaction():
                self._publish(connection, bundle)
            with connection.transaction():
                attempt = self._claim_attempt(connection, bundle)
            retry_at = datetime.now(UTC) + timedelta(minutes=1)
            with connection.transaction():
                result = connection.execute(
                    "SELECT * FROM complete_ingestion_dispatch_attempt(%s, %s, %s, 'retry_wait', 'network_timeout', NULL, %s, NULL, 'request_not_sent')",
                    (
                        attempt["claim_id"],
                        attempt["worker_identity"],
                        attempt["lease_token"],
                        retry_at,
                    ),
                ).fetchone()
            self.assertEqual(result[0], "committed")
            bundle_counts = connection.execute(
                """
                SELECT
                    (SELECT count(*) FROM ingestion_quota_reservation WHERE ingestion_dispatch_id = %s AND attempt_number = 2),
                    (SELECT count(*) FROM ingestion_dispatch_outbox WHERE ingestion_dispatch_id = %s AND attempt_number = 2),
                    (SELECT count(*) FROM ingestion_dispatch_transition WHERE ingestion_dispatch_id = %s AND state = 'retry_wait' AND attempt_count = 1)
                """,
                (bundle["dispatch_id"], bundle["dispatch_id"], bundle["dispatch_id"]),
            ).fetchone()
            self.assertEqual(bundle_counts, (1, 1, 1))

    def test_runtime_ledger_rows_reject_mutation_deletion_and_truncation(self) -> None:
        import psycopg

        with psycopg.connect(_DATABASE_URL, autocommit=True) as connection:
            with connection.transaction():
                self._create_initial_bundle(connection)
            publisher_identity = f"publisher-{uuid4().hex[:10]}"
            with connection.transaction():
                claim = connection.execute(
                    "SELECT * FROM claim_ingestion_outbox_publication(%s, %s)",
                    (publisher_identity, uuid4()),
                ).fetchone()
            statements = (
                (
                    "UPDATE ingestion_outbox_publication_claim SET publisher_identity = publisher_identity WHERE id = %s",
                    (claim[1],),
                ),
                (
                    "DELETE FROM ingestion_outbox_publication_claim WHERE id = %s",
                    (claim[1],),
                ),
                ("TRUNCATE ingestion_outbox_publication_claim CASCADE", None),
            )
            for statement, parameters in statements:
                with self.subTest(statement=statement):
                    with self.assertRaises(psycopg.Error):
                        with connection.transaction():
                            connection.execute(statement, parameters)


if __name__ == "__main__":
    unittest.main()
