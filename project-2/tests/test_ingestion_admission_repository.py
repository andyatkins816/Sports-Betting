"""Focused unit tests for the inactive PostgreSQL admission repository."""

from __future__ import annotations

import copy
import unittest
from contextlib import AbstractContextManager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

from sam_analytics.ingestion_admission_repository import (
    IngestionAdmissionRepositoryConflict,
    IngestionAdmissionRepositoryUnavailable,
    PostgresIngestionAdmissionRepository,
)
from sam_analytics.ingestion_dispatch import (
    DispatchBlockReason,
    DispatchCandidate,
    DispatchPolicy,
    DispatchValidationError,
    retry_schedule_sha256,
)
from sam_analytics.provider_contracts import ProviderUse


class _FakeDatabaseError(RuntimeError):
    def __init__(self, message: str, *, sqlstate: str | None = None) -> None:
        super().__init__(message)
        self.sqlstate = sqlstate


class _FakeTransaction(AbstractContextManager[None]):
    def __init__(self, database: "_FakeDatabase") -> None:
        self.database = database
        self.snapshot = None

    def __enter__(self) -> None:
        self.snapshot = copy.deepcopy(self.database.writes)
        return None

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_type is not None:
            self.database.writes = self.snapshot
        return False


class _FakeCursor:
    def __init__(self, database: "_FakeDatabase") -> None:
        self.database = database
        self.rows = []

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return False

    def execute(self, query, params=None) -> None:
        normalized = " ".join(query.split())
        self.database.executions.append((normalized, params))
        self.rows = []
        if self.database.fail_prefix and normalized.startswith(self.database.fail_prefix):
            failure = self.database.failure
            self.database.fail_prefix = None
            raise failure

        if normalized.startswith("SELECT lock_ingestion_provider"):
            self.rows = [(None,)]
            return
        if normalized == "SELECT clock_timestamp()":
            self.rows = [(self.database.now,)]
            return
        if normalized.startswith("SELECT id, authorization_manifest_sha256"):
            if self.database.authorization is not None:
                self.rows = [self.database.authorization]
            return
        if normalized.startswith("SELECT id, provider_quota_remaining"):
            if self.database.quota_receipt is not None:
                self.rows = [self.database.quota_receipt]
            return
        if normalized.startswith("SELECT dispatch.idempotency_key,"):
            self.rows = list(self.database.reservations)
            return
        if normalized.startswith("SELECT idempotency_key"):
            requested = set(params[1])
            self.rows = [
                (key,)
                for key in sorted(self.database.existing_keys)
                if key in requested
            ]
            return
        if normalized.startswith("SELECT max(transition.occurred_at)"):
            self.rows = [(self.database.latest_activity_at,)]
            return
        if normalized.startswith("INSERT INTO ingestion_dispatch ("):
            self.database.writes.append(("dispatch", params))
            return
        if normalized.startswith("INSERT INTO ingestion_quota_reservation"):
            self.database.writes.append(("reservation", params))
            return
        if normalized.startswith("INSERT INTO ingestion_dispatch_outbox"):
            self.database.writes.append(("outbox", params))
            return
        if normalized.startswith("INSERT INTO ingestion_dispatch_transition"):
            self.database.writes.append(("transition", params))
            return
        raise AssertionError(f"unexpected repository SQL: {normalized}")

    def fetchone(self):
        return self.rows.pop(0) if self.rows else None

    def fetchall(self):
        rows = tuple(self.rows)
        self.rows = []
        return rows


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
    def __init__(self, now: datetime) -> None:
        self.now = now
        self.authorization_id = uuid4()
        self.authorization = (self.authorization_id, "a" * 64)
        self.quota_receipt_id = uuid4()
        self.quota_receipt = (
            self.quota_receipt_id,
            10,
            now - timedelta(minutes=5),
        )
        self.reservations = []
        self.existing_keys = set()
        self.latest_activity_at = None
        self.writes = []
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


class IngestionAdmissionRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 9, 6, 18, tzinfo=UTC)
        self.database_url = "postgresql://sam:never-show-this@db.example/sam"
        self.database = _FakeDatabase(self.now)
        self.repository = PostgresIngestionAdmissionRepository(
            self.database_url,
            connection_factory=self.database.connect,
        )
        self.policy = DispatchPolicy(
            provider="the_odds_api",
            policy_version="admission-v1",
            enabled=True,
            allowed_source_types=frozenset({"odds"}),
            max_batch_size=2,
            max_attempts=3,
            min_request_interval=timedelta(0),
            quota_floor=2,
            quota_max_age=timedelta(minutes=10),
            retry_delays=(timedelta(seconds=30), timedelta(minutes=5)),
        )
        self.provider_use = ProviderUse(
            provider="the_odds_api",
            license_scope="internal-derived-only",
            license_version="terms-2026-09-06",
            source_type="odds",
            exposure="private_raw",
        )

    def _candidate(self, offset: int = 0, *, cost: int = 1) -> DispatchCandidate:
        return DispatchCandidate(
            provider="the_odds_api",
            source_type="odds",
            request_fingerprint_sha256=f"{offset + 1:064x}",
            window_start=self.now + timedelta(hours=offset),
            window_end=self.now + timedelta(hours=offset + 1),
            estimated_cost=cost,
        )

    def test_disabled_policy_returns_before_database_or_authorization_lookup(self) -> None:
        def fail_if_connected(database_url: str):
            raise AssertionError(f"database must not be opened: {database_url}")

        repository = PostgresIngestionAdmissionRepository(
            self.database_url,
            connection_factory=fail_if_connected,
        )

        result = repository.admit(
            [self._candidate(), self._candidate(1)],
            policy=replace(self.policy, enabled=False),
        )

        self.assertEqual(result.persisted, ())
        self.assertEqual(
            [blocked.reason for blocked in result.decision.blocked],
            [DispatchBlockReason.DISABLED, DispatchBlockReason.DISABLED],
        )

    def test_enabled_admission_uses_locked_database_snapshot_and_atomic_bundles(self) -> None:
        result = self.repository.admit(
            [self._candidate(), self._candidate(1)],
            policy=self.policy,
            provider_use=self.provider_use,
        )

        self.assertEqual(len(result.decision.admitted), 2)
        self.assertEqual(len(result.persisted), 2)
        self.assertEqual(
            [persisted.idempotency_key for persisted in result.persisted],
            [plan.idempotency_key for plan in result.decision.admitted],
        )
        self.assertEqual(self.database.supplied_urls, [self.database_url])
        self.assertEqual(self.database.closed_connections, 1)
        self.assertEqual(
            [sql.split(" ", 2)[:2] for sql, _ in self.database.executions[:2]],
            [["SELECT", "lock_ingestion_provider(%s)"], ["SELECT", "clock_timestamp()"]],
        )

        authorization_execution = self.database.executions[2]
        self.assertEqual(
            authorization_execution[1],
            (
                "the_odds_api",
                "internal-derived-only",
                "terms-2026-09-06",
                "odds",
                "private_raw",
                self.now,
                self.now,
                self.now,
            ),
        )
        quota_execution = self.database.executions[3]
        self.assertEqual(
            quota_execution[1],
            (
                "the_odds_api",
                "internal-derived-only",
                "terms-2026-09-06",
            ),
        )
        self.assertIn(
            "ORDER BY received_at DESC, created_at DESC, "
            "provider_quota_remaining ASC, id DESC",
            " ".join(quota_execution[0].split()),
        )

        self.assertEqual(
            [kind for kind, _ in self.database.writes],
            [
                "dispatch",
                "reservation",
                "outbox",
                "transition",
                "dispatch",
                "reservation",
                "outbox",
                "transition",
            ],
        )
        for index in (0, 4):
            dispatch = self.database.writes[index][1]
            reservation = self.database.writes[index + 1][1]
            outbox = self.database.writes[index + 2][1]
            transition = self.database.writes[index + 3][1]
            self.assertEqual(dispatch[9], self.now)
            self.assertEqual(dispatch[11], self.database.authorization_id)
            self.assertEqual(dispatch[15], retry_schedule_sha256(self.policy))
            self.assertEqual(reservation[0], dispatch[0])
            self.assertEqual(reservation[2], self.now)
            self.assertEqual(reservation[3], self.database.quota_receipt_id)
            self.assertEqual(outbox, (dispatch[0], self.now))
            self.assertEqual(transition, (dispatch[0], self.now))

    def test_missing_quota_fails_closed_without_loading_or_writing_work(self) -> None:
        self.database.quota_receipt = None

        result = self.repository.admit(
            [self._candidate()],
            policy=self.policy,
            provider_use=self.provider_use,
        )

        self.assertEqual(result.persisted, ())
        self.assertEqual(
            result.decision.blocked[0].reason,
            DispatchBlockReason.QUOTA_UNAVAILABLE,
        )
        self.assertEqual(self.database.writes, [])
        self.assertEqual(len(self.database.executions), 4)

    def test_all_historical_reservations_remain_outstanding(self) -> None:
        duplicate = self._candidate()
        self.database.reservations = [
            (duplicate.idempotency_key, 1, self.now - timedelta(minutes=4)),
            ("f" * 64, 6, self.now - timedelta(minutes=3)),
        ]

        result = self.repository.admit(
            [duplicate, self._candidate(1, cost=2)],
            policy=self.policy,
            provider_use=self.provider_use,
        )

        self.assertEqual(result.persisted, ())
        self.assertEqual(
            [blocked.reason for blocked in result.decision.blocked],
            [DispatchBlockReason.DUPLICATE, DispatchBlockReason.QUOTA_FLOOR],
        )
        self.assertEqual(result.decision.quota_remaining_after_reservations, 3)
        self.assertEqual(self.database.writes, [])

    def test_terminal_worker_activity_still_enforces_provider_spacing(self) -> None:
        self.database.latest_activity_at = self.now - timedelta(seconds=30)
        policy = replace(
            self.policy,
            min_request_interval=timedelta(minutes=1),
        )

        result = self.repository.admit(
            [self._candidate()],
            policy=policy,
            provider_use=self.provider_use,
        )

        self.assertEqual(result.persisted, ())
        self.assertEqual(
            result.decision.blocked[0].reason,
            DispatchBlockReason.RATE_SPACING,
        )

    def test_missing_authorization_is_a_safe_conflict(self) -> None:
        self.database.authorization = None

        with self.assertRaises(IngestionAdmissionRepositoryConflict) as caught:
            self.repository.admit(
                [self._candidate()],
                policy=self.policy,
                provider_use=self.provider_use,
            )

        self.assertEqual(
            str(caught.exception),
            "an effective provider use authorization is not available",
        )
        self.assertNotIn(self.database_url, str(caught.exception))
        self.assertEqual(self.database.writes, [])

    def test_zero_authorization_id_is_rejected_before_any_write(self) -> None:
        self.database.authorization = (UUID(int=0), "a" * 64)

        with self.assertRaisesRegex(
            IngestionAdmissionRepositoryUnavailable,
            "stored provider use authorization is invalid",
        ):
            self.repository.admit(
                [self._candidate()],
                policy=self.policy,
                provider_use=self.provider_use,
            )

        self.assertEqual(self.database.writes, [])

    def test_zero_quota_receipt_id_is_rejected_before_any_write(self) -> None:
        self.database.quota_receipt = (
            UUID(int=0),
            10,
            self.now - timedelta(minutes=1),
        )

        with self.assertRaisesRegex(
            IngestionAdmissionRepositoryUnavailable,
            "stored provider quota observation is invalid",
        ):
            self.repository.admit(
                [self._candidate()],
                policy=self.policy,
                provider_use=self.provider_use,
            )

        self.assertEqual(self.database.writes, [])

    def test_insert_failure_rolls_back_and_redacts_driver_details(self) -> None:
        self.database.fail_once(
            "INSERT INTO ingestion_quota_reservation",
            message=f"leaked {self.database_url} and provider secret",
        )

        with self.assertRaises(IngestionAdmissionRepositoryUnavailable) as caught:
            self.repository.admit(
                [self._candidate()],
                policy=self.policy,
                provider_use=self.provider_use,
            )

        self.assertEqual(
            str(caught.exception),
            "ingestion admission database evaluation failed",
        )
        self.assertNotIn(self.database_url, str(caught.exception))
        self.assertNotIn("provider secret", str(caught.exception))
        self.assertIsNone(caught.exception.__context__)
        self.assertEqual(self.database.writes, [])
        self.assertNotIn(self.database_url, repr(self.repository))

    def test_integrity_failure_is_reported_as_a_safe_conflict(self) -> None:
        self.database.fail_once(
            "INSERT INTO ingestion_dispatch (",
            message=f"duplicate leaked {self.database_url}",
            sqlstate="23505",
        )

        with self.assertRaises(IngestionAdmissionRepositoryConflict) as caught:
            self.repository.admit(
                [self._candidate()],
                policy=self.policy,
                provider_use=self.provider_use,
            )

        self.assertEqual(
            str(caught.exception),
            "ingestion admission conflicts with stored facts",
        )
        self.assertIsNone(caught.exception.__context__)
        self.assertEqual(self.database.writes, [])

    def test_enabled_admission_requires_one_exact_private_raw_use(self) -> None:
        invalid_use = replace(self.provider_use, exposure="derived")

        with self.assertRaisesRegex(DispatchValidationError, "private raw"):
            self.repository.admit(
                [self._candidate()],
                policy=self.policy,
                provider_use=invalid_use,
            )

        self.assertEqual(self.database.supplied_urls, [])

    def test_repository_remains_unwired_from_every_runtime_entrypoint(self) -> None:
        project = Path(__file__).resolve().parents[1]
        for relative_path in (
            "worker.py",
            "provider_worker.py",
            "app.py",
            "main.py",
            "routes/api.py",
            "routes/views.py",
        ):
            with self.subTest(relative_path=relative_path):
                source = (project / relative_path).read_text(encoding="utf-8")
                self.assertNotIn("ingestion_admission_repository", source)
                self.assertNotIn("PostgresIngestionAdmissionRepository", source)


if __name__ == "__main__":
    unittest.main()
