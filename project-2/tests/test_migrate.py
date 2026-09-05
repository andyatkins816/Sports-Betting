"""Unit tests for the credential-safe PostgreSQL migration runner."""

from __future__ import annotations

import unittest
from contextlib import AbstractContextManager
from pathlib import Path
from tempfile import TemporaryDirectory

from sam_analytics.migrate import (
    DatabaseMigrationError,
    MigrationIntegrityError,
    MigrationLockError,
    MigrationTransactionControlError,
    apply_migrations,
    discover_migrations,
    run_migrations,
)


class _FakeTransaction(AbstractContextManager[None]):
    def __init__(self, connection: "_FakeConnection"):
        self.connection = connection

    def __enter__(self):
        self.connection.transaction_depth += 1
        self.connection.transaction_count += 1
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        self.connection.transaction_depth -= 1
        return False


class _FakeCursor:
    def __init__(self, connection: "_FakeConnection"):
        self.connection = connection
        self._rows = []
        self._row = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def execute(self, query, params=None):
        normalized = " ".join(query.split())
        self.connection.executions.append((normalized, params, self.connection.transaction_depth))
        if "pg_try_advisory_lock" in normalized:
            self._row = (self.connection.lock_available,)
        elif normalized.startswith("SELECT version, filename, checksum_sha256"):
            self._rows = list(self.connection.applied_rows)
        elif normalized.startswith("INSERT INTO sam_schema_migrations"):
            self.connection.applied_rows.append(tuple(params))

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._row


class _FakeConnection:
    def __init__(self, *, applied_rows=(), lock_available=True):
        self.applied_rows = list(applied_rows)
        self.lock_available = lock_available
        self.executions = []
        self.transaction_depth = 0
        self.transaction_count = 0
        self.closed = False

    def cursor(self):
        return _FakeCursor(self)

    def transaction(self):
        return _FakeTransaction(self)

    def commit(self):
        pass

    def rollback(self):
        pass

    def close(self):
        self.closed = True


class MigrationRunnerTests(unittest.TestCase):
    def _migrations_dir(self, files):
        temporary = TemporaryDirectory()
        directory = Path(temporary.name)
        for filename, sql in files.items():
            (directory / filename).write_text(sql, encoding="utf-8")
        self.addCleanup(temporary.cleanup)
        return directory

    def test_discovers_numbered_files_in_numeric_order_and_ignores_function_body_begin(self):
        directory = self._migrations_dir(
            {
                "010_second.sql": "CREATE TABLE second_table (id integer);",
                "001_first.sql": """
                    CREATE OR REPLACE FUNCTION audit_example() RETURNS trigger AS $$
                    BEGIN
                        RETURN NEW;
                    END;
                    $$ LANGUAGE plpgsql;
                """,
            }
        )

        migrations = discover_migrations(directory)

        self.assertEqual([migration.version for migration in migrations], ["001", "010"])
        self.assertTrue(all(len(migration.checksum_sha256) == 64 for migration in migrations))

    def test_rejects_top_level_transaction_control(self):
        directory = self._migrations_dir(
            {"001_initial.sql": "BEGIN; CREATE TABLE should_not_run (id integer); COMMIT;"}
        )

        with self.assertRaises(MigrationTransactionControlError):
            discover_migrations(directory)

    def test_ledgered_migrations_apply_once_with_each_execute_inside_a_transaction(self):
        directory = self._migrations_dir(
            {
                "001_initial.sql": "CREATE TABLE first_table (id integer);",
                "002_next.sql": "CREATE TABLE second_table (id integer);",
            }
        )
        connection = _FakeConnection()

        first_result = apply_migrations(connection, directory)
        second_result = apply_migrations(connection, directory)

        self.assertEqual(first_result.applied_versions, ("001", "002"))
        self.assertEqual(second_result.applied_versions, ())
        self.assertEqual([row[0] for row in connection.applied_rows], ["001", "002"])
        migration_sql = [
            (query, transaction_depth)
            for query, _, transaction_depth in connection.executions
            if query.startswith("CREATE TABLE first_table") or query.startswith("CREATE TABLE second_table")
        ]
        self.assertEqual(len(migration_sql), 2)
        self.assertTrue(all(transaction_depth == 1 for _, transaction_depth in migration_sql))

    def test_changed_applied_checksum_fails_before_executing_migration_sql(self):
        directory = self._migrations_dir(
            {"001_initial.sql": "CREATE TABLE first_table (id integer);"}
        )
        connection = _FakeConnection(
            applied_rows=(
                ("001", "001_initial.sql", "0" * 64),
            )
        )

        with self.assertRaises(MigrationIntegrityError):
            apply_migrations(connection, directory)

        self.assertFalse(
            any(query.startswith("CREATE TABLE first_table") for query, _, _ in connection.executions)
        )

    def test_missing_earlier_migration_is_not_backfilled_after_a_later_ledger_row(self):
        directory = self._migrations_dir(
            {
                "001_initial.sql": "CREATE TABLE first_table (id integer);",
                "002_next.sql": "CREATE TABLE second_table (id integer);",
            }
        )
        later = discover_migrations(directory)[1]
        connection = _FakeConnection(
            applied_rows=(("002", later.filename, later.checksum_sha256),)
        )

        with self.assertRaises(MigrationIntegrityError):
            apply_migrations(connection, directory)

        self.assertFalse(
            any(query.startswith("CREATE TABLE first_table") for query, _, _ in connection.executions)
        )

    def test_contention_fails_before_creating_ledger_or_running_migrations(self):
        directory = self._migrations_dir(
            {"001_initial.sql": "CREATE TABLE first_table (id integer);"}
        )
        connection = _FakeConnection(lock_available=False)

        with self.assertRaises(MigrationLockError):
            apply_migrations(connection, directory)

        self.assertEqual(len(connection.executions), 1)
        self.assertIn("pg_try_advisory_lock", connection.executions[0][0])

    def test_run_migrations_closes_connection_without_exposing_the_dsn(self):
        directory = self._migrations_dir(
            {"001_initial.sql": "CREATE TABLE first_table (id integer);"}
        )
        connection = _FakeConnection()
        database_url = "postgresql://sam:credential-that-must-not-be-logged@db.example/sam"

        result = run_migrations(
            database_url,
            directory,
            connection_factory=lambda supplied_url: self._connection_for_url(
                supplied_url, database_url, connection
            ),
        )

        self.assertEqual(result.applied_versions, ("001",))
        self.assertTrue(connection.closed)

    def test_connection_error_is_replaced_with_a_credential_safe_message(self):
        directory = self._migrations_dir(
            {"001_initial.sql": "CREATE TABLE first_table (id integer);"}
        )
        database_url = "postgresql://sam:credential-that-must-not-be-logged@db.example/sam"

        with self.assertRaises(DatabaseMigrationError) as context:
            run_migrations(
                database_url,
                directory,
                connection_factory=lambda _: self._raise_connection_error(database_url),
            )

        self.assertNotIn(database_url, str(context.exception))

    def test_checked_in_provider_receipt_migration_has_an_immutable_audit_path(self):
        migrations_dir = Path(__file__).resolve().parents[1] / "migrations"
        migration = next(
            item
            for item in discover_migrations(migrations_dir)
            if item.filename == "004_provider_payload_receipts.sql"
        )

        self.assertIn("CREATE TABLE provider_payload_receipt", migration.sql)
        self.assertIn("license_version TEXT NOT NULL", migration.sql)
        self.assertIn("provider_payload_receipt_id UUID", migration.sql)
        self.assertIn("raw_data_provenance_receipt_required_for_new_records", migration.sql)
        self.assertIn("ADD COLUMN bookmaker TEXT", migration.sql)
        self.assertIn("ADD COLUMN primary_provenance_id UUID", migration.sql)
        self.assertIn("DROP CONSTRAINT IF EXISTS odds_snapshot_provider_provider_quote_id_captured_at_key", migration.sql)
        self.assertIn("odds_snapshot_provider_evidence_required_for_new_records", migration.sql)
        self.assertIn("enforce_raw_provenance_receipt_integrity", migration.sql)
        self.assertIn("enforce_odds_snapshot_primary_provenance", migration.sql)
        self.assertIn("sports_event_append_only", migration.sql)
        self.assertIn("provider_payload_receipt_append_only", migration.sql)

    def _connection_for_url(self, supplied_url, expected_url, connection):
        self.assertEqual(supplied_url, expected_url)
        return connection

    def _raise_connection_error(self, database_url):
        raise RuntimeError(f"database failed for {database_url}")


if __name__ == "__main__":
    unittest.main()
