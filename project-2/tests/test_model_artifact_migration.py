"""Contract tests for immutable, database-verified model artifacts."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import unittest
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

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


class ModelArtifactMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        migrations_dir = Path(__file__).resolve().parents[1] / "migrations"
        cls.migration = next(
            migration
            for migration in discover_migrations(migrations_dir)
            if migration.filename == "009_model_artifact.sql"
        )
        cls.sql = cls.migration.sql
        cls.normalized = " ".join(cls.sql.split())

    def test_adds_only_the_artifact_format_and_bytes_columns(self) -> None:
        self.assertIn("ADD COLUMN artifact_format TEXT", self.normalized)
        self.assertIn("ADD COLUMN artifact_bytes BYTEA", self.normalized)
        self.assertNotIn("CREATE TABLE", self.sql.upper())
        self.assertNotIn("INSERT INTO", self.sql.upper())

    def test_new_records_require_a_complete_artifact(self) -> None:
        self.assertIn(
            "model_registry_artifact_required_for_new_records CHECK ( "
            "artifact_format IS NOT NULL AND length(btrim(artifact_format)) > 0 "
            "AND artifact_bytes IS NOT NULL AND artifact_sha256 IS NOT NULL ) NOT VALID",
            self.normalized,
        )

    def test_postgres_verifies_the_artifact_sha256_without_rewriting_history(self) -> None:
        self.assertIn(
            "model_registry_artifact_sha256_verified_for_new_records CHECK",
            self.normalized,
        )
        self.assertIn(
            "artifact_sha256 = encode(public.digest(artifact_bytes, 'sha256'), 'hex')",
            self.normalized,
        )
        self.assertEqual(self.normalized.upper().count(") NOT VALID"), 2)
        self.assertNotIn("CREATE EXTENSION", self.sql.upper())

    def test_artifact_fields_cannot_be_changed_after_registration(self) -> None:
        self.assertIn("forbid_model_artifact_mutation", self.sql)
        self.assertIn(
            "BEFORE UPDATE OF artifact_format, artifact_bytes, artifact_sha256 "
            "ON model_registry",
            self.normalized,
        )
        for field in ("artifact_format", "artifact_bytes", "artifact_sha256"):
            with self.subTest(field=field):
                self.assertIn(
                    f"OLD.{field} IS DISTINCT FROM NEW.{field}",
                    self.normalized,
                )


@unittest.skipUnless(
    _PSYCOPG_AVAILABLE and _is_disposable_database_url(_DATABASE_URL),
    "requires the disposable loopback PostgreSQL test database",
)
class ModelArtifactPostgresTests(unittest.TestCase):
    @staticmethod
    def _candidate(version: str, artifact: bytes, artifact_sha256: str) -> tuple[object, ...]:
        return (
            version,
            "baseball_mlb",
            "home_team_wins",
            "a" * 64,
            f"postgresql:model_registry/{version}",
            artifact_sha256,
            "sam-joblib-envelope-v1",
            artifact,
        )

    @staticmethod
    def _insert_candidate(connection, values: tuple[object, ...]):
        return connection.execute(
            """
            INSERT INTO model_registry (
                version, sport, target_definition, feature_contract_sha256,
                artifact_uri, artifact_sha256, artifact_format, artifact_bytes,
                training_data_cutoff, validation_report, approval_status
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s,
                '2026-09-06 12:00:00+00', '{}'::jsonb, 'candidate'
            )
            ON CONFLICT (version) DO NOTHING
            RETURNING id
            """,
            values,
        ).fetchone()

    def test_valid_artifact_persists_and_exact_replay_is_idempotent(self) -> None:
        import psycopg

        artifact = b"immutable-ci-model-artifact"
        artifact_sha256 = hashlib.sha256(artifact).hexdigest()
        version = f"ci-model-artifact-{uuid4().hex}"
        values = self._candidate(version, artifact, artifact_sha256)

        with psycopg.connect(_DATABASE_URL) as connection:
            with connection.transaction():
                inserted = self._insert_candidate(connection, values)

            with connection.transaction():
                replayed = self._insert_candidate(connection, values)

            stored = connection.execute(
                """
                SELECT artifact_format, artifact_bytes, artifact_sha256
                FROM model_registry
                WHERE version = %s
                """,
                (version,),
            ).fetchone()

        self.assertIsNotNone(inserted)
        self.assertIsNone(replayed)
        self.assertEqual(stored, ("sam-joblib-envelope-v1", artifact, artifact_sha256))

    def test_database_rejects_bad_digest_and_artifact_mutation(self) -> None:
        import psycopg

        artifact = b"immutable-ci-model-artifact"
        artifact_sha256 = hashlib.sha256(artifact).hexdigest()
        version = f"ci-model-artifact-{uuid4().hex}"

        with psycopg.connect(_DATABASE_URL) as connection:
            with connection.transaction():
                self._insert_candidate(
                    connection,
                    self._candidate(version, artifact, artifact_sha256),
                )

            with self.assertRaisesRegex(
                psycopg.Error,
                "model_registry_artifact_sha256_verified_for_new_records",
            ):
                with connection.transaction():
                    self._insert_candidate(
                        connection,
                        self._candidate(
                            f"ci-model-artifact-{uuid4().hex}",
                            artifact,
                            "0" * 64,
                        ),
                    )

            with self.assertRaisesRegex(
                psycopg.Error,
                "registered model artifact fields are immutable",
            ):
                with connection.transaction():
                    connection.execute(
                        """
                        UPDATE model_registry
                        SET artifact_format = 'changed-format'
                        WHERE version = %s
                        """,
                        (version,),
                    )

            stored_format = connection.execute(
                "SELECT artifact_format FROM model_registry WHERE version = %s",
                (version,),
            ).fetchone()

        self.assertEqual(stored_format, ("sam-joblib-envelope-v1",))


if __name__ == "__main__":
    unittest.main()
