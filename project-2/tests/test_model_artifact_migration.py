"""Contract tests for immutable, database-verified model artifacts."""

from __future__ import annotations

import unittest
from pathlib import Path

from sam_analytics.migrate import discover_migrations


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


if __name__ == "__main__":
    unittest.main()
