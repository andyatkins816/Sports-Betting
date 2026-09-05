"""Tests for the exact synthetic staging worker admission boundary."""

from __future__ import annotations

import unittest
from dataclasses import fields

from sam_analytics.worker_settings import (
    SYNTHETIC_STAGING_EVIDENCE_URI,
    SYNTHETIC_STAGING_MAX_BYTES,
    PrivateWorkerConfigurationError,
    PrivateWorkerSettings,
)


class PrivateWorkerSettingsTests(unittest.TestCase):
    @staticmethod
    def _environment() -> dict[str, str]:
        return {
            "APP_ENV": "staging",
            "DATABASE_URL": "postgresql://sam:opaque@dpg-sam-test-a/sam",
            "REDIS_URL": "redis://red-sam-test:6379/0",
            "SAM_WORKER_ROLE": "private_ingestion",
            "SAM_WORKER_MODE": "synthetic_storage_probe",
            "SAM_INGESTION_ENABLED": "false",
            "SAM_RAW_EVIDENCE_STORE_BACKEND": "cloudflare_r2",
            "SAM_RAW_EVIDENCE_STORE_URI": SYNTHETIC_STAGING_EVIDENCE_URI,
            "SAM_RAW_EVIDENCE_S3_REGION": "auto",
            "SAM_RAW_EVIDENCE_S3_ENDPOINT_URL": (
                "https://0123456789abcdef0123456789abcdef.r2.cloudflarestorage.com"
            ),
            "SAM_RAW_EVIDENCE_MAX_BYTES": str(SYNTHETIC_STAGING_MAX_BYTES),
            "SAM_RAW_EVIDENCE_S3_ACCESS_KEY_ID": "scoped-r2-access-id",
            "SAM_RAW_EVIDENCE_S3_SECRET_ACCESS_KEY": "scoped-r2-secret",
        }

    def test_accepts_only_credential_free_synthetic_staging_description(self) -> None:
        environment = self._environment()

        settings = PrivateWorkerSettings.from_environment(environment)

        self.assertEqual(settings.environment, "staging")
        self.assertEqual(settings.role, "private_ingestion")
        self.assertEqual(settings.mode, "synthetic_storage_probe")
        self.assertEqual(settings.raw_evidence_store_backend, "cloudflare_r2")
        self.assertEqual(
            settings.raw_evidence_store_uri,
            "s3://sam-raw-evidence-staging/raw/synthetic",
        )
        self.assertEqual(settings.raw_evidence_s3_region, "auto")
        self.assertEqual(settings.raw_evidence_max_bytes, 1024 * 1024)

        retained_field_names = {field.name for field in fields(settings)}
        self.assertNotIn("database_url", retained_field_names)
        self.assertNotIn("redis_url", retained_field_names)
        self.assertNotIn("access_key_id", retained_field_names)
        self.assertNotIn("secret_access_key", retained_field_names)
        rendered = repr(settings)
        for private_value in (
            environment["DATABASE_URL"],
            environment["REDIS_URL"],
            environment["SAM_RAW_EVIDENCE_S3_ACCESS_KEY_ID"],
            environment["SAM_RAW_EVIDENCE_S3_SECRET_ACCESS_KEY"],
        ):
            self.assertNotIn(private_value, rendered)

    def test_live_ingestion_switch_may_only_be_absent_or_literal_false(self) -> None:
        without_switch = self._environment()
        without_switch.pop("SAM_INGESTION_ENABLED")
        PrivateWorkerSettings.from_environment(without_switch)

        for unsafe_value in ("true", "True", "yes", "0", " false "):
            with self.subTest(unsafe_value=unsafe_value):
                environment = self._environment()
                environment["SAM_INGESTION_ENABLED"] = unsafe_value
                with self.assertRaisesRegex(
                    PrivateWorkerConfigurationError, "live ingestion"
                ):
                    PrivateWorkerSettings.from_environment(environment)

    def test_rejects_any_boundary_value_other_than_the_exact_staging_probe(self) -> None:
        changes = (
            ("APP_ENV", "production"),
            ("APP_ENV", "development"),
            ("SAM_WORKER_ROLE", "worker"),
            ("SAM_WORKER_MODE", "provider_shadow"),
            ("SAM_RAW_EVIDENCE_STORE_BACKEND", "aws_s3"),
            (
                "SAM_RAW_EVIDENCE_STORE_URI",
                "s3://sam-raw-evidence-prod/raw/synthetic",
            ),
            (
                "SAM_RAW_EVIDENCE_STORE_URI",
                "s3://sam-raw-evidence-staging/raw/odds",
            ),
            ("SAM_RAW_EVIDENCE_S3_REGION", "us-west-2"),
            ("SAM_RAW_EVIDENCE_MAX_BYTES", "20971520"),
            ("SAM_RAW_EVIDENCE_MAX_BYTES", "01048576"),
            ("SAM_RAW_EVIDENCE_MAX_BYTES", "+1048576"),
        )
        for name, value in changes:
            with self.subTest(name=name, value=value):
                environment = self._environment()
                environment[name] = value
                with self.assertRaises(PrivateWorkerConfigurationError):
                    PrivateWorkerSettings.from_environment(environment)

        for missing_name in (
            "APP_ENV",
            "SAM_WORKER_ROLE",
            "SAM_WORKER_MODE",
            "SAM_RAW_EVIDENCE_STORE_BACKEND",
            "SAM_RAW_EVIDENCE_STORE_URI",
            "SAM_RAW_EVIDENCE_S3_REGION",
            "SAM_RAW_EVIDENCE_MAX_BYTES",
        ):
            with self.subTest(missing_name=missing_name):
                environment = self._environment()
                environment.pop(missing_name)
                with self.assertRaises(PrivateWorkerConfigurationError):
                    PrivateWorkerSettings.from_environment(environment)

    def test_rejects_non_r2_or_credential_bearing_endpoint_without_echoing_it(self) -> None:
        endpoints = (
            "http://0123456789abcdef0123456789abcdef.r2.cloudflarestorage.com",
            "https://example.invalid",
            "https://0123456789abcdef0123456789abcdef.r2.cloudflarestorage.com/bucket",
            "https://0123456789abcdef0123456789abcdef.r2.cloudflarestorage.com?token=sentinel",
            "https://user:sentinel@0123456789abcdef0123456789abcdef.r2.cloudflarestorage.com",
            "https://0123456789abcdef0123456789abcdef.r2.cloudflarestorage.com:bad",
        )
        for endpoint in endpoints:
            with self.subTest(endpoint=endpoint):
                environment = self._environment()
                environment["SAM_RAW_EVIDENCE_S3_ENDPOINT_URL"] = endpoint
                with self.assertRaises(PrivateWorkerConfigurationError) as raised:
                    PrivateWorkerSettings.from_environment(environment)
                self.assertNotIn(endpoint, str(raised.exception))
                self.assertNotIn("sentinel", str(raised.exception))
                self.assertIsNone(raised.exception.__context__)

    def test_rejects_missing_or_invalid_private_dependencies_without_echoing_values(self) -> None:
        changes = (
            ("DATABASE_URL", ""),
            ("DATABASE_URL", "https://sentinel.invalid/database"),
            ("REDIS_URL", ""),
            ("REDIS_URL", "amqp://sentinel.invalid/queue"),
            ("SAM_RAW_EVIDENCE_S3_ACCESS_KEY_ID", ""),
            ("SAM_RAW_EVIDENCE_S3_SECRET_ACCESS_KEY", ""),
            ("SAM_RAW_EVIDENCE_S3_SECRET_ACCESS_KEY", "sentinel\nsecret"),
        )
        for name, value in changes:
            with self.subTest(name=name):
                environment = self._environment()
                environment[name] = value
                with self.assertRaises(PrivateWorkerConfigurationError) as raised:
                    PrivateWorkerSettings.from_environment(environment)
                if value:
                    self.assertNotIn(value, str(raised.exception))
                self.assertNotIn("sentinel", str(raised.exception))

    def test_rejects_provider_public_base44_and_ambient_cloud_settings(self) -> None:
        forbidden_names = (
            "ODDS_PROVIDER",
            "ODDS_PROVIDER_API_KEY",
            "NFL_API_KEY",
            "OPENAI_API_KEY",
            "RESULTS_PROVIDER",
            "RESULTS_PROVIDER_API_KEY",
            "SESSION_SECRET",
            "SAM_API_KEY",
            "SAM_STATUS_API_KEY",
            "ALLOWED_ORIGINS",
            "BASE44_EVIDENCE_WEBHOOK_URL",
            "BASE44_EVIDENCE_WEBHOOK_TOKEN",
            "BASE44_EVIDENCE_WEBHOOK_HOST",
            "AWS_ACCESS_KEY_ID",
            "PGHOSTADDR",
            "PGHOST",
            "PGPORT",
            "PGDATABASE",
            "PGUSER",
            "PGPASSWORD",
            "PGPASSFILE",
            "PGSERVICE",
            "PGSERVICEFILE",
            "PGOPTIONS",
            "PGSSLMODE",
            "PGSSLROOTCERT",
            "CELERY_BROKER_URL",
            "CELERY_BROKER_READ_URL",
            "CELERY_BROKER_WRITE_URL",
            "CELERY_RESULT_BACKEND",
            "CELERY_CONFIG_MODULE",
            "CELERY_LOADER",
            "CELERY_DUMMY_PROXY",
            "CELERY_LOG_LEVEL",
            "CELERY_LOG_REDIRECT",
            "CELERY_LOG_REDIRECT_LEVEL",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_SESSION_TOKEN",
            "AWS_PROFILE",
            "AWS_DEFAULT_PROFILE",
            "AWS_CONFIG_FILE",
            "AWS_SHARED_CREDENTIALS_FILE",
            "AWS_ROLE_ARN",
            "AWS_ROLE_SESSION_NAME",
            "AWS_WEB_IDENTITY_TOKEN_FILE",
            "AWS_CONTAINER_CREDENTIALS_FULL_URI",
            "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
            "AWS_CONTAINER_AUTHORIZATION_TOKEN",
            "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
            "CLOUDFLARE_API_TOKEN",
            "CLOUDFLARE_API_KEY",
            "CLOUDFLARE_EMAIL",
            "CF_API_TOKEN",
            "CF_API_KEY",
            "CLOUDFLARE_API_USER_SERVICE_KEY",
            "CF_API_USER_SERVICE_KEY",
            "CF_API_EMAIL",
        )
        sentinel = "private-sentinel-value"
        for name in forbidden_names:
            with self.subTest(name=name):
                environment = self._environment()
                environment[name] = sentinel
                with self.assertRaises(PrivateWorkerConfigurationError) as raised:
                    PrivateWorkerSettings.from_environment(environment)
                self.assertNotIn(sentinel, str(raised.exception))
                self.assertIsNone(raised.exception.__context__)

    def test_admits_only_known_celery_worker_runtime_markers(self) -> None:
        environment = self._environment()
        environment.update(
            {
                "CELERY_LOG_LEVEL": "20",
                "CELERY_LOG_REDIRECT": "1",
                "CELERY_LOG_REDIRECT_LEVEL": "WARNING",
                "celery_dummy_proxy": "set_by_celeryd",
            }
        )

        settings = PrivateWorkerSettings.from_environment(environment)

        self.assertEqual(settings.environment, "staging")

        for name in (
            "CELERY_LOG_LEVEL",
            "CELERY_LOG_REDIRECT",
            "CELERY_LOG_REDIRECT_LEVEL",
            "celery_dummy_proxy",
        ):
            with self.subTest(name=name):
                changed = dict(environment)
                changed[name] = "operator-supplied"
                with self.assertRaises(PrivateWorkerConfigurationError):
                    PrivateWorkerSettings.from_environment(changed)

    def test_rejects_new_secret_like_names_without_needing_an_allowlist_update(self) -> None:
        sentinel = "private-sentinel-value"
        for name in (
            "NEW_PROVIDER_API_KEY",
            "RAPIDAPI_KEY",
            "X_RAPIDAPI_KEY",
            "CUSTOM_BEARER_TOKEN",
            "BROKER_PASSWORD",
            "SERVICE_AUTHORIZATION",
            "PRIVATE_CREDENTIAL",
            "SESSION_COOKIE_SECRET",
        ):
            with self.subTest(name=name):
                environment = self._environment()
                environment[name] = sentinel
                with self.assertRaises(PrivateWorkerConfigurationError) as raised:
                    PrivateWorkerSettings.from_environment(environment)
                self.assertNotIn(name, str(raised.exception))
                self.assertNotIn(sentinel, str(raised.exception))

    def test_admits_only_the_named_scoped_r2_credential_pair(self) -> None:
        environment = self._environment()

        settings = PrivateWorkerSettings.from_environment(environment)

        self.assertEqual(settings.mode, "synthetic_storage_probe")

    def test_rejects_public_database_and_broker_hosts(self) -> None:
        changes = (
            ("DATABASE_URL", "postgresql://sam:opaque@dpg-sam-test-a.oregon-postgres.render.com/sam"),
            ("DATABASE_URL", "postgresql://sam:opaque@public-db.example/sam"),
            (
                "DATABASE_URL",
                "postgresql://sam:opaque@dpg-sam-test-a/sam?hostaddr=203.0.113.10",
            ),
            ("REDIS_URL", "rediss://default:opaque@oregon-redis.render.com:6379/0"),
            ("REDIS_URL", "redis://public-cache.example:6379/0"),
            ("REDIS_URL", "redis://red-sam-test:6379/0?host=public-cache.example"),
        )
        for name, value in changes:
            with self.subTest(name=name, value=value):
                environment = self._environment()
                environment[name] = value
                with self.assertRaises(PrivateWorkerConfigurationError) as raised:
                    PrivateWorkerSettings.from_environment(environment)
                self.assertNotIn(value, str(raised.exception))

    def test_allows_explicit_local_and_compose_service_hosts_for_verification(self) -> None:
        environment = self._environment()
        environment["DATABASE_URL"] = "postgresql://sam:opaque@postgres:5432/sam"
        environment["REDIS_URL"] = "redis://redis:6379/0"
        PrivateWorkerSettings.from_environment(environment)

        environment["DATABASE_URL"] = "postgresql://sam:opaque@127.0.0.1:5432/sam"
        environment["REDIS_URL"] = "redis://localhost:6379/0"
        PrivateWorkerSettings.from_environment(environment)

    def test_empty_forbidden_settings_do_not_create_an_authority(self) -> None:
        environment = self._environment()
        environment.update(
            {
                "ODDS_PROVIDER_API_KEY": "",
                "SESSION_SECRET": "",
                "BASE44_EVIDENCE_WEBHOOK_TOKEN": "",
                "AWS_ACCESS_KEY_ID": "",
            }
        )

        settings = PrivateWorkerSettings.from_environment(environment)

        self.assertEqual(settings.mode, "synthetic_storage_probe")


if __name__ == "__main__":
    unittest.main()
