"""Tests for the sealed The Odds API provider-shadow configuration."""

from __future__ import annotations

import unittest
from dataclasses import fields

from sam_analytics.provider_shadow_settings import (
    PROVIDER_SHADOW_ADMISSION_RUN_ID,
    PROVIDER_SHADOW_ALLOWED_SPORT_KEYS,
    PROVIDER_SHADOW_EVIDENCE_URI,
    PROVIDER_SHADOW_LICENSE_SCOPE,
    PROVIDER_SHADOW_LICENSE_VERSION,
    PROVIDER_SHADOW_MARKETS,
    PROVIDER_SHADOW_MAX_BYTES,
    PROVIDER_SHADOW_REGIONS,
    ProviderShadowConfigurationError,
    ProviderShadowSettings,
)


class ProviderShadowSettingsTests(unittest.TestCase):
    @staticmethod
    def _environment() -> dict[str, str]:
        return {
            "APP_ENV": "staging",
            "DATABASE_URL": "postgresql://sam:opaque@dpg-sam-test-a/sam",
            "REDIS_URL": "redis://red-sam-test:6379/0",
            "SAM_WORKER_ROLE": "private_ingestion",
            "SAM_WORKER_MODE": "provider_shadow",
            "SAM_INGESTION_ENABLED": "true",
            "ODDS_PROVIDER": "the_odds_api",
            "ODDS_PROVIDER_API_KEY": "scoped-provider-key",
            "SAM_ODDS_SPORT_KEY": "basketball_nba",
            "SAM_ODDS_REGIONS": PROVIDER_SHADOW_REGIONS,
            "SAM_ODDS_MARKETS": PROVIDER_SHADOW_MARKETS,
            "SAM_PROVIDER_LICENSE_SCOPE": PROVIDER_SHADOW_LICENSE_SCOPE,
            "SAM_PROVIDER_LICENSE_VERSION": PROVIDER_SHADOW_LICENSE_VERSION,
            "SAM_RAW_EVIDENCE_STORE_BACKEND": "cloudflare_r2",
            "SAM_RAW_EVIDENCE_STORE_URI": PROVIDER_SHADOW_EVIDENCE_URI,
            "SAM_RAW_EVIDENCE_S3_REGION": "auto",
            "SAM_RAW_EVIDENCE_S3_ENDPOINT_URL": (
                "https://0123456789abcdef0123456789abcdef.r2.cloudflarestorage.com"
            ),
            "SAM_RAW_EVIDENCE_MAX_BYTES": str(PROVIDER_SHADOW_MAX_BYTES),
            "SAM_RAW_EVIDENCE_S3_ACCESS_KEY_ID": "scoped-r2-access-id",
            "SAM_RAW_EVIDENCE_S3_SECRET_ACCESS_KEY": "scoped-r2-secret",
        }

    def test_accepts_only_the_bounded_non_secret_shadow_description(self) -> None:
        environment = self._environment()

        settings = ProviderShadowSettings.from_environment(environment)

        self.assertEqual(settings.environment, "staging")
        self.assertEqual(settings.role, "private_ingestion")
        self.assertEqual(settings.mode, "provider_shadow")
        self.assertEqual(settings.provider, "the_odds_api")
        self.assertEqual(settings.sport_key, "basketball_nba")
        self.assertEqual(settings.regions, ("us",))
        self.assertEqual(settings.markets, ("h2h",))
        self.assertEqual(settings.license_scope, "internal_analytics_only")
        self.assertEqual(settings.license_version, "terms-2026-08-31")
        self.assertEqual(
            str(PROVIDER_SHADOW_ADMISSION_RUN_ID),
            "f3cd3650-568a-4f36-89b8-acde937c23a1",
        )
        self.assertEqual(
            settings.raw_evidence_store_uri,
            "s3://sam-raw-evidence-staging/raw/the_odds_api",
        )
        self.assertEqual(settings.raw_evidence_max_bytes, 10 * 1024 * 1024)

        retained_field_names = {field.name for field in fields(settings)}
        for secret_field in (
            "database_url",
            "redis_url",
            "provider_api_key",
            "access_key_id",
            "secret_access_key",
        ):
            self.assertNotIn(secret_field, retained_field_names)
        rendered = repr(settings)
        for private_value in (
            environment["DATABASE_URL"],
            environment["REDIS_URL"],
            environment["ODDS_PROVIDER_API_KEY"],
            environment["SAM_RAW_EVIDENCE_S3_ACCESS_KEY_ID"],
            environment["SAM_RAW_EVIDENCE_S3_SECRET_ACCESS_KEY"],
        ):
            self.assertNotIn(private_value, rendered)

    def test_every_capability_and_request_scope_value_is_exact(self) -> None:
        changes = (
            ("APP_ENV", "production"),
            ("APP_ENV", "development"),
            ("SAM_WORKER_ROLE", "worker"),
            ("SAM_WORKER_MODE", "synthetic_storage_probe"),
            ("SAM_INGESTION_ENABLED", "false"),
            ("SAM_INGESTION_ENABLED", "True"),
            ("ODDS_PROVIDER", "another_provider"),
            ("SAM_ODDS_REGIONS", "us,eu"),
            ("SAM_ODDS_MARKETS", "spreads"),
            ("SAM_PROVIDER_LICENSE_SCOPE", "public_display"),
            ("SAM_PROVIDER_LICENSE_VERSION", "terms-unreviewed"),
            ("SAM_RAW_EVIDENCE_STORE_BACKEND", "aws_s3"),
            (
                "SAM_RAW_EVIDENCE_STORE_URI",
                "s3://sam-raw-evidence-prod/raw/the_odds_api",
            ),
            (
                "SAM_RAW_EVIDENCE_STORE_URI",
                "s3://sam-raw-evidence-staging/raw/synthetic",
            ),
            ("SAM_RAW_EVIDENCE_S3_REGION", "us-west-2"),
            ("SAM_RAW_EVIDENCE_MAX_BYTES", "20971520"),
            ("SAM_RAW_EVIDENCE_MAX_BYTES", "010485760"),
        )
        for name, value in changes:
            with self.subTest(name=name, value=value):
                environment = self._environment()
                environment[name] = value
                with self.assertRaises(ProviderShadowConfigurationError):
                    ProviderShadowSettings.from_environment(environment)

        required_names = (
            "APP_ENV",
            "DATABASE_URL",
            "REDIS_URL",
            "SAM_WORKER_ROLE",
            "SAM_WORKER_MODE",
            "SAM_INGESTION_ENABLED",
            "ODDS_PROVIDER",
            "SAM_ODDS_SPORT_KEY",
            "SAM_ODDS_REGIONS",
            "SAM_ODDS_MARKETS",
            "SAM_PROVIDER_LICENSE_SCOPE",
            "SAM_PROVIDER_LICENSE_VERSION",
            "SAM_RAW_EVIDENCE_STORE_BACKEND",
            "SAM_RAW_EVIDENCE_STORE_URI",
            "SAM_RAW_EVIDENCE_S3_REGION",
            "SAM_RAW_EVIDENCE_S3_ENDPOINT_URL",
            "SAM_RAW_EVIDENCE_MAX_BYTES",
        )
        for name in required_names:
            with self.subTest(missing_name=name):
                environment = self._environment()
                environment.pop(name)
                with self.assertRaises(ProviderShadowConfigurationError):
                    ProviderShadowSettings.from_environment(environment)

    def test_accepts_only_one_of_the_three_reviewed_sports(self) -> None:
        self.assertEqual(
            PROVIDER_SHADOW_ALLOWED_SPORT_KEYS,
            frozenset({"americanfootball_nfl", "baseball_mlb", "basketball_nba"}),
        )
        for sport_key in PROVIDER_SHADOW_ALLOWED_SPORT_KEYS:
            with self.subTest(sport_key=sport_key):
                environment = self._environment()
                environment["SAM_ODDS_SPORT_KEY"] = sport_key
                settings = ProviderShadowSettings.from_environment(environment)
                self.assertEqual(settings.sport_key, sport_key)
                self.assertEqual(settings.regions, ("us",))
                self.assertEqual(settings.markets, ("h2h",))

        for sport_key in (
            "icehockey_nhl",
            "Basketball_nba",
            "basketball_nba,baseball_mlb",
            " basketball_nba",
            "",
        ):
            with self.subTest(rejected_sport_key=sport_key):
                environment = self._environment()
                environment["SAM_ODDS_SPORT_KEY"] = sport_key
                with self.assertRaises(ProviderShadowConfigurationError):
                    ProviderShadowSettings.from_environment(environment)

    def test_requires_only_the_three_named_credentials(self) -> None:
        for name in (
            "ODDS_PROVIDER_API_KEY",
            "SAM_RAW_EVIDENCE_S3_ACCESS_KEY_ID",
            "SAM_RAW_EVIDENCE_S3_SECRET_ACCESS_KEY",
        ):
            for invalid in (None, "", " leading", "trailing ", "line\nbreak"):
                with self.subTest(name=name, invalid=invalid):
                    environment = self._environment()
                    if invalid is None:
                        environment.pop(name)
                    else:
                        environment[name] = invalid
                    with self.assertRaises(ProviderShadowConfigurationError) as raised:
                        ProviderShadowSettings.from_environment(environment)
                    self.assertNotIn(name, str(raised.exception))
                    if invalid:
                        self.assertNotIn(invalid, str(raised.exception))

    def test_rejects_public_base44_results_scheduler_and_admin_authority(self) -> None:
        forbidden_names = (
            "SESSION_SECRET",
            "SAM_API_KEY",
            "SAM_STATUS_API_KEY",
            "ALLOWED_ORIGINS",
            "SAM_ADMIN_ENABLED",
            "PUBLIC_SERVICE_URL",
            "SESSION_COOKIE_NAME",
            "SAM_RESULT_WRITER_ENABLED",
            "BASE44_EVIDENCE_WEBHOOK_URL",
            "BASE44_EVIDENCE_WEBHOOK_TOKEN",
            "RESULTS_PROVIDER",
            "RESULTS_PROVIDER_API_KEY",
            "SAM_RESULTS_ENABLED",
            "SAM_SETTLEMENT_ENABLED",
            "SAM_SCHEDULER_ENABLED",
            "SAM_CRON_ENABLED",
            "SCHEDULER_INTERVAL",
            "CRON_SCHEDULE",
            "CELERY_BEAT_SCHEDULE",
            "CELERYBEAT_SCHEDULE_FILENAME",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
            "AWS_PROFILE",
            "CLOUDFLARE_API_TOKEN",
            "CLOUDFLARE_ACCOUNT_ID",
            "CF_API_TOKEN",
            "PGHOST",
            "PGPASSWORD",
            "CELERY_BROKER_URL",
            "NEW_PROVIDER_API_KEY",
            "R2_ACCESS_KEY_ID",
            "SSH_PRIVATE_KEY",
            "GITHUB_PAT",
            "BASIC_AUTH",
            "BROKER_PASSWORD",
            "SERVICE_AUTHORIZATION",
        )
        sentinel = "private-sentinel-value"
        for name in forbidden_names:
            with self.subTest(name=name):
                environment = self._environment()
                environment[name] = sentinel
                with self.assertRaises(ProviderShadowConfigurationError) as raised:
                    ProviderShadowSettings.from_environment(environment)
                self.assertNotIn(name, str(raised.exception))
                self.assertNotIn(sentinel, str(raised.exception))
                self.assertIsNone(raised.exception.__context__)

    def test_admits_only_known_celery_runtime_markers(self) -> None:
        environment = self._environment()
        environment.update(
            {
                "CELERY_LOG_LEVEL": "20",
                "CELERY_LOG_REDIRECT": "1",
                "CELERY_LOG_REDIRECT_LEVEL": "WARNING",
                "celery_dummy_proxy": "set_by_celeryd",
            }
        )

        settings = ProviderShadowSettings.from_environment(environment)

        self.assertEqual(settings.mode, "provider_shadow")
        for name in (
            "CELERY_LOG_LEVEL",
            "CELERY_LOG_REDIRECT",
            "CELERY_LOG_REDIRECT_LEVEL",
            "celery_dummy_proxy",
        ):
            with self.subTest(name=name):
                changed = dict(environment)
                changed[name] = "operator-supplied"
                with self.assertRaises(ProviderShadowConfigurationError):
                    ProviderShadowSettings.from_environment(changed)

    def test_accepts_only_internal_render_database_and_broker_urls(self) -> None:
        invalid_urls = (
            (
                "DATABASE_URL",
                "postgresql://sam:opaque@dpg-sam-test-a.oregon-postgres.render.com/sam",
            ),
            ("DATABASE_URL", "postgresql://sam:opaque@postgres:5432/sam"),
            ("DATABASE_URL", "postgresql://sam:opaque@127.0.0.1:5432/sam"),
            (
                "DATABASE_URL",
                "postgresql://sam:opaque@dpg-sam-test-a/sam?hostaddr=203.0.113.10",
            ),
            (
                "REDIS_URL",
                "rediss://default:opaque@oregon-redis.render.com:6379/0",
            ),
            ("REDIS_URL", "redis://redis:6379/0"),
            ("REDIS_URL", "redis://localhost:6379/0"),
            (
                "REDIS_URL",
                "redis://red-sam-test:6379/0?host=public-cache.example",
            ),
        )
        for name, value in invalid_urls:
            with self.subTest(name=name, value=value):
                environment = self._environment()
                environment[name] = value
                with self.assertRaises(ProviderShadowConfigurationError) as raised:
                    ProviderShadowSettings.from_environment(environment)
                self.assertNotIn(value, str(raised.exception))
                self.assertIsNone(raised.exception.__context__)

    def test_rejects_non_r2_or_credential_bearing_endpoint(self) -> None:
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
                with self.assertRaises(ProviderShadowConfigurationError) as raised:
                    ProviderShadowSettings.from_environment(environment)
                self.assertNotIn(endpoint, str(raised.exception))
                self.assertNotIn("sentinel", str(raised.exception))
                self.assertIsNone(raised.exception.__context__)

    def test_empty_forbidden_settings_do_not_grant_authority(self) -> None:
        environment = self._environment()
        environment.update(
            {
                "SESSION_SECRET": "",
                "BASE44_EVIDENCE_WEBHOOK_TOKEN": "",
                "RESULTS_PROVIDER_API_KEY": "",
                "AWS_ACCESS_KEY_ID": "",
                "CELERY_BEAT_SCHEDULE": "",
            }
        )

        settings = ProviderShadowSettings.from_environment(environment)

        self.assertEqual(settings.provider, "the_odds_api")


if __name__ == "__main__":
    unittest.main()
