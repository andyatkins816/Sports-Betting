import os
import unittest
from unittest.mock import patch

from sam_analytics.settings import Settings


class SettingsEnvironmentTests(unittest.TestCase):
    def test_staging_and_production_reject_private_worker_only_settings_without_echoing_values(self):
        sentinel = "private-value-that-must-not-appear-in-an-error"
        worker_only_names = (
            "SAM_WORKER_ROLE",
            "SAM_WORKER_MODE",
            "SAM_INGESTION_ENABLED",
            "SAM_RAW_EVIDENCE_STORE_URI",
            "SAM_RAW_EVIDENCE_STORE_BACKEND",
            "SAM_RAW_EVIDENCE_S3_REGION",
            "SAM_RAW_EVIDENCE_S3_ENDPOINT_URL",
            "SAM_RAW_EVIDENCE_MAX_BYTES",
            "ODDS_PROVIDER",
            "ODDS_PROVIDER_API_KEY",
            "NFL_API_KEY",
            "OPENAI_API_KEY",
            "RESULTS_PROVIDER",
            "RESULTS_PROVIDER_API_KEY",
            "SAM_RAW_EVIDENCE_S3_ACCESS_KEY_ID",
            "SAM_RAW_EVIDENCE_S3_SECRET_ACCESS_KEY",
            "AWS_ACCESS_KEY_ID",
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
            "CLOUDFLARE_API_USER_SERVICE_KEY",
            "CLOUDFLARE_EMAIL",
            "CF_API_TOKEN",
            "CF_API_KEY",
            "CF_API_USER_SERVICE_KEY",
            "CF_API_EMAIL",
        )

        for environment in ("staging", "production"):
            for setting_name in worker_only_names:
                with self.subTest(environment=environment, setting_name=setting_name):
                    with patch.dict(
                        os.environ,
                        {"APP_ENV": environment, setting_name: sentinel},
                        clear=True,
                    ):
                        with self.assertRaises(ValueError) as raised:
                            Settings.from_environment()

                    self.assertEqual(
                        str(raised.exception),
                        "worker-only or ambient connection settings cannot be configured "
                        "in the web process",
                    )
                    self.assertNotIn(sentinel, str(raised.exception))

    def test_staging_and_production_reject_ambient_postgres_overrides(self):
        sentinel = "203.0.113.10"
        for environment in ("staging", "production"):
            for setting_name in (
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
            ):
                with self.subTest(environment=environment, setting_name=setting_name):
                    with patch.dict(
                        os.environ,
                        {"APP_ENV": environment, setting_name: sentinel},
                        clear=True,
                    ):
                        with self.assertRaises(ValueError) as raised:
                            Settings.from_environment()

                    self.assertNotIn(sentinel, str(raised.exception))

    def test_development_and_test_allow_fixture_private_worker_only_settings(self):
        for environment in ("development", "test"):
            with self.subTest(environment=environment):
                with patch.dict(
                    os.environ,
                    {
                        "APP_ENV": environment,
                        "ODDS_PROVIDER_API_KEY": "fixture-provider-key",
                        "RESULTS_PROVIDER_API_KEY": "fixture-results-key",
                        "SAM_RAW_EVIDENCE_S3_ACCESS_KEY_ID": "fixture-access-key",
                        "SAM_RAW_EVIDENCE_S3_SECRET_ACCESS_KEY": "fixture-secret-key",
                        "SAM_WORKER_MODE": "fixture-worker-mode",
                        "AWS_PROFILE": "fixture-profile",
                    },
                    clear=True,
                ):
                    settings = Settings.from_environment()

                self.assertEqual(settings.environment, environment)


if __name__ == "__main__":
    unittest.main()
