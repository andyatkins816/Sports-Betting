import os
import unittest
from unittest.mock import patch

from sam_analytics.settings import Settings


class SettingsEnvironmentTests(unittest.TestCase):
    def test_staging_and_production_reject_private_worker_only_secrets_without_echoing_values(self):
        sentinel = "private-value-that-must-not-appear-in-an-error"
        secret_names = (
            "ODDS_PROVIDER_API_KEY",
            "RESULTS_PROVIDER_API_KEY",
            "SAM_RAW_EVIDENCE_S3_ACCESS_KEY_ID",
            "SAM_RAW_EVIDENCE_S3_SECRET_ACCESS_KEY",
        )

        for environment in ("staging", "production"):
            for secret_name in secret_names:
                with self.subTest(environment=environment, secret_name=secret_name):
                    with patch.dict(
                        os.environ,
                        {"APP_ENV": environment, secret_name: sentinel},
                        clear=True,
                    ):
                        with self.assertRaises(ValueError) as raised:
                            Settings.from_environment()

                    self.assertEqual(
                        str(raised.exception),
                        "private worker-only credentials cannot be configured in the web process",
                    )
                    self.assertNotIn(sentinel, str(raised.exception))

    def test_development_and_test_allow_fixture_private_worker_only_secrets(self):
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
                    },
                    clear=True,
                ):
                    settings = Settings.from_environment()

                self.assertEqual(settings.environment, environment)


if __name__ == "__main__":
    unittest.main()
