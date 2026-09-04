import unittest
from datetime import datetime, timedelta, timezone

from app import create_app
from sam_analytics.service_contract import OperationalSignals, build_integration_status
from sam_analytics.settings import Settings


class ServiceContractTests(unittest.TestCase):
    def test_default_contract_is_blocked_instead_of_claiming_a_live_model(self):
        now = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)

        status = build_integration_status(
            database_configured=False,
            redis_configured=False,
            quote_max_age_seconds=300,
            now=now,
        )

        self.assertEqual(
            set(status),
            {"status", "generated_at", "data_freshness", "model_health", "risk_status", "deployment"},
        )
        self.assertEqual(status["status"], "blocked")
        self.assertEqual(status["deployment"]["contract_version"], "v1")
        self.assertEqual(status["data_freshness"]["status"], "unavailable")
        self.assertEqual(status["model_health"]["status"], "unavailable")
        self.assertEqual(status["deployment"]["prediction_delivery"], "disabled")
        self.assertEqual(status["risk_status"]["wager_submission"], "unsupported")

    def test_only_fresh_verified_and_approved_signals_make_delivery_ready(self):
        now = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
        signals = OperationalSignals(
            provider="licensed-odds-provider",
            latest_quote_at=now - timedelta(seconds=30),
            model_version="nba-moneyline-2026.09.01",
            model_approved=True,
            model_artifact_verified=True,
            model_evaluated_at=now - timedelta(hours=1),
            audit_repository_healthy=True,
            worker_queue_healthy=True,
        )

        status = build_integration_status(
            database_configured=True,
            redis_configured=True,
            quote_max_age_seconds=300,
            signals=signals,
            now=now,
        )

        self.assertEqual(status["status"], "ready")
        self.assertEqual(status["data_freshness"]["status"], "fresh")
        self.assertEqual(status["model_health"]["status"], "healthy")
        self.assertTrue(status["model_health"]["serving_allowed"])
        self.assertEqual(status["deployment"]["prediction_delivery"], "available")

    def test_bad_or_stale_timestamps_fail_closed(self):
        now = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
        signals = OperationalSignals(
            latest_quote_at=now + timedelta(seconds=1),
            model_version="candidate-v1",
            model_approved=True,
            model_artifact_verified=True,
            model_evaluated_at=now + timedelta(seconds=1),
        )

        status = build_integration_status(
            database_configured=True,
            redis_configured=True,
            quote_max_age_seconds=300,
            signals=signals,
            now=now,
        )

        self.assertEqual(status["status"], "blocked")
        self.assertEqual(status["data_freshness"]["status"], "invalid")
        self.assertEqual(status["model_health"]["status"], "invalid")

    def test_malformed_or_old_operational_signals_fail_closed(self):
        now = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
        signals = OperationalSignals(
            provider="licensed-odds-provider",
            latest_quote_at="not-a-timestamp",
            model_version="nba-v1",
            model_approved="true",
            model_artifact_verified=True,
            model_evaluated_at=now - timedelta(days=2),
            audit_repository_healthy=True,
            worker_queue_healthy=True,
        )

        status = build_integration_status(
            database_configured=True,
            redis_configured=True,
            quote_max_age_seconds=300,
            model_evaluation_max_age_seconds=60,
            signals=signals,
            now=now,
        )

        self.assertEqual(status["status"], "blocked")
        self.assertEqual(status["data_freshness"]["status"], "invalid")
        self.assertEqual(status["model_health"]["status"], "unapproved")

        stale_model = OperationalSignals(
            provider="licensed-odds-provider",
            latest_quote_at=now,
            model_version="nba-v1",
            model_approved=True,
            model_artifact_verified=True,
            model_evaluated_at=now - timedelta(hours=2),
            audit_repository_healthy=True,
            worker_queue_healthy=True,
        )
        stale_status = build_integration_status(
            database_configured=True,
            redis_configured=True,
            quote_max_age_seconds=300,
            model_evaluation_max_age_seconds=60,
            signals=stale_model,
            now=now,
        )
        self.assertEqual(stale_status["model_health"]["status"], "monitoring_stale")

    def test_gateway_status_requires_api_key_and_never_returns_the_key(self):
        settings = Settings(
            environment="test",
            secret_key="test-secret",
            api_key="test-gateway-secret",
            status_api_key="test-status-secret",
            database_url=None,
            redis_url=None,
            allowed_origins=(),
            quote_max_age_seconds=300,
        )
        app = create_app(settings)
        client = app.test_client()

        self.assertEqual(client.get("/api/v1/integration/status").status_code, 401)
        response = client.get(
            "/api/v1/integration/status", headers={"X-API-Key": "test-status-secret"}
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["Cache-Control"], "no-store, max-age=0")
        self.assertEqual(response.headers["Pragma"], "no-cache")
        self.assertNotIn("test-gateway-secret", response.get_data(as_text=True))
        self.assertNotIn("test-status-secret", response.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
