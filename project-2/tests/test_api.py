import unittest
from datetime import datetime, timezone

from app import create_app
from sam_analytics.settings import Settings


class ApiTests(unittest.TestCase):
    def setUp(self):
        settings = Settings(
            environment="test",
            secret_key="test-secret",
            api_key=None,
            database_url=None,
            redis_url=None,
            allowed_origins=(),
            quote_max_age_seconds=300,
            approved_model_versions=("nba-v1",),
        )
        self.client = create_app(settings).test_client()

    def test_health_is_safe_and_legacy_routes_are_retired(self):
        response = self.client.get("/api/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(self.client.get("/api/simulate_predictions").status_code, 410)

    def test_evaluation_requires_timestamped_quote_and_model_version(self):
        response = self.client.post(
            "/api/v1/evaluate",
            json={
                "event_id": "event-1",
                "model_probability": 0.60,
                "decimal_odds": 2.0,
                "quote_captured_at": datetime.now(timezone.utc).isoformat(),
                "model_version": "nba-v1",
                "bankroll": 1000,
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["decision"]["status"], "accepted")
        self.assertEqual(payload["decision"]["stake"], 10.0)
        self.assertTrue(payload["model_is_approved"])

    def test_evaluation_does_not_allow_callers_to_self_approve_versions(self):
        response = self.client.post(
            "/api/v1/evaluate",
            json={
                "event_id": "event-1",
                "model_probability": 0.60,
                "decimal_odds": 2.0,
                "quote_captured_at": datetime.now(timezone.utc).isoformat(),
                "model_version": "unregistered-model",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertFalse(payload["model_is_approved"])
        self.assertEqual(payload["decision"]["status"], "rejected")
        self.assertIn("not approved", " ".join(payload["decision"]["reasons"]))

    def test_evaluation_rejects_untimestamped_input(self):
        response = self.client.post(
            "/api/v1/evaluate",
            json={
                "event_id": "event-1",
                "model_probability": 0.60,
                "decimal_odds": 2.0,
                "quote_captured_at": "2026-01-01T12:00:00",
                "model_version": "nba-v1",
            },
        )
        self.assertEqual(response.status_code, 400)

    def test_production_settings_fail_closed_without_required_infrastructure(self):
        settings = Settings(
            environment="production",
            secret_key="short",
            api_key=None,
            database_url=None,
            redis_url=None,
            allowed_origins=(),
            quote_max_age_seconds=300,
        )
        with self.assertRaisesRegex(ValueError, "SESSION_SECRET"):
            settings.validate()

    def test_status_gateway_key_cannot_invoke_research_evaluation(self):
        settings = Settings(
            environment="test",
            secret_key="test-secret",
            api_key="research-key",
            status_api_key="status-key",
            database_url=None,
            redis_url=None,
            allowed_origins=(),
            quote_max_age_seconds=300,
            approved_model_versions=("nba-v1",),
        )
        client = create_app(settings).test_client()

        self.assertEqual(
            client.get("/api/v1/integration/status", headers={"X-API-Key": "research-key"}).status_code,
            401,
        )
        self.assertEqual(
            client.get("/api/v1/integration/status", headers={"X-API-Key": "status-key"}).status_code,
            200,
        )
        self.assertEqual(
            client.post("/api/v1/evaluate", headers={"X-API-Key": "status-key"}, json={}).status_code,
            401,
        )

    def test_production_disables_client_supplied_research_evaluation(self):
        settings = Settings(
            environment="production",
            secret_key="x" * 32,
            api_key="research-key",
            status_api_key="status-key",
            database_url="postgresql://example",
            redis_url="redis://example",
            allowed_origins=(),
            quote_max_age_seconds=300,
        )
        response = create_app(settings).test_client().post(
            "/api/v1/evaluate",
            headers={"X-API-Key": "research-key"},
            json={},
        )
        self.assertEqual(response.status_code, 403)
