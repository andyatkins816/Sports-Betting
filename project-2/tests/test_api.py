import unittest
from datetime import UTC, datetime, timedelta

from app import create_app
from sam_analytics.readiness import DependencyReadiness
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

    def test_readiness_fails_closed_without_live_dependencies(self):
        response = self.client.get("/api/readyz")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json(), {"status": "not_ready"})

    def test_readiness_reports_only_safe_status_after_dependency_probes_pass(self):
        app = create_app(
            Settings(
                environment="test",
                secret_key="test-secret",
                api_key=None,
                database_url="postgresql://database-secret",
                redis_url="redis://queue-secret",
                allowed_origins=(),
                quote_max_age_seconds=300,
            )
        )
        app.config["SAM_DEPENDENCY_READINESS_PROBE"] = lambda *_: DependencyReadiness(
            database_reachable=True,
            migrations_current=True,
            queue_reachable=True,
        )
        response = app.test_client().get("/api/readyz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"status": "ready"})
        self.assertNotIn("secret", response.get_data(as_text=True))

    def test_evaluation_requires_timestamped_quote_and_model_version(self):
        response = self.client.post(
            "/api/v1/evaluate",
            json={
                "event_id": "event-1",
                "model_probability": 0.60,
                "decimal_odds": 2.0,
                "quote_captured_at": datetime.now(UTC).isoformat(),
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
                "quote_captured_at": datetime.now(UTC).isoformat(),
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

    def test_predictions_return_current_market_consensus_without_client_probabilities(self):
        settings = Settings(
            environment="test",
            secret_key="test-secret",
            api_key=None,
            status_api_key=None,
            database_url="postgresql://opaque-test-value",
            redis_url=None,
            allowed_origins=(),
            quote_max_age_seconds=300,
        )
        app = create_app(settings)
        now = datetime.now(UTC)
        starts_at = now + timedelta(hours=2)
        captured_at = now - timedelta(seconds=30)
        rows = [
            self._quote_row("book-a", "Home", 2.0, starts_at, captured_at),
            self._quote_row("book-a", "Away", 2.0, starts_at, captured_at),
            self._quote_row("book-b", "Home", 2.2, starts_at, captured_at),
            self._quote_row("book-b", "Away", 1.8, starts_at, captured_at),
            self._quote_row(
                "stale-book", "Home", 3.0, starts_at, now - timedelta(seconds=301)
            ),
            self._quote_row(
                "stale-book", "Away", 3.0, starts_at, now - timedelta(seconds=301)
            ),
        ]
        app.config["SAM_PREDICTION_READER"] = lambda **_: rows

        response = app.test_client().get(
            "/api/v1/predictions?limit=10&model_probability=0.99"
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["classification"], "market_consensus_baseline")
        self.assertEqual(payload["method"], "market_consensus_v1")
        self.assertFalse(payload["trained_model"])
        self.assertEqual(payload["count"], 1)
        prediction = payload["predictions"][0]
        self.assertEqual(prediction["book_count"], 2)
        self.assertAlmostEqual(prediction["home"]["consensus_probability"], 0.475)
        self.assertAlmostEqual(prediction["away"]["consensus_probability"], 0.525)
        self.assertEqual(prediction["home"]["best_available_price"]["bookmaker"], "book-b")
        self.assertEqual(prediction["home"]["best_available_price"]["decimal_odds"], 2.2)
        self.assertAlmostEqual(prediction["home"]["consensus_expected_roi"], 0.045)
        self.assertNotIn("model_probability", response.get_data(as_text=True))
        self.assertNotIn("opaque-test-value", response.get_data(as_text=True))
        self.assertEqual(response.headers["Cache-Control"], "no-store, max-age=0")

    def test_predictions_validate_limit_and_fail_closed_without_database(self):
        for value in ("0", "101", "1.5", "01", "nope"):
            with self.subTest(value=value):
                response = self.client.get(f"/api/v1/predictions?limit={value}")
                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.get_json(), {"error": "limit must be an integer from 1 to 100"})
        response = self.client.get("/api/v1/predictions")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json(), {"error": "predictions unavailable"})

    def test_predictions_use_status_key_and_hide_database_failures(self):
        settings = Settings(
            environment="production",
            secret_key="x" * 32,
            api_key="research-key",
            status_api_key="status-key",
            database_url="postgresql://opaque-test-value",
            redis_url="redis://opaque-test-value",
            allowed_origins=("https://sam.vegas",),
            quote_max_age_seconds=300,
        )
        app = create_app(settings)

        def unavailable_reader(**_):
            raise RuntimeError("database failed at postgresql://user:secret@private-host/db")

        app.config["SAM_PREDICTION_READER"] = unavailable_reader
        client = app.test_client()
        self.assertEqual(
            client.get("/api/v1/predictions", headers={"X-API-Key": "research-key"}).status_code,
            401,
        )
        response = client.get("/api/v1/predictions", headers={"X-API-Key": "status-key"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.get_json(), {"error": "predictions unavailable"})
        self.assertNotIn("secret", response.get_data(as_text=True))

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

    def test_production_requires_status_key_and_safe_origins(self):
        settings = Settings(
            environment="production",
            secret_key="x" * 32,
            api_key="research-key",
            status_api_key=None,
            database_url="postgresql://database",
            redis_url="redis://queue",
            allowed_origins=("https://sam.vegas",),
            quote_max_age_seconds=300,
        )
        with self.assertRaisesRegex(ValueError, "SAM_STATUS_API_KEY"):
            settings.validate()

        unsafe_origins = Settings(
            environment="production",
            secret_key="x" * 32,
            api_key="research-key",
            status_api_key="status-key",
            database_url="postgresql://database",
            redis_url="redis://queue",
            allowed_origins=("http://sam.vegas/path",),
            quote_max_age_seconds=300,
        )
        with self.assertRaisesRegex(ValueError, "ALLOWED_ORIGINS"):
            unsafe_origins.validate()

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

    @staticmethod
    def _quote_row(
        bookmaker: str,
        selection: str,
        decimal_odds: float,
        starts_at: datetime,
        captured_at: datetime,
    ) -> dict[str, object]:
        return {
            "event_id": "event-1",
            "provider": "the_odds_api",
            "provider_event_id": "provider-event-1",
            "sport": "basketball_nba",
            "league": "NBA",
            "starts_at": starts_at,
            "home_team": "Home",
            "away_team": "Away",
            "bookmaker": bookmaker,
            "selection": selection,
            "decimal_odds": decimal_odds,
            "captured_at": captured_at,
        }
