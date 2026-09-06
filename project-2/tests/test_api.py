import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

from app import create_app
from routes import api as api_routes
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
        app.config["SAM_APPROVED_PREDICTION_READER"] = lambda **_: []

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
        self.assertFalse(prediction["trained_model"])
        self.assertEqual(prediction["prediction_source"], "market_consensus_v1")
        self.assertIsNone(prediction["model"])
        self.assertNotIn("model_probability", response.get_data(as_text=True))
        self.assertNotIn("opaque-test-value", response.get_data(as_text=True))
        self.assertEqual(response.headers["Cache-Control"], "no-store, max-age=0")

    def test_predictions_overlay_only_persisted_approved_model_output(self):
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
        quote_rows = [
            self._quote_row("book-a", "Home", 2.0, starts_at, captured_at),
            self._quote_row("book-a", "Away", 2.0, starts_at, captured_at),
            self._quote_row("book-b", "Home", 2.2, starts_at, captured_at),
            self._quote_row("book-b", "Away", 1.8, starts_at, captured_at),
        ]
        model_row = self._approved_prediction_row(now=now, starts_at=starts_at)
        app.config["SAM_PREDICTION_READER"] = lambda **_: quote_rows
        app.config["SAM_APPROVED_PREDICTION_READER"] = lambda **_: [model_row]

        response = app.test_client().get(
            "/api/v1/predictions?model_probability=0.99&model_version=attacker-model"
        )

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["method"], "approved_trained_model_v1")
        self.assertEqual(payload["classification"], "approved_trained_model")
        self.assertTrue(payload["trained_model"])
        self.assertEqual(payload["trained_model_count"], 1)
        self.assertEqual(payload["market_consensus_fallback_count"], 0)
        prediction = payload["predictions"][0]
        self.assertEqual(prediction["method"], "market_consensus_v1")
        self.assertAlmostEqual(prediction["home"]["consensus_probability"], 0.475)
        self.assertTrue(prediction["trained_model"])
        self.assertEqual(prediction["prediction_source"], "approved_trained_model_v1")
        self.assertEqual(
            prediction["model"],
            {
                "prediction_id": "prediction-1",
                "version": "sam-approved-v1",
                "as_of": model_row["as_of"].isoformat().replace("+00:00", "Z"),
                "features_available_at": (
                    model_row["features_available_at"].isoformat().replace("+00:00", "Z")
                ),
                "home_win_probability": 0.6,
                "away_win_probability": 0.4,
            },
        )
        self.assertAlmostEqual(prediction["home"]["trained_model_expected_roi"], 0.32)
        self.assertAlmostEqual(prediction["home"]["probability_edge_vs_consensus"], 0.125)
        self.assertAlmostEqual(prediction["away"]["trained_model_expected_roi"], -0.2)
        self.assertAlmostEqual(prediction["away"]["probability_edge_vs_consensus"], -0.125)
        self.assertNotIn("attacker-model", response.get_data(as_text=True))
        self.assertNotIn("opaque-test-value", response.get_data(as_text=True))

    def test_predictions_reject_inconsistent_governance_artifact_timing_and_fields(self):
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
        quote_rows = [
            self._quote_row("book-a", "Home", 2.0, starts_at, captured_at),
            self._quote_row("book-a", "Away", 2.0, starts_at, captured_at),
            self._quote_row("book-b", "Home", 2.2, starts_at, captured_at),
            self._quote_row("book-b", "Away", 1.8, starts_at, captured_at),
        ]
        invalid_overrides = (
            {"governance_decision": "suspended"},
            {"governance_decision": "rejected"},
            {"registry_approval_status": "retired"},
            {"decided_by": "different-reviewer"},
            {"target_definition": "away_team_wins"},
            {"model_sport": "americanfootball_nfl"},
            {"artifact_bytes_present": False},
            {"artifact_digest_verified": False},
            {"artifact_format": "joblib-sklearn-v1"},
            {"artifact_sha256": "not-a-digest"},
            {"as_of": now - timedelta(hours=2)},
            {"as_of": starts_at},
            {"features_available_at": now + timedelta(minutes=1)},
            {"prediction_created_at": now - timedelta(hours=2)},
            {"prediction_created_at": now + timedelta(minutes=1)},
            {"home_win_probability": float("nan")},
            {"home_win_probability": 1.1},
            {"feature_values_uri": "https://untrusted.invalid/features.json"},
        )
        model_rows = []
        for index, overrides in enumerate(invalid_overrides):
            row = self._approved_prediction_row(now=now, starts_at=starts_at)
            row.update(overrides)
            row["prediction_id"] = f"invalid-prediction-{index}"
            model_rows.append(row)
        app.config["SAM_PREDICTION_READER"] = lambda **_: quote_rows
        app.config["SAM_APPROVED_PREDICTION_READER"] = lambda **_: model_rows

        response = app.test_client().get("/api/v1/predictions")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["method"], "market_consensus_v1")
        self.assertEqual(payload["classification"], "market_consensus_baseline")
        self.assertFalse(payload["trained_model"])
        prediction = payload["predictions"][0]
        self.assertFalse(prediction["trained_model"])
        self.assertEqual(prediction["prediction_source"], "market_consensus_v1")
        self.assertIsNone(prediction["model"])
        self.assertNotIn("trained_model_expected_roi", prediction["home"])

    def test_predictions_fall_back_to_consensus_if_trained_lookup_fails(self):
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
        app.config["SAM_PREDICTION_READER"] = lambda **_: [
            self._quote_row("book-a", "Home", 2.0, starts_at, captured_at),
            self._quote_row("book-a", "Away", 2.0, starts_at, captured_at),
            self._quote_row("book-b", "Home", 2.2, starts_at, captured_at),
            self._quote_row("book-b", "Away", 1.8, starts_at, captured_at),
        ]

        def unavailable_reader(**_):
            raise RuntimeError("trained lookup failed at postgresql://user:secret@host/db")

        app.config["SAM_APPROVED_PREDICTION_READER"] = unavailable_reader
        response = app.test_client().get("/api/v1/predictions")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertFalse(payload["trained_model"])
        self.assertIsNone(payload["predictions"][0]["model"])
        self.assertNotIn("secret", response.get_data(as_text=True))

    def test_approved_prediction_query_ranks_all_governance_states_before_filtering(self):
        connection = MagicMock()
        cursor = connection.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = []
        now = datetime.now(UTC)

        with patch("psycopg.connect", return_value=connection):
            rows = api_routes._read_current_approved_predictions(
                database_url="postgresql://opaque-test-value",
                now=now,
                event_ids=("00000000-0000-0000-0000-000000000001",),
            )

        self.assertEqual(rows, [])
        sql = " ".join(cursor.execute.call_args.args[0].split())
        latest_decision_sql = sql.split("eligible_predictions AS", maxsplit=1)[0]
        self.assertNotIn("decision.decision =", latest_decision_sql)
        self.assertIn("decision.decided_at <= %(now)s", latest_decision_sql)
        self.assertIn(
            "ORDER BY decision.model_id, decision.decided_at DESC, "
            "decision.created_at DESC, decision.id DESC",
            latest_decision_sql,
        )
        self.assertIn("latest.decision = 'approved'", sql)
        self.assertIn("model.approved_by = latest.decided_by", sql)
        self.assertIn("model.approved_at = latest.decided_at", sql)
        self.assertIn("event.sport = model.sport", sql)
        self.assertIn("model.target_definition = 'home_team_wins'", sql)
        self.assertIn("public.digest(model.artifact_bytes, 'sha256')", sql)
        self.assertIn("model.artifact_format = 'sam-joblib-envelope-v1'", sql)
        self.assertIn("prediction.as_of >= latest.decided_at", sql)
        self.assertIn("prediction.as_of < event.starts_at", sql)
        self.assertIn("prediction.created_at >= latest.decided_at", sql)
        self.assertIn("prediction.created_at BETWEEN prediction.as_of AND %(now)s", sql)
        self.assertIn(
            "data:application/vnd.sam.feature-vector+json;base64,%",
            sql,
        )
        connection.close.assert_called_once_with()

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

    @staticmethod
    def _approved_prediction_row(*, now: datetime, starts_at: datetime) -> dict[str, object]:
        released_at = now - timedelta(hours=1)
        return {
            "prediction_id": "prediction-1",
            "event_id": "event-1",
            "model_id": "model-1",
            "as_of": now - timedelta(minutes=2),
            "features_available_at": now - timedelta(minutes=3),
            "prediction_created_at": now - timedelta(minutes=1),
            "home_win_probability": 0.6,
            "feature_values_uri": (
                "data:application/vnd.sam.feature-vector+json;base64,e30="
            ),
            "feature_values_sha256": "b" * 64,
            "event_sport": "basketball_nba",
            "starts_at": starts_at,
            "version": "sam-approved-v1",
            "model_sport": "basketball_nba",
            "target_definition": "home_team_wins",
            "registry_approval_status": "approved",
            "approved_by": "independent-reviewer",
            "approved_at": released_at,
            "model_created_at": now - timedelta(hours=2),
            "artifact_format": "sam-joblib-envelope-v1",
            "artifact_sha256": "a" * 64,
            "artifact_bytes_present": True,
            "artifact_digest_verified": True,
            "governance_decision": "approved",
            "decided_by": "independent-reviewer",
            "released_at": released_at,
        }
