import unittest
from datetime import datetime, timedelta, timezone

from app import create_app
from sam_analytics.ingestion_health import (
    IngestionHealthFacts,
    IngestionHealthPolicy,
    evaluate_ingestion_health,
)
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
        self.assertEqual(status["deployment"]["contract_version"], "v2")
        self.assertEqual(status["deployment"]["ingestion"]["status"], "unavailable")
        self.assertEqual(
            status["deployment"]["ingestion"]["worker_activity"]["basis"],
            "durable_activity_only",
        )
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
            ingestion_health=self._healthy_ingestion(now),
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
        self.assertEqual(status["deployment"]["ingestion"]["status"], "healthy")
        self.assertTrue(status["model_health"]["serving_allowed"])
        self.assertEqual(status["deployment"]["prediction_delivery"], "available")

    def test_missing_or_malformed_ingestion_health_cannot_enable_delivery(self):
        now = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
        common = dict(
            provider="licensed-odds-provider",
            latest_quote_at=now,
            model_version="nba-v1",
            model_approved=True,
            model_artifact_verified=True,
            model_evaluated_at=now,
            audit_repository_healthy=True,
            worker_queue_healthy=True,
        )

        for value in (None, {"status": "healthy"}, "healthy"):
            with self.subTest(value=value):
                signals = OperationalSignals(**common, ingestion_health=value)
                status = build_integration_status(
                    database_configured=True,
                    redis_configured=True,
                    quote_max_age_seconds=300,
                    signals=signals,
                    now=now,
                )

                self.assertEqual(status["status"], "blocked")
                self.assertEqual(
                    status["deployment"]["ingestion"]["status"],
                    "unavailable",
                )
                self.assertIn(
                    "ingestion health is unavailable",
                    status["deployment"]["blockers"],
                )

    def test_truthy_non_boolean_configuration_flags_fail_closed(self):
        now = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
        signals = OperationalSignals(
            provider="licensed-odds-provider",
            latest_quote_at=now,
            model_version="nba-v1",
            model_approved=True,
            model_artifact_verified=True,
            model_evaluated_at=now,
            audit_repository_healthy=True,
            worker_queue_healthy=True,
            ingestion_health=self._healthy_ingestion(now),
        )
        for database_configured, redis_configured, dependency in (
            (1, True, "audit_database"),
            (True, "true", "worker_queue"),
        ):
            with self.subTest(dependency=dependency):
                status = build_integration_status(
                    database_configured=database_configured,
                    redis_configured=redis_configured,
                    quote_max_age_seconds=300,
                    signals=signals,
                    now=now,
                )

                self.assertEqual(status["status"], "blocked")
                self.assertEqual(status["deployment"][dependency], "unconfigured")
                self.assertEqual(status["deployment"]["prediction_delivery"], "disabled")

    def test_ingestion_health_for_another_provider_cannot_enable_delivery(self):
        now = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
        signals = OperationalSignals(
            provider="different-provider",
            latest_quote_at=now,
            model_version="nba-v1",
            model_approved=True,
            model_artifact_verified=True,
            model_evaluated_at=now,
            audit_repository_healthy=True,
            worker_queue_healthy=True,
            ingestion_health=self._healthy_ingestion(now),
        )

        status = build_integration_status(
            database_configured=True,
            redis_configured=True,
            quote_max_age_seconds=300,
            signals=signals,
            now=now,
        )

        self.assertEqual(status["status"], "blocked")
        self.assertEqual(status["deployment"]["ingestion"]["status"], "unavailable")
        self.assertIn(
            "monitoring_invalid",
            status["deployment"]["ingestion"]["alert_codes"],
        )

    def test_malformed_provider_identity_is_not_reflected(self):
        now = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
        unsafe_provider = "provider.example.com?credential=do-not-reflect"
        status = build_integration_status(
            database_configured=True,
            redis_configured=True,
            quote_max_age_seconds=300,
            signals=OperationalSignals(
                provider=unsafe_provider,
                latest_quote_at=now,
                ingestion_health=self._healthy_ingestion(now),
            ),
            now=now,
        )

        self.assertIsNone(status["data_freshness"]["provider"])
        self.assertNotIn(unsafe_provider, str(status))
        self.assertEqual(status["deployment"]["ingestion"]["status"], "unavailable")

    def test_malformed_model_version_is_not_reflected(self):
        now = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
        unsafe_version = "https://example.invalid/model?credential=do-not-reflect"
        status = build_integration_status(
            database_configured=True,
            redis_configured=True,
            quote_max_age_seconds=300,
            signals=OperationalSignals(
                model_version=unsafe_version,
                model_approved=True,
                model_artifact_verified=True,
                model_evaluated_at=now,
            ),
            now=now,
        )

        self.assertIsNone(status["model_health"]["version"])
        self.assertNotIn(unsafe_version, str(status))
        self.assertEqual(status["model_health"]["status"], "unavailable")

    def test_cached_ingestion_health_expires_before_it_can_enable_delivery(self):
        evaluated_at = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
        health = self._healthy_ingestion(evaluated_at)
        for now in (
            evaluated_at + timedelta(seconds=6),
            evaluated_at - timedelta(microseconds=1),
        ):
            with self.subTest(now=now):
                signals = OperationalSignals(
                    provider="licensed-odds-provider",
                    latest_quote_at=now,
                    model_version="nba-v1",
                    model_approved=True,
                    model_artifact_verified=True,
                    model_evaluated_at=now,
                    audit_repository_healthy=True,
                    worker_queue_healthy=True,
                    ingestion_health=health,
                )

                status = build_integration_status(
                    database_configured=True,
                    redis_configured=True,
                    quote_max_age_seconds=300,
                    signals=signals,
                    now=now,
                )

                self.assertEqual(status["status"], "blocked")
                self.assertEqual(
                    status["deployment"]["ingestion"]["status"],
                    "unavailable",
                )
                self.assertIn(
                    "monitoring_invalid",
                    status["deployment"]["ingestion"]["alert_codes"],
                )

    def test_ingestion_health_is_accepted_at_exact_reuse_boundary(self):
        evaluated_at = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
        now = evaluated_at + timedelta(seconds=5)
        status = build_integration_status(
            database_configured=True,
            redis_configured=True,
            quote_max_age_seconds=300,
            signals=OperationalSignals(
                provider="licensed-odds-provider",
                latest_quote_at=now,
                model_version="nba-v1",
                model_approved=True,
                model_artifact_verified=True,
                model_evaluated_at=now,
                audit_repository_healthy=True,
                worker_queue_healthy=True,
                ingestion_health=self._healthy_ingestion(evaluated_at),
            ),
            now=now,
        )

        self.assertEqual(status["status"], "ready")
        self.assertEqual(status["deployment"]["ingestion"]["status"], "healthy")

    def test_ingestion_health_expires_at_the_next_component_age_boundary(self):
        evaluated_at = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
        now = evaluated_at + timedelta(microseconds=1)
        status = build_integration_status(
            database_configured=True,
            redis_configured=True,
            quote_max_age_seconds=300,
            signals=OperationalSignals(
                provider="licensed-odds-provider",
                latest_quote_at=now,
                model_version="nba-v1",
                model_approved=True,
                model_artifact_verified=True,
                model_evaluated_at=now,
                audit_repository_healthy=True,
                worker_queue_healthy=True,
                ingestion_health=self._healthy_ingestion(
                    evaluated_at,
                    worker_age_seconds=300,
                ),
            ),
            now=now,
        )

        self.assertEqual(status["status"], "blocked")
        self.assertEqual(status["deployment"]["ingestion"]["status"], "unavailable")

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

    def test_quote_just_past_freshness_boundary_is_stale_before_rounding(self):
        now = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)
        status = build_integration_status(
            database_configured=True,
            redis_configured=True,
            quote_max_age_seconds=300,
            signals=OperationalSignals(
                provider="licensed-odds-provider",
                latest_quote_at=now - timedelta(seconds=300, microseconds=400),
                model_version="nba-v1",
                model_approved=True,
                model_artifact_verified=True,
                model_evaluated_at=now,
                audit_repository_healthy=True,
                worker_queue_healthy=True,
                ingestion_health=self._healthy_ingestion(now),
            ),
            now=now,
        )

        self.assertEqual(status["data_freshness"]["age_seconds"], 300.0)
        self.assertEqual(status["data_freshness"]["status"], "stale")
        self.assertEqual(status["status"], "blocked")

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

    @staticmethod
    def _healthy_ingestion(now, *, worker_age_seconds=20):
        return evaluate_ingestion_health(
            policy=IngestionHealthPolicy(
                quote_max_age_seconds=300,
                worker_activity_max_age_seconds=300,
                quota_max_age_seconds=900,
                queue_oldest_max_age_seconds=300,
                queue_depth_warning=10,
                quota_low_watermark=5,
            ),
            facts=IngestionHealthFacts(
                provider="licensed-odds-provider",
                latest_quote_received_at=now - timedelta(seconds=30),
                latest_worker_activity_at=now - timedelta(seconds=worker_age_seconds),
                quota_remaining=100,
                quota_reserved=0,
                quota_observed_at=now - timedelta(seconds=30),
                backlog_count=0,
                oldest_outstanding_at=None,
                retry_wait_count=0,
                dead_letter_count=0,
            ),
            now=now,
        )


if __name__ == "__main__":
    unittest.main()
