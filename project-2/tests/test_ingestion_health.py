from __future__ import annotations

import unittest
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta

from sam_analytics.ingestion_health import (
    ALERT_CODES,
    IngestionHealth,
    IngestionHealthFacts,
    IngestionHealthPolicy,
    evaluate_ingestion_health,
)


class IngestionHealthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 9, 5, 20, 0, tzinfo=UTC)
        self.policy = IngestionHealthPolicy(
            quote_max_age_seconds=300,
            worker_activity_max_age_seconds=120,
            quota_max_age_seconds=900,
            queue_oldest_max_age_seconds=180,
            queue_depth_warning=10,
            quota_low_watermark=25,
        )
        self.healthy = IngestionHealthFacts(
            provider="licensed-odds-provider",
            latest_quote_received_at=self.now - timedelta(seconds=30),
            latest_worker_activity_at=self.now - timedelta(seconds=20),
            quota_remaining=100,
            quota_reserved=4,
            quota_observed_at=self.now - timedelta(seconds=60),
            backlog_count=0,
            oldest_outstanding_at=None,
            retry_wait_count=0,
            dead_letter_count=0,
        )

    def evaluate(self, signals: IngestionHealthFacts | None = None):
        return evaluate_ingestion_health(
            policy=self.policy,
            facts=self.healthy if signals is None else signals,
            now=self.now,
        ).to_public_dict()

    def test_complete_recent_clear_observations_are_healthy(self) -> None:
        health = self.evaluate()

        self.assertEqual(health["status"], "healthy")
        self.assertEqual(health["quote_freshness"]["status"], "fresh")
        self.assertEqual(health["worker_activity"]["status"], "active")
        self.assertEqual(health["worker_activity"]["basis"], "durable_activity_only")
        self.assertEqual(health["queue"]["status"], "clear")
        self.assertEqual(health["queue"]["depth_band"], "empty")
        self.assertEqual(health["quota"]["status"], "healthy")
        self.assertEqual(health["quota"]["remaining_band"], "adequate")
        self.assertEqual(health["dead_letter"], {"status": "clear", "count_band": "zero"})
        self.assertEqual(health["alert_codes"], [])
        self.assertNotIn("provider", health)
        self.assertNotIn("evaluated_at", health)
        self.assertNotIn("valid_until", health)

    def test_all_missing_is_unavailable_and_never_claims_worker_liveness(self) -> None:
        health = self.evaluate(IngestionHealthFacts())

        self.assertEqual(health["status"], "unavailable")
        self.assertEqual(health["worker_activity"]["status"], "unavailable")
        self.assertNotIn("heartbeat", repr(health).lower())
        self.assertIn("feed_unavailable", health["alert_codes"])
        self.assertIn("quota_unavailable", health["alert_codes"])
        self.assertIn("queue_unavailable", health["alert_codes"])
        self.assertIn("dead_letter_unavailable", health["alert_codes"])

    def test_age_boundaries_are_inclusive_then_become_stale_or_stalled(self) -> None:
        at_boundary = replace(
            self.healthy,
            latest_quote_received_at=self.now - timedelta(seconds=300),
            latest_worker_activity_at=self.now - timedelta(seconds=120),
            quota_observed_at=self.now - timedelta(seconds=900),
            backlog_count=1,
            oldest_outstanding_at=self.now - timedelta(seconds=180),
        )
        boundary_health = self.evaluate(at_boundary)

        self.assertEqual(boundary_health["quote_freshness"]["status"], "fresh")
        self.assertEqual(boundary_health["worker_activity"]["status"], "active")
        self.assertEqual(boundary_health["quota"]["status"], "healthy")
        self.assertEqual(boundary_health["queue"]["status"], "clear")

        over_boundary = replace(
            at_boundary,
            latest_quote_received_at=self.now - timedelta(seconds=300, microseconds=1),
            latest_worker_activity_at=self.now - timedelta(seconds=120, microseconds=1),
            quota_observed_at=self.now - timedelta(seconds=900, microseconds=1),
            oldest_outstanding_at=self.now - timedelta(seconds=180, microseconds=1),
        )
        health = self.evaluate(over_boundary)

        self.assertEqual(health["status"], "blocked")
        self.assertEqual(health["quote_freshness"]["status"], "stale")
        self.assertEqual(health["worker_activity"]["status"], "stalled")
        self.assertEqual(health["quota"]["status"], "stale")
        self.assertEqual(health["queue"]["status"], "stalled")
        self.assertIn("worker_stalled", health["alert_codes"])

    def test_queue_depth_boundary_and_retry_wait_are_banded(self) -> None:
        at_depth = replace(
            self.healthy,
            backlog_count=10,
            oldest_outstanding_at=self.now - timedelta(seconds=30),
        )
        at_depth_health = self.evaluate(at_depth)
        self.assertEqual(at_depth_health["queue"]["status"], "clear")
        self.assertEqual(at_depth_health["queue"]["depth_band"], "normal")

        above_depth = replace(at_depth, backlog_count=11)
        above_health = self.evaluate(above_depth)
        self.assertEqual(above_health["status"], "degraded")
        self.assertEqual(above_health["queue"]["status"], "backlogged")
        self.assertEqual(above_health["queue"]["depth_band"], "high")
        self.assertIn("queue_backlogged", above_health["alert_codes"])

        retry_wait = replace(
            self.healthy,
            retry_wait_count=1,
            oldest_outstanding_at=self.now - timedelta(seconds=30),
        )
        retry_health = self.evaluate(retry_wait)
        self.assertEqual(retry_health["status"], "degraded")
        self.assertEqual(retry_health["queue"]["retry_wait"], "present")
        self.assertIn("retry_wait_present", retry_health["alert_codes"])

    def test_worker_activity_is_not_invented_for_an_idle_or_stalled_worker(self) -> None:
        idle = replace(self.healthy, latest_worker_activity_at=None)
        idle_health = self.evaluate(idle)
        self.assertEqual(idle_health["status"], "degraded")
        self.assertEqual(idle_health["worker_activity"]["status"], "idle_unverified")

        stalled = replace(
            self.healthy,
            latest_worker_activity_at=self.now - timedelta(seconds=121),
            backlog_count=1,
            oldest_outstanding_at=self.now - timedelta(seconds=60),
        )
        stalled_health = self.evaluate(stalled)
        self.assertEqual(stalled_health["status"], "blocked")
        self.assertEqual(stalled_health["worker_activity"]["status"], "stalled")
        self.assertIn("worker_activity_stale", stalled_health["alert_codes"])
        self.assertIn("worker_stalled", stalled_health["alert_codes"])

    def test_effective_quota_subtracts_reservations_before_banding(self) -> None:
        exactly_low = replace(self.healthy, quota_remaining=30, quota_reserved=5)
        low_health = self.evaluate(exactly_low)
        self.assertEqual(low_health["status"], "degraded")
        self.assertEqual(low_health["quota"]["status"], "low")
        self.assertEqual(low_health["quota"]["remaining_band"], "low")
        self.assertNotIn("remaining", low_health["quota"])
        self.assertNotIn("reserved", low_health["quota"])

        above_low = replace(exactly_low, quota_remaining=31)
        self.assertEqual(self.evaluate(above_low)["quota"]["status"], "healthy")

        exhausted = replace(self.healthy, quota_remaining=8, quota_reserved=8)
        exhausted_health = self.evaluate(exhausted)
        self.assertEqual(exhausted_health["status"], "blocked")
        self.assertEqual(exhausted_health["quota"]["status"], "exhausted")
        self.assertIn("quota_exhausted", exhausted_health["alert_codes"])

        over_reserved = replace(self.healthy, quota_remaining=8, quota_reserved=9)
        self.assertEqual(self.evaluate(over_reserved)["quota"]["status"], "invalid")

    def test_dead_letters_are_present_without_exposing_an_exact_count(self) -> None:
        health = self.evaluate(replace(self.healthy, dead_letter_count=37))

        self.assertEqual(health["status"], "blocked")
        self.assertEqual(
            health["dead_letter"],
            {"status": "present", "count_band": "one_or_more"},
        )
        self.assertNotIn("37", repr(health))
        self.assertIn("dead_letter_present", health["alert_codes"])

    def test_naive_future_and_wrong_type_timestamps_fail_closed(self) -> None:
        bad_values = (
            datetime(2026, 9, 5, 20, 0),
            self.now + timedelta(microseconds=1),
            "2026-09-05T20:00:00Z",
            True,
        )
        timestamp_fields = (
            "latest_quote_received_at",
            "latest_worker_activity_at",
            "quota_observed_at",
        )
        for field_name in timestamp_fields:
            for bad_value in bad_values:
                with self.subTest(field=field_name, value=bad_value):
                    health = self.evaluate(replace(self.healthy, **{field_name: bad_value}))
                    self.assertEqual(health["status"], "blocked")
                    self.assertIn("monitoring_invalid", health["alert_codes"])

        queue_values = bad_values[:3]
        for bad_value in queue_values:
            with self.subTest(field="oldest_outstanding_at", value=bad_value):
                signals = replace(
                    self.healthy,
                    backlog_count=1,
                    oldest_outstanding_at=bad_value,
                )
                health = self.evaluate(signals)
                self.assertEqual(health["queue"]["status"], "invalid")
                self.assertEqual(health["status"], "blocked")

    def test_boolean_and_malformed_counts_never_pass_as_integers(self) -> None:
        for field_name in (
            "quota_remaining",
            "quota_reserved",
            "backlog_count",
            "retry_wait_count",
            "dead_letter_count",
        ):
            for bad_value in (True, False, -1, 1.5, "1"):
                with self.subTest(field=field_name, value=bad_value):
                    health = self.evaluate(replace(self.healthy, **{field_name: bad_value}))
                    self.assertEqual(health["status"], "blocked")
                    self.assertIn("monitoring_invalid", health["alert_codes"])

    def test_incomplete_or_inconsistent_observations_fail_closed(self) -> None:
        incomplete_quota = replace(self.healthy, quota_reserved=None)
        self.assertEqual(self.evaluate(incomplete_quota)["quota"]["status"], "invalid")

        missing_oldest = replace(
            self.healthy,
            backlog_count=1,
            oldest_outstanding_at=None,
        )
        self.assertEqual(self.evaluate(missing_oldest)["queue"]["status"], "unavailable")

        impossible_oldest = replace(
            self.healthy,
            backlog_count=0,
            retry_wait_count=0,
            oldest_outstanding_at=self.now,
        )
        self.assertEqual(self.evaluate(impossible_oldest)["queue"]["status"], "invalid")

        missing_provider = replace(self.healthy, provider=None)
        missing_provider_health = self.evaluate(missing_provider)
        self.assertEqual(missing_provider_health["status"], "blocked")
        self.assertIn("monitoring_invalid", missing_provider_health["alert_codes"])

        malformed_provider = replace(self.healthy, provider="provider.example.com")
        malformed_provider_health = self.evaluate(malformed_provider)
        self.assertEqual(malformed_provider_health["status"], "blocked")
        self.assertIn("monitoring_invalid", malformed_provider_health["alert_codes"])

    def test_alerts_are_finite_unique_and_deterministically_ordered(self) -> None:
        health = self.evaluate(IngestionHealthFacts())

        self.assertEqual(len(health["alert_codes"]), len(set(health["alert_codes"])))
        self.assertTrue(set(health["alert_codes"]) <= ALERT_CODES)
        self.assertEqual(
            health["alert_codes"],
            [code for code in (
                "feed_unavailable",
                "worker_activity_unavailable",
                "queue_unavailable",
                "quota_unavailable",
                "dead_letter_unavailable",
            )],
        )

    def test_policy_and_now_reject_boolean_zero_or_naive_values(self) -> None:
        for field_name in (
            "quote_max_age_seconds",
            "worker_activity_max_age_seconds",
            "quota_max_age_seconds",
            "queue_oldest_max_age_seconds",
        ):
            with self.subTest(field=field_name):
                with self.assertRaises(ValueError):
                    replace(self.policy, **{field_name: False})

        for field_name in ("queue_depth_warning", "quota_low_watermark"):
            with self.subTest(field=field_name):
                with self.assertRaises(ValueError):
                    replace(self.policy, **{field_name: True})

        with self.assertRaises(ValueError):
            evaluate_ingestion_health(
                policy=self.policy,
                facts=self.healthy,
                now=datetime(2026, 9, 5, 20, 0),
            )
        with self.assertRaises(ValueError):
            evaluate_ingestion_health(
                policy=self.policy,
                facts={},
                now=self.now,
            )

    def test_result_is_frozen_and_public_dict_is_a_fresh_projection(self) -> None:
        result = evaluate_ingestion_health(
            policy=self.policy,
            facts=self.healthy,
            now=self.now,
        )
        self.assertIsInstance(result, IngestionHealth)
        first = result.to_public_dict()
        first["alert_codes"].append("not-approved")
        first["quota"]["status"] = "invented"

        second = result.to_public_dict()
        self.assertEqual(second["alert_codes"], [])
        self.assertEqual(second["quota"]["status"], "healthy")
        with self.assertRaises((AttributeError, TypeError)):
            result.status = "invented"

    def test_health_value_cannot_be_hand_built_or_replaced_around_the_evaluator(self) -> None:
        result = evaluate_ingestion_health(
            policy=self.policy,
            facts=self.healthy,
            now=self.now,
        )

        with self.assertRaisesRegex(ValueError, "created by the evaluator"):
            IngestionHealth(**asdict(result))
        with self.assertRaisesRegex(ValueError, "created by the evaluator"):
            replace(result, status="blocked")


if __name__ == "__main__":
    unittest.main()
