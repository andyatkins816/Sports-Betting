"""Tests for the pure, disabled provider-dispatch safety policy."""

from __future__ import annotations

import ast
import hashlib
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

from sam_analytics.ingestion_dispatch import (
    DeadLetterReason,
    DispatchBlockReason,
    DispatchCandidate,
    DispatchPolicy,
    DispatchValidationError,
    ProviderActivitySnapshot,
    QuotaReservation,
    QuotaSnapshot,
    RetryDisposition,
    admit_dispatch,
    plan_retry,
    retry_schedule_sha256,
)
from sam_analytics.ingestion_runs import IngestionFailureCode


class IngestionDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 9, 6, 15, tzinfo=UTC)
        self.policy = DispatchPolicy(
            provider="the_odds_api",
            policy_version="provider-safety-v1",
            enabled=True,
            allowed_source_types=frozenset({"odds"}),
            max_batch_size=2,
            max_attempts=3,
            min_request_interval=timedelta(0),
            quota_floor=2,
            quota_max_age=timedelta(minutes=10),
            retry_delays=(timedelta(seconds=30), timedelta(minutes=5)),
            max_retry_delay=timedelta(hours=1),
        )
        self.quota = QuotaSnapshot(
            provider="the_odds_api",
            remaining=10,
            observed_at=self.now - timedelta(minutes=1),
        )
        self.activity = ProviderActivitySnapshot(
            provider="the_odds_api",
            observed_at=self.now,
            latest_attempt_at=None,
        )

    def _candidate(self, offset: int = 0, *, cost: int = 1) -> DispatchCandidate:
        start = self.now.replace(minute=0, second=0) + timedelta(hours=offset)
        return DispatchCandidate(
            provider="the_odds_api",
            source_type="odds",
            request_fingerprint_sha256=f"{offset + 1:064x}",
            window_start=start,
            window_end=start + timedelta(hours=1),
            estimated_cost=cost,
        )

    def test_policy_is_disabled_by_default_and_needs_no_quota_to_fail_closed(self) -> None:
        policy = DispatchPolicy(
            provider="the_odds_api",
            policy_version="provider-safety-v1",
        )

        decision = admit_dispatch(
            [self._candidate()],
            policy=policy,
            quota=None,
            now=self.now,
        )

        self.assertEqual(decision.admitted, ())
        self.assertEqual(
            [blocked.reason for blocked in decision.blocked],
            [DispatchBlockReason.DISABLED],
        )
        self.assertIsNone(decision.quota_remaining_after_reservations)

    def test_idempotency_is_canonical_stable_and_not_changed_by_policy(self) -> None:
        candidate = self._candidate()
        equivalent = replace(
            candidate,
            window_start=candidate.window_start.astimezone(timezone(timedelta(hours=-5))),
            window_end=candidate.window_end.astimezone(timezone(timedelta(hours=-5))),
        )
        changed_scope = replace(candidate, request_fingerprint_sha256="f" * 64)

        expected_preimage = (
            "sam-ingestion-dispatch-v1|the_odds_api|odds|"
            "0000000000000000000000000000000000000000000000000000000000000001|"
            "2026-09-06T15:00:00.000000Z|2026-09-06T16:00:00.000000Z"
        )
        self.assertEqual(candidate.idempotency_preimage, expected_preimage)
        self.assertEqual(
            candidate.idempotency_key,
            hashlib.sha256(expected_preimage.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(candidate.idempotency_key, equivalent.idempotency_key)
        self.assertNotEqual(candidate.idempotency_key, changed_scope.idempotency_key)
        first = admit_dispatch(
            [candidate],
            policy=self.policy,
            quota=self.quota,
            provider_activity=self.activity,
            now=self.now,
        ).admitted[0]
        revised_policy = replace(self.policy, policy_version="provider-safety-v2")
        second = admit_dispatch(
            [candidate],
            policy=revised_policy,
            quota=self.quota,
            provider_activity=self.activity,
            now=self.now,
        ).admitted[0]
        self.assertEqual(first.idempotency_key, second.idempotency_key)
        self.assertEqual(len(first.idempotency_key), 64)

    def test_existing_and_in_batch_duplicates_never_reserve_twice(self) -> None:
        first = self._candidate()
        second = self._candidate(1)

        decision = admit_dispatch(
            [first, first, second],
            policy=self.policy,
            quota=self.quota,
            provider_activity=self.activity,
            existing_idempotency_keys=(second.idempotency_key,),
            now=self.now,
        )

        self.assertEqual([item.candidate for item in decision.admitted], [first])
        self.assertEqual(
            [item.reason for item in decision.blocked],
            [DispatchBlockReason.DUPLICATE, DispatchBlockReason.DUPLICATE],
        )
        self.assertEqual(decision.quota_remaining_after_reservations, 9)

    def test_outstanding_reservation_alone_blocks_the_same_request_identity(self) -> None:
        candidate = self._candidate()
        reservation = QuotaReservation(
            provider=self.policy.provider,
            idempotency_key=candidate.idempotency_key,
            credits=candidate.estimated_cost,
            reserved_at=self.now - timedelta(seconds=1),
        )

        decision = admit_dispatch(
            [candidate],
            policy=self.policy,
            quota=self.quota,
            provider_activity=self.activity,
            reservations=(reservation,),
            now=self.now,
        )

        self.assertEqual(decision.admitted, ())
        self.assertEqual(decision.blocked[0].reason, DispatchBlockReason.DUPLICATE)
        self.assertEqual(decision.quota_remaining_after_reservations, 9)

    def test_missing_stale_and_future_quota_fail_closed(self) -> None:
        candidate = self._candidate()
        missing = admit_dispatch(
            [candidate],
            policy=self.policy,
            quota=None,
            provider_activity=self.activity,
            now=self.now,
        )
        stale = admit_dispatch(
            [candidate],
            policy=self.policy,
            quota=replace(
                self.quota,
                observed_at=self.now - self.policy.quota_max_age - timedelta(microseconds=1),
            ),
            provider_activity=self.activity,
            reservations=(
                QuotaReservation(
                    provider="the_odds_api",
                    idempotency_key="c" * 64,
                    credits=1,
                    reserved_at=self.now - timedelta(minutes=20),
                ),
            ),
            now=self.now,
        )

        self.assertEqual(missing.blocked[0].reason, DispatchBlockReason.QUOTA_UNAVAILABLE)
        self.assertEqual(stale.blocked[0].reason, DispatchBlockReason.QUOTA_STALE)
        self.assertEqual(stale.quota_remaining_after_reservations, 9)
        with self.assertRaisesRegex(DispatchValidationError, "future"):
            admit_dispatch(
                [candidate],
                policy=self.policy,
                quota=replace(self.quota, observed_at=self.now + timedelta(seconds=1)),
                provider_activity=self.activity,
                now=self.now,
            )

    def test_all_outstanding_reservations_are_subtracted_and_floor_is_preserved(self) -> None:
        reservation = QuotaReservation(
            provider="the_odds_api",
            idempotency_key="a" * 64,
            credits=3,
            reserved_at=self.quota.observed_at + timedelta(seconds=1),
        )
        old_reservation = replace(
            reservation,
            idempotency_key="b" * 64,
            credits=2,
            reserved_at=self.quota.observed_at - timedelta(seconds=1),
        )

        decision = admit_dispatch(
            [self._candidate(cost=6), self._candidate(1, cost=3)],
            policy=self.policy,
            quota=self.quota,
            provider_activity=self.activity,
            reservations=(old_reservation, reservation),
            now=self.now,
        )

        self.assertEqual(
            [admitted.candidate.estimated_cost for admitted in decision.admitted],
            [3],
        )
        self.assertEqual(
            [blocked.reason for blocked in decision.blocked],
            [DispatchBlockReason.QUOTA_FLOOR],
        )
        self.assertEqual(decision.quota_remaining_after_reservations, 2)

    def test_enabled_policy_allows_only_explicit_source_types(self) -> None:
        result = admit_dispatch(
            [replace(self._candidate(), source_type="results")],
            policy=self.policy,
            quota=self.quota,
            provider_activity=self.activity,
            now=self.now,
        )

        self.assertEqual(result.admitted, ())
        self.assertEqual(result.blocked[0].reason, DispatchBlockReason.SOURCE_NOT_ALLOWED)

        with self.assertRaisesRegex(DispatchValidationError, "allowed source"):
            replace(self.policy, allowed_source_types=frozenset())

    def test_batch_cap_is_exact_and_remaining_candidates_are_blocked(self) -> None:
        decision = admit_dispatch(
            [self._candidate(0), self._candidate(1), self._candidate(2)],
            policy=self.policy,
            quota=self.quota,
            provider_activity=self.activity,
            now=self.now,
        )

        self.assertEqual(len(decision.admitted), 2)
        self.assertEqual(decision.blocked[0].reason, DispatchBlockReason.BATCH_LIMIT)
        self.assertEqual(decision.quota_remaining_after_reservations, 8)

    def test_spacing_uses_both_last_quota_observation_and_latest_reservation(self) -> None:
        policy = replace(self.policy, min_request_interval=timedelta(minutes=2))
        blocked_by_observation = admit_dispatch(
            [self._candidate()],
            policy=policy,
            quota=self.quota,
            provider_activity=self.activity,
            now=self.now,
        )
        older_quota = replace(self.quota, observed_at=self.now - timedelta(minutes=5))
        recent_reservation = QuotaReservation(
            provider="the_odds_api",
            idempotency_key="c" * 64,
            credits=1,
            reserved_at=self.now - timedelta(minutes=1),
        )
        blocked_by_reservation = admit_dispatch(
            [self._candidate()],
            policy=policy,
            quota=older_quota,
            provider_activity=self.activity,
            reservations=(recent_reservation,),
            now=self.now,
        )
        admitted = admit_dispatch(
            [self._candidate()],
            policy=policy,
            quota=older_quota,
            provider_activity=self.activity,
            now=self.now,
        )

        self.assertEqual(
            blocked_by_observation.blocked[0].reason,
            DispatchBlockReason.RATE_SPACING,
        )
        self.assertEqual(
            blocked_by_reservation.blocked[0].reason,
            DispatchBlockReason.RATE_SPACING,
        )
        self.assertEqual(len(admitted.admitted), 1)

    def test_a_positive_spacing_allows_only_one_immediate_batch_item(self) -> None:
        policy = replace(
            self.policy,
            min_request_interval=timedelta(seconds=1),
            max_batch_size=3,
        )
        quota = replace(self.quota, observed_at=self.now - timedelta(minutes=5))

        decision = admit_dispatch(
            [self._candidate(0), self._candidate(1)],
            policy=policy,
            quota=quota,
            provider_activity=self.activity,
            now=self.now,
        )

        self.assertEqual(len(decision.admitted), 1)
        self.assertEqual(decision.blocked[0].reason, DispatchBlockReason.RATE_SPACING)

    def test_recent_terminal_attempt_still_enforces_spacing(self) -> None:
        policy = replace(self.policy, min_request_interval=timedelta(minutes=2))
        older_quota = replace(self.quota, observed_at=self.now - timedelta(minutes=5))
        activity = replace(
            self.activity,
            latest_attempt_at=self.now - timedelta(minutes=1),
        )

        decision = admit_dispatch(
            [self._candidate()],
            policy=policy,
            quota=older_quota,
            provider_activity=activity,
            now=self.now,
        )

        self.assertEqual(decision.admitted, ())
        self.assertEqual(decision.blocked[0].reason, DispatchBlockReason.RATE_SPACING)

    def test_missing_stale_or_future_provider_activity_fails_closed(self) -> None:
        missing = admit_dispatch(
            [self._candidate()],
            policy=self.policy,
            quota=self.quota,
            now=self.now,
        )
        stale = admit_dispatch(
            [self._candidate()],
            policy=self.policy,
            quota=self.quota,
            provider_activity=replace(
                self.activity,
                observed_at=(
                    self.now
                    - self.policy.provider_activity_max_age
                    - timedelta(microseconds=1)
                ),
            ),
            now=self.now,
        )

        self.assertEqual(
            missing.blocked[0].reason,
            DispatchBlockReason.ACTIVITY_UNAVAILABLE,
        )
        self.assertEqual(stale.blocked[0].reason, DispatchBlockReason.ACTIVITY_STALE)
        with self.assertRaisesRegex(DispatchValidationError, "future"):
            admit_dispatch(
                [self._candidate()],
                policy=self.policy,
                quota=self.quota,
                provider_activity=replace(
                    self.activity,
                    observed_at=self.now + timedelta(seconds=1),
                ),
                now=self.now,
            )

    def test_retry_uses_larger_of_local_backoff_and_retry_after(self) -> None:
        local = plan_retry(
            failure_code=IngestionFailureCode.NETWORK_TIMEOUT,
            completed_attempts=1,
            policy=self.policy,
            now=self.now,
            retry_after=timedelta(seconds=10),
        )
        provider = plan_retry(
            failure_code=IngestionFailureCode.PROVIDER_RATE_LIMITED,
            completed_attempts=1,
            policy=self.policy,
            now=self.now,
            retry_after=timedelta(minutes=2),
        )

        self.assertEqual(local.disposition, RetryDisposition.RETRY)
        self.assertEqual(local.next_attempt_at, self.now + timedelta(seconds=30))
        self.assertEqual(provider.next_attempt_at, self.now + timedelta(minutes=2))

    def test_nonretryable_exhausted_and_excessive_retry_after_dead_letter(self) -> None:
        nonretryable = plan_retry(
            failure_code=IngestionFailureCode.PROVIDER_RESPONSE_INVALID,
            completed_attempts=1,
            policy=self.policy,
            now=self.now,
        )
        exhausted = plan_retry(
            failure_code=IngestionFailureCode.NETWORK_TIMEOUT,
            completed_attempts=3,
            policy=self.policy,
            now=self.now,
        )
        excessive = plan_retry(
            failure_code=IngestionFailureCode.PROVIDER_RATE_LIMITED,
            completed_attempts=1,
            policy=self.policy,
            now=self.now,
            retry_after=timedelta(hours=2),
        )
        extremely_long = plan_retry(
            failure_code=IngestionFailureCode.PROVIDER_RATE_LIMITED,
            completed_attempts=1,
            policy=self.policy,
            now=self.now,
            retry_after=timedelta(days=30),
        )

        self.assertEqual(nonretryable.disposition, RetryDisposition.DEAD_LETTER)
        self.assertEqual(nonretryable.dead_letter_reason, DeadLetterReason.NON_RETRYABLE)
        self.assertEqual(exhausted.dead_letter_reason, DeadLetterReason.ATTEMPTS_EXHAUSTED)
        self.assertEqual(
            excessive.dead_letter_reason,
            DeadLetterReason.RETRY_AFTER_EXCEEDS_LIMIT,
        )
        self.assertEqual(extremely_long.disposition, RetryDisposition.DEAD_LETTER)
        self.assertEqual(
            extremely_long.dead_letter_reason,
            DeadLetterReason.RETRY_AFTER_EXCEEDS_LIMIT,
        )
        self.assertIsNone(excessive.next_attempt_at)

        with self.assertRaisesRegex(DispatchValidationError, "reviewed policy limit"):
            plan_retry(
                failure_code=IngestionFailureCode.NETWORK_TIMEOUT,
                completed_attempts=self.policy.max_attempts + 1,
                policy=self.policy,
                now=self.now,
            )

    def test_disabled_policy_cannot_plan_a_retry(self) -> None:
        disabled = plan_retry(
            failure_code=IngestionFailureCode.NETWORK_TIMEOUT,
            completed_attempts=1,
            policy=replace(self.policy, enabled=False),
            now=self.now,
        )

        self.assertEqual(disabled.disposition, RetryDisposition.DEAD_LETTER)
        self.assertEqual(disabled.dead_letter_reason, DeadLetterReason.POLICY_DISABLED)

    def test_retry_schedule_fingerprint_binds_order_count_and_maximum(self) -> None:
        baseline = retry_schedule_sha256(self.policy)

        self.assertEqual(
            baseline,
            "7251b5da350a580a36c74d5780a56e62267b1f9cc273147fd15991c2ad07d3bb",
        )
        for changed in (
            replace(
                self.policy,
                retry_delays=(timedelta(minutes=5), timedelta(seconds=30)),
            ),
            replace(
                self.policy,
                retry_delays=(
                    timedelta(seconds=30),
                    timedelta(minutes=5),
                    timedelta(minutes=10),
                ),
            ),
            replace(self.policy, max_retry_delay=timedelta(hours=2)),
        ):
            with self.subTest(changed=changed):
                self.assertNotEqual(retry_schedule_sha256(changed), baseline)

    def test_policy_and_candidate_bounds_reject_unsafe_inputs(self) -> None:
        with self.assertRaises(DispatchValidationError):
            replace(self.policy, max_batch_size=11)
        with self.assertRaises(DispatchValidationError):
            replace(self.policy, max_attempts=4)
        with self.assertRaises(DispatchValidationError):
            replace(self.policy, min_request_interval=timedelta(seconds=-1))
        with self.assertRaises(DispatchValidationError):
            replace(self._candidate(), request_fingerprint_sha256="not-a-digest")
        with self.assertRaises(DispatchValidationError):
            replace(self._candidate(), window_end=self._candidate().window_start)
        with self.assertRaises(DispatchValidationError):
            replace(
                self._candidate(),
                window_end=self._candidate().window_start + timedelta(days=8),
            )
        with self.assertRaisesRegex(DispatchValidationError, "Retry-After"):
            plan_retry(
                failure_code=IngestionFailureCode.NETWORK_TIMEOUT,
                completed_attempts=1,
                policy=self.policy,
                now=self.now,
                retry_after=timedelta(seconds=-1),
            )

    def test_module_has_no_runtime_or_external_integration_imports(self) -> None:
        module_path = (
            Path(__file__).resolve().parents[1]
            / "sam_analytics"
            / "ingestion_dispatch.py"
        )
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        imported_roots = {
            alias.name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported_roots.update(
            (node.module or "").split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        )

        self.assertTrue(
            imported_roots
            <= {
                "__future__",
                "collections",
                "dataclasses",
                "datetime",
                "enum",
                "hashlib",
                "json",
                "re",
                "sam_analytics",
            }
        )


if __name__ == "__main__":
    unittest.main()
