"""Focused tests for the intentionally unwired ingestion outbox boundary."""

from __future__ import annotations

import ast
import inspect
import unittest
from dataclasses import fields, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

from sam_analytics.ingestion_dispatch import (
    DeadLetterReason,
    DispatchPolicy,
    RetryDisposition,
    RetryPlan,
    retry_schedule_sha256,
)
from sam_analytics.ingestion_outbox_runtime import (
    AttemptClaimDisposition,
    AttemptCompletion,
    AttemptCompletionCommit,
    AttemptCompletionOutcome,
    AttemptExecutionFailure,
    AttemptRetrySafety,
    BrokerPublishAcknowledgement,
    ClaimedDispatchAttempt,
    ConsumeDisposition,
    DispatchAttemptClaim,
    IngestionOutboxConfigurationError,
    IngestionOutboxConsumer,
    IngestionOutboxPublisher,
    IngestionOutboxUnavailable,
    OutboxMessage,
    OutboxPublicationClaim,
    PersistedProviderAttemptResult,
    ProviderAttemptResultStatus,
    PublicationCommit,
    PublicationDisposition,
)
from sam_analytics.ingestion_runs import IngestionFailureCode
from sam_analytics.provider_contracts import ProviderUse


class _FakePublisherRepository:
    def __init__(self, message: OutboxMessage, now: datetime) -> None:
        self.message = message
        self.now = now
        self.events: list[str] = []
        self.active_claim: OutboxPublicationClaim | None = None
        self.recorded = False
        self.fail_claim = False
        self.fail_record = False

    def claim_ingestion_outbox_publication(
        self,
        *,
        publisher_identity: str,
        lease_token: UUID,
    ) -> OutboxPublicationClaim | None:
        self.events.append("claim")
        if self.fail_claim:
            raise RuntimeError("postgresql://user:secret@private.invalid/database")
        if self.recorded or self.active_claim is not None:
            return None
        self.active_claim = OutboxPublicationClaim(
            publication_claim_id=uuid4(),
            message=self.message,
            publisher_identity=publisher_identity,
            lease_token=lease_token,
            claimed_at=self.now,
            lease_expires_at=self.now + timedelta(minutes=2),
        )
        return self.active_claim

    def record_ingestion_outbox_publication(
        self,
        claim: OutboxPublicationClaim,
    ) -> PublicationCommit:
        self.events.append("record")
        if self.fail_record:
            self.fail_record = False
            raise RuntimeError("database password=do-not-leak")
        if self.recorded:
            return PublicationCommit.ALREADY_RECORDED
        if claim != self.active_claim:
            raise RuntimeError("stale publication claim")
        self.recorded = True
        self.active_claim = None
        return PublicationCommit.RECORDED

    def expire_publication_lease(self) -> None:
        self.active_claim = None


class _FakePolicyResolver:
    def __init__(self, policy: DispatchPolicy, events: list[str]) -> None:
        self.policy = policy
        self.events = events
        self.fail = False

    def resolve_dispatch_policy(
        self,
        *,
        provider: str,
        policy_version: str,
    ) -> DispatchPolicy:
        self.events.append("resolve_policy")
        if self.fail:
            raise RuntimeError("secret registry detail")
        if (
            provider != self.policy.provider
            or policy_version != self.policy.policy_version
        ):
            raise KeyError("reviewed policy not found")
        return self.policy


class _FakeConsumerRepository:
    def __init__(
        self,
        *,
        now: datetime,
        provider_use: ProviderUse,
        policy: DispatchPolicy,
    ) -> None:
        self.now = now
        self.provider_use = provider_use
        self.policy = policy
        self.events: list[str] = []
        self.claim_disposition = AttemptClaimDisposition.STARTED
        self.running = False
        self.terminal = False
        self.fail_claim = False
        self.fail_time = False
        self.fail_time_on_read: int | None = None
        self.time_read_count = 0
        self.fail_completion = False
        self.completion = None
        self.started: ClaimedDispatchAttempt | None = None
        self.authorization_effective_until = self.now + timedelta(days=1)

    def claim_ingestion_dispatch_attempt(
        self,
        message: OutboxMessage,
        *,
        worker_identity: str,
        lease_token: UUID,
    ) -> DispatchAttemptClaim:
        self.events.append("claim")
        if self.fail_claim:
            raise RuntimeError("postgresql://sam:super-secret@db.invalid/sam")
        if self.terminal:
            return DispatchAttemptClaim(
                message=message,
                disposition=AttemptClaimDisposition.TERMINAL,
            )
        if self.running:
            return DispatchAttemptClaim(
                message=message,
                disposition=AttemptClaimDisposition.INCONCLUSIVE,
            )
        if self.claim_disposition is not AttemptClaimDisposition.STARTED:
            return DispatchAttemptClaim(
                message=message,
                disposition=self.claim_disposition,
            )
        self.running = True
        self.started = ClaimedDispatchAttempt(
            attempt_claim_id=uuid4(),
            message=message,
            worker_identity=worker_identity,
            lease_token=lease_token,
            running_transition_id=uuid4(),
            provider_use_authorization_id=uuid4(),
            quota_receipt_id=uuid4(),
            provider_use=self.provider_use,
            request_fingerprint_sha256="a" * 64,
            estimated_cost=1,
            policy_version=self.policy.policy_version,
            max_attempts=self.policy.max_attempts,
            min_request_interval=self.policy.min_request_interval,
            quota_floor=self.policy.quota_floor,
            quota_max_age=self.policy.quota_max_age,
            retry_schedule_sha256=retry_schedule_sha256(self.policy),
            window_start=self.now - timedelta(hours=1),
            window_end=self.now + timedelta(hours=1),
            authorization_effective_until=self.authorization_effective_until,
            claimed_at=self.now,
            lease_expires_at=self.now + timedelta(minutes=5),
        )
        # This event represents the transaction commit, not merely an insert.
        self.events.append("running_committed")
        return DispatchAttemptClaim(
            message=message,
            disposition=AttemptClaimDisposition.STARTED,
            started=self.started,
        )

    def read_ingestion_dispatch_attempt_time(
        self,
        attempt: ClaimedDispatchAttempt,
    ) -> datetime:
        self.events.append("read_database_time")
        self.time_read_count += 1
        if self.fail_time or self.fail_time_on_read == self.time_read_count:
            raise RuntimeError("database unavailable with credential=hidden")
        if attempt != self.started:
            raise RuntimeError("wrong attempt")
        return self.now + timedelta(seconds=5)

    def complete_ingestion_dispatch_attempt(
        self,
        attempt: ClaimedDispatchAttempt,
        completion,
    ) -> AttemptCompletionCommit:
        self.events.append("complete")
        if self.fail_completion:
            raise RuntimeError("database password must not escape")
        if attempt != self.started:
            raise RuntimeError("wrong completion attempt")
        if self.completion is not None:
            if completion == self.completion:
                return AttemptCompletionCommit.ALREADY_COMMITTED
            raise RuntimeError("conflicting completion")
        self.completion = completion
        self.running = False
        self.terminal = completion.outcome in {
            AttemptCompletionOutcome.SUCCEEDED,
            AttemptCompletionOutcome.DEAD_LETTERED,
        }
        return AttemptCompletionCommit.COMMITTED


class IngestionOutboxPublisherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 9, 6, 19, tzinfo=UTC)
        self.message = OutboxMessage(dispatch_id=uuid4(), attempt_number=1)
        self.repository = _FakePublisherRepository(self.message, self.now)

    def test_claim_precedes_broker_and_record_and_envelope_has_only_two_fields(self) -> None:
        seen = []

        def publish(message: OutboxMessage) -> BrokerPublishAcknowledgement:
            self.repository.events.append("broker")
            seen.append(message)
            return BrokerPublishAcknowledgement.ACCEPTED

        publisher = IngestionOutboxPublisher(
            self.repository,
            publish,
            publisher_identity="outbox-publisher-1",
        )

        result = publisher.publish_available(limit=3)

        self.assertEqual(
            self.repository.events,
            ["claim", "broker", "record", "claim"],
        )
        self.assertEqual(seen, [self.message])
        self.assertEqual(
            {field.name for field in fields(seen[0])},
            {"dispatch_id", "attempt_number"},
        )
        self.assertEqual(
            result.results[0].disposition,
            PublicationDisposition.PUBLISHED,
        )

    def test_broker_failure_never_records_queued_transition(self) -> None:
        def fail(message: OutboxMessage) -> BrokerPublishAcknowledgement:
            self.repository.events.append("broker")
            raise RuntimeError("bearer credential that must stay private")

        publisher = IngestionOutboxPublisher(
            self.repository,
            fail,
            publisher_identity="outbox-publisher-1",
        )

        result = publisher.publish_available()

        self.assertEqual(self.repository.events, ["claim", "broker"])
        self.assertFalse(self.repository.recorded)
        self.assertEqual(
            result.results[0].disposition,
            PublicationDisposition.BROKER_UNAVAILABLE,
        )

    def test_post_send_commit_crash_releases_only_by_expiry_and_may_duplicate(self) -> None:
        delivered = []

        def publish(message: OutboxMessage) -> BrokerPublishAcknowledgement:
            delivered.append(message)
            return BrokerPublishAcknowledgement.ACCEPTED

        publisher = IngestionOutboxPublisher(
            self.repository,
            publish,
            publisher_identity="outbox-publisher-1",
        )
        self.repository.fail_record = True

        first = publisher.publish_available()
        self.assertEqual(
            first.results[0].disposition,
            PublicationDisposition.COMMIT_UNCERTAIN,
        )
        self.assertFalse(self.repository.recorded)

        # The database, not this runtime, decides that the two-minute lease has
        # expired. Re-leasing and sending a duplicate is safer than row loss.
        self.repository.expire_publication_lease()
        second = publisher.publish_available()

        self.assertEqual(delivered, [self.message, self.message])
        self.assertTrue(self.repository.recorded)
        self.assertEqual(
            second.results[0].disposition,
            PublicationDisposition.PUBLISHED,
        )

    def test_claim_failure_is_sanitized_without_exception_context(self) -> None:
        self.repository.fail_claim = True
        publisher = IngestionOutboxPublisher(
            self.repository,
            lambda message: BrokerPublishAcknowledgement.ACCEPTED,
            publisher_identity="outbox-publisher-1",
        )

        with self.assertRaises(IngestionOutboxUnavailable) as caught:
            publisher.publish_available()

        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)
        self.assertNotIn("secret", str(caught.exception).lower())

    def test_publication_limits_and_identities_are_fail_closed(self) -> None:
        with self.assertRaises(IngestionOutboxConfigurationError):
            IngestionOutboxPublisher(
                self.repository,
                lambda message: BrokerPublishAcknowledgement.ACCEPTED,
                publisher_identity="api_token_value",
            )
        publisher = IngestionOutboxPublisher(
            self.repository,
            lambda message: BrokerPublishAcknowledgement.ACCEPTED,
            publisher_identity="outbox-publisher-1",
        )
        for invalid in (0, 101, True, "10"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(IngestionOutboxConfigurationError):
                    publisher.publish_available(limit=invalid)


class IngestionOutboxConsumerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 9, 6, 20, tzinfo=UTC)
        self.message = OutboxMessage(dispatch_id=uuid4(), attempt_number=1)
        self.provider_use = ProviderUse(
            provider="the_odds_api",
            license_scope="internal-derived-only",
            license_version="terms-2026-09-06",
            source_type="odds",
            exposure="private_raw",
        )
        self.policy = DispatchPolicy(
            provider="the_odds_api",
            policy_version="admission-v1",
            enabled=True,
            allowed_source_types=frozenset({"odds"}),
            max_batch_size=1,
            max_attempts=3,
            min_request_interval=timedelta(seconds=30),
            quota_floor=2,
            quota_max_age=timedelta(minutes=5),
            retry_delays=(timedelta(seconds=30), timedelta(minutes=5)),
        )
        self.repository = _FakeConsumerRepository(
            now=self.now,
            provider_use=self.provider_use,
            policy=self.policy,
        )
        self.resolver = _FakePolicyResolver(self.policy, self.repository.events)
        self.callback_count = 0

    def _result_for(
        self,
        attempt: ClaimedDispatchAttempt,
    ) -> PersistedProviderAttemptResult:
        self.callback_count += 1
        self.repository.events.append("provider_callback")
        return PersistedProviderAttemptResult(
            dispatch_id=attempt.message.dispatch_id,
            attempt_number=attempt.message.attempt_number,
            provider_use_authorization_id=attempt.provider_use_authorization_id,
            provider_payload_receipt_id=uuid4(),
            status=ProviderAttemptResultStatus.ACCEPTED,
        )

    def _consumer(self, executor=None, resolver=None) -> IngestionOutboxConsumer:
        return IngestionOutboxConsumer(
            self.repository,
            resolver or self.resolver,
            executor or self._result_for,
            worker_identity="private-worker-1",
        )

    def test_running_is_committed_before_callback_and_success_binds_exact_facts(self) -> None:
        result = self._consumer().consume(
            self.message.dispatch_id,
            self.message.attempt_number,
        )

        self.assertEqual(
            self.repository.events,
            [
                "claim",
                "running_committed",
                "resolve_policy",
                "read_database_time",
                "provider_callback",
                "complete",
            ],
        )
        self.assertEqual(result.disposition, ConsumeDisposition.SUCCEEDED)
        completion = self.repository.completion
        self.assertEqual(completion.message, self.message)
        self.assertEqual(
            completion.attempt_claim_id,
            self.repository.started.attempt_claim_id,
        )
        self.assertEqual(
            completion.provider_use_authorization_id,
            self.repository.started.provider_use_authorization_id,
        )
        self.assertIsNotNone(completion.provider_payload_receipt_id)

    def test_not_ready_is_an_explicit_requeue_without_callback(self) -> None:
        self.repository.claim_disposition = AttemptClaimDisposition.NOT_READY

        result = self._consumer().consume(self.message.dispatch_id, 1)

        self.assertEqual(result.disposition, ConsumeDisposition.REQUEUE)
        self.assertEqual(self.callback_count, 0)
        self.assertEqual(self.repository.events, ["claim"])

    def test_running_or_terminal_redelivery_never_calls_provider(self) -> None:
        for claim_disposition, expected in (
            (AttemptClaimDisposition.INCONCLUSIVE, ConsumeDisposition.INCONCLUSIVE),
            (AttemptClaimDisposition.TERMINAL, ConsumeDisposition.TERMINAL),
        ):
            with self.subTest(claim_disposition=claim_disposition):
                self.repository.events.clear()
                self.repository.claim_disposition = claim_disposition
                result = self._consumer().consume(self.message.dispatch_id, 1)
                self.assertEqual(result.disposition, expected)
                self.assertEqual(self.callback_count, 0)
                self.assertEqual(self.repository.events, ["claim"])

    def test_duplicate_broker_messages_execute_provider_only_once(self) -> None:
        consumer = self._consumer()

        first = consumer.consume(self.message.dispatch_id, 1)
        second = consumer.consume(self.message.dispatch_id, 1)

        self.assertEqual(first.disposition, ConsumeDisposition.SUCCEEDED)
        self.assertEqual(second.disposition, ConsumeDisposition.TERMINAL)
        self.assertEqual(self.callback_count, 1)

    def test_unsent_retry_uses_database_time_and_reviewed_local_delay(self) -> None:
        def fail(attempt: ClaimedDispatchAttempt):
            self.callback_count += 1
            self.repository.events.append("provider_callback")
            raise AttemptExecutionFailure(
                code=IngestionFailureCode.NETWORK_TIMEOUT,
                retry_safety=AttemptRetrySafety.REQUEST_NOT_SENT,
            )

        result = self._consumer(executor=fail).consume(self.message.dispatch_id, 1)

        self.assertEqual(result.disposition, ConsumeDisposition.RETRY_SCHEDULED)
        self.assertEqual(result.failure_code, IngestionFailureCode.NETWORK_TIMEOUT)
        plan = self.repository.completion.retry_plan
        self.assertEqual(plan.disposition, RetryDisposition.RETRY)
        self.assertEqual(
            plan.next_attempt_at,
            self.now + timedelta(seconds=35),
        )
        self.assertEqual(
            self.repository.completion.retry_safety,
            AttemptRetrySafety.REQUEST_NOT_SENT,
        )
        self.assertIsNone(self.repository.completion.provider_payload_receipt_id)
        self.assertEqual(
            self.repository.events,
            [
                "claim",
                "running_committed",
                "resolve_policy",
                "read_database_time",
                "provider_callback",
                "read_database_time",
                "complete",
            ],
        )

    def test_nonretryable_failure_dead_letters_deterministically(self) -> None:
        def fail(attempt: ClaimedDispatchAttempt):
            raise AttemptExecutionFailure(
                code=IngestionFailureCode.PROVIDER_RESPONSE_INVALID
            )

        result = self._consumer(executor=fail).consume(self.message.dispatch_id, 1)

        self.assertEqual(result.disposition, ConsumeDisposition.DEAD_LETTERED)
        plan = self.repository.completion.retry_plan
        self.assertEqual(plan.disposition, RetryDisposition.DEAD_LETTER)
        self.assertEqual(plan.dead_letter_reason, DeadLetterReason.NON_RETRYABLE)

    def test_retryable_failure_without_safety_proof_dead_letters_as_ambiguous(self) -> None:
        def fail(attempt: ClaimedDispatchAttempt):
            raise AttemptExecutionFailure(code=IngestionFailureCode.NETWORK_TIMEOUT)

        result = self._consumer(executor=fail).consume(self.message.dispatch_id, 1)

        self.assertEqual(result.disposition, ConsumeDisposition.DEAD_LETTERED)
        self.assertEqual(result.failure_code, IngestionFailureCode.IDEMPOTENCY_CONFLICT)
        self.assertEqual(
            self.repository.completion.retry_plan.dead_letter_reason,
            DeadLetterReason.NON_RETRYABLE,
        )

    def test_unknown_callback_error_dead_letters_as_ambiguous_side_effect(self) -> None:
        def fail(attempt: ClaimedDispatchAttempt):
            raise RuntimeError("Authorization: Bearer do-not-leak")

        result = self._consumer(executor=fail).consume(self.message.dispatch_id, 1)

        self.assertEqual(result.disposition, ConsumeDisposition.DEAD_LETTERED)
        self.assertEqual(result.failure_code, IngestionFailureCode.IDEMPOTENCY_CONFLICT)
        self.assertEqual(
            self.repository.completion.retry_plan.failure_code,
            IngestionFailureCode.IDEMPOTENCY_CONFLICT,
        )

    def test_cross_attempt_provider_result_fails_evidence_validation(self) -> None:
        def wrong_result(attempt: ClaimedDispatchAttempt):
            return PersistedProviderAttemptResult(
                dispatch_id=uuid4(),
                attempt_number=attempt.message.attempt_number,
                provider_use_authorization_id=attempt.provider_use_authorization_id,
                provider_payload_receipt_id=uuid4(),
                status=ProviderAttemptResultStatus.ACCEPTED,
            )

        result = self._consumer(executor=wrong_result).consume(
            self.message.dispatch_id,
            1,
        )

        self.assertEqual(result.disposition, ConsumeDisposition.DEAD_LETTERED)
        self.assertEqual(
            result.failure_code,
            IngestionFailureCode.EVIDENCE_VALIDATION_FAILED,
        )
        self.assertIsNone(self.repository.completion.provider_payload_receipt_id)

    def test_completion_failure_is_inconclusive_and_redelivery_does_not_execute(self) -> None:
        self.repository.fail_completion = True
        consumer = self._consumer()

        first = consumer.consume(self.message.dispatch_id, 1)
        # A provider-attempt lease is never an authorization to try the call
        # again. Even well past its five-minute expiry, DB state stays
        # inconclusive and blocks the provider lane until a future recovery
        # design can prove that the original external call cannot resume.
        self.repository.now += timedelta(minutes=30)
        second = consumer.consume(self.message.dispatch_id, 1)

        self.assertEqual(first.disposition, ConsumeDisposition.COMPLETION_UNCERTAIN)
        self.assertEqual(second.disposition, ConsumeDisposition.INCONCLUSIVE)
        self.assertEqual(self.callback_count, 1)

    def test_string_like_commit_dispositions_fail_closed(self) -> None:
        for raw_disposition in ("committed", "already_committed"):
            with self.subTest(raw_disposition=raw_disposition):
                self.repository.running = False
                self.repository.terminal = False
                self.repository.started = None

                def invalid_commit(attempt, completion, result=raw_disposition):
                    return result

                self.repository.complete_ingestion_dispatch_attempt = invalid_commit
                result = self._consumer().consume(self.message.dispatch_id, 1)

                self.assertEqual(
                    result.disposition,
                    ConsumeDisposition.COMPLETION_UNCERTAIN,
                )

    def test_policy_conflict_dead_letters_before_provider_callback(self) -> None:
        conflicts = (
            {"max_attempts": 2},
            {"min_request_interval": timedelta(seconds=31)},
            {"quota_floor": 3},
            {"quota_max_age": timedelta(minutes=6)},
            {
                "retry_delays": (
                    timedelta(seconds=31),
                    timedelta(minutes=5),
                )
            },
            {"max_retry_delay": timedelta(hours=2)},
        )
        for conflict in conflicts:
            with self.subTest(conflict=conflict):
                self.repository.events.clear()
                self.repository.running = False
                self.repository.terminal = False
                self.repository.completion = None
                self.repository.started = None
                self.callback_count = 0
                self.resolver.policy = replace(self.policy, **conflict)

                result = self._consumer().consume(self.message.dispatch_id, 1)

                self.assertEqual(result.disposition, ConsumeDisposition.DEAD_LETTERED)
                self.assertEqual(
                    result.failure_code,
                    IngestionFailureCode.CONFIGURATION_INVALID,
                )
                self.assertEqual(self.callback_count, 0)
                self.assertNotIn("provider_callback", self.repository.events)

    def test_retry_that_cannot_fit_authorization_is_dead_lettered(self) -> None:
        self.repository.authorization_effective_until = (
            self.now + timedelta(minutes=5, seconds=30)
        )

        def fail(attempt: ClaimedDispatchAttempt):
            raise AttemptExecutionFailure(
                code=IngestionFailureCode.NETWORK_TIMEOUT,
                retry_safety=AttemptRetrySafety.REQUEST_NOT_SENT,
            )

        result = self._consumer(executor=fail).consume(self.message.dispatch_id, 1)

        self.assertEqual(result.disposition, ConsumeDisposition.DEAD_LETTERED)
        self.assertEqual(
            result.failure_code,
            IngestionFailureCode.LICENSE_NOT_PERMITTED,
        )
        self.assertIsNone(self.repository.completion.retry_safety)

    def test_retry_safety_requires_noncontradictory_durable_evidence(self) -> None:
        receipt_id = uuid4()
        with self.assertRaises(IngestionOutboxConfigurationError):
            AttemptExecutionFailure(
                code=IngestionFailureCode.NETWORK_TIMEOUT,
                provider_payload_receipt_id=receipt_id,
                retry_safety=AttemptRetrySafety.REQUEST_NOT_SENT,
            )
        with self.assertRaises(IngestionOutboxConfigurationError):
            AttemptExecutionFailure(
                code=IngestionFailureCode.PROVIDER_RATE_LIMITED,
                retry_after=timedelta(minutes=1),
                retry_safety=AttemptRetrySafety.REQUEST_NOT_SENT,
            )
        self.assertEqual(
            tuple(AttemptRetrySafety),
            (AttemptRetrySafety.REQUEST_NOT_SENT,),
        )

        attempt = self.repository.claim_ingestion_dispatch_attempt(
            self.message,
            worker_identity="private-worker-1",
            lease_token=uuid4(),
        ).started
        retry_plan = RetryPlan(
            disposition=RetryDisposition.RETRY,
            failure_code=IngestionFailureCode.NETWORK_TIMEOUT,
            completed_attempts=1,
            next_attempt_at=self.now + timedelta(minutes=1),
        )
        with self.assertRaises(IngestionOutboxConfigurationError):
            AttemptCompletion(
                attempt_claim_id=attempt.attempt_claim_id,
                message=attempt.message,
                provider_use_authorization_id=attempt.provider_use_authorization_id,
                outcome=AttemptCompletionOutcome.RETRY_WAIT,
                provider_payload_receipt_id=receipt_id,
                retry_plan=retry_plan,
                retry_safety=AttemptRetrySafety.REQUEST_NOT_SENT,
            )
    def test_claim_error_is_sanitized_and_callback_cannot_run(self) -> None:
        self.repository.fail_claim = True

        with self.assertRaises(IngestionOutboxUnavailable) as caught:
            self._consumer().consume(self.message.dispatch_id, 1)

        self.assertEqual(self.callback_count, 0)
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)
        self.assertNotIn("secret", str(caught.exception).lower())

    def test_invalid_pre_execution_gate_prevents_callback(self) -> None:
        self.repository.fail_time = True

        def fail(attempt: ClaimedDispatchAttempt):
            self.callback_count += 1
            raise AttemptExecutionFailure(code=IngestionFailureCode.NETWORK_TIMEOUT)

        result = self._consumer(executor=fail).consume(self.message.dispatch_id, 1)

        self.assertEqual(result.disposition, ConsumeDisposition.COMPLETION_UNCERTAIN)
        self.assertEqual(self.callback_count, 0)
        self.assertIsNone(self.repository.completion)

    def test_invalid_database_time_after_callback_is_completion_uncertain(self) -> None:
        self.repository.fail_time_on_read = 2

        def fail(attempt: ClaimedDispatchAttempt):
            self.callback_count += 1
            raise AttemptExecutionFailure(
                code=IngestionFailureCode.NETWORK_TIMEOUT,
                retry_safety=AttemptRetrySafety.REQUEST_NOT_SENT,
            )

        result = self._consumer(executor=fail).consume(self.message.dispatch_id, 1)

        self.assertEqual(result.disposition, ConsumeDisposition.COMPLETION_UNCERTAIN)
        self.assertEqual(self.callback_count, 1)
        self.assertIsNone(self.repository.completion)

    def test_consumer_accepts_only_dispatch_id_and_attempt_from_broker(self) -> None:
        parameters = list(inspect.signature(IngestionOutboxConsumer.consume).parameters)
        self.assertEqual(parameters, ["self", "dispatch_id", "attempt_number"])

    def test_runtime_module_has_no_live_wiring_or_unsafe_imports(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        runtime_path = project_root / "sam_analytics" / "ingestion_outbox_runtime.py"
        tree = ast.parse(runtime_path.read_text(encoding="utf-8"))
        imported_roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])
        self.assertTrue(
            imported_roots.isdisjoint(
                {"celery", "boto3", "requests", "urllib", "os", "psycopg"}
            )
        )

        runtime_module = "ingestion_outbox_runtime"
        entrypoints = (
            project_root / "app.py",
            project_root / "main.py",
            project_root / "worker.py",
            project_root / "provider_worker.py",
            project_root / "wsgi.py",
            project_root / "routes" / "api.py",
            project_root / "routes" / "views.py",
        )
        for entrypoint in entrypoints:
            if entrypoint.exists():
                with self.subTest(entrypoint=entrypoint.name):
                    self.assertNotIn(
                        runtime_module,
                        entrypoint.read_text(encoding="utf-8"),
                    )


if __name__ == "__main__":
    unittest.main()
