"""Unwired, fail-closed orchestration for ingestion outbox delivery.

This module defines two deliberately narrow runtime boundaries without making
either one live.  The publisher leases durable outbox intent, sends an
at-least-once broker message containing only a dispatch UUID and attempt
number, and only then records publication.  The consumer turns that message
into a committed ``running`` transition before an injected provider callback
can run.

PostgreSQL remains the authority for provider serialization, leases, exact
authorization, attempt state, receipt binding, and append-only transitions.
Implementations of the repository protocols are expected to map to the stored
functions with the corresponding long names.  There is intentionally no SQL,
scheduler, queue client, provider client, credential loading, Celery task, or
entry-point import here.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Protocol
from uuid import UUID, uuid4

from sam_analytics.ingestion_dispatch import (
    DeadLetterReason,
    DispatchPolicy,
    RetryDisposition,
    RetryPlan,
    plan_retry,
    retry_schedule_sha256,
)
from sam_analytics.ingestion_runs import IngestionFailureCode
from sam_analytics.provider_contracts import (
    ProviderContractViolation,
    ProviderUse,
    validate_provider_use,
)

_IDENTITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SENSITIVE_IDENTITY_RE = re.compile(
    r"(?:api[-_]?key|token|secret|password|authorization|credential|cookie|bearer)",
    re.IGNORECASE,
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_ATTEMPTS = 5
_MAX_PUBLICATION_LEASE = timedelta(minutes=2)
_MAX_PROVIDER_LEASE = timedelta(minutes=5)
_MAX_PUBLISH_BATCH = 100


class IngestionOutboxRuntimeError(RuntimeError):
    """Base error with a fixed, credential-safe public message."""


class IngestionOutboxConfigurationError(IngestionOutboxRuntimeError, ValueError):
    """A dependency or durable fact violates the closed runtime contract."""


class IngestionOutboxUnavailable(IngestionOutboxRuntimeError):
    """A durable outbox operation could not be confirmed safely."""


@dataclass(frozen=True)
class OutboxMessage:
    """The complete broker envelope; no provider detail or secret is allowed."""

    dispatch_id: UUID
    attempt_number: int

    def __post_init__(self) -> None:
        _validate_uuid(self.dispatch_id, "dispatch id")
        _validate_attempt(self.attempt_number)


@dataclass(frozen=True)
class OutboxPublicationClaim:
    """A database-owned, renewable lease on one unpublished outbox row."""

    publication_claim_id: UUID
    message: OutboxMessage
    publisher_identity: str
    lease_token: UUID
    claimed_at: datetime
    lease_expires_at: datetime

    def __post_init__(self) -> None:
        _validate_uuid(self.publication_claim_id, "publication claim id")
        if not isinstance(self.message, OutboxMessage):
            raise IngestionOutboxConfigurationError("publication claim message is invalid")
        _validate_identity(self.publisher_identity, "publisher identity")
        _validate_uuid(self.lease_token, "publication lease token")
        _validate_lease(
            self.claimed_at,
            self.lease_expires_at,
            maximum=_MAX_PUBLICATION_LEASE,
            label="publication lease",
        )


class BrokerPublishAcknowledgement(StrEnum):
    """The only acknowledgement that permits a durable publication commit."""

    ACCEPTED = "accepted"


class PublicationCommit(StrEnum):
    """Idempotent database result after the broker accepts a message."""

    RECORDED = "recorded"
    ALREADY_RECORDED = "already_recorded"


class PublicationDisposition(StrEnum):
    """Finite, sanitized outcome for one publication claim."""

    PUBLISHED = "published"
    ALREADY_PUBLISHED = "already_published"
    BROKER_UNAVAILABLE = "broker_unavailable"
    COMMIT_UNCERTAIN = "commit_uncertain"


@dataclass(frozen=True)
class PublicationResult:
    """Safe result for a claimed outbox row."""

    message: OutboxMessage
    disposition: PublicationDisposition

    def __post_init__(self) -> None:
        if not isinstance(self.message, OutboxMessage):
            raise IngestionOutboxConfigurationError("publication result message is invalid")
        if not isinstance(self.disposition, PublicationDisposition):
            raise IngestionOutboxConfigurationError("publication disposition is invalid")


@dataclass(frozen=True)
class PublicationBatchResult:
    """Bounded publisher work completed during one caller-controlled tick."""

    results: tuple[PublicationResult, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.results, tuple) or not all(
            isinstance(result, PublicationResult) for result in self.results
        ):
            raise IngestionOutboxConfigurationError(
                "publication batch results must be an immutable validated tuple"
            )


class IngestionOutboxPublisherRepository(Protocol):
    """Durable publisher seam, normally backed by two stored functions.

    ``claim_ingestion_outbox_publication`` must durably lease at most one due
    row and may re-lease it only after the database-owned two-minute lease
    expires. ``record_ingestion_outbox_publication`` must validate the exact
    claim, identity, and lease token and atomically append/confirm the queued
    transition. Repeating the latter for the same completed claim is a success;
    a conflicting or stale claim must fail closed.
    """

    def claim_ingestion_outbox_publication(
        self,
        *,
        publisher_identity: str,
        lease_token: UUID,
    ) -> OutboxPublicationClaim | None:
        ...

    def record_ingestion_outbox_publication(
        self,
        claim: OutboxPublicationClaim,
    ) -> PublicationCommit:
        ...


BrokerPublisher = Callable[[OutboxMessage], BrokerPublishAcknowledgement]


class IngestionOutboxPublisher:
    """Publish leased outbox intent with broker-first, DB-second durability."""

    def __init__(
        self,
        repository: IngestionOutboxPublisherRepository,
        broker_publish: BrokerPublisher,
        *,
        publisher_identity: str,
    ) -> None:
        _validate_identity(publisher_identity, "publisher identity")
        if repository is None or not callable(broker_publish):
            raise IngestionOutboxConfigurationError("publisher dependencies are required")
        self._repository = repository
        self._broker_publish = broker_publish
        self._publisher_identity = publisher_identity

    def __repr__(self) -> str:
        return "IngestionOutboxPublisher()"

    def publish_available(self, *, limit: int = 10) -> PublicationBatchResult:
        """Publish at most ``limit`` rows, stopping on any uncertain outcome.

        A broker failure leaves the durable claim available for re-lease after
        expiry.  If the broker accepts but the database commit is uncertain,
        a later publisher may send a duplicate.  This is intentional
        at-least-once behavior: an outbox row is never silently discarded.
        """

        _validate_publish_limit(limit)
        results: list[PublicationResult] = []
        for _ in range(limit):
            lease_token = uuid4()
            claim_failed = False
            try:
                claim = self._repository.claim_ingestion_outbox_publication(
                    publisher_identity=self._publisher_identity,
                    lease_token=lease_token,
                )
            except Exception:
                claim = None
                claim_failed = True
            if claim_failed:
                raise IngestionOutboxUnavailable(
                    "ingestion outbox publication claim is unavailable"
                ) from None

            if claim is None:
                break
            if (
                not isinstance(claim, OutboxPublicationClaim)
                or claim.publisher_identity != self._publisher_identity
                or claim.lease_token != lease_token
            ):
                raise IngestionOutboxUnavailable(
                    "ingestion outbox publication claim is invalid"
                ) from None

            try:
                acknowledgement = self._broker_publish(claim.message)
            except Exception:
                results.append(
                    PublicationResult(
                        message=claim.message,
                        disposition=PublicationDisposition.BROKER_UNAVAILABLE,
                    )
                )
                break
            if acknowledgement is not BrokerPublishAcknowledgement.ACCEPTED:
                results.append(
                    PublicationResult(
                        message=claim.message,
                        disposition=PublicationDisposition.BROKER_UNAVAILABLE,
                    )
                )
                break

            try:
                committed = self._repository.record_ingestion_outbox_publication(claim)
            except Exception:
                results.append(
                    PublicationResult(
                        message=claim.message,
                        disposition=PublicationDisposition.COMMIT_UNCERTAIN,
                    )
                )
                break
            if committed is PublicationCommit.RECORDED:
                disposition = PublicationDisposition.PUBLISHED
            elif committed is PublicationCommit.ALREADY_RECORDED:
                disposition = PublicationDisposition.ALREADY_PUBLISHED
            else:
                disposition = PublicationDisposition.COMMIT_UNCERTAIN
            results.append(PublicationResult(message=claim.message, disposition=disposition))
            if disposition is PublicationDisposition.COMMIT_UNCERTAIN:
                break

        return PublicationBatchResult(results=tuple(results))


class AttemptClaimDisposition(StrEnum):
    """Database decision for a broker delivery before provider execution."""

    STARTED = "started"
    NOT_READY = "not_ready"
    INCONCLUSIVE = "inconclusive"
    TERMINAL = "terminal"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ClaimedDispatchAttempt:
    """Exact attempt facts returned only after ``running`` has committed."""

    attempt_claim_id: UUID
    message: OutboxMessage
    worker_identity: str
    lease_token: UUID
    running_transition_id: UUID
    provider_use_authorization_id: UUID
    quota_receipt_id: UUID
    provider_use: ProviderUse
    request_fingerprint_sha256: str
    estimated_cost: int
    policy_version: str
    max_attempts: int
    min_request_interval: timedelta
    quota_floor: int
    quota_max_age: timedelta
    retry_schedule_sha256: str
    window_start: datetime
    window_end: datetime
    authorization_effective_until: datetime
    claimed_at: datetime
    lease_expires_at: datetime

    def __post_init__(self) -> None:
        _validate_uuid(self.attempt_claim_id, "attempt claim id")
        if not isinstance(self.message, OutboxMessage):
            raise IngestionOutboxConfigurationError("claimed attempt message is invalid")
        _validate_identity(self.worker_identity, "worker identity")
        _validate_uuid(self.lease_token, "attempt lease token")
        _validate_uuid(self.running_transition_id, "running transition id")
        _validate_uuid(
            self.provider_use_authorization_id,
            "provider use authorization id",
        )
        _validate_uuid(self.quota_receipt_id, "quota receipt id")
        try:
            validate_provider_use(self.provider_use)
        except (ProviderContractViolation, TypeError, ValueError):
            raise IngestionOutboxConfigurationError(
                "claimed provider authorization is invalid"
            ) from None
        if self.provider_use.exposure != "private_raw":
            raise IngestionOutboxConfigurationError(
                "claimed provider authorization must remain private raw"
            )
        if not isinstance(self.request_fingerprint_sha256, str) or not _SHA256_RE.fullmatch(
            self.request_fingerprint_sha256
        ):
            raise IngestionOutboxConfigurationError("request fingerprint is invalid")
        if (
            isinstance(self.estimated_cost, bool)
            or not isinstance(self.estimated_cost, int)
            or not 1 <= self.estimated_cost <= 10_000
        ):
            raise IngestionOutboxConfigurationError("estimated request cost is invalid")
        if (
            not isinstance(self.policy_version, str)
            or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", self.policy_version)
        ):
            raise IngestionOutboxConfigurationError("dispatch policy version is invalid")
        _validate_attempt(self.max_attempts, label="maximum attempts")
        if self.message.attempt_number > self.max_attempts:
            raise IngestionOutboxConfigurationError(
                "claimed attempt exceeds the durable dispatch limit"
            )
        _validate_duration(
            self.min_request_interval,
            label="minimum request interval",
            allow_zero=True,
        )
        if (
            isinstance(self.quota_floor, bool)
            or not isinstance(self.quota_floor, int)
            or not 0 <= self.quota_floor <= 2_147_483_647
        ):
            raise IngestionOutboxConfigurationError("quota floor is invalid")
        _validate_duration(self.quota_max_age, label="quota maximum age")
        if (
            not isinstance(self.retry_schedule_sha256, str)
            or not _SHA256_RE.fullmatch(self.retry_schedule_sha256)
        ):
            raise IngestionOutboxConfigurationError(
                "retry schedule fingerprint is invalid"
            )
        _validate_aware_time(self.window_start, "dispatch window start")
        _validate_aware_time(self.window_end, "dispatch window end")
        if self.window_end <= self.window_start:
            raise IngestionOutboxConfigurationError("dispatch window is invalid")
        if self.window_end - self.window_start > timedelta(days=7):
            raise IngestionOutboxConfigurationError(
                "dispatch window cannot exceed seven days"
            )
        _validate_aware_time(
            self.authorization_effective_until,
            "provider authorization expiry",
        )
        _validate_lease(
            self.claimed_at,
            self.lease_expires_at,
            maximum=_MAX_PROVIDER_LEASE,
            label="provider attempt lease",
        )
        if self.lease_expires_at > self.authorization_effective_until:
            raise IngestionOutboxConfigurationError(
                "provider attempt lease exceeds its authorization"
            )


@dataclass(frozen=True)
class DispatchAttemptClaim:
    """Finite claim result; only ``STARTED`` may carry provider-call facts."""

    message: OutboxMessage
    disposition: AttemptClaimDisposition
    started: ClaimedDispatchAttempt | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.message, OutboxMessage):
            raise IngestionOutboxConfigurationError("attempt claim message is invalid")
        if not isinstance(self.disposition, AttemptClaimDisposition):
            raise IngestionOutboxConfigurationError("attempt claim disposition is invalid")
        if self.disposition is AttemptClaimDisposition.STARTED:
            if not isinstance(self.started, ClaimedDispatchAttempt):
                raise IngestionOutboxConfigurationError(
                    "started attempt claim requires committed running facts"
                )
            if self.started.message != self.message:
                raise IngestionOutboxConfigurationError(
                    "started attempt does not match its broker message"
                )
        elif self.started is not None:
            raise IngestionOutboxConfigurationError(
                "non-started attempt claims cannot carry provider-call facts"
            )


class ProviderAttemptResultStatus(StrEnum):
    """Sanitized result categories returned by an injected provider boundary."""

    ACCEPTED = "accepted"
    ACCEPTED_EMPTY = "accepted_empty"


class AttemptRetrySafety(StrEnum):
    """Proof that a failed call may be repeated without an ambiguous replay."""

    REQUEST_NOT_SENT = "request_not_sent"


@dataclass(frozen=True)
class PersistedProviderAttemptResult:
    """Private receipt identity bound to the exact executed attempt."""

    dispatch_id: UUID
    attempt_number: int
    provider_use_authorization_id: UUID
    provider_payload_receipt_id: UUID
    status: ProviderAttemptResultStatus

    def __post_init__(self) -> None:
        _validate_uuid(self.dispatch_id, "result dispatch id")
        _validate_attempt(self.attempt_number)
        _validate_uuid(
            self.provider_use_authorization_id,
            "result provider use authorization id",
        )
        _validate_uuid(self.provider_payload_receipt_id, "provider payload receipt id")
        if not isinstance(self.status, ProviderAttemptResultStatus):
            raise IngestionOutboxConfigurationError("provider attempt result is invalid")


class AttemptExecutionFailure(Exception):
    """A provider callback's finite failure, containing no exception text."""

    def __init__(
        self,
        *,
        code: IngestionFailureCode,
        retry_after: timedelta | None = None,
        provider_payload_receipt_id: UUID | None = None,
        retry_safety: AttemptRetrySafety | None = None,
    ) -> None:
        if not isinstance(code, IngestionFailureCode):
            raise IngestionOutboxConfigurationError("execution failure code is invalid")
        if retry_after is not None and (
            not isinstance(retry_after, timedelta) or retry_after < timedelta(0)
        ):
            raise IngestionOutboxConfigurationError("Retry-After is invalid")
        if provider_payload_receipt_id is not None:
            _validate_uuid(provider_payload_receipt_id, "provider payload receipt id")
        if retry_safety is not None and not isinstance(
            retry_safety, AttemptRetrySafety
        ):
            raise IngestionOutboxConfigurationError(
                "execution retry safety is invalid"
            )
        if (
            retry_safety is AttemptRetrySafety.REQUEST_NOT_SENT
            and (
                provider_payload_receipt_id is not None
                or retry_after is not None
            )
        ):
            raise IngestionOutboxConfigurationError(
                "an unsent request cannot carry provider response evidence"
            )
        self.code = code
        self.retry_after = retry_after
        self.provider_payload_receipt_id = provider_payload_receipt_id
        self.retry_safety = retry_safety
        super().__init__(code.value)


class AttemptCompletionOutcome(StrEnum):
    """Exact durable completion requested for a claimed attempt."""

    SUCCEEDED = "succeeded"
    RETRY_WAIT = "retry_wait"
    DEAD_LETTERED = "dead_lettered"


@dataclass(frozen=True)
class AttemptCompletion:
    """Safe, exact completion passed to the durable repository."""

    attempt_claim_id: UUID
    message: OutboxMessage
    provider_use_authorization_id: UUID
    outcome: AttemptCompletionOutcome
    provider_payload_receipt_id: UUID | None = None
    retry_plan: RetryPlan | None = None
    retry_safety: AttemptRetrySafety | None = None

    def __post_init__(self) -> None:
        _validate_uuid(self.attempt_claim_id, "completion claim id")
        if not isinstance(self.message, OutboxMessage):
            raise IngestionOutboxConfigurationError("completion message is invalid")
        _validate_uuid(
            self.provider_use_authorization_id,
            "completion provider use authorization id",
        )
        if not isinstance(self.outcome, AttemptCompletionOutcome):
            raise IngestionOutboxConfigurationError("completion outcome is invalid")
        if self.provider_payload_receipt_id is not None:
            _validate_uuid(self.provider_payload_receipt_id, "provider payload receipt id")
        if self.outcome is AttemptCompletionOutcome.SUCCEEDED:
            if (
                self.provider_payload_receipt_id is None
                or self.retry_plan is not None
                or self.retry_safety is not None
            ):
                raise IngestionOutboxConfigurationError(
                    "successful completion requires only an exact persisted result"
                )
            return
        if not isinstance(self.retry_plan, RetryPlan):
            raise IngestionOutboxConfigurationError(
                "failed completion requires only a reviewed retry plan"
            )
        if not isinstance(self.retry_plan.failure_code, IngestionFailureCode):
            raise IngestionOutboxConfigurationError(
                "retry plan failure code is invalid"
            )
        expected = (
            RetryDisposition.RETRY
            if self.outcome is AttemptCompletionOutcome.RETRY_WAIT
            else RetryDisposition.DEAD_LETTER
        )
        if self.retry_plan.disposition is not expected:
            raise IngestionOutboxConfigurationError(
                "completion outcome conflicts with its retry plan"
            )
        if self.retry_plan.completed_attempts != self.message.attempt_number:
            raise IngestionOutboxConfigurationError(
                "retry plan does not match the completed attempt"
            )
        if self.outcome is AttemptCompletionOutcome.RETRY_WAIT:
            if not isinstance(self.retry_safety, AttemptRetrySafety):
                raise IngestionOutboxConfigurationError(
                    "retry completion requires explicit replay-safety proof"
                )
            if (
                self.retry_safety is AttemptRetrySafety.REQUEST_NOT_SENT
                and self.provider_payload_receipt_id is not None
            ):
                raise IngestionOutboxConfigurationError(
                    "an unsent request cannot carry a provider response receipt"
                )
            _validate_aware_time(self.retry_plan.next_attempt_at, "next attempt time")
        else:
            if self.retry_safety is not None:
                raise IngestionOutboxConfigurationError(
                    "dead-letter completion cannot carry retry-safety proof"
                )
            if not isinstance(self.retry_plan.dead_letter_reason, DeadLetterReason):
                raise IngestionOutboxConfigurationError(
                    "dead-letter reason is invalid"
                )


class AttemptCompletionCommit(StrEnum):
    """Idempotent repository result for an exact attempt completion."""

    COMMITTED = "committed"
    ALREADY_COMMITTED = "already_committed"


class ConsumeDisposition(StrEnum):
    """Finite broker-handling outcome with no provider or exception content."""

    SUCCEEDED = "succeeded"
    RETRY_SCHEDULED = "retry_scheduled"
    DEAD_LETTERED = "dead_lettered"
    REQUEUE = "requeue"
    INCONCLUSIVE = "inconclusive"
    TERMINAL = "terminal"
    REJECTED = "rejected"
    COMPLETION_UNCERTAIN = "completion_uncertain"


@dataclass(frozen=True)
class ConsumeResult:
    """Credential-safe outcome returned to a future queue adapter."""

    message: OutboxMessage
    disposition: ConsumeDisposition
    failure_code: IngestionFailureCode | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.message, OutboxMessage):
            raise IngestionOutboxConfigurationError("consume result message is invalid")
        if not isinstance(self.disposition, ConsumeDisposition):
            raise IngestionOutboxConfigurationError("consume disposition is invalid")
        if self.failure_code is not None and not isinstance(
            self.failure_code, IngestionFailureCode
        ):
            raise IngestionOutboxConfigurationError("consume failure code is invalid")
        failure_dispositions = {
            ConsumeDisposition.RETRY_SCHEDULED,
            ConsumeDisposition.DEAD_LETTERED,
        }
        if self.disposition in failure_dispositions and self.failure_code is None:
            raise IngestionOutboxConfigurationError(
                "failed consume outcome requires a safe failure code"
            )
        if (
            self.disposition not in failure_dispositions
            and self.disposition is not ConsumeDisposition.COMPLETION_UNCERTAIN
            and self.failure_code is not None
        ):
            raise IngestionOutboxConfigurationError(
                "non-failure consume outcome cannot carry a failure code"
            )


class IngestionOutboxConsumerRepository(Protocol):
    """Durable attempt seam, normally backed by stored functions.

    ``claim_ingestion_dispatch_attempt`` must acquire the provider lock,
    validate the exact authorization/dispatch/outbox/reservation/attempt, append
    ``running``, and *commit that transaction* before returning ``STARTED``.
    A redelivery of a running attempt, even after its lease expires, must return
    ``INCONCLUSIVE`` and must never silently authorize a second provider call.

    ``complete_ingestion_dispatch_attempt`` must reacquire the provider lock,
    revalidate the exact claim/identity/token/dispatch/attempt/authorization,
    bind any payload receipt to those exact facts, and atomically append either
    success, retry reservation+outbox, or dead-letter facts. Exact repetition
    is idempotent; conflicting completion fails closed. Expired running claims
    remain unresolved and block the provider lane until a later design supplies
    an execution fence that can prove the original call cannot resume.
    """

    def claim_ingestion_dispatch_attempt(
        self,
        message: OutboxMessage,
        *,
        worker_identity: str,
        lease_token: UUID,
    ) -> DispatchAttemptClaim:
        ...

    def read_ingestion_dispatch_attempt_time(
        self,
        attempt: ClaimedDispatchAttempt,
    ) -> datetime:
        """Revalidate the active claim and return provider-locked database time.

        The consumer calls this immediately before execution and again before
        planning a failed attempt. The implementation must reject an expired,
        completed, unauthorized, non-latest, or provider-lane-conflicting claim.
        """

        ...

    def complete_ingestion_dispatch_attempt(
        self,
        attempt: ClaimedDispatchAttempt,
        completion: AttemptCompletion,
    ) -> AttemptCompletionCommit:
        ...


class DispatchPolicyResolver(Protocol):
    """Resolve one code-reviewed policy by exact durable identity."""

    def resolve_dispatch_policy(
        self,
        *,
        provider: str,
        policy_version: str,
    ) -> DispatchPolicy:
        ...


ProviderAttemptExecutor = Callable[
    [ClaimedDispatchAttempt], PersistedProviderAttemptResult
]


class IngestionOutboxConsumer:
    """Claim, execute once, and durably complete one broker delivery.

    The injected executor is deliberately unwired. A production adapter must
    add an immediately pre-send database execution fence plus an absolute
    deadline; constructing this consumer alone never authorizes live traffic.
    """

    def __init__(
        self,
        repository: IngestionOutboxConsumerRepository,
        policy_resolver: DispatchPolicyResolver,
        execute_provider_attempt: ProviderAttemptExecutor,
        *,
        worker_identity: str,
    ) -> None:
        _validate_identity(worker_identity, "worker identity")
        if (
            repository is None
            or policy_resolver is None
            or not callable(execute_provider_attempt)
        ):
            raise IngestionOutboxConfigurationError("consumer dependencies are required")
        self._repository = repository
        self._policy_resolver = policy_resolver
        self._execute_provider_attempt = execute_provider_attempt
        self._worker_identity = worker_identity

    def __repr__(self) -> str:
        return "IngestionOutboxConsumer()"

    def consume(self, dispatch_id: UUID, attempt_number: int) -> ConsumeResult:
        """Consume only an exact dispatch UUID and attempt number.

        No injected provider callback can run unless the repository has already
        returned a valid ``STARTED`` fact, which contractually means the
        ``running`` transition committed. Any completion ambiguity returns a
        finite inconclusive result and this call never invokes the provider a
        second time.
        """

        message = OutboxMessage(dispatch_id=dispatch_id, attempt_number=attempt_number)
        lease_token = uuid4()
        claim_failed = False
        try:
            claim = self._repository.claim_ingestion_dispatch_attempt(
                message,
                worker_identity=self._worker_identity,
                lease_token=lease_token,
            )
        except Exception:
            claim = None
            claim_failed = True
        if claim_failed:
            raise IngestionOutboxUnavailable(
                "ingestion dispatch attempt claim is unavailable"
            ) from None
        if not isinstance(claim, DispatchAttemptClaim) or claim.message != message:
            raise IngestionOutboxUnavailable(
                "ingestion dispatch attempt claim is invalid"
            ) from None

        if claim.disposition is not AttemptClaimDisposition.STARTED:
            return ConsumeResult(
                message=message,
                disposition=_claim_consume_disposition(claim.disposition),
            )

        attempt = claim.started
        if (
            not isinstance(attempt, ClaimedDispatchAttempt)
            or attempt.worker_identity != self._worker_identity
            or attempt.lease_token != lease_token
        ):
            raise IngestionOutboxUnavailable(
                "committed ingestion dispatch attempt is invalid"
            ) from None

        try:
            policy = self._policy_resolver.resolve_dispatch_policy(
                provider=attempt.provider_use.provider,
                policy_version=attempt.policy_version,
            )
            _validate_exact_policy(policy, attempt)
        except Exception:
            plan = RetryPlan(
                disposition=RetryDisposition.DEAD_LETTER,
                failure_code=IngestionFailureCode.CONFIGURATION_INVALID,
                completed_attempts=message.attempt_number,
                dead_letter_reason=DeadLetterReason.NON_RETRYABLE,
            )
            return self._commit_failure(
                attempt,
                plan=plan,
                provider_payload_receipt_id=None,
                retry_safety=None,
            )

        try:
            execution_time = self._repository.read_ingestion_dispatch_attempt_time(
                attempt
            )
            _validate_aware_time(execution_time, "provider execution gate time")
            if (
                execution_time < attempt.claimed_at
                or execution_time >= attempt.lease_expires_at
            ):
                raise IngestionOutboxConfigurationError(
                    "provider execution gate is outside the active attempt lease"
                )
        except Exception:
            return ConsumeResult(
                message=attempt.message,
                disposition=ConsumeDisposition.COMPLETION_UNCERTAIN,
            )

        try:
            provider_result = self._execute_provider_attempt(attempt)
        except AttemptExecutionFailure as failure:
            return self._plan_and_commit_failure(attempt, policy=policy, failure=failure)
        except Exception:
            # An unclassified exception may have happened after the provider
            # accepted a request. Retrying it could duplicate a licensed call,
            # so only an explicitly classified AttemptExecutionFailure may
            # enter reviewed retry policy.
            failure = AttemptExecutionFailure(code=IngestionFailureCode.IDEMPOTENCY_CONFLICT)
            return self._plan_and_commit_failure(attempt, policy=policy, failure=failure)

        try:
            _validate_exact_provider_result(provider_result, attempt)
        except Exception:
            failure = AttemptExecutionFailure(
                code=IngestionFailureCode.EVIDENCE_VALIDATION_FAILED
            )
            return self._plan_and_commit_failure(attempt, policy=policy, failure=failure)

        completion = AttemptCompletion(
            attempt_claim_id=attempt.attempt_claim_id,
            message=message,
            provider_use_authorization_id=attempt.provider_use_authorization_id,
            outcome=AttemptCompletionOutcome.SUCCEEDED,
            provider_payload_receipt_id=provider_result.provider_payload_receipt_id,
        )
        return self._commit_completion(attempt, completion)

    def _plan_and_commit_failure(
        self,
        attempt: ClaimedDispatchAttempt,
        *,
        policy: DispatchPolicy,
        failure: AttemptExecutionFailure,
    ) -> ConsumeResult:
        try:
            decision_time = self._repository.read_ingestion_dispatch_attempt_time(attempt)
            _validate_aware_time(decision_time, "retry decision time")
            if decision_time < attempt.claimed_at:
                raise IngestionOutboxConfigurationError(
                    "retry decision time precedes the committed attempt"
                )
            retry = plan_retry(
                failure_code=failure.code,
                completed_attempts=attempt.message.attempt_number,
                policy=policy,
                now=decision_time,
                retry_after=failure.retry_after,
            )
            if (
                retry.disposition is RetryDisposition.RETRY
                and failure.retry_safety is None
            ):
                retry = RetryPlan(
                    disposition=RetryDisposition.DEAD_LETTER,
                    failure_code=IngestionFailureCode.IDEMPOTENCY_CONFLICT,
                    completed_attempts=attempt.message.attempt_number,
                    dead_letter_reason=DeadLetterReason.NON_RETRYABLE,
                )
            if (
                retry.disposition is RetryDisposition.RETRY
                and retry.next_attempt_at is not None
                and retry.next_attempt_at + _MAX_PROVIDER_LEASE
                > attempt.authorization_effective_until
            ):
                retry = RetryPlan(
                    disposition=RetryDisposition.DEAD_LETTER,
                    failure_code=IngestionFailureCode.LICENSE_NOT_PERMITTED,
                    completed_attempts=attempt.message.attempt_number,
                    dead_letter_reason=DeadLetterReason.NON_RETRYABLE,
                )
        except Exception:
            return ConsumeResult(
                message=attempt.message,
                disposition=ConsumeDisposition.COMPLETION_UNCERTAIN,
                failure_code=failure.code,
            )
        return self._commit_failure(
            attempt,
            plan=retry,
            provider_payload_receipt_id=failure.provider_payload_receipt_id,
            retry_safety=(
                failure.retry_safety
                if retry.disposition is RetryDisposition.RETRY
                else None
            ),
        )

    def _commit_failure(
        self,
        attempt: ClaimedDispatchAttempt,
        *,
        plan: RetryPlan,
        provider_payload_receipt_id: UUID | None,
        retry_safety: AttemptRetrySafety | None,
    ) -> ConsumeResult:
        outcome = (
            AttemptCompletionOutcome.RETRY_WAIT
            if plan.disposition is RetryDisposition.RETRY
            else AttemptCompletionOutcome.DEAD_LETTERED
        )
        completion = AttemptCompletion(
            attempt_claim_id=attempt.attempt_claim_id,
            message=attempt.message,
            provider_use_authorization_id=attempt.provider_use_authorization_id,
            outcome=outcome,
            provider_payload_receipt_id=provider_payload_receipt_id,
            retry_plan=plan,
            retry_safety=retry_safety,
        )
        return self._commit_completion(attempt, completion)

    def _commit_completion(
        self,
        attempt: ClaimedDispatchAttempt,
        completion: AttemptCompletion,
    ) -> ConsumeResult:
        try:
            committed = self._repository.complete_ingestion_dispatch_attempt(
                attempt,
                completion,
            )
        except Exception:
            return ConsumeResult(
                message=attempt.message,
                disposition=ConsumeDisposition.COMPLETION_UNCERTAIN,
                failure_code=(
                    completion.retry_plan.failure_code
                    if completion.retry_plan is not None
                    else None
                ),
            )
        if (
            committed is not AttemptCompletionCommit.COMMITTED
            and committed is not AttemptCompletionCommit.ALREADY_COMMITTED
        ):
            return ConsumeResult(
                message=attempt.message,
                disposition=ConsumeDisposition.COMPLETION_UNCERTAIN,
                failure_code=(
                    completion.retry_plan.failure_code
                    if completion.retry_plan is not None
                    else None
                ),
            )
        if completion.outcome is AttemptCompletionOutcome.SUCCEEDED:
            disposition = ConsumeDisposition.SUCCEEDED
            failure_code = None
        elif completion.outcome is AttemptCompletionOutcome.RETRY_WAIT:
            disposition = ConsumeDisposition.RETRY_SCHEDULED
            failure_code = completion.retry_plan.failure_code
        else:
            disposition = ConsumeDisposition.DEAD_LETTERED
            failure_code = completion.retry_plan.failure_code
        return ConsumeResult(
            message=attempt.message,
            disposition=disposition,
            failure_code=failure_code,
        )


def _claim_consume_disposition(disposition: AttemptClaimDisposition) -> ConsumeDisposition:
    mapping = {
        AttemptClaimDisposition.NOT_READY: ConsumeDisposition.REQUEUE,
        AttemptClaimDisposition.INCONCLUSIVE: ConsumeDisposition.INCONCLUSIVE,
        AttemptClaimDisposition.TERMINAL: ConsumeDisposition.TERMINAL,
        AttemptClaimDisposition.REJECTED: ConsumeDisposition.REJECTED,
    }
    try:
        return mapping[disposition]
    except KeyError:
        raise IngestionOutboxUnavailable(
            "ingestion dispatch attempt claim is invalid"
        ) from None


def _validate_exact_policy(policy: object, attempt: ClaimedDispatchAttempt) -> None:
    if not isinstance(policy, DispatchPolicy):
        raise IngestionOutboxConfigurationError("reviewed dispatch policy is unavailable")
    if (
        policy.provider != attempt.provider_use.provider
        or policy.policy_version != attempt.policy_version
        or policy.max_attempts != attempt.max_attempts
        or policy.min_request_interval != attempt.min_request_interval
        or policy.quota_floor != attempt.quota_floor
        or policy.quota_max_age != attempt.quota_max_age
        or retry_schedule_sha256(policy) != attempt.retry_schedule_sha256
        or attempt.provider_use.source_type not in policy.allowed_source_types
    ):
        raise IngestionOutboxConfigurationError(
            "reviewed dispatch policy conflicts with durable dispatch facts"
        )
    if not policy.enabled:
        raise IngestionOutboxConfigurationError("reviewed dispatch policy is disabled")


def _validate_exact_provider_result(
    result: object,
    attempt: ClaimedDispatchAttempt,
) -> None:
    if not isinstance(result, PersistedProviderAttemptResult):
        raise IngestionOutboxConfigurationError("provider result is invalid")
    if (
        result.dispatch_id != attempt.message.dispatch_id
        or result.attempt_number != attempt.message.attempt_number
        or result.provider_use_authorization_id
        != attempt.provider_use_authorization_id
    ):
        raise IngestionOutboxConfigurationError(
            "provider result does not match the claimed dispatch attempt"
        )


def _validate_identity(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or not _IDENTITY_RE.fullmatch(value)
        or _SENSITIVE_IDENTITY_RE.search(value)
    ):
        raise IngestionOutboxConfigurationError(f"{label} must be a safe identifier")


def _validate_uuid(value: object, label: str) -> None:
    if not isinstance(value, UUID) or value.int == 0:
        raise IngestionOutboxConfigurationError(f"{label} must be a non-zero UUID")


def _validate_attempt(value: object, *, label: str = "attempt number") -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= _MAX_ATTEMPTS:
        raise IngestionOutboxConfigurationError(f"{label} must be between 1 and 5")


def _validate_duration(
    value: object,
    *,
    label: str,
    allow_zero: bool = False,
) -> None:
    minimum = timedelta(0)
    if (
        not isinstance(value, timedelta)
        or value < minimum
        or (not allow_zero and value == minimum)
        or value > timedelta(days=7)
    ):
        raise IngestionOutboxConfigurationError(f"{label} is invalid")


def _validate_aware_time(value: object, label: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise IngestionOutboxConfigurationError(f"{label} must be timezone-aware")


def _validate_lease(
    claimed_at: object,
    expires_at: object,
    *,
    maximum: timedelta,
    label: str,
) -> None:
    _validate_aware_time(claimed_at, f"{label} claim time")
    _validate_aware_time(expires_at, f"{label} expiry")
    duration = expires_at - claimed_at
    if duration <= timedelta(0) or duration > maximum:
        raise IngestionOutboxConfigurationError(f"{label} duration is invalid")


def _validate_publish_limit(value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 1 <= value <= _MAX_PUBLISH_BATCH
    ):
        raise IngestionOutboxConfigurationError(
            "publication limit must be between 1 and 100"
        )
