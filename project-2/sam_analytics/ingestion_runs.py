"""Sanitized, append-only state contracts for future manual shadow ingestion.

This module models only *operational facts* about an ingestion run: a safe
provider identifier, a safe job identifier, state transitions, attempts, and
an enumerated failure code.  It deliberately has no field for a provider
credential, request URL, response body, exception text, or retained object
reference.  Raw evidence belongs exclusively behind the private evidence
store/receipt boundary.

The functions mirror ``005_ingestion_run_audit.sql``.  A future private
dispatcher can use them to form values for that append-only audit trail before
any licensed provider request is enabled.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4


_PROVIDER_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_SOURCE_TYPE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_JOB_IDENTITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SENSITIVE_IDENTITY_RE = re.compile(
    r"(?:api[-_]?key|token|secret|password|authorization|credential|cookie|bearer)",
    re.IGNORECASE,
)
_MAX_ATTEMPTS = 5


class IngestionRunValidationError(ValueError):
    """Raised when an audit fact is malformed or could contain unsafe data."""


class IngestionRunTransitionError(IngestionRunValidationError):
    """Raised when a requested run-state transition is not audit-valid."""


class IngestionRunState(str, Enum):
    """The only persisted states for a bounded manual shadow run."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class IngestionFailureClass(str, Enum):
    """Whether a failed attempt may be manually retried."""

    RETRYABLE = "retryable"
    NON_RETRYABLE = "non_retryable"


class IngestionFailureCode(str, Enum):
    """Safe, finite failure reasons; exception text is intentionally absent."""

    PROVIDER_RATE_LIMITED = "provider_rate_limited"
    PROVIDER_TEMPORARY_UNAVAILABLE = "provider_temporary_unavailable"
    NETWORK_TIMEOUT = "network_timeout"
    STORAGE_UNAVAILABLE = "storage_unavailable"
    DATABASE_UNAVAILABLE = "database_unavailable"
    QUEUE_UNAVAILABLE = "queue_unavailable"
    INTERNAL_TRANSIENT = "internal_transient"
    PROVIDER_CONTRACT_UNAPPROVED = "provider_contract_unapproved"
    LICENSE_NOT_PERMITTED = "license_not_permitted"
    CONFIGURATION_INVALID = "configuration_invalid"
    PROVIDER_RESPONSE_INVALID = "provider_response_invalid"
    EVIDENCE_VALIDATION_FAILED = "evidence_validation_failed"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"


_FAILURE_CLASS_BY_CODE = {
    IngestionFailureCode.PROVIDER_RATE_LIMITED: IngestionFailureClass.RETRYABLE,
    IngestionFailureCode.PROVIDER_TEMPORARY_UNAVAILABLE: IngestionFailureClass.RETRYABLE,
    IngestionFailureCode.NETWORK_TIMEOUT: IngestionFailureClass.RETRYABLE,
    IngestionFailureCode.STORAGE_UNAVAILABLE: IngestionFailureClass.RETRYABLE,
    IngestionFailureCode.DATABASE_UNAVAILABLE: IngestionFailureClass.RETRYABLE,
    IngestionFailureCode.QUEUE_UNAVAILABLE: IngestionFailureClass.RETRYABLE,
    IngestionFailureCode.INTERNAL_TRANSIENT: IngestionFailureClass.RETRYABLE,
    IngestionFailureCode.PROVIDER_CONTRACT_UNAPPROVED: IngestionFailureClass.NON_RETRYABLE,
    IngestionFailureCode.LICENSE_NOT_PERMITTED: IngestionFailureClass.NON_RETRYABLE,
    IngestionFailureCode.CONFIGURATION_INVALID: IngestionFailureClass.NON_RETRYABLE,
    IngestionFailureCode.PROVIDER_RESPONSE_INVALID: IngestionFailureClass.NON_RETRYABLE,
    IngestionFailureCode.EVIDENCE_VALIDATION_FAILED: IngestionFailureClass.NON_RETRYABLE,
    IngestionFailureCode.IDEMPOTENCY_CONFLICT: IngestionFailureClass.NON_RETRYABLE,
}


@dataclass(frozen=True)
class IngestionFailure:
    """One safe classification, without an exception or provider content."""

    code: IngestionFailureCode

    def __post_init__(self) -> None:
        if not isinstance(self.code, IngestionFailureCode):
            raise IngestionRunValidationError("failure code must be an approved safe code")

    @property
    def classification(self) -> IngestionFailureClass:
        """Return the fixed retry policy for this failure code."""

        return _FAILURE_CLASS_BY_CODE[self.code]


@dataclass(frozen=True)
class IngestionRun:
    """Immutable identity for one manually initiated shadow-ingestion run."""

    id: UUID
    provider: str
    job_identity: str
    source_type: str
    max_attempts: int
    created_at: datetime
    run_mode: str = "manual_shadow"

    def __post_init__(self) -> None:
        if not isinstance(self.id, UUID):
            raise IngestionRunValidationError("run id must be a UUID")
        _validate_safe_text(self.provider, _PROVIDER_RE, "provider")
        _validate_safe_text(self.source_type, _SOURCE_TYPE_RE, "source type")
        _validate_job_identity(self.job_identity)
        if self.run_mode != "manual_shadow":
            raise IngestionRunValidationError("run mode must be manual_shadow")
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int):
            raise IngestionRunValidationError("max attempts must be an integer")
        if not 1 <= self.max_attempts <= _MAX_ATTEMPTS:
            raise IngestionRunValidationError("max attempts must be between 1 and 5")
        _validate_aware_time(self.created_at, "created at")


@dataclass(frozen=True)
class IngestionRunStateTransition:
    """One append-only, credential-free state fact for an ingestion run."""

    ingestion_run_id: UUID
    state_sequence: int
    state: IngestionRunState
    attempt_count: int
    occurred_at: datetime
    failure: IngestionFailure | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.ingestion_run_id, UUID):
            raise IngestionRunValidationError("ingestion run id must be a UUID")
        if isinstance(self.state_sequence, bool) or not isinstance(self.state_sequence, int):
            raise IngestionRunValidationError("state sequence must be an integer")
        if self.state_sequence < 1:
            raise IngestionRunValidationError("state sequence must be positive")
        if not isinstance(self.state, IngestionRunState):
            raise IngestionRunValidationError("state must be an approved ingestion run state")
        if isinstance(self.attempt_count, bool) or not isinstance(self.attempt_count, int):
            raise IngestionRunValidationError("attempt count must be an integer")
        if self.attempt_count < 0:
            raise IngestionRunValidationError("attempt count cannot be negative")
        _validate_aware_time(self.occurred_at, "occurred at")
        if self.state in {IngestionRunState.FAILED, IngestionRunState.BLOCKED}:
            if not isinstance(self.failure, IngestionFailure):
                raise IngestionRunValidationError("failed or blocked states require a safe failure code")
        elif self.failure is not None:
            raise IngestionRunValidationError("only failed or blocked states may have a failure code")
        if self.state == IngestionRunState.BLOCKED and (
            self.failure is None
            or self.failure.classification != IngestionFailureClass.NON_RETRYABLE
        ):
            raise IngestionRunValidationError("blocked states require a non-retryable failure code")
        if self.state == IngestionRunState.QUEUED and self.attempt_count != 0:
            raise IngestionRunValidationError("queued states must have zero attempts")
        if self.state in {
            IngestionRunState.RUNNING,
            IngestionRunState.SUCCEEDED,
            IngestionRunState.FAILED,
        } and self.attempt_count < 1:
            raise IngestionRunValidationError("active or completed attempts must have a positive count")

    @property
    def can_retry(self) -> bool:
        """Return whether the failure class permits another manual attempt."""

        return self.state == IngestionRunState.FAILED and bool(
            self.failure and self.failure.classification == IngestionFailureClass.RETRYABLE
        )


def new_manual_shadow_run(
    *,
    provider: str,
    job_identity: str,
    source_type: str,
    max_attempts: int = 1,
    created_at: datetime | None = None,
    run_id: UUID | None = None,
) -> tuple[IngestionRun, IngestionRunStateTransition]:
    """Create a run identity plus its mandatory initial queued audit fact."""

    timestamp = created_at or datetime.now(timezone.utc)
    run = IngestionRun(
        id=run_id or uuid4(),
        provider=provider,
        job_identity=job_identity,
        source_type=source_type,
        max_attempts=max_attempts,
        created_at=timestamp,
    )
    return run, IngestionRunStateTransition(
        ingestion_run_id=run.id,
        state_sequence=1,
        state=IngestionRunState.QUEUED,
        attempt_count=0,
        occurred_at=timestamp,
    )


def start_next_attempt(
    run: IngestionRun,
    previous: IngestionRunStateTransition,
    *,
    occurred_at: datetime,
) -> IngestionRunStateTransition:
    """Record a new manual attempt only from queued/retryable-failed state."""

    _validate_previous(run, previous, occurred_at)
    if previous.state == IngestionRunState.QUEUED:
        next_attempt = 1
    elif previous.can_retry:
        next_attempt = previous.attempt_count + 1
    else:
        raise IngestionRunTransitionError("only queued or retryable failed runs may start an attempt")
    if next_attempt > run.max_attempts:
        raise IngestionRunTransitionError("run has reached its approved attempt limit")
    return _next_transition(
        previous,
        state=IngestionRunState.RUNNING,
        attempt_count=next_attempt,
        occurred_at=occurred_at,
    )


def mark_succeeded(
    run: IngestionRun,
    previous: IngestionRunStateTransition,
    *,
    occurred_at: datetime,
) -> IngestionRunStateTransition:
    """Finish an active attempt successfully without retaining provider data."""

    _validate_previous(run, previous, occurred_at)
    _require_running(previous)
    return _next_transition(
        previous,
        state=IngestionRunState.SUCCEEDED,
        attempt_count=previous.attempt_count,
        occurred_at=occurred_at,
    )


def mark_failed(
    run: IngestionRun,
    previous: IngestionRunStateTransition,
    *,
    failure_code: IngestionFailureCode,
    occurred_at: datetime,
) -> IngestionRunStateTransition:
    """Finish an active attempt with an enumerated, safe failure category."""

    _validate_previous(run, previous, occurred_at)
    _require_running(previous)
    return _next_transition(
        previous,
        state=IngestionRunState.FAILED,
        attempt_count=previous.attempt_count,
        occurred_at=occurred_at,
        failure=IngestionFailure(failure_code),
    )


def mark_blocked(
    run: IngestionRun,
    previous: IngestionRunStateTransition,
    *,
    failure_code: IngestionFailureCode,
    occurred_at: datetime,
) -> IngestionRunStateTransition:
    """Record a terminal policy/configuration block without making a request."""

    _validate_previous(run, previous, occurred_at)
    if previous.state not in {IngestionRunState.QUEUED, IngestionRunState.RUNNING}:
        raise IngestionRunTransitionError("only queued or running runs may be blocked")
    failure = IngestionFailure(failure_code)
    if failure.classification != IngestionFailureClass.NON_RETRYABLE:
        raise IngestionRunTransitionError("blocked runs require a non-retryable failure code")
    return _next_transition(
        previous,
        state=IngestionRunState.BLOCKED,
        attempt_count=previous.attempt_count,
        occurred_at=occurred_at,
        failure=failure,
    )


def mark_cancelled(
    run: IngestionRun,
    previous: IngestionRunStateTransition,
    *,
    occurred_at: datetime,
) -> IngestionRunStateTransition:
    """Record a human cancellation before a run reaches a terminal outcome."""

    _validate_previous(run, previous, occurred_at)
    if previous.state not in {
        IngestionRunState.QUEUED,
        IngestionRunState.RUNNING,
        IngestionRunState.FAILED,
    }:
        raise IngestionRunTransitionError("only queued, running, or failed runs may be cancelled")
    return _next_transition(
        previous,
        state=IngestionRunState.CANCELLED,
        attempt_count=previous.attempt_count,
        occurred_at=occurred_at,
    )


def _next_transition(
    previous: IngestionRunStateTransition,
    *,
    state: IngestionRunState,
    attempt_count: int,
    occurred_at: datetime,
    failure: IngestionFailure | None = None,
) -> IngestionRunStateTransition:
    return IngestionRunStateTransition(
        ingestion_run_id=previous.ingestion_run_id,
        state_sequence=previous.state_sequence + 1,
        state=state,
        attempt_count=attempt_count,
        occurred_at=occurred_at,
        failure=failure,
    )


def _validate_previous(
    run: IngestionRun,
    previous: IngestionRunStateTransition,
    occurred_at: datetime,
) -> None:
    if not isinstance(run, IngestionRun):
        raise IngestionRunValidationError("run must be an IngestionRun")
    if not isinstance(previous, IngestionRunStateTransition):
        raise IngestionRunValidationError("previous state must be an ingestion run state transition")
    if previous.ingestion_run_id != run.id:
        raise IngestionRunTransitionError("previous state belongs to another ingestion run")
    _validate_aware_time(occurred_at, "occurred at")
    if occurred_at < previous.occurred_at:
        raise IngestionRunTransitionError("state transition time cannot move backwards")


def _require_running(previous: IngestionRunStateTransition) -> None:
    if previous.state != IngestionRunState.RUNNING:
        raise IngestionRunTransitionError("only running attempts may be completed")


def _validate_safe_text(value: object, pattern: re.Pattern[str], label: str) -> None:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise IngestionRunValidationError(f"{label} must be a safe identifier")


def _validate_job_identity(value: object) -> None:
    if not isinstance(value, str) or not _JOB_IDENTITY_RE.fullmatch(value):
        raise IngestionRunValidationError("job identity must be a safe identifier")
    if _SENSITIVE_IDENTITY_RE.search(value):
        raise IngestionRunValidationError("job identity cannot contain secret-like text")


def _validate_aware_time(value: object, label: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise IngestionRunValidationError(f"{label} must be timezone-aware")
