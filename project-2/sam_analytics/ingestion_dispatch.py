"""Pure admission and retry policy for a future provider dispatcher.

This module deliberately has no worker, queue, database, configuration, clock,
or provider dependency.  Callers must supply already-sanitized candidates,
durable idempotency facts, quota observations, reservations, and the current
time.  The default policy is disabled, so importing or constructing these
objects can never authorize a provider request.  A current durable
provider-activity snapshot is also required so terminal attempts cannot vanish
from request-spacing decisions.

The decisions here are plans only.  A future persistence boundary must reserve
quota and an outbox record atomically before publishing any work.  In
particular, a returned ``PlannedDispatch`` is not evidence that a task was
published or that a provider was contacted.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from sam_analytics.ingestion_runs import (
    IngestionFailure,
    IngestionFailureClass,
    IngestionFailureCode,
)

_SAFE_PROVIDER_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_SAFE_SOURCE_TYPE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SAFE_POLICY_VERSION_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_BATCH_SIZE = 10
_MAX_ATTEMPTS = 5
_MAX_POLICY_DURATION = timedelta(days=7)
DISPATCH_IDEMPOTENCY_SCHEME = "sam-ingestion-dispatch-v1"
RETRY_SCHEDULE_FINGERPRINT_SCHEME = "sam-ingestion-retry-schedule-v1"


class DispatchValidationError(ValueError):
    """A dispatcher input is malformed or outside the fixed safety bounds."""


class DispatchBlockReason(StrEnum):
    """Finite, credential-safe reasons why a candidate was not admitted."""

    DISABLED = "disabled"
    SOURCE_NOT_ALLOWED = "source_not_allowed"
    DUPLICATE = "duplicate"
    ACTIVITY_UNAVAILABLE = "activity_unavailable"
    ACTIVITY_STALE = "activity_stale"
    QUOTA_UNAVAILABLE = "quota_unavailable"
    QUOTA_STALE = "quota_stale"
    QUOTA_FLOOR = "quota_floor"
    RATE_SPACING = "rate_spacing"
    BATCH_LIMIT = "batch_limit"


class RetryDisposition(StrEnum):
    """The only two outcomes after a failed dispatch attempt."""

    RETRY = "retry"
    DEAD_LETTER = "dead_letter"


class DeadLetterReason(StrEnum):
    """Finite reasons a failed attempt requires review rather than retry."""

    POLICY_DISABLED = "policy_disabled"
    NON_RETRYABLE = "non_retryable"
    ATTEMPTS_EXHAUSTED = "attempts_exhausted"
    RETRY_AFTER_EXCEEDS_LIMIT = "retry_after_exceeds_limit"


@dataclass(frozen=True)
class DispatchPolicy:
    """Code-reviewed limits for one provider's future dispatcher.

    ``enabled`` intentionally defaults to ``False``.  Enabling this value only
    permits this pure function to return a plan; it does not create a runtime
    path or grant credentials.
    """

    provider: str
    policy_version: str
    enabled: bool = False
    allowed_source_types: frozenset[str] = frozenset()
    max_batch_size: int = 1
    max_attempts: int = 3
    min_request_interval: timedelta = timedelta(seconds=30)
    quota_floor: int = 1
    quota_max_age: timedelta = timedelta(minutes=5)
    provider_activity_max_age: timedelta = timedelta(seconds=5)
    retry_delays: tuple[timedelta, ...] = (
        timedelta(seconds=30),
        timedelta(minutes=5),
    )
    max_retry_delay: timedelta = timedelta(hours=1)

    def __post_init__(self) -> None:
        _validate_safe_text(self.provider, _SAFE_PROVIDER_RE, "provider")
        _validate_safe_text(
            self.policy_version,
            _SAFE_POLICY_VERSION_RE,
            "policy version",
        )
        if not isinstance(self.enabled, bool):
            raise DispatchValidationError("enabled must be a boolean")
        if not isinstance(self.allowed_source_types, frozenset):
            raise DispatchValidationError("allowed source types must be an immutable set")
        for source_type in self.allowed_source_types:
            _validate_safe_text(
                source_type,
                _SAFE_SOURCE_TYPE_RE,
                "allowed source type",
            )
        if self.enabled and not self.allowed_source_types:
            raise DispatchValidationError(
                "enabled policies must name at least one allowed source type"
            )
        _validate_bounded_int(
            self.max_batch_size,
            label="max batch size",
            minimum=1,
            maximum=_MAX_BATCH_SIZE,
        )
        _validate_bounded_int(
            self.max_attempts,
            label="max attempts",
            minimum=1,
            maximum=_MAX_ATTEMPTS,
        )
        _validate_duration(
            self.min_request_interval,
            label="minimum request interval",
            allow_zero=True,
        )
        _validate_bounded_int(
            self.quota_floor,
            label="quota floor",
            minimum=0,
            maximum=2_147_483_647,
        )
        _validate_duration(self.quota_max_age, label="quota maximum age")
        _validate_duration(
            self.provider_activity_max_age,
            label="provider activity maximum age",
        )
        if not isinstance(self.retry_delays, tuple):
            raise DispatchValidationError("retry delays must be an immutable tuple")
        if len(self.retry_delays) < self.max_attempts - 1:
            raise DispatchValidationError("retry delays do not cover every permitted retry")
        if len(self.retry_delays) > _MAX_ATTEMPTS - 1:
            raise DispatchValidationError("too many retry delays are configured")
        for delay in self.retry_delays:
            _validate_duration(delay, label="retry delay")
        _validate_duration(self.max_retry_delay, label="maximum retry delay")
        if any(delay > self.max_retry_delay for delay in self.retry_delays):
            raise DispatchValidationError("retry delay exceeds the maximum retry delay")


def retry_schedule_sha256(policy: DispatchPolicy) -> str:
    """Return an exact digest of the retry inputs consumed after admission."""

    if not isinstance(policy, DispatchPolicy):
        raise DispatchValidationError("dispatch policy is required")
    components = [
        RETRY_SCHEDULE_FINGERPRINT_SCHEME,
        f"max_retry_delay_us={_timedelta_microseconds(policy.max_retry_delay)}",
        f"retry_delay_count={len(policy.retry_delays)}",
    ]
    components.extend(
        f"retry_delay_{index}_us={_timedelta_microseconds(delay)}"
        for index, delay in enumerate(policy.retry_delays)
    )
    return hashlib.sha256("\n".join(components).encode("ascii")).hexdigest()


@dataclass(frozen=True)
class DispatchCandidate:
    """One credential-free logical request proposed for a bounded time window."""

    provider: str
    source_type: str
    request_fingerprint_sha256: str
    window_start: datetime
    window_end: datetime
    estimated_cost: int = 1

    def __post_init__(self) -> None:
        _validate_safe_text(self.provider, _SAFE_PROVIDER_RE, "provider")
        _validate_safe_text(self.source_type, _SAFE_SOURCE_TYPE_RE, "source type")
        _validate_sha256(self.request_fingerprint_sha256, "request fingerprint")
        _validate_aware_time(self.window_start, "window start")
        _validate_aware_time(self.window_end, "window end")
        if self.window_end <= self.window_start:
            raise DispatchValidationError("dispatch window end must be after its start")
        if self.window_end - self.window_start > _MAX_POLICY_DURATION:
            raise DispatchValidationError("dispatch windows cannot exceed seven days")
        _validate_bounded_int(
            self.estimated_cost,
            label="estimated request cost",
            minimum=1,
            maximum=10_000,
        )

    @property
    def idempotency_preimage(self) -> str:
        """Return the versioned canonical text hashed for idempotency.

        Every variable field is validated to exclude ``|`` and timestamps are
        fixed-width UTC text.  PostgreSQL can reproduce the timestamp form with
        ``to_char(value AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"')``.
        The exact preimage is therefore suitable for a database uniqueness
        check without depending on JSON serialization details.

        Policy versions and task IDs are intentionally absent.  Changing an
        operational policy or redelivering a queue task must not authorize the
        same provider/scope/window request a second time.
        """

        return "|".join(
            (
                DISPATCH_IDEMPOTENCY_SCHEME,
                self.provider,
                self.source_type,
                self.request_fingerprint_sha256,
                _utc_text(self.window_start),
                _utc_text(self.window_end),
            )
        )

    @property
    def idempotency_key(self) -> str:
        """Return the lowercase SHA-256 of :attr:`idempotency_preimage`.

        Policy versions and task IDs are intentionally absent.  Changing an
        operational policy or redelivering a queue task must not authorize the
        same provider/scope/window request a second time.
        """

        return hashlib.sha256(self.idempotency_preimage.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class QuotaSnapshot:
    """Last trusted provider quota observation, containing no response data."""

    provider: str
    remaining: int
    observed_at: datetime

    def __post_init__(self) -> None:
        _validate_safe_text(self.provider, _SAFE_PROVIDER_RE, "provider")
        _validate_bounded_int(
            self.remaining,
            label="quota remaining",
            minimum=0,
            maximum=2_147_483_647,
        )
        _validate_aware_time(self.observed_at, "quota observation time")


@dataclass(frozen=True)
class ProviderActivitySnapshot:
    """Current durable view of the latest attempt that could contact a provider.

    ``latest_attempt_at=None`` means the trusted repository observed no prior
    attempt.  A missing snapshot is different: admission fails closed because
    it cannot safely enforce spacing after terminal attempts.
    """

    provider: str
    observed_at: datetime
    latest_attempt_at: datetime | None = None

    def __post_init__(self) -> None:
        _validate_safe_text(self.provider, _SAFE_PROVIDER_RE, "provider")
        _validate_aware_time(self.observed_at, "provider activity observation time")
        if self.latest_attempt_at is not None:
            _validate_aware_time(self.latest_attempt_at, "latest provider attempt time")
            if self.latest_attempt_at > self.observed_at:
                raise DispatchValidationError(
                    "latest provider attempt cannot follow its activity observation"
                )


@dataclass(frozen=True)
class QuotaReservation:
    """One durable local reservation made against a quota observation."""

    provider: str
    idempotency_key: str
    credits: int
    reserved_at: datetime

    def __post_init__(self) -> None:
        _validate_safe_text(self.provider, _SAFE_PROVIDER_RE, "provider")
        _validate_sha256(self.idempotency_key, "reservation idempotency key")
        _validate_bounded_int(
            self.credits,
            label="reserved credits",
            minimum=1,
            maximum=10_000,
        )
        _validate_aware_time(self.reserved_at, "reservation time")


@dataclass(frozen=True)
class PlannedDispatch:
    """Safe plan that still requires atomic persistence before publication."""

    candidate: DispatchCandidate
    idempotency_key: str
    policy_version: str
    max_attempts: int
    admitted_at: datetime


@dataclass(frozen=True)
class BlockedDispatch:
    """One candidate and its finite admission refusal."""

    candidate: DispatchCandidate
    idempotency_key: str
    reason: DispatchBlockReason


@dataclass(frozen=True)
class DispatchBatchDecision:
    """Deterministic result for one bounded dispatcher evaluation."""

    admitted: tuple[PlannedDispatch, ...]
    blocked: tuple[BlockedDispatch, ...]
    quota_remaining_after_reservations: int | None


@dataclass(frozen=True)
class RetryPlan:
    """Bounded next action for one already-failed attempt."""

    disposition: RetryDisposition
    failure_code: IngestionFailureCode
    completed_attempts: int
    next_attempt_at: datetime | None = None
    dead_letter_reason: DeadLetterReason | None = None

    def __post_init__(self) -> None:
        if self.disposition == RetryDisposition.RETRY:
            if self.next_attempt_at is None or self.dead_letter_reason is not None:
                raise DispatchValidationError("retry plans require only a next-attempt time")
        elif self.disposition == RetryDisposition.DEAD_LETTER:
            if self.next_attempt_at is not None or self.dead_letter_reason is None:
                raise DispatchValidationError("dead-letter plans require only a terminal reason")
        else:
            raise DispatchValidationError("retry disposition is invalid")


def admit_dispatch(
    candidates: Iterable[DispatchCandidate],
    *,
    policy: DispatchPolicy,
    quota: QuotaSnapshot | None,
    provider_activity: ProviderActivitySnapshot | None = None,
    reservations: Iterable[QuotaReservation] = (),
    existing_idempotency_keys: Iterable[str] = (),
    now: datetime,
) -> DispatchBatchDecision:
    """Plan an immediate bounded batch without performing any side effect.

    Every supplied reservation must still be outstanding and is subtracted,
    regardless of when it was made.  A reservation is released only by a
    separately audited reconciliation; a newer provider observation alone is
    not proof that an inconclusive dispatch consumed its reserved credit.  The
    required activity snapshot distinguishes "no prior attempt" from an
    unavailable query and preserves spacing after terminal attempts.
    """

    if not isinstance(policy, DispatchPolicy):
        raise DispatchValidationError("dispatch policy is required")
    _validate_aware_time(now, "dispatch decision time")
    candidate_values = _validated_candidates(candidates, policy)
    reservation_values = _validated_reservations(reservations, now=now)
    existing_keys = _validated_idempotency_keys(existing_idempotency_keys)

    if not policy.enabled:
        return _block_all(
            candidate_values,
            DispatchBlockReason.DISABLED,
            quota_remaining=None,
        )
    if quota is None:
        return _block_all(
            candidate_values,
            DispatchBlockReason.QUOTA_UNAVAILABLE,
            quota_remaining=None,
        )
    if not isinstance(quota, QuotaSnapshot) or quota.provider != policy.provider:
        raise DispatchValidationError("quota observation does not match the dispatch policy")
    if quota.observed_at > now:
        raise DispatchValidationError("quota observation time cannot be in the future")
    provider_reservations = tuple(
        reservation
        for reservation in reservation_values
        if reservation.provider == policy.provider
    )
    outstanding_credits = sum(
        reservation.credits for reservation in provider_reservations
    )
    quota_remaining = max(0, quota.remaining - outstanding_credits)

    if now - quota.observed_at > policy.quota_max_age:
        return _block_all(
            candidate_values,
            DispatchBlockReason.QUOTA_STALE,
            quota_remaining=quota_remaining,
        )

    if provider_activity is None:
        return _block_all(
            candidate_values,
            DispatchBlockReason.ACTIVITY_UNAVAILABLE,
            quota_remaining=quota_remaining,
        )
    if (
        not isinstance(provider_activity, ProviderActivitySnapshot)
        or provider_activity.provider != policy.provider
    ):
        raise DispatchValidationError(
            "provider activity observation does not match the dispatch policy"
        )
    if provider_activity.observed_at > now:
        raise DispatchValidationError(
            "provider activity observation time cannot be in the future"
        )
    if now - provider_activity.observed_at > policy.provider_activity_max_age:
        return _block_all(
            candidate_values,
            DispatchBlockReason.ACTIVITY_STALE,
            quota_remaining=quota_remaining,
        )

    activity_times = [quota.observed_at]
    activity_times.extend(
        reservation.reserved_at for reservation in provider_reservations
    )
    if provider_activity.latest_attempt_at is not None:
        activity_times.append(provider_activity.latest_attempt_at)
    last_activity_at = max(
        activity_times,
    )

    admitted: list[PlannedDispatch] = []
    blocked: list[BlockedDispatch] = []
    # A durable outstanding reservation is itself evidence that this logical
    # request has already crossed the admission boundary.  Requiring callers
    # to repeat those keys in ``existing_idempotency_keys`` would create a
    # replay gap if the two read models ever diverged.
    decided_keys = set(existing_keys)
    decided_keys.update(reservation.idempotency_key for reservation in reservation_values)
    for candidate in candidate_values:
        key = candidate.idempotency_key
        if candidate.source_type not in policy.allowed_source_types:
            blocked.append(_blocked(candidate, DispatchBlockReason.SOURCE_NOT_ALLOWED))
            continue
        if key in decided_keys:
            blocked.append(_blocked(candidate, DispatchBlockReason.DUPLICATE))
            continue
        decided_keys.add(key)
        if len(admitted) >= policy.max_batch_size:
            blocked.append(_blocked(candidate, DispatchBlockReason.BATCH_LIMIT))
            continue
        if now < last_activity_at + policy.min_request_interval:
            blocked.append(_blocked(candidate, DispatchBlockReason.RATE_SPACING))
            continue
        if quota_remaining - candidate.estimated_cost < policy.quota_floor:
            blocked.append(_blocked(candidate, DispatchBlockReason.QUOTA_FLOOR))
            continue

        admitted.append(
            PlannedDispatch(
                candidate=candidate,
                idempotency_key=key,
                policy_version=policy.policy_version,
                max_attempts=policy.max_attempts,
                admitted_at=now,
            )
        )
        quota_remaining -= candidate.estimated_cost
        last_activity_at = now

    return DispatchBatchDecision(
        admitted=tuple(admitted),
        blocked=tuple(blocked),
        quota_remaining_after_reservations=quota_remaining,
    )


def plan_retry(
    *,
    failure_code: IngestionFailureCode,
    completed_attempts: int,
    policy: DispatchPolicy,
    now: datetime,
    retry_after: timedelta | None = None,
) -> RetryPlan:
    """Return a bounded retry time or a terminal dead-letter decision.

    A supplied provider ``Retry-After`` is a lower bound.  It is never shortened
    to the local backoff.  If it exceeds the reviewed maximum, the attempt is
    dead-lettered for human review instead of waiting without a bound.
    """

    if not isinstance(policy, DispatchPolicy):
        raise DispatchValidationError("dispatch policy is required")
    if not isinstance(failure_code, IngestionFailureCode):
        raise DispatchValidationError("failure code must be an approved safe code")
    _validate_bounded_int(
        completed_attempts,
        label="completed attempts",
        minimum=1,
        maximum=_MAX_ATTEMPTS,
    )
    _validate_aware_time(now, "retry decision time")
    if completed_attempts > policy.max_attempts:
        raise DispatchValidationError(
            "completed attempts cannot exceed the reviewed policy limit"
        )
    if retry_after is not None:
        _validate_retry_after(retry_after)

    if not policy.enabled:
        return _dead_letter(
            failure_code,
            completed_attempts,
            DeadLetterReason.POLICY_DISABLED,
        )
    failure = IngestionFailure(failure_code)
    if failure.classification != IngestionFailureClass.RETRYABLE:
        return _dead_letter(
            failure_code,
            completed_attempts,
            DeadLetterReason.NON_RETRYABLE,
        )
    if completed_attempts >= policy.max_attempts:
        return _dead_letter(
            failure_code,
            completed_attempts,
            DeadLetterReason.ATTEMPTS_EXHAUSTED,
        )
    if retry_after is not None and retry_after > policy.max_retry_delay:
        return _dead_letter(
            failure_code,
            completed_attempts,
            DeadLetterReason.RETRY_AFTER_EXCEEDS_LIMIT,
        )

    local_delay = policy.retry_delays[completed_attempts - 1]
    effective_delay = max(local_delay, retry_after or timedelta(0))
    return RetryPlan(
        disposition=RetryDisposition.RETRY,
        failure_code=failure_code,
        completed_attempts=completed_attempts,
        next_attempt_at=now + effective_delay,
    )


def _validated_candidates(
    candidates: Iterable[DispatchCandidate],
    policy: DispatchPolicy,
) -> tuple[DispatchCandidate, ...]:
    if isinstance(candidates, (str, bytes)):
        raise DispatchValidationError("dispatch candidates must be an iterable of candidates")
    try:
        values = tuple(candidates)
    except TypeError:
        raise DispatchValidationError(
            "dispatch candidates must be an iterable of candidates"
        ) from None
    if not all(isinstance(candidate, DispatchCandidate) for candidate in values):
        raise DispatchValidationError("every dispatch candidate must be validated")
    if any(candidate.provider != policy.provider for candidate in values):
        raise DispatchValidationError("dispatch candidate does not match the policy provider")
    return values


def _validated_reservations(
    reservations: Iterable[QuotaReservation],
    *,
    now: datetime,
) -> tuple[QuotaReservation, ...]:
    if isinstance(reservations, (str, bytes)):
        raise DispatchValidationError("quota reservations must be an iterable of reservations")
    try:
        values = tuple(reservations)
    except TypeError:
        raise DispatchValidationError(
            "quota reservations must be an iterable of reservations"
        ) from None
    if not all(isinstance(reservation, QuotaReservation) for reservation in values):
        raise DispatchValidationError("every quota reservation must be validated")
    if any(reservation.reserved_at > now for reservation in values):
        raise DispatchValidationError("quota reservation time cannot be in the future")
    return values


def _validated_idempotency_keys(values: Iterable[str]) -> frozenset[str]:
    if isinstance(values, (str, bytes)):
        raise DispatchValidationError("existing idempotency keys must be an iterable")
    try:
        keys = frozenset(values)
    except TypeError:
        raise DispatchValidationError("existing idempotency keys must be an iterable") from None
    for key in keys:
        _validate_sha256(key, "existing idempotency key")
    return keys


def _block_all(
    candidates: tuple[DispatchCandidate, ...],
    reason: DispatchBlockReason,
    *,
    quota_remaining: int | None,
) -> DispatchBatchDecision:
    return DispatchBatchDecision(
        admitted=(),
        blocked=tuple(_blocked(candidate, reason) for candidate in candidates),
        quota_remaining_after_reservations=quota_remaining,
    )


def _blocked(candidate: DispatchCandidate, reason: DispatchBlockReason) -> BlockedDispatch:
    return BlockedDispatch(
        candidate=candidate,
        idempotency_key=candidate.idempotency_key,
        reason=reason,
    )


def _dead_letter(
    failure_code: IngestionFailureCode,
    completed_attempts: int,
    reason: DeadLetterReason,
) -> RetryPlan:
    return RetryPlan(
        disposition=RetryDisposition.DEAD_LETTER,
        failure_code=failure_code,
        completed_attempts=completed_attempts,
        dead_letter_reason=reason,
    )


def _validate_safe_text(
    value: object,
    pattern: re.Pattern[str],
    label: str,
) -> None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise DispatchValidationError(f"{label} must be a safe identifier")


def _validate_sha256(value: object, label: str) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise DispatchValidationError(f"{label} must be a lowercase SHA-256 digest")


def _validate_bounded_int(
    value: object,
    *,
    label: str,
    minimum: int,
    maximum: int,
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise DispatchValidationError(f"{label} must be between {minimum} and {maximum}")


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
        or value > _MAX_POLICY_DURATION
    ):
        qualifier = "non-negative" if allow_zero else "positive"
        raise DispatchValidationError(
            f"{label} must be a {qualifier} duration no longer than seven days"
        )


def _validate_retry_after(value: object) -> None:
    # A provider may return an arbitrarily long wait.  It is valid input, but
    # anything above the reviewed policy limit becomes a finite dead-letter
    # decision rather than an uncaught validation error.
    if not isinstance(value, timedelta) or value < timedelta(0):
        raise DispatchValidationError("Retry-After must be a non-negative duration")


def _validate_aware_time(value: object, label: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise DispatchValidationError(f"{label} must be timezone-aware")


def _timedelta_microseconds(value: timedelta) -> int:
    return (
        (value.days * 86_400 + value.seconds) * 1_000_000
        + value.microseconds
    )


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )
