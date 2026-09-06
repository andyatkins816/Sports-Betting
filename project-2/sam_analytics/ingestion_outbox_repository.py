"""PostgreSQL adapter for the inactive ingestion outbox runtime.

Every method calls one narrow migration-owned function and returns only after
its transaction commits.  The adapter has no broker, provider client,
credential loader, scheduler, environment lookup, or reconciliation entry
point.  In particular, it cannot turn an expired provider-attempt claim into a
second provider call.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any, Protocol, TypeVar
from uuid import UUID

from sam_analytics.ingestion_outbox_runtime import (
    AttemptClaimDisposition,
    AttemptCompletion,
    AttemptCompletionCommit,
    AttemptCompletionOutcome,
    ClaimedDispatchAttempt,
    DispatchAttemptClaim,
    IngestionOutboxConfigurationError,
    OutboxMessage,
    OutboxPublicationClaim,
    PublicationCommit,
)
from sam_analytics.provider_contracts import ProviderUse


class IngestionOutboxRepositoryError(RuntimeError):
    """Base error containing only a credential-safe repository message."""


class IngestionOutboxRepositoryUnavailable(IngestionOutboxRepositoryError):
    """A database operation could not be confirmed safely."""


class _Cursor(Protocol):
    def execute(self, query: str, params: Sequence[Any] | None = None) -> Any:
        ...

    def fetchone(self) -> Sequence[Any] | None:
        ...

    def __enter__(self) -> _Cursor:
        ...

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        ...


class _Connection(Protocol):
    def cursor(self) -> _Cursor:
        ...

    def transaction(self) -> Any:
        ...

    def close(self) -> None:
        ...


ConnectionFactory = Callable[[str], _Connection]
_Decoded = TypeVar("_Decoded")

_IDENTITY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SENSITIVE_IDENTITY_RE = re.compile(
    r"(?:api[-_]?key|token|secret|password|authorization|credential|cookie|bearer)",
    re.IGNORECASE,
)


class PostgresIngestionOutboxRepository:
    """Map the broker-neutral runtime to migration 008 stored functions."""

    def __init__(
        self,
        database_url: str,
        *,
        connection_factory: ConnectionFactory | None = None,
    ) -> None:
        if not isinstance(database_url, str) or not database_url.strip():
            raise IngestionOutboxConfigurationError(
                "a database connection must be configured for the ingestion outbox"
            )
        self._database_url = database_url
        self._connection_factory = connection_factory or _connect_postgres

    def __repr__(self) -> str:
        return "PostgresIngestionOutboxRepository()"

    def claim_ingestion_outbox_publication(
        self,
        *,
        publisher_identity: str,
        lease_token: UUID,
    ) -> OutboxPublicationClaim | None:
        _validate_call_identity(publisher_identity, "publisher identity")
        _validate_call_uuid(lease_token, "publication lease token")
        return self._call_one(
            """
            SELECT disposition, publication_claim_id, outbox_id,
                   dispatch_id, attempt_number, claimed_at, lease_expires_at
            FROM claim_ingestion_outbox_publication(%s, %s)
            """,
            (publisher_identity, lease_token),
            lambda row: _decode_publication_claim(
                row,
                publisher_identity=publisher_identity,
                lease_token=lease_token,
            ),
        )

    def record_ingestion_outbox_publication(
        self,
        claim: OutboxPublicationClaim,
    ) -> PublicationCommit:
        if not isinstance(claim, OutboxPublicationClaim):
            raise IngestionOutboxConfigurationError(
                "an exact publication claim is required"
            )
        return self._call_one(
            """
            SELECT disposition, delivery_id
            FROM record_ingestion_outbox_publication(%s, %s, %s)
            """,
            (
                claim.publication_claim_id,
                claim.publisher_identity,
                claim.lease_token,
            ),
            _decode_publication_commit,
        )

    def claim_ingestion_dispatch_attempt(
        self,
        message: OutboxMessage,
        *,
        worker_identity: str,
        lease_token: UUID,
    ) -> DispatchAttemptClaim:
        if not isinstance(message, OutboxMessage):
            raise IngestionOutboxConfigurationError(
                "an exact outbox message is required"
            )
        _validate_call_identity(worker_identity, "worker identity")
        _validate_call_uuid(lease_token, "attempt lease token")
        return self._call_one(
            """
            SELECT disposition, provider_call_permitted, claim_id,
                   running_transition_id, provider_use_authorization_id,
                   quota_receipt_id, provider, source_type,
                   request_fingerprint_sha256, window_start, window_end,
                   estimated_cost, policy_version, max_attempts,
                   license_scope, license_version, exposure,
                   started_at, lease_expires_at, min_request_interval,
                   quota_floor, quota_max_age, retry_schedule_sha256,
                   authorization_effective_until
            FROM claim_ingestion_dispatch_attempt(%s, %s, %s, %s)
            """,
            (
                message.dispatch_id,
                message.attempt_number,
                worker_identity,
                lease_token,
            ),
            lambda row: _decode_attempt_claim(
                row,
                message=message,
                worker_identity=worker_identity,
                lease_token=lease_token,
            ),
        )

    def read_ingestion_dispatch_attempt_time(
        self,
        attempt: ClaimedDispatchAttempt,
    ) -> datetime:
        if not isinstance(attempt, ClaimedDispatchAttempt):
            raise IngestionOutboxConfigurationError(
                "an exact claimed attempt is required"
            )
        return self._call_one(
            "SELECT read_ingestion_dispatch_attempt_time(%s, %s, %s)",
            (
                attempt.attempt_claim_id,
                attempt.worker_identity,
                attempt.lease_token,
            ),
            _decode_attempt_time,
        )

    def complete_ingestion_dispatch_attempt(
        self,
        attempt: ClaimedDispatchAttempt,
        completion: AttemptCompletion,
    ) -> AttemptCompletionCommit:
        _validate_exact_completion(attempt, completion)
        outcome, failure_code, dead_letter_reason, retry_not_before_at, retry_safety = (
            _completion_parameters(completion)
        )
        return self._call_one(
            """
            SELECT disposition, completion_id
            FROM complete_ingestion_dispatch_attempt(
                %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            """,
            (
                attempt.attempt_claim_id,
                attempt.worker_identity,
                attempt.lease_token,
                outcome,
                failure_code,
                dead_letter_reason,
                retry_not_before_at,
                completion.provider_payload_receipt_id,
                retry_safety,
            ),
            _decode_attempt_completion_commit,
        )

    def _call_one(
        self,
        query: str,
        params: Sequence[Any],
        decode: Callable[[Sequence[Any] | None], _Decoded],
    ) -> _Decoded:
        connection: _Connection | None = None
        repository_error: IngestionOutboxRepositoryError | None = None
        operation_failed = False
        try:
            connection = self._connection_factory(self._database_url)
            with connection.transaction():
                with connection.cursor() as cursor:
                    cursor.execute(query, params)
                    # Decode before the transaction exits. A malformed result
                    # must roll back any claim/completion facts it just wrote.
                    result = decode(cursor.fetchone())
        except IngestionOutboxRepositoryError as error:
            repository_error = error
        except Exception:
            operation_failed = True
        finally:
            _close_safely(connection)
        if repository_error is not None:
            raise repository_error from None
        if operation_failed:
            raise IngestionOutboxRepositoryUnavailable(
                "ingestion outbox database operation failed"
            ) from None
        return result


def _decode_publication_claim(
    row: Sequence[Any] | None,
    *,
    publisher_identity: str,
    lease_token: UUID,
) -> OutboxPublicationClaim | None:
    if row is None:
        return None
    if len(row) != 7 or row[0] != "publishable":
        raise IngestionOutboxRepositoryUnavailable(
            "stored ingestion publication claim is invalid"
        )
    _require_uuid(row[1], "publication claim id")
    _require_uuid(row[2], "outbox id")
    _require_uuid(row[3], "dispatch id")
    try:
        return OutboxPublicationClaim(
            publication_claim_id=row[1],
            message=OutboxMessage(
                dispatch_id=row[3],
                attempt_number=row[4],
            ),
            publisher_identity=publisher_identity,
            lease_token=lease_token,
            claimed_at=row[5],
            lease_expires_at=row[6],
        )
    except (TypeError, ValueError):
        raise IngestionOutboxRepositoryUnavailable(
            "stored ingestion publication claim is invalid"
        ) from None


def _decode_publication_commit(
    row: Sequence[Any] | None,
) -> PublicationCommit:
    if row is None or len(row) != 2:
        raise IngestionOutboxRepositoryUnavailable(
            "stored ingestion publication result is invalid"
        )
    _require_uuid(row[1], "publication delivery id")
    mapping = {
        "recorded": PublicationCommit.RECORDED,
        "already_recorded": PublicationCommit.ALREADY_RECORDED,
    }
    try:
        return mapping[row[0]]
    except (KeyError, TypeError):
        raise IngestionOutboxRepositoryUnavailable(
            "stored ingestion publication result is invalid"
        ) from None


def _decode_attempt_claim(
    row: Sequence[Any] | None,
    *,
    message: OutboxMessage,
    worker_identity: str,
    lease_token: UUID,
) -> DispatchAttemptClaim:
    if row is None or len(row) != 24:
        raise IngestionOutboxRepositoryUnavailable(
            "stored ingestion attempt claim is invalid"
        )
    try:
        disposition = AttemptClaimDisposition(row[0])
    except (TypeError, ValueError):
        raise IngestionOutboxRepositoryUnavailable(
            "stored ingestion attempt claim is invalid"
        ) from None
    if disposition is not AttemptClaimDisposition.STARTED:
        if row[1] is not False:
            raise IngestionOutboxRepositoryUnavailable(
                "stored ingestion attempt permission is invalid"
            )
        return DispatchAttemptClaim(message=message, disposition=disposition)
    if row[1] is not True:
        raise IngestionOutboxRepositoryUnavailable(
            "stored ingestion attempt permission is invalid"
        )
    _require_estimated_cost(row[11])
    try:
        started = ClaimedDispatchAttempt(
            attempt_claim_id=row[2],
            message=message,
            worker_identity=worker_identity,
            lease_token=lease_token,
            running_transition_id=row[3],
            provider_use_authorization_id=row[4],
            quota_receipt_id=row[5],
            provider_use=ProviderUse(
                provider=row[6],
                source_type=row[7],
                license_scope=row[14],
                license_version=row[15],
                exposure=row[16],
            ),
            request_fingerprint_sha256=row[8],
            estimated_cost=row[11],
            policy_version=row[12],
            max_attempts=row[13],
            min_request_interval=row[19],
            quota_floor=row[20],
            quota_max_age=row[21],
            retry_schedule_sha256=row[22],
            window_start=row[9],
            window_end=row[10],
            authorization_effective_until=row[23],
            claimed_at=row[17],
            lease_expires_at=row[18],
        )
    except (TypeError, ValueError):
        raise IngestionOutboxRepositoryUnavailable(
            "stored ingestion attempt claim is invalid"
        ) from None
    return DispatchAttemptClaim(
        message=message,
        disposition=AttemptClaimDisposition.STARTED,
        started=started,
    )


def _decode_attempt_time(row: Sequence[Any] | None) -> datetime:
    if row is None or len(row) != 1 or not _aware(row[0]):
        raise IngestionOutboxRepositoryUnavailable(
            "stored ingestion attempt time is invalid"
        )
    return row[0]


def _decode_attempt_completion_commit(
    row: Sequence[Any] | None,
) -> AttemptCompletionCommit:
    if row is None or len(row) != 2:
        raise IngestionOutboxRepositoryUnavailable(
            "stored ingestion attempt completion is invalid"
        )
    _require_uuid(row[1], "attempt completion id")
    mapping = {
        "committed": AttemptCompletionCommit.COMMITTED,
        "already_committed": AttemptCompletionCommit.ALREADY_COMMITTED,
    }
    try:
        return mapping[row[0]]
    except (KeyError, TypeError):
        raise IngestionOutboxRepositoryUnavailable(
            "stored ingestion attempt completion is invalid"
        ) from None


def _completion_parameters(
    completion: AttemptCompletion,
) -> tuple[str, str | None, str | None, datetime | None, str | None]:
    if completion.outcome is AttemptCompletionOutcome.SUCCEEDED:
        return "succeeded", None, None, None, None
    plan = completion.retry_plan
    if plan is None:
        raise IngestionOutboxConfigurationError(
            "a failed completion requires a retry plan"
        )
    if completion.outcome is AttemptCompletionOutcome.RETRY_WAIT:
        retry_safety = completion.retry_safety
        if retry_safety is None:
            raise IngestionOutboxConfigurationError(
                "a retry completion requires replay-safety proof"
            )
        return (
            "retry_wait",
            plan.failure_code.value,
            None,
            plan.next_attempt_at,
            retry_safety.value,
        )
    if completion.outcome is AttemptCompletionOutcome.DEAD_LETTERED:
        reason = plan.dead_letter_reason
        if reason is None:
            raise IngestionOutboxConfigurationError(
                "a dead-letter completion requires a reason"
            )
        return "dead_lettered", plan.failure_code.value, reason.value, None, None
    raise IngestionOutboxConfigurationError("attempt completion outcome is invalid")


def _validate_exact_completion(
    attempt: ClaimedDispatchAttempt,
    completion: AttemptCompletion,
) -> None:
    if not isinstance(attempt, ClaimedDispatchAttempt) or not isinstance(
        completion, AttemptCompletion
    ):
        raise IngestionOutboxConfigurationError(
            "an exact claimed attempt and completion are required"
        )
    if (
        completion.attempt_claim_id != attempt.attempt_claim_id
        or completion.message != attempt.message
        or completion.provider_use_authorization_id
        != attempt.provider_use_authorization_id
    ):
        raise IngestionOutboxConfigurationError(
            "attempt completion conflicts with its claimed attempt"
        )


def _require_uuid(value: object, label: str) -> None:
    if not isinstance(value, UUID) or value.int == 0:
        raise IngestionOutboxRepositoryUnavailable(f"stored {label} is invalid")


def _validate_call_identity(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or not _IDENTITY_RE.fullmatch(value)
        or _SENSITIVE_IDENTITY_RE.search(value)
    ):
        raise IngestionOutboxConfigurationError(f"{label} must be a safe identifier")


def _validate_call_uuid(value: object, label: str) -> None:
    if not isinstance(value, UUID) or value.int == 0:
        raise IngestionOutboxConfigurationError(f"{label} must be a non-zero UUID")


def _require_estimated_cost(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 10_000:
        raise IngestionOutboxRepositoryUnavailable(
            "stored dispatch estimated cost is invalid"
        )


def _aware(value: object) -> bool:
    return (
        isinstance(value, datetime)
        and value.tzinfo is not None
        and value.utcoffset() is not None
    )


def _close_safely(connection: _Connection | None) -> None:
    if connection is None:
        return
    try:
        connection.close()
    except Exception:
        return


def _connect_postgres(database_url: str) -> _Connection:
    try:
        import psycopg

        return psycopg.connect(
            database_url,
            application_name="sam-analytics-ingestion-outbox",
            connect_timeout=5,
            options="-c statement_timeout=10000",
        )
    except Exception:
        raise IngestionOutboxRepositoryUnavailable(
            "ingestion outbox database is unavailable"
        ) from None
