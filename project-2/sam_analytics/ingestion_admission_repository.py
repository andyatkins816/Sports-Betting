"""Transactional PostgreSQL boundary for an inactive ingestion admission plan.

This adapter deliberately stops at durable ``pending`` outbox intent.  It does
not publish a message, construct a provider client, read a credential, import a
worker, or activate a scheduler.  The pure policy in
:mod:`sam_analytics.ingestion_dispatch` remains the only admission decision
maker; this module supplies its trusted PostgreSQL observations and persists an
admitted initial bundle atomically.

Disabled policies return before a database connection is attempted.  Enabled
decisions are serialized per provider by the database advisory-lock function,
and use one database-owned timestamp for both planning and persistence.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import UUID, uuid4

from sam_analytics.ingestion_dispatch import (
    DispatchBatchDecision,
    DispatchCandidate,
    DispatchPolicy,
    DispatchValidationError,
    PlannedDispatch,
    ProviderActivitySnapshot,
    QuotaReservation,
    QuotaSnapshot,
    admit_dispatch,
)
from sam_analytics.provider_contracts import (
    ProviderContractViolation,
    ProviderUse,
    validate_provider_use,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class IngestionAdmissionRepositoryError(RuntimeError):
    """Base error containing only a credential-safe admission message."""


class IngestionAdmissionRepositoryUnavailable(IngestionAdmissionRepositoryError):
    """The durable admission boundary could not be evaluated safely."""


class IngestionAdmissionRepositoryConflict(IngestionAdmissionRepositoryError):
    """Stored authorization or admission facts conflict with this request."""


@dataclass(frozen=True)
class PersistedAdmission:
    """Safe identity of one dispatch bundle committed by this call."""

    ingestion_dispatch_id: UUID
    idempotency_key: str


@dataclass(frozen=True)
class AdmissionRepositoryResult:
    """The pure decision and identities of only the newly persisted plans."""

    decision: DispatchBatchDecision
    persisted: tuple[PersistedAdmission, ...]


class IngestionAdmissionRepository(Protocol):
    """Narrow persistence seam for one provider admission evaluation."""

    def admit(
        self,
        candidates: Iterable[DispatchCandidate],
        *,
        policy: DispatchPolicy,
        provider_use: ProviderUse | None = None,
    ) -> AdmissionRepositoryResult:
        """Evaluate and atomically persist a bounded initial admission batch."""

        ...


class _Cursor(Protocol):
    def execute(self, query: str, params: Sequence[Any] | None = None) -> Any:
        ...

    def fetchone(self) -> Sequence[Any] | None:
        ...

    def fetchall(self) -> Sequence[Sequence[Any]]:
        ...

    def __enter__(self) -> "_Cursor":
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


class PostgresIngestionAdmissionRepository:
    """Plan and append inactive dispatch bundles under one provider lock."""

    def __init__(
        self,
        database_url: str,
        *,
        connection_factory: ConnectionFactory | None = None,
    ) -> None:
        if not isinstance(database_url, str) or not database_url.strip():
            raise DispatchValidationError(
                "a database connection must be configured for ingestion admission"
            )
        self._database_url = database_url
        self._connection_factory = connection_factory or _connect_postgres

    def __repr__(self) -> str:
        return "PostgresIngestionAdmissionRepository()"

    def admit(
        self,
        candidates: Iterable[DispatchCandidate],
        *,
        policy: DispatchPolicy,
        provider_use: ProviderUse | None = None,
    ) -> AdmissionRepositoryResult:
        """Evaluate and persist a bounded initial dispatch batch.

        A disabled policy is evaluated without opening PostgreSQL.  Enabled
        policies require an exact, currently effective private-raw use.  All
        durable reads, the pure policy call, and every admitted four-row bundle
        occur in a single provider-serialized transaction.
        """

        candidate_values = _materialize_candidates(candidates)
        if not isinstance(policy, DispatchPolicy):
            raise DispatchValidationError("dispatch policy is required")

        if not policy.enabled:
            decision = admit_dispatch(
                candidate_values,
                policy=policy,
                quota=None,
                now=datetime.now(UTC),
            )
            return AdmissionRepositoryResult(decision=decision, persisted=())

        use = _validate_enabled_inputs(candidate_values, policy, provider_use)
        connection: _Connection | None = None
        failure_kind: str | None = None
        result: AdmissionRepositoryResult | None = None
        try:
            connection = self._connection_factory(self._database_url)
            with connection.transaction():
                with connection.cursor() as cursor:
                    cursor.execute(
                        "SELECT lock_ingestion_provider(%s)",
                        (policy.provider,),
                    )
                    cursor.fetchone()

                    cursor.execute("SELECT clock_timestamp()")
                    now = _decode_database_time(cursor.fetchone())
                    authorization_id = _read_authorization_id(
                        cursor,
                        provider_use=use,
                        now=now,
                    )
                    quota_receipt_id, quota = _read_quota_snapshot(
                        cursor,
                        provider_use=use,
                    )

                    if quota is None or quota_receipt_id is None:
                        decision = admit_dispatch(
                            candidate_values,
                            policy=policy,
                            quota=None,
                            now=now,
                        )
                        result = AdmissionRepositoryResult(
                            decision=decision,
                            persisted=(),
                        )
                    else:
                        reservations = _read_reservations(
                            cursor,
                            provider=policy.provider,
                        )
                        existing_keys = _read_existing_keys(
                            cursor,
                            provider=policy.provider,
                            candidate_keys=tuple(
                                candidate.idempotency_key
                                for candidate in candidate_values
                            ),
                        )
                        activity = _read_provider_activity(
                            cursor,
                            provider=policy.provider,
                            observed_at=now,
                        )
                        decision = admit_dispatch(
                            candidate_values,
                            policy=policy,
                            quota=quota,
                            provider_activity=activity,
                            reservations=reservations,
                            existing_idempotency_keys=existing_keys,
                            now=now,
                        )
                        persisted = tuple(
                            _insert_initial_bundle(
                                cursor,
                                plan=plan,
                                provider_use_authorization_id=authorization_id,
                                provider_payload_receipt_id=quota_receipt_id,
                                now=now,
                            )
                            for plan in decision.admitted
                        )
                        result = AdmissionRepositoryResult(
                            decision=decision,
                            persisted=persisted,
                        )
        except IngestionAdmissionRepositoryError:
            raise
        except Exception as error:
            failure_kind = _write_failure_kind(error)
        finally:
            _close_safely(connection)

        if failure_kind == "conflict":
            raise IngestionAdmissionRepositoryConflict(
                "ingestion admission conflicts with stored facts"
            )
        if failure_kind is not None or result is None:
            raise IngestionAdmissionRepositoryUnavailable(
                "ingestion admission database evaluation failed"
            )
        return result


def _materialize_candidates(
    candidates: Iterable[DispatchCandidate],
) -> tuple[DispatchCandidate, ...]:
    if isinstance(candidates, (str, bytes)):
        raise DispatchValidationError(
            "dispatch candidates must be an iterable of candidates"
        )
    try:
        return tuple(candidates)
    except TypeError:
        raise DispatchValidationError(
            "dispatch candidates must be an iterable of candidates"
        ) from None


def _validate_enabled_inputs(
    candidates: tuple[DispatchCandidate, ...],
    policy: DispatchPolicy,
    provider_use: ProviderUse | None,
) -> ProviderUse:
    if not all(isinstance(candidate, DispatchCandidate) for candidate in candidates):
        raise DispatchValidationError("every dispatch candidate must be validated")
    if any(candidate.provider != policy.provider for candidate in candidates):
        raise DispatchValidationError(
            "dispatch candidate does not match the policy provider"
        )
    if provider_use is None:
        raise DispatchValidationError(
            "an exact private provider use is required for enabled admission"
        )
    try:
        validate_provider_use(provider_use)
    except ProviderContractViolation:
        raise DispatchValidationError(
            "an exact private provider use is required for enabled admission"
        ) from None
    if provider_use.exposure != "private_raw":
        raise DispatchValidationError(
            "enabled admission requires private raw provider authorization"
        )
    if provider_use.provider != policy.provider:
        raise DispatchValidationError(
            "provider use does not match the dispatch policy"
        )
    if any(candidate.source_type != provider_use.source_type for candidate in candidates):
        raise DispatchValidationError(
            "dispatch candidates must match the exact provider use source type"
        )
    return provider_use


def _decode_database_time(row: Sequence[Any] | None) -> datetime:
    if row is None or len(row) != 1 or not _aware(row[0]):
        raise IngestionAdmissionRepositoryUnavailable(
            "ingestion admission database time is unavailable"
        )
    return row[0]


def _read_authorization_id(
    cursor: _Cursor,
    *,
    provider_use: ProviderUse,
    now: datetime,
) -> UUID:
    cursor.execute(
        """
        SELECT id, authorization_manifest_sha256
        FROM provider_use_authorization
        WHERE provider = %s
          AND license_scope = %s
          AND license_version = %s
          AND source_type = %s
          AND exposure = %s
          AND reviewed_at <= %s
          AND effective_from <= %s
          AND effective_until > %s
        """,
        (
            provider_use.provider,
            provider_use.license_scope,
            provider_use.license_version,
            provider_use.source_type,
            provider_use.exposure,
            now,
            now,
            now,
        ),
    )
    row = cursor.fetchone()
    if row is None:
        raise IngestionAdmissionRepositoryConflict(
            "an effective provider use authorization is not available"
        )
    if (
        len(row) != 2
        or not isinstance(row[0], UUID)
        or not isinstance(row[1], str)
        or _SHA256_RE.fullmatch(row[1]) is None
    ):
        raise IngestionAdmissionRepositoryUnavailable(
            "stored provider use authorization is invalid"
        )
    return row[0]


def _read_quota_snapshot(
    cursor: _Cursor,
    *,
    provider_use: ProviderUse,
) -> tuple[UUID | None, QuotaSnapshot | None]:
    cursor.execute(
        """
        SELECT id, provider_quota_remaining, received_at
        FROM provider_payload_receipt
        WHERE provider = %s
          AND license_scope = %s
          AND license_version = %s
          AND provider_quota_remaining IS NOT NULL
        ORDER BY received_at DESC, created_at DESC,
                 provider_quota_remaining ASC, id DESC
        LIMIT 1
        """,
        (
            provider_use.provider,
            provider_use.license_scope,
            provider_use.license_version,
        ),
    )
    row = cursor.fetchone()
    if row is None:
        return None, None
    if len(row) != 3 or not isinstance(row[0], UUID):
        raise IngestionAdmissionRepositoryUnavailable(
            "stored provider quota observation is invalid"
        )
    try:
        quota = QuotaSnapshot(
            provider=provider_use.provider,
            remaining=row[1],
            observed_at=row[2],
        )
    except (TypeError, ValueError):
        raise IngestionAdmissionRepositoryUnavailable(
            "stored provider quota observation is invalid"
        ) from None
    return row[0], quota


def _read_reservations(
    cursor: _Cursor,
    *,
    provider: str,
) -> tuple[QuotaReservation, ...]:
    # No append-only release/reconciliation fact exists yet.  Every historical
    # reservation therefore remains outstanding, even after a newer receipt or
    # a terminal dispatch transition.
    cursor.execute(
        """
        SELECT dispatch.idempotency_key,
               reservation.reserved_credits,
               reservation.reserved_at
        FROM ingestion_quota_reservation AS reservation
        JOIN ingestion_dispatch AS dispatch
          ON dispatch.id = reservation.ingestion_dispatch_id
        WHERE dispatch.provider = %s
        ORDER BY reservation.reserved_at,
                 reservation.ingestion_dispatch_id,
                 reservation.attempt_number
        """,
        (provider,),
    )
    rows = cursor.fetchall()
    if any(len(row) != 3 for row in rows):
        raise IngestionAdmissionRepositoryUnavailable(
            "stored provider quota reservations are invalid"
        )
    try:
        return tuple(
            QuotaReservation(
                provider=provider,
                idempotency_key=row[0],
                credits=row[1],
                reserved_at=row[2],
            )
            for row in rows
        )
    except (TypeError, ValueError):
        raise IngestionAdmissionRepositoryUnavailable(
            "stored provider quota reservations are invalid"
        ) from None


def _read_existing_keys(
    cursor: _Cursor,
    *,
    provider: str,
    candidate_keys: tuple[str, ...],
) -> tuple[str, ...]:
    if not candidate_keys:
        return ()
    cursor.execute(
        """
        SELECT idempotency_key
        FROM ingestion_dispatch
        WHERE provider = %s
          AND idempotency_key = ANY(%s::text[])
        ORDER BY idempotency_key
        """,
        (provider, list(candidate_keys)),
    )
    rows = cursor.fetchall()
    if any(
        len(row) != 1
        or not isinstance(row[0], str)
        or _SHA256_RE.fullmatch(row[0]) is None
        for row in rows
    ):
        raise IngestionAdmissionRepositoryUnavailable(
            "stored ingestion dispatch identities are invalid"
        )
    return tuple(row[0] for row in rows)


def _read_provider_activity(
    cursor: _Cursor,
    *,
    provider: str,
    observed_at: datetime,
) -> ProviderActivitySnapshot:
    cursor.execute(
        """
        SELECT max(transition.occurred_at)
        FROM ingestion_dispatch_transition AS transition
        JOIN ingestion_dispatch AS dispatch
          ON dispatch.id = transition.ingestion_dispatch_id
        WHERE dispatch.provider = %s
          AND transition.worker_identity IS NOT NULL
        """,
        (provider,),
    )
    row = cursor.fetchone()
    if row is None or len(row) != 1:
        raise IngestionAdmissionRepositoryUnavailable(
            "stored provider activity is unavailable"
        )
    try:
        return ProviderActivitySnapshot(
            provider=provider,
            observed_at=observed_at,
            latest_attempt_at=row[0],
        )
    except (TypeError, ValueError):
        raise IngestionAdmissionRepositoryUnavailable(
            "stored provider activity is invalid"
        ) from None


def _insert_initial_bundle(
    cursor: _Cursor,
    *,
    plan: PlannedDispatch,
    provider_use_authorization_id: UUID,
    provider_payload_receipt_id: UUID,
    now: datetime,
) -> PersistedAdmission:
    dispatch_id = uuid4()
    candidate = plan.candidate
    cursor.execute(
        """
        INSERT INTO ingestion_dispatch (
            id, provider, source_type, request_fingerprint_sha256,
            window_start, window_end, estimated_cost, policy_version,
            max_attempts, admitted_at, idempotency_key,
            provider_use_authorization_id
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            dispatch_id,
            candidate.provider,
            candidate.source_type,
            candidate.request_fingerprint_sha256,
            candidate.window_start,
            candidate.window_end,
            candidate.estimated_cost,
            plan.policy_version,
            plan.max_attempts,
            now,
            plan.idempotency_key,
            provider_use_authorization_id,
        ),
    )
    cursor.execute(
        """
        INSERT INTO ingestion_quota_reservation (
            ingestion_dispatch_id, attempt_number, reserved_credits,
            reserved_at, provider_payload_receipt_id
        ) VALUES (%s, 1, %s, %s, %s)
        """,
        (
            dispatch_id,
            candidate.estimated_cost,
            now,
            provider_payload_receipt_id,
        ),
    )
    cursor.execute(
        """
        INSERT INTO ingestion_dispatch_outbox (
            ingestion_dispatch_id, attempt_number, available_at
        ) VALUES (%s, 1, %s)
        """,
        (dispatch_id, now),
    )
    cursor.execute(
        """
        INSERT INTO ingestion_dispatch_transition (
            ingestion_dispatch_id, state_sequence, state,
            attempt_count, occurred_at
        ) VALUES (%s, 1, 'pending', 0, %s)
        """,
        (dispatch_id, now),
    )
    return PersistedAdmission(
        ingestion_dispatch_id=dispatch_id,
        idempotency_key=plan.idempotency_key,
    )


def _write_failure_kind(error: Exception) -> str:
    """Classify database integrity failures without reading their messages."""

    sqlstate = getattr(error, "sqlstate", None)
    if isinstance(sqlstate, str) and (
        sqlstate.startswith("23") or sqlstate == "P0001"
    ):
        return "conflict"
    return "unavailable"


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
            application_name="sam-analytics-ingestion-admission",
            connect_timeout=5,
            options="-c statement_timeout=10000",
        )
    except Exception:
        raise IngestionAdmissionRepositoryUnavailable(
            "ingestion admission database is unavailable"
        ) from None
