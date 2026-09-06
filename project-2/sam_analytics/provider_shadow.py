"""Credential-free orchestration for one manual provider-shadow ingestion run.

This module is deliberately not a worker, scheduler, configuration loader, or
provider client.  The separate private composition root injects one configured,
zero-argument provider fetch together with the reviewed odds ledger and
append-only ingestion-run repository.  This boundary never reads an API key,
constructs a network client, consults process environment, retries, or
publishes provider data.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import UUID

from .ingestion_run_repository import IngestionRunRepository
from .ingestion_runs import (
    IngestionFailureCode,
    IngestionRun,
    IngestionRunStateTransition,
    mark_failed,
    mark_succeeded,
    new_manual_shadow_run,
    start_next_attempt,
)
from .odds_ledger import (
    LedgerWriteResult,
    OddsLedgerValidationError,
    PreparedOddsPayload,
    prepare_the_odds_api_payload,
)
from .provider_contracts import ProviderUse, validate_provider_use

THE_ODDS_API_PROVIDER = "the_odds_api"
PROVIDER_SHADOW_SOURCE_TYPE = "odds"
PROVIDER_SHADOW_SCHEMA_VERSION = "v4"

_FETCH_FAILURE_CODES = frozenset(
    {
        IngestionFailureCode.PROVIDER_RATE_LIMITED,
        IngestionFailureCode.PROVIDER_TEMPORARY_UNAVAILABLE,
        IngestionFailureCode.NETWORK_TIMEOUT,
        IngestionFailureCode.PROVIDER_RESPONSE_INVALID,
    }
)
_ACCEPTED_LEDGER_STATUSES = frozenset({"accepted", "accepted_empty"})


class ProviderShadowError(RuntimeError):
    """Base error containing no provider response, URL, or credential detail."""


class ProviderShadowConfigurationError(ValueError):
    """The injected boundary or non-secret run plan is invalid."""


class ProviderShadowUnavailable(ProviderShadowError):
    """The append-only audit could not conclusively record the run."""


class ProviderShadowRunFailed(ProviderShadowError):
    """One audited shadow attempt ended with an enumerated safe failure."""

    def __init__(self, *, ingestion_run_id: UUID, failure_code: IngestionFailureCode) -> None:
        self.ingestion_run_id = ingestion_run_id
        self.failure_code = failure_code
        super().__init__("manual provider-shadow ingestion failed")


class ProviderShadowFetchFailure(ProviderShadowError):
    """A provider boundary's sanitized, explicitly classified fetch failure."""

    def __init__(self, failure_code: IngestionFailureCode) -> None:
        if (
            not isinstance(failure_code, IngestionFailureCode)
            or failure_code not in _FETCH_FAILURE_CODES
        ):
            raise ProviderShadowConfigurationError("provider fetch failure code is not permitted")
        self.failure_code = failure_code
        super().__init__("provider fetch failed")


class OddsLedgerWriter(Protocol):
    """Narrow injected surface of :class:`~sam_analytics.odds_ledger.OddsLedger`."""

    def persist(
        self, payload: PreparedOddsPayload, *, now: datetime | None = None
    ) -> LedgerWriteResult: ...


@dataclass(frozen=True)
class ProviderShadowResult:
    """Credential-free summary of one fully audited accepted shadow receipt."""

    ingestion_run_id: UUID
    completed_at: datetime
    ledger_status: str
    events_created: int
    snapshots_created: int
    snapshots_replayed: int
    incidents_created: int


class ManualProviderShadowOrchestrator:
    """Run one injected The Odds API fetch through the reviewed evidence ledger."""

    def __init__(
        self,
        *,
        provider_fetch: Callable[[], Any],
        odds_ledger: OddsLedgerWriter,
        ingestion_run_repository: IngestionRunRepository,
    ) -> None:
        if not callable(provider_fetch):
            raise ProviderShadowConfigurationError("provider fetch is not configured")
        if not callable(getattr(odds_ledger, "persist", None)):
            raise ProviderShadowConfigurationError("odds ledger is not configured")
        if not callable(getattr(ingestion_run_repository, "create_run", None)) or not callable(
            getattr(ingestion_run_repository, "append_transition", None)
        ):
            raise ProviderShadowConfigurationError("ingestion run repository is not configured")
        self._provider_fetch = provider_fetch
        self._odds_ledger = odds_ledger
        self._ingestion_run_repository = ingestion_run_repository

    def run(
        self,
        *,
        job_identity: str,
        license_scope: str,
        license_version: str,
        now: datetime | None = None,
        run_id: UUID | None = None,
    ) -> ProviderShadowResult:
        """Execute exactly one audited attempt without constructing dependencies.

        ``provider_fetch`` is called once after ``queued`` and ``running`` are
        durable.  Every post-start failure is collapsed to a finite code and
        recorded before a sanitized exception is raised.  There is deliberately
        no retry path here; the run identity is created with ``max_attempts=1``.
        """

        fixed_time = _require_aware_time(now) if now is not None else None
        _validate_non_secret_plan(license_scope, license_version)
        created_at = fixed_time or _utc_now()
        run, queued = new_manual_shadow_run(
            provider=THE_ODDS_API_PROVIDER,
            job_identity=job_identity,
            source_type=PROVIDER_SHADOW_SOURCE_TYPE,
            max_attempts=1,
            created_at=created_at,
            run_id=run_id,
        )
        self._create_run(run, queued)

        running_at = fixed_time or _utc_now()
        running = start_next_attempt(run, queued, occurred_at=running_at)
        self._append_transition(run, queued, running)

        fetch, failure_code = self._fetch_once()
        if failure_code is not None:
            self._fail_run(run, running, failure_code, fixed_time)

        payload, failure_code = _prepare_payload(
            fetch,
            license_scope=license_scope,
            license_version=license_version,
        )
        if failure_code is not None:
            self._fail_run(run, running, failure_code, fixed_time)

        ledger_result, failure_code = self._persist_payload(payload, fixed_time)
        if failure_code is not None:
            self._fail_run(run, running, failure_code, fixed_time)
        if ledger_result is None:
            self._fail_run(
                run,
                running,
                IngestionFailureCode.EVIDENCE_VALIDATION_FAILED,
                fixed_time,
            )

        completed_at = fixed_time or _utc_now()
        succeeded = mark_succeeded(run, running, occurred_at=completed_at)
        self._append_transition(run, running, succeeded)
        return ProviderShadowResult(
            ingestion_run_id=run.id,
            completed_at=completed_at,
            ledger_status=ledger_result.status,
            events_created=ledger_result.events_created,
            snapshots_created=ledger_result.snapshots_created,
            snapshots_replayed=ledger_result.snapshots_replayed,
            incidents_created=ledger_result.incidents_created,
        )

    def _fetch_once(self) -> tuple[object | None, IngestionFailureCode | None]:
        fetch: object | None = None
        failure_code: IngestionFailureCode | None = None
        try:
            fetch = self._provider_fetch()
        except ProviderShadowFetchFailure as error:
            failure_code = error.failure_code
        except Exception:
            failure_code = IngestionFailureCode.INTERNAL_TRANSIENT
        return fetch, failure_code

    def _persist_payload(
        self,
        payload: PreparedOddsPayload | None,
        fixed_time: datetime | None,
    ) -> tuple[LedgerWriteResult | None, IngestionFailureCode | None]:
        result: object | None = None
        failure_code: IngestionFailureCode | None = None
        validation_time = fixed_time or _utc_now()
        try:
            result = self._odds_ledger.persist(payload, now=validation_time)
        except OddsLedgerValidationError:
            failure_code = IngestionFailureCode.EVIDENCE_VALIDATION_FAILED
        except Exception:
            failure_code = IngestionFailureCode.INTERNAL_TRANSIENT
        if failure_code is None and (
            not isinstance(result, LedgerWriteResult)
            or result.status not in _ACCEPTED_LEDGER_STATUSES
        ):
            failure_code = IngestionFailureCode.EVIDENCE_VALIDATION_FAILED
        return result if isinstance(result, LedgerWriteResult) else None, failure_code

    def _create_run(
        self,
        run: IngestionRun,
        queued: IngestionRunStateTransition,
    ) -> None:
        persisted: object | None = None
        repository_failed = False
        try:
            persisted = self._ingestion_run_repository.create_run(run, queued)
        except Exception:
            repository_failed = True
        if repository_failed or persisted != queued:
            raise ProviderShadowUnavailable("provider-shadow audit repository is unavailable")

    def _append_transition(
        self,
        run: IngestionRun,
        previous: IngestionRunStateTransition,
        transition: IngestionRunStateTransition,
    ) -> None:
        persisted: object | None = None
        repository_failed = False
        try:
            persisted = self._ingestion_run_repository.append_transition(run, previous, transition)
        except Exception:
            repository_failed = True
        if repository_failed or persisted != transition:
            raise ProviderShadowUnavailable("provider-shadow audit repository is unavailable")

    def _fail_run(
        self,
        run: IngestionRun,
        running: IngestionRunStateTransition,
        failure_code: IngestionFailureCode,
        fixed_time: datetime | None,
    ) -> None:
        failed_at = fixed_time or _utc_now()
        failed = mark_failed(
            run,
            running,
            failure_code=failure_code,
            occurred_at=failed_at,
        )
        self._append_transition(run, running, failed)
        raise ProviderShadowRunFailed(
            ingestion_run_id=run.id,
            failure_code=failure_code,
        )


def _prepare_payload(
    fetch: object | None,
    *,
    license_scope: str,
    license_version: str,
) -> tuple[PreparedOddsPayload | None, IngestionFailureCode | None]:
    payload: PreparedOddsPayload | None = None
    preparation_failed = False
    try:
        payload = prepare_the_odds_api_payload(
            fetch,
            license_scope=license_scope,
            license_version=license_version,
            source_type=PROVIDER_SHADOW_SOURCE_TYPE,
            schema_version=PROVIDER_SHADOW_SCHEMA_VERSION,
        )
    except Exception:
        preparation_failed = True
    return (
        (None, IngestionFailureCode.PROVIDER_RESPONSE_INVALID)
        if preparation_failed
        else (payload, None)
    )


def _validate_non_secret_plan(license_scope: str, license_version: str) -> None:
    invalid = False
    try:
        validate_provider_use(
            ProviderUse(
                provider=THE_ODDS_API_PROVIDER,
                license_scope=license_scope,
                license_version=license_version,
                source_type=PROVIDER_SHADOW_SOURCE_TYPE,
            )
        )
    except Exception:
        invalid = True
    if invalid:
        raise ProviderShadowConfigurationError("provider-shadow license plan is invalid")


def _require_aware_time(value: object) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ProviderShadowConfigurationError("provider-shadow time must be timezone-aware")
    return value


def _utc_now() -> datetime:
    return datetime.now(UTC)
