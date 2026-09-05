"""A credential-free fixture for the private evidence-storage probe.

This module intentionally writes only a fixed, synthetic JSON document.  It
does not import provider adapters, read environment variables, make network
requests, or normalize market data. Its purpose is to give the admitted
private worker one bounded way to prove that its injected evidence store and
ingestion audit repository can retain an auditable object before any licensed
provider request is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable
from uuid import UUID, uuid4

from .ingestion_runs import (
    IngestionFailureCode,
    IngestionRun,
    IngestionRunStateTransition,
    mark_failed,
    mark_succeeded,
    new_manual_shadow_run,
    start_next_attempt,
)
from .raw_payload_store import (
    RawPayloadMetadata,
    RawPayloadStore,
    StoredRawPayload,
    validate_private_payload_uri,
)


SYNTHETIC_EVIDENCE_PROVIDER = "sam_synthetic"
SYNTHETIC_EVIDENCE_SOURCE_TYPE = "storage_probe"
SYNTHETIC_EVIDENCE_SCHEMA_VERSION = "v1"
SYNTHETIC_EVIDENCE_PAYLOAD_BYTE_COUNT = 61
SYNTHETIC_EVIDENCE_PAYLOAD_SHA256 = (
    "5dc961d33ef2a18a1e47b6ffc52475bf0442bf7ba3959787a4718e5fd5015aa1"
)
_SYNTHETIC_EVIDENCE_PAYLOAD = (
    b'{"kind":"sam_synthetic_evidence_probe","schema_version":"v1"}'
)


class SyntheticEvidenceProbeError(RuntimeError):
    """Base error for a probe failure with no provider or credential detail."""


class SyntheticEvidenceProbeConfigurationError(ValueError):
    """Raised when the injected private boundaries are not usable."""


class SyntheticEvidenceProbeUnavailable(SyntheticEvidenceProbeError):
    """Raised when a required private dependency cannot record the probe."""


class SyntheticEvidenceProbeEvidenceError(SyntheticEvidenceProbeError):
    """Raised when a storage receipt does not prove the synthetic bytes."""


@runtime_checkable
class IngestionRunRepository(Protocol):
    """Private, append-only persistence boundary for ingestion-run facts.

    An adapter must insert the immutable run and its initial queued state in
    one transaction, then append every later state in sequence. It must not
    persist provider content, credentials, request URLs, or exception text.
    """

    def create_run(
        self,
        run: IngestionRun,
        initial_transition: IngestionRunStateTransition,
    ) -> IngestionRunStateTransition:
        """Atomically persist an identity and its initial queued fact."""

    def append_transition(
        self,
        run: IngestionRun,
        previous: IngestionRunStateTransition,
        transition: IngestionRunStateTransition,
    ) -> IngestionRunStateTransition:
        """Append one fact only if ``previous`` is still the latest state."""


@dataclass(frozen=True)
class SyntheticEvidenceProbeResult:
    """A non-sensitive summary of one fully audited probe run.

    The result deliberately excludes raw bytes, object URIs, provider URLs,
    and any credentials.  An operator can correlate it with the private audit
    repository through ``ingestion_run_id`` and with an evidence receipt by
    SHA-256 without making either object public.
    """

    ingestion_run_id: UUID
    payload_sha256: str
    byte_count: int
    completed_at: datetime


class SyntheticEvidenceProbe:
    """Run one safe evidence-store verification using injected boundaries."""

    def __init__(
        self,
        *,
        raw_payload_store: RawPayloadStore,
        ingestion_run_repository: IngestionRunRepository,
    ) -> None:
        if not isinstance(raw_payload_store, RawPayloadStore):
            raise SyntheticEvidenceProbeConfigurationError("raw payload store is not configured")
        if not isinstance(ingestion_run_repository, IngestionRunRepository):
            raise SyntheticEvidenceProbeConfigurationError(
                "ingestion run repository is not configured"
            )
        self._raw_payload_store = raw_payload_store
        self._ingestion_run_repository = ingestion_run_repository

    def run(
        self,
        *,
        job_identity: str,
        now: datetime | None = None,
        run_id: UUID | None = None,
    ) -> SyntheticEvidenceProbeResult:
        """Write and verify only fixed synthetic bytes, then audit success.

        A storage or audit exception is converted into a generic error outside
        the caught exception handler.  This prevents exception context from
        retaining an SDK message that might include a signed URL or secret.
        """

        occurred_at = _require_aware_time(now or datetime.now(timezone.utc))
        selected_run_id = run_id or uuid4()
        run, queued = new_manual_shadow_run(
            provider=SYNTHETIC_EVIDENCE_PROVIDER,
            job_identity=job_identity,
            source_type=SYNTHETIC_EVIDENCE_SOURCE_TYPE,
            max_attempts=1,
            created_at=occurred_at,
            run_id=selected_run_id,
        )
        self._create_run(run, queued)

        running = start_next_attempt(run, queued, occurred_at=occurred_at)
        self._append_transition(run, queued, running)

        metadata = RawPayloadMetadata(
            provider=SYNTHETIC_EVIDENCE_PROVIDER,
            provider_record_id=f"synthetic-fixture-v1:{selected_run_id}",
            source_type=SYNTHETIC_EVIDENCE_SOURCE_TYPE,
            captured_at=occurred_at,
            received_at=occurred_at,
            schema_version=SYNTHETIC_EVIDENCE_SCHEMA_VERSION,
            license_scope="synthetic_test_only",
            license_version="fixture-v1",
        )
        receipt, storage_failed = self._store_synthetic_payload(metadata, occurred_at)
        if storage_failed:
            self._append_failure(
                run,
                running,
                failure_code=IngestionFailureCode.STORAGE_UNAVAILABLE,
                occurred_at=occurred_at,
            )
            raise SyntheticEvidenceProbeUnavailable("synthetic evidence storage probe failed")

        if not _is_expected_receipt(receipt, metadata=metadata, stored_at=occurred_at):
            self._append_failure(
                run,
                running,
                failure_code=IngestionFailureCode.EVIDENCE_VALIDATION_FAILED,
                occurred_at=occurred_at,
            )
            raise SyntheticEvidenceProbeEvidenceError(
                "synthetic evidence receipt verification failed"
            )

        succeeded = mark_succeeded(run, running, occurred_at=occurred_at)
        self._append_transition(run, running, succeeded)
        return SyntheticEvidenceProbeResult(
            ingestion_run_id=run.id,
            payload_sha256=receipt.payload_sha256,
            byte_count=receipt.byte_count,
            completed_at=succeeded.occurred_at,
        )

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
            raise SyntheticEvidenceProbeUnavailable(
                "synthetic evidence probe audit repository is unavailable"
            )

    def _append_transition(
        self,
        run: IngestionRun,
        previous: IngestionRunStateTransition,
        transition: IngestionRunStateTransition,
    ) -> None:
        persisted: object | None = None
        repository_failed = False
        try:
            persisted = self._ingestion_run_repository.append_transition(
                run, previous, transition
            )
        except Exception:
            repository_failed = True
        if repository_failed or persisted != transition:
            raise SyntheticEvidenceProbeUnavailable(
                "synthetic evidence probe audit repository is unavailable"
            )

    def _append_failure(
        self,
        run: IngestionRun,
        running: IngestionRunStateTransition,
        *,
        failure_code: IngestionFailureCode,
        occurred_at: datetime,
    ) -> None:
        failed = mark_failed(run, running, failure_code=failure_code, occurred_at=occurred_at)
        self._append_transition(run, running, failed)

    def _store_synthetic_payload(
        self,
        metadata: RawPayloadMetadata,
        stored_at: datetime,
    ) -> tuple[object | None, bool]:
        receipt: object | None = None
        storage_failed = False
        try:
            receipt = self._raw_payload_store.store(
                _SYNTHETIC_EVIDENCE_PAYLOAD,
                metadata=metadata,
                stored_at=stored_at,
            )
        except Exception:
            storage_failed = True
        return receipt, storage_failed


def _require_aware_time(value: object) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise SyntheticEvidenceProbeConfigurationError("probe time must be timezone-aware")
    return value


def _is_expected_receipt(
    receipt: object | None,
    *,
    metadata: RawPayloadMetadata,
    stored_at: datetime,
) -> bool:
    """Verify the injected store proved the exact probe bytes without leaks."""

    if not isinstance(receipt, StoredRawPayload):
        return False
    try:
        if receipt.payload_sha256 != SYNTHETIC_EVIDENCE_PAYLOAD_SHA256:
            return False
        if receipt.byte_count != SYNTHETIC_EVIDENCE_PAYLOAD_BYTE_COUNT:
            return False
        if receipt.metadata != metadata or receipt.stored_at != stored_at:
            return False
        validate_private_payload_uri(
            receipt.payload_uri,
            payload_sha256=SYNTHETIC_EVIDENCE_PAYLOAD_SHA256,
        )
    except Exception:
        return False
    return True
