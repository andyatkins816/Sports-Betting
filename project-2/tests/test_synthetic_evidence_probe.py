"""Focused tests for the credential-free synthetic evidence-store probe."""

from __future__ import annotations

import hashlib
import unittest
from datetime import datetime, timedelta, timezone
from uuid import UUID

from sam_analytics.ingestion_runs import IngestionFailureCode, IngestionRunState
from sam_analytics.raw_payload_store import InMemoryRawPayloadStore
from sam_analytics.synthetic_evidence_probe import (
    SYNTHETIC_EVIDENCE_PROVIDER,
    SYNTHETIC_EVIDENCE_PAYLOAD_BYTE_COUNT,
    SYNTHETIC_EVIDENCE_PAYLOAD_SHA256,
    SYNTHETIC_EVIDENCE_SOURCE_TYPE,
    SyntheticEvidenceProbe,
    SyntheticEvidenceProbeConfigurationError,
    SyntheticEvidenceProbeEvidenceError,
    SyntheticEvidenceProbeUnavailable,
)


class _RecordingRunRepository:
    def __init__(self) -> None:
        self.runs = []
        self.transitions = []

    def create_run(self, run, initial_transition):
        self.runs.append(run)
        self.transitions.append(initial_transition)
        return initial_transition

    def append_transition(self, run, previous, transition):
        self.transitions.append(transition)
        return transition


class _RecordingStore:
    def __init__(self) -> None:
        self.delegate = InMemoryRawPayloadStore(namespace="sam-synthetic-probe-test")
        self.calls = []

    def store(self, payload, *, metadata, stored_at=None):
        self.calls.append((payload, metadata, stored_at))
        return self.delegate.store(payload, metadata=metadata, stored_at=stored_at)


class _FailingStore:
    def store(self, payload, *, metadata, stored_at=None):
        raise RuntimeError("simulated storage error with an opaque credential-like value")


class _InvalidReceiptStore:
    def store(self, payload, *, metadata, stored_at=None):
        return object()


class _FailingRunRepository:
    def __init__(self) -> None:
        self.create_calls = 0
        self.append_calls = 0

    def create_run(self, run, initial_transition) -> None:
        self.create_calls += 1
        raise RuntimeError("simulated database error with an opaque connection value")

    def append_transition(self, run, previous, transition) -> None:
        self.append_calls += 1


class _FailingAppendRunRepository(_RecordingRunRepository):
    def __init__(self, *, fail_on_append: int) -> None:
        super().__init__()
        self.fail_on_append = fail_on_append
        self.append_calls = 0

    def append_transition(self, run, previous, transition):
        self.append_calls += 1
        if self.append_calls == self.fail_on_append:
            raise RuntimeError(
                "simulated database error with an opaque connection value"
            )
        return super().append_transition(run, previous, transition)


class SyntheticEvidenceProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 9, 4, 18, tzinfo=timezone.utc)
        self.run_id = UUID("12345678-1234-5678-1234-567812345678")

    def test_writes_only_fixed_synthetic_bytes_and_records_a_complete_audit_lifecycle(self) -> None:
        store = _RecordingStore()
        repository = _RecordingRunRepository()

        result = SyntheticEvidenceProbe(
            raw_payload_store=store,
            ingestion_run_repository=repository,
        ).run(
            job_identity="celery:12345678-1234-5678-1234-567812345678",
            now=self.now,
            run_id=self.run_id,
        )

        self.assertEqual(result.ingestion_run_id, self.run_id)
        self.assertEqual(result.completed_at, self.now)
        approved_payload = (
            b'{"kind":"sam_synthetic_evidence_probe","schema_version":"v1"}'
        )
        approved_digest = (
            "5dc961d33ef2a18a1e47b6ffc52475bf"
            "0442bf7ba3959787a4718e5fd5015aa1"
        )
        self.assertEqual(store.calls[0][0], approved_payload)
        self.assertEqual(len(approved_payload), 61)
        self.assertEqual(hashlib.sha256(approved_payload).hexdigest(), approved_digest)
        self.assertEqual(SYNTHETIC_EVIDENCE_PAYLOAD_BYTE_COUNT, 61)
        self.assertEqual(SYNTHETIC_EVIDENCE_PAYLOAD_SHA256, approved_digest)
        self.assertEqual(result.byte_count, 61)
        self.assertEqual(result.payload_sha256, approved_digest)
        self.assertEqual(store.delegate.stored_count, 1)
        self.assertEqual(len(repository.runs), 1)
        self.assertEqual(repository.runs[0].provider, SYNTHETIC_EVIDENCE_PROVIDER)
        self.assertEqual(repository.runs[0].source_type, SYNTHETIC_EVIDENCE_SOURCE_TYPE)
        self.assertEqual(
            repository.runs[0].job_identity,
            "celery:12345678-1234-5678-1234-567812345678",
        )
        self.assertEqual(
            [transition.state for transition in repository.transitions],
            [IngestionRunState.QUEUED, IngestionRunState.RUNNING, IngestionRunState.SUCCEEDED],
        )

        payload, metadata, stored_at = store.calls[0]
        self.assertEqual(metadata.provider, SYNTHETIC_EVIDENCE_PROVIDER)
        self.assertEqual(metadata.source_type, SYNTHETIC_EVIDENCE_SOURCE_TYPE)
        self.assertEqual(metadata.captured_at, self.now)
        self.assertEqual(metadata.received_at, self.now)
        self.assertEqual(stored_at, self.now)
        self.assertNotIn(payload.decode("utf-8"), repr(result))

    def test_reuses_the_same_content_addressed_synthetic_object_across_distinct_runs(self) -> None:
        store = _RecordingStore()
        repository = _RecordingRunRepository()
        probe = SyntheticEvidenceProbe(
            raw_payload_store=store,
            ingestion_run_repository=repository,
        )

        first = probe.run(
            job_identity="celery:12345678-1234-5678-1234-567812345678",
            now=self.now,
            run_id=self.run_id,
        )
        second = probe.run(
            job_identity="celery:87654321-4321-8765-4321-876543218765",
            now=self.now + timedelta(seconds=1),
            run_id=UUID("87654321-4321-8765-4321-876543218765"),
        )

        self.assertEqual(first.payload_sha256, second.payload_sha256)
        self.assertEqual(store.delegate.stored_count, 1)
        self.assertEqual(len(repository.runs), 2)
        self.assertEqual(
            [transition.state for transition in repository.transitions[-3:]],
            [IngestionRunState.QUEUED, IngestionRunState.RUNNING, IngestionRunState.SUCCEEDED],
        )

    def test_storage_failure_is_sanitized_and_recorded_as_a_safe_failed_transition(self) -> None:
        repository = _RecordingRunRepository()
        probe = SyntheticEvidenceProbe(
            raw_payload_store=_FailingStore(),
            ingestion_run_repository=repository,
        )

        with self.assertRaises(SyntheticEvidenceProbeUnavailable) as caught:
            probe.run(
                job_identity="celery:12345678-1234-5678-1234-567812345678",
                now=self.now,
                run_id=self.run_id,
            )

        self.assertEqual(str(caught.exception), "synthetic evidence storage probe failed")
        self.assertIsNone(caught.exception.__context__)
        self.assertEqual(repository.transitions[-1].state, IngestionRunState.FAILED)
        self.assertEqual(
            repository.transitions[-1].failure.code,
            IngestionFailureCode.STORAGE_UNAVAILABLE,
        )

    def test_invalid_storage_receipt_is_blocked_without_exposing_store_details(self) -> None:
        repository = _RecordingRunRepository()
        probe = SyntheticEvidenceProbe(
            raw_payload_store=_InvalidReceiptStore(),
            ingestion_run_repository=repository,
        )

        with self.assertRaises(SyntheticEvidenceProbeEvidenceError) as caught:
            probe.run(
                job_identity="celery:12345678-1234-5678-1234-567812345678",
                now=self.now,
                run_id=self.run_id,
            )

        self.assertEqual(str(caught.exception), "synthetic evidence receipt verification failed")
        self.assertEqual(repository.transitions[-1].state, IngestionRunState.FAILED)
        self.assertEqual(
            repository.transitions[-1].failure.code,
            IngestionFailureCode.EVIDENCE_VALIDATION_FAILED,
        )

    def test_repository_failure_stops_before_the_probe_can_write_evidence(self) -> None:
        store = _RecordingStore()
        repository = _FailingRunRepository()
        probe = SyntheticEvidenceProbe(
            raw_payload_store=store,
            ingestion_run_repository=repository,
        )

        with self.assertRaises(SyntheticEvidenceProbeUnavailable) as caught:
            probe.run(
                job_identity="celery:12345678-1234-5678-1234-567812345678",
                now=self.now,
                run_id=self.run_id,
            )

        self.assertEqual(
            str(caught.exception),
            "synthetic evidence probe audit repository is unavailable",
        )
        self.assertIsNone(caught.exception.__context__)
        self.assertEqual(repository.create_calls, 1)
        self.assertEqual(repository.append_calls, 0)
        self.assertEqual(store.calls, [])

    def test_running_append_failure_stops_before_the_object_store_write(self) -> None:
        store = _RecordingStore()
        repository = _FailingAppendRunRepository(fail_on_append=1)

        with self.assertRaises(SyntheticEvidenceProbeUnavailable) as caught:
            SyntheticEvidenceProbe(
                raw_payload_store=store,
                ingestion_run_repository=repository,
            ).run(
                job_identity="celery:12345678-1234-5678-1234-567812345678",
                now=self.now,
                run_id=self.run_id,
            )

        self.assertEqual(
            str(caught.exception),
            "synthetic evidence probe audit repository is unavailable",
        )
        self.assertIsNone(caught.exception.__context__)
        self.assertEqual(store.calls, [])
        self.assertEqual(
            [transition.state for transition in repository.transitions],
            [IngestionRunState.QUEUED],
        )

    def test_success_append_failure_leaves_an_inconclusive_running_audit(self) -> None:
        store = _RecordingStore()
        repository = _FailingAppendRunRepository(fail_on_append=2)

        with self.assertRaises(SyntheticEvidenceProbeUnavailable) as caught:
            SyntheticEvidenceProbe(
                raw_payload_store=store,
                ingestion_run_repository=repository,
            ).run(
                job_identity="celery:12345678-1234-5678-1234-567812345678",
                now=self.now,
                run_id=self.run_id,
            )

        self.assertEqual(
            str(caught.exception),
            "synthetic evidence probe audit repository is unavailable",
        )
        self.assertIsNone(caught.exception.__context__)
        self.assertEqual(store.delegate.stored_count, 1)
        self.assertEqual(
            [transition.state for transition in repository.transitions],
            [IngestionRunState.QUEUED, IngestionRunState.RUNNING],
        )

    def test_failure_append_failure_preserves_only_safe_nonterminal_facts(self) -> None:
        repository = _FailingAppendRunRepository(fail_on_append=2)

        with self.assertRaises(SyntheticEvidenceProbeUnavailable) as caught:
            SyntheticEvidenceProbe(
                raw_payload_store=_FailingStore(),
                ingestion_run_repository=repository,
            ).run(
                job_identity="celery:12345678-1234-5678-1234-567812345678",
                now=self.now,
                run_id=self.run_id,
            )

        self.assertEqual(
            str(caught.exception),
            "synthetic evidence probe audit repository is unavailable",
        )
        self.assertIsNone(caught.exception.__context__)
        self.assertEqual(
            [transition.state for transition in repository.transitions],
            [IngestionRunState.QUEUED, IngestionRunState.RUNNING],
        )

    def test_requires_injected_private_boundaries_and_an_aware_probe_time(self) -> None:
        repository = _RecordingRunRepository()
        with self.assertRaises(SyntheticEvidenceProbeConfigurationError):
            SyntheticEvidenceProbe(
                raw_payload_store=object(),  # type: ignore[arg-type]
                ingestion_run_repository=repository,
            )
        with self.assertRaises(SyntheticEvidenceProbeConfigurationError):
            SyntheticEvidenceProbe(
                raw_payload_store=_RecordingStore(),
                ingestion_run_repository=object(),  # type: ignore[arg-type]
            )

        probe = SyntheticEvidenceProbe(
            raw_payload_store=_RecordingStore(),
            ingestion_run_repository=repository,
        )
        with self.assertRaises(SyntheticEvidenceProbeConfigurationError):
            probe.run(
                job_identity="celery:12345678-1234-5678-1234-567812345678",
                now=datetime(2026, 9, 4, 18),
                run_id=self.run_id,
            )


if __name__ == "__main__":
    unittest.main()
