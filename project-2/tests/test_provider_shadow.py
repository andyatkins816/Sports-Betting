"""Focused tests for credential-free manual provider-shadow orchestration."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sam_analytics.ingestion import RawOddsQuote
from sam_analytics.ingestion_runs import IngestionFailureCode, IngestionRunState
from sam_analytics.odds_ledger import LedgerWriteResult, OddsLedgerValidationError
from sam_analytics.provider_shadow import (
    ManualProviderShadowOrchestrator,
    ProviderShadowConfigurationError,
    ProviderShadowFetchFailure,
    ProviderShadowRunFailed,
    ProviderShadowUnavailable,
)
from sam_analytics.providers.the_odds_api import OddsApiFetch, OddsApiRequestScope


class _RecordingFetch:
    def __init__(self, result) -> None:
        self.result = result
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class _RecordingLedger:
    def __init__(self, result=None) -> None:
        self.result = result or _accepted_ledger_result()
        self.calls = []

    def persist(self, payload, *, now=None):
        self.calls.append((payload, now))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


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


class _FailingRunRepository(_RecordingRunRepository):
    def __init__(self, *, fail_create=False, fail_on_append=None) -> None:
        super().__init__()
        self.fail_create = fail_create
        self.fail_on_append = fail_on_append
        self.append_calls = 0

    def create_run(self, run, initial_transition):
        if self.fail_create:
            raise RuntimeError("opaque database connection and password")
        return super().create_run(run, initial_transition)

    def append_transition(self, run, previous, transition):
        self.append_calls += 1
        if self.append_calls == self.fail_on_append:
            raise RuntimeError("opaque database connection and password")
        return super().append_transition(run, previous, transition)


class _OneShotRunRepository(_RecordingRunRepository):
    """Model the durable primary-key rejection used by PostgreSQL."""

    def __init__(self) -> None:
        super().__init__()
        self.run_ids = set()

    def create_run(self, run, initial_transition):
        if run.id in self.run_ids:
            raise RuntimeError("duplicate admission run")
        self.run_ids.add(run.id)
        return super().create_run(run, initial_transition)


class ManualProviderShadowOrchestratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 9, 6, 12, tzinfo=UTC)
        self.run_id = UUID("12345678-1234-5678-1234-567812345678")
        self.fetch = _RecordingFetch(_valid_fetch(self.now))
        self.ledger = _RecordingLedger()
        self.repository = _RecordingRunRepository()

    def _orchestrator(self, **overrides):
        return ManualProviderShadowOrchestrator(
            provider_fetch=overrides.get("provider_fetch", self.fetch),
            odds_ledger=overrides.get("odds_ledger", self.ledger),
            ingestion_run_repository=overrides.get("ingestion_run_repository", self.repository),
        )

    def _run(self, orchestrator=None, **overrides):
        return (orchestrator or self._orchestrator()).run(
            job_identity=overrides.get(
                "job_identity", "manual:12345678-1234-5678-1234-567812345678"
            ),
            license_scope=overrides.get("license_scope", "internal_analytics_only"),
            license_version=overrides.get("license_version", "terms-2026-09-06"),
            now=overrides.get("now", self.now),
            run_id=overrides.get("run_id", self.run_id),
        )

    def test_runs_one_fetch_and_records_queued_running_succeeded(self) -> None:
        result = self._run()

        self.assertEqual(self.fetch.calls, 1)
        self.assertEqual(len(self.ledger.calls), 1)
        self.assertEqual(result.ingestion_run_id, self.run_id)
        self.assertEqual(result.completed_at, self.now)
        self.assertEqual(result.ledger_status, "accepted")
        self.assertEqual(result.events_created, 1)
        self.assertEqual(result.snapshots_created, 2)
        self.assertEqual(result.snapshots_replayed, 0)
        self.assertEqual(result.incidents_created, 0)
        self.assertEqual(len(self.repository.runs), 1)
        self.assertEqual(self.repository.runs[0].provider, "the_odds_api")
        self.assertEqual(self.repository.runs[0].source_type, "odds")
        self.assertEqual(self.repository.runs[0].max_attempts, 1)
        self.assertEqual(
            [transition.state for transition in self.repository.transitions],
            [
                IngestionRunState.QUEUED,
                IngestionRunState.RUNNING,
                IngestionRunState.SUCCEEDED,
            ],
        )

        payload, validation_time = self.ledger.calls[0]
        self.assertEqual(validation_time, self.now)
        self.assertEqual(payload.provider, "the_odds_api")
        self.assertEqual(payload.source_type, "odds")
        self.assertEqual(payload.schema_version, "v4")
        self.assertEqual(payload.license_scope, "internal_analytics_only")
        self.assertEqual(payload.license_version, "terms-2026-09-06")
        self.assertEqual(payload.raw_payload, b'{"provider":"fixture"}')
        self.assertNotIn(payload.raw_payload.decode(), repr(result))

    def test_accepted_empty_ledger_result_records_success_and_exposes_only_status(self) -> None:
        fetch = _RecordingFetch(_empty_fetch(self.now))
        ledger = _RecordingLedger(_accepted_empty_ledger_result())
        orchestrator = self._orchestrator(provider_fetch=fetch, odds_ledger=ledger)

        result = self._run(orchestrator)

        self.assertEqual(result.ledger_status, "accepted_empty")
        self.assertEqual(result.events_created, 0)
        self.assertEqual(result.snapshots_created, 0)
        self.assertEqual(fetch.calls, 1)
        self.assertEqual(ledger.calls[0][0].quotes, ())
        self.assertEqual(ledger.calls[0][0].captured_at, self.now)
        self.assertEqual(
            [transition.state for transition in self.repository.transitions],
            [
                IngestionRunState.QUEUED,
                IngestionRunState.RUNNING,
                IngestionRunState.SUCCEEDED,
            ],
        )

    def test_classified_fetch_failure_is_recorded_once_and_sanitized(self) -> None:
        fetch = _RecordingFetch(
            ProviderShadowFetchFailure(IngestionFailureCode.PROVIDER_RATE_LIMITED)
        )
        ledger = _RecordingLedger()
        orchestrator = self._orchestrator(provider_fetch=fetch, odds_ledger=ledger)

        with self.assertRaises(ProviderShadowRunFailed) as caught:
            self._run(orchestrator)

        self.assertEqual(fetch.calls, 1)
        self.assertEqual(ledger.calls, [])
        self.assertEqual(caught.exception.ingestion_run_id, self.run_id)
        self.assertEqual(
            caught.exception.failure_code,
            IngestionFailureCode.PROVIDER_RATE_LIMITED,
        )
        self.assertEqual(str(caught.exception), "manual provider-shadow ingestion failed")
        self.assertIsNone(caught.exception.__context__)
        self._assert_failed_transition(IngestionFailureCode.PROVIDER_RATE_LIMITED)

    def test_unexpected_fetch_exception_becomes_finite_internal_failure(self) -> None:
        fetch = _RecordingFetch(RuntimeError("https://provider.invalid?apiKey=must-not-leak"))
        orchestrator = self._orchestrator(provider_fetch=fetch)

        with self.assertRaises(ProviderShadowRunFailed) as caught:
            self._run(orchestrator)

        self.assertEqual(fetch.calls, 1)
        self.assertNotIn("apiKey", str(caught.exception))
        self.assertIsNone(caught.exception.__context__)
        self._assert_failed_transition(IngestionFailureCode.INTERNAL_TRANSIENT)

    def test_malformed_fetch_is_recorded_as_provider_response_invalid(self) -> None:
        fetch = _RecordingFetch(object())
        ledger = _RecordingLedger()
        orchestrator = self._orchestrator(provider_fetch=fetch, odds_ledger=ledger)

        with self.assertRaises(ProviderShadowRunFailed):
            self._run(orchestrator)

        self.assertEqual(fetch.calls, 1)
        self.assertEqual(ledger.calls, [])
        self._assert_failed_transition(IngestionFailureCode.PROVIDER_RESPONSE_INVALID)

    def test_ledger_rejection_is_recorded_as_evidence_validation_failure(self) -> None:
        ledger = _RecordingLedger(
            OddsLedgerValidationError("opaque provider payload and object location")
        )
        orchestrator = self._orchestrator(odds_ledger=ledger)

        with self.assertRaises(ProviderShadowRunFailed) as caught:
            self._run(orchestrator)

        self.assertNotIn("object location", str(caught.exception))
        self.assertIsNone(caught.exception.__context__)
        self._assert_failed_transition(IngestionFailureCode.EVIDENCE_VALIDATION_FAILED)

    def test_nonaccepted_ledger_status_never_records_success(self) -> None:
        ledger = _RecordingLedger(_blocked_ledger_result())
        orchestrator = self._orchestrator(odds_ledger=ledger)

        with self.assertRaises(ProviderShadowRunFailed):
            self._run(orchestrator)

        self._assert_failed_transition(IngestionFailureCode.EVIDENCE_VALIDATION_FAILED)

    def test_audit_failure_stops_before_fetch_or_leaves_running_inconclusive(self) -> None:
        for repository, expected_states in (
            (_FailingRunRepository(fail_create=True), []),
            (_FailingRunRepository(fail_on_append=1), [IngestionRunState.QUEUED]),
        ):
            with self.subTest(expected_states=expected_states):
                fetch = _RecordingFetch(_valid_fetch(self.now))
                orchestrator = self._orchestrator(
                    provider_fetch=fetch,
                    ingestion_run_repository=repository,
                )
                with self.assertRaises(ProviderShadowUnavailable) as caught:
                    self._run(orchestrator)
                self.assertEqual(fetch.calls, 0)
                self.assertEqual(
                    str(caught.exception),
                    "provider-shadow audit repository is unavailable",
                )
                self.assertIsNone(caught.exception.__context__)
                self.assertEqual(
                    [transition.state for transition in repository.transitions],
                    expected_states,
                )

    def test_terminal_append_failure_leaves_running_without_retry(self) -> None:
        repository = _FailingRunRepository(fail_on_append=2)
        fetch = _RecordingFetch(_valid_fetch(self.now))
        ledger = _RecordingLedger()
        orchestrator = self._orchestrator(
            provider_fetch=fetch,
            odds_ledger=ledger,
            ingestion_run_repository=repository,
        )

        with self.assertRaises(ProviderShadowUnavailable):
            self._run(orchestrator)

        self.assertEqual(fetch.calls, 1)
        self.assertEqual(len(ledger.calls), 1)
        self.assertEqual(
            [transition.state for transition in repository.transitions],
            [IngestionRunState.QUEUED, IngestionRunState.RUNNING],
        )

    def test_fixed_admission_run_id_blocks_a_second_fetch(self) -> None:
        repository = _OneShotRunRepository()
        fetch = _RecordingFetch(_valid_fetch(self.now))
        ledger = _RecordingLedger()
        orchestrator = self._orchestrator(
            provider_fetch=fetch,
            odds_ledger=ledger,
            ingestion_run_repository=repository,
        )

        self._run(orchestrator)
        with self.assertRaises(ProviderShadowUnavailable):
            self._run(
                orchestrator,
                job_identity="manual:87654321-4321-8765-4321-876543218765",
            )

        self.assertEqual(fetch.calls, 1)
        self.assertEqual(len(ledger.calls), 1)
        self.assertEqual(repository.run_ids, {self.run_id})

    def test_invalid_dependencies_plan_and_time_fail_before_any_fetch(self) -> None:
        for kwargs in (
            {"provider_fetch": object()},
            {"odds_ledger": object()},
            {"ingestion_run_repository": object()},
        ):
            with self.subTest(kwargs=tuple(kwargs)):
                with self.assertRaises(ProviderShadowConfigurationError):
                    ManualProviderShadowOrchestrator(
                        provider_fetch=kwargs.get("provider_fetch", self.fetch),
                        odds_ledger=kwargs.get("odds_ledger", self.ledger),
                        ingestion_run_repository=kwargs.get(
                            "ingestion_run_repository", self.repository
                        ),
                    )

        for overrides in (
            {"license_scope": "not valid"},
            {"license_version": ""},
            {"now": datetime(2026, 9, 6, 12)},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ProviderShadowConfigurationError):
                    self._run(**overrides)
        self.assertEqual(self.fetch.calls, 0)
        self.assertEqual(self.repository.transitions, [])

    def test_fetch_failure_wrapper_accepts_only_provider_failure_codes(self) -> None:
        with self.assertRaises(ProviderShadowConfigurationError):
            ProviderShadowFetchFailure(IngestionFailureCode.DATABASE_UNAVAILABLE)

    def _assert_failed_transition(self, failure_code) -> None:
        self.assertEqual(
            [transition.state for transition in self.repository.transitions],
            [
                IngestionRunState.QUEUED,
                IngestionRunState.RUNNING,
                IngestionRunState.FAILED,
            ],
        )
        failed = self.repository.transitions[-1]
        self.assertEqual(failed.attempt_count, 1)
        self.assertEqual(failed.failure.code, failure_code)
        self.assertEqual(self.repository.runs[0].max_attempts, 1)


def _valid_fetch(now: datetime) -> OddsApiFetch:
    quote = RawOddsQuote(
        provider="the_odds_api",
        provider_quote_id="quote-1",
        event_id="event-1",
        sport="basketball_nba",
        market="h2h",
        selection="Home",
        american_odds=-110.0,
        line=None,
        captured_at=now - timedelta(seconds=10),
        starts_at=now + timedelta(hours=2),
        bookmaker="example_book",
        league="NBA",
        home_team="Home",
        away_team="Away",
    )
    return OddsApiFetch(
        quotes=[quote],
        requests_remaining=99,
        requests_used=1,
        request_cost=1,
        skipped_live_events=0,
        raw_payload=b'{"provider":"fixture"}',
        received_at=now,
        request_scope=OddsApiRequestScope(
            sport_key="basketball_nba",
            regions=("us",),
            markets=("h2h",),
        ),
    )


def _empty_fetch(now: datetime) -> OddsApiFetch:
    return OddsApiFetch(
        quotes=[],
        requests_remaining=98,
        requests_used=2,
        request_cost=1,
        skipped_live_events=0,
        raw_payload=b"[]",
        received_at=now,
        request_scope=OddsApiRequestScope(
            sport_key="basketball_nba",
            regions=("us",),
            markets=("h2h",),
        ),
    )


def _accepted_ledger_result() -> LedgerWriteResult:
    return LedgerWriteResult(
        status="accepted",
        receipt_sha256="a" * 64,
        provenance_sha256="b" * 64,
        events_created=1,
        snapshots_created=2,
        snapshots_replayed=0,
        provenance_links_created=2,
        incidents_created=0,
    )


def _blocked_ledger_result() -> LedgerWriteResult:
    return LedgerWriteResult(
        status="blocked_event_identity",
        receipt_sha256="a" * 64,
        provenance_sha256="b" * 64,
        events_created=0,
        snapshots_created=0,
        snapshots_replayed=0,
        provenance_links_created=0,
        incidents_created=1,
    )


def _accepted_empty_ledger_result() -> LedgerWriteResult:
    return LedgerWriteResult(
        status="accepted_empty",
        receipt_sha256="a" * 64,
        provenance_sha256="b" * 64,
        events_created=0,
        snapshots_created=0,
        snapshots_replayed=0,
        provenance_links_created=0,
        incidents_created=0,
    )


if __name__ == "__main__":
    unittest.main()
