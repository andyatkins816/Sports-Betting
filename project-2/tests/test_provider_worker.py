"""Tests for the one-request The Odds API provider-shadow worker."""

from __future__ import annotations

import importlib.util
import inspect
import os
import sys
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

from sam_analytics.ingestion import RawOddsQuote
from sam_analytics.ingestion_runs import IngestionFailureCode
from sam_analytics.provider_shadow import ProviderShadowFetchFailure
from sam_analytics.providers.the_odds_api import (
    OddsApiFetch,
    OddsApiRequestScope,
    TheOddsApiError,
)

_CELERY_AVAILABLE = importlib.util.find_spec("celery") is not None
_WORKER_PATH = Path(__file__).resolve().parents[1] / "provider_worker.py"


@unittest.skipUnless(
    _CELERY_AVAILABLE,
    "Celery is installed in the deployment/CI dependency set",
)
class ProviderWorkerTests(unittest.TestCase):
    @staticmethod
    def _environment() -> dict[str, str]:
        return {
            "APP_ENV": "staging",
            "DATABASE_URL": "postgresql://sam:opaque@dpg-sam-test-a/sam",
            "REDIS_URL": "redis://red-sam-test:6379/0",
            "SAM_WORKER_ROLE": "private_ingestion",
            "SAM_WORKER_MODE": "provider_shadow",
            "SAM_INGESTION_ENABLED": "true",
            "ODDS_PROVIDER": "the_odds_api",
            "ODDS_PROVIDER_API_KEY": "scoped-provider-key",
            "SAM_ODDS_SPORT_KEY": "baseball_mlb",
            "SAM_ODDS_REGIONS": "us",
            "SAM_ODDS_MARKETS": "h2h",
            "SAM_PROVIDER_LICENSE_SCOPE": "internal_analytics_only",
            "SAM_PROVIDER_LICENSE_VERSION": "terms-2026-08-31",
            "SAM_RAW_EVIDENCE_STORE_BACKEND": "cloudflare_r2",
            "SAM_RAW_EVIDENCE_STORE_URI": ("s3://sam-raw-evidence-staging/raw/the_odds_api"),
            "SAM_RAW_EVIDENCE_S3_REGION": "auto",
            "SAM_RAW_EVIDENCE_S3_ENDPOINT_URL": (
                "https://0123456789abcdef0123456789abcdef.r2.cloudflarestorage.com"
            ),
            "SAM_RAW_EVIDENCE_MAX_BYTES": "10485760",
            "SAM_RAW_EVIDENCE_S3_ACCESS_KEY_ID": "scoped-r2-access-id",
            "SAM_RAW_EVIDENCE_S3_SECRET_ACCESS_KEY": "scoped-r2-secret",
        }

    def _load_worker(self, environment: dict[str, str]):
        module_name = f"provider_worker_test_{uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, _WORKER_PATH)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        self.addCleanup(sys.modules.pop, module_name, None)
        with patch.dict(os.environ, environment, clear=True):
            spec.loader.exec_module(module)
        return module

    def test_import_requires_exact_provider_shadow_boundary(self) -> None:
        valid = self._environment()
        self._load_worker(valid)

        for name, value in (
            ("APP_ENV", "production"),
            ("SAM_WORKER_MODE", "synthetic_storage_probe"),
            ("SAM_INGESTION_ENABLED", "false"),
            ("SAM_ODDS_MARKETS", "spreads"),
        ):
            with self.subTest(name=name):
                changed = dict(valid)
                changed[name] = value
                with self.assertRaises(RuntimeError):
                    self._load_worker(changed)

    def test_worker_has_one_manual_queue_and_no_automatic_path(self) -> None:
        environment = self._environment()
        worker = self._load_worker(environment)
        app = worker.create_celery_app(environment)

        self.assertEqual(app.conf.broker_url, environment["REDIS_URL"])
        self.assertEqual(app.conf.result_backend, "disabled://")
        self.assertTrue(app.conf.task_ignore_result)
        self.assertFalse(app.conf.task_store_errors_even_if_ignored)
        self.assertFalse(app.conf.task_acks_late)
        self.assertFalse(app.conf.task_reject_on_worker_lost)
        self.assertEqual(app.conf.worker_prefetch_multiplier, 1)
        self.assertEqual(app.conf.worker_concurrency, 1)
        self.assertFalse(app.conf.task_create_missing_queues)
        self.assertEqual(app.conf.beat_schedule, {})
        self.assertFalse(app.conf.task_send_sent_event)
        self.assertFalse(app.conf.worker_send_task_events)
        self.assertFalse(app.conf.worker_enable_remote_control)
        self.assertEqual(
            app.conf.task_routes["sam_analytics.ingest_the_odds_api_shadow"]["queue"],
            "sam_provider_shadow",
        )
        self.assertEqual(
            {queue.name for queue in app.conf.task_queues},
            {"sam_provider_shadow"},
        )

    def test_task_is_zero_argument_no_retry_and_returns_nothing(self) -> None:
        worker = self._load_worker(self._environment())
        task_id = str(uuid4())

        self.assertEqual(
            tuple(inspect.signature(worker.ingest_the_odds_api_shadow.run).parameters),
            (),
        )
        self.assertFalse(worker.ingest_the_odds_api_shadow.acks_late)
        self.assertFalse(worker.ingest_the_odds_api_shadow.reject_on_worker_lost)
        self.assertEqual(worker.ingest_the_odds_api_shadow.max_retries, 0)

        with patch.object(worker, "_execute_provider_shadow", return_value=None) as execute:
            result = worker.ingest_the_odds_api_shadow.apply(
                task_id=task_id,
                throw=True,
            )

        self.assertTrue(result.successful())
        self.assertIsNone(result.result)
        execute.assert_called_once_with(task_id=task_id)

    def test_execution_revalidates_before_constructing_dependencies(self) -> None:
        worker = self._load_worker(self._environment())
        environment = self._environment()
        environment["APP_ENV"] = "production"

        with patch.object(worker, "TheOddsApiClient") as client_factory:
            with self.assertRaises(worker.WorkerConfigurationError):
                worker._execute_provider_shadow(
                    task_id=str(uuid4()),
                    environ=environment,
                )

        client_factory.assert_not_called()

    def test_execution_injects_only_reviewed_private_dependencies(self) -> None:
        environment = self._environment()
        worker = self._load_worker(environment)
        task_id = str(uuid4())
        client = MagicMock()
        store = MagicMock()
        ledger = MagicMock()
        repository = MagicMock()
        orchestrator = MagicMock()

        with (
            patch.object(
                worker,
                "TheOddsApiClient",
                return_value=client,
            ) as client_factory,
            patch.object(
                worker.S3CompatibleRawPayloadStore,
                "from_environment",
                return_value=store,
            ) as store_factory,
            patch.object(
                worker,
                "OddsLedger",
                return_value=ledger,
            ) as ledger_factory,
            patch.object(
                worker,
                "PostgresIngestionRunRepository",
                return_value=repository,
            ) as repository_factory,
            patch.object(
                worker,
                "ManualProviderShadowOrchestrator",
                return_value=orchestrator,
            ) as orchestrator_factory,
        ):
            result = worker._execute_provider_shadow(
                task_id=task_id,
                environ=environment,
            )

        self.assertIsNone(result)
        client_factory.assert_called_once_with(
            environment["ODDS_PROVIDER_API_KEY"],
            max_response_bytes=10485760,
        )
        store_factory.assert_called_once_with(environment)
        repository_factory.assert_called_once_with(environment["DATABASE_URL"])
        ledger_factory.assert_called_once()
        self.assertEqual(ledger_factory.call_args.args, (environment["DATABASE_URL"],))
        self.assertEqual(ledger_factory.call_args.kwargs["raw_payload_store"], store)
        orchestrator_factory.assert_called_once()
        self.assertEqual(
            orchestrator_factory.call_args.kwargs["odds_ledger"],
            ledger,
        )
        self.assertEqual(
            orchestrator_factory.call_args.kwargs["ingestion_run_repository"],
            repository,
        )
        provider_fetch = orchestrator_factory.call_args.kwargs["provider_fetch"]
        self.assertTrue(callable(provider_fetch))
        orchestrator.run.assert_called_once_with(
            job_identity=f"celery:{task_id}",
            license_scope="internal_analytics_only",
            license_version="terms-2026-08-31",
            run_id=worker.PROVIDER_SHADOW_ADMISSION_RUN_ID,
        )

    def test_only_canonical_uuid_task_ids_enter_the_audit_ledger(self) -> None:
        worker = self._load_worker(self._environment())
        task_id = str(uuid4())

        self.assertEqual(worker._celery_job_identity(task_id), f"celery:{task_id}")
        for unsafe in (None, "", "manual-task", task_id + "?key=secret"):
            with self.subTest(unsafe=unsafe):
                with self.assertRaises(worker.WorkerConfigurationError):
                    worker._celery_job_identity(unsafe)

    def test_provider_errors_are_finitely_classified_without_chaining(self) -> None:
        worker = self._load_worker(self._environment())
        cases = (
            ("The Odds API returned HTTP 429", IngestionFailureCode.PROVIDER_RATE_LIMITED),
            (
                "The Odds API returned HTTP 408",
                IngestionFailureCode.PROVIDER_RESPONSE_INVALID,
            ),
            (
                "The Odds API returned HTTP 503",
                IngestionFailureCode.PROVIDER_TEMPORARY_UNAVAILABLE,
            ),
            (
                "The Odds API request failed",
                IngestionFailureCode.PROVIDER_TEMPORARY_UNAVAILABLE,
            ),
            (
                "The Odds API returned invalid JSON",
                IngestionFailureCode.PROVIDER_RESPONSE_INVALID,
            ),
        )
        for message, expected in cases:
            with self.subTest(message=message):
                client = MagicMock()
                client.fetch_pregame_odds.side_effect = TheOddsApiError(message)
                with self.assertRaises(ProviderShadowFetchFailure) as raised:
                    worker._fetch_once(
                        client,
                        MagicMock(
                            sport_key="baseball_mlb",
                            regions=("us",),
                            markets=("h2h",),
                        ),
                    )
                self.assertEqual(raised.exception.failure_code, expected)
                self.assertIsNone(raised.exception.__context__)

    def test_fetch_is_exactly_scoped_and_requires_bounded_quota_receipt(self) -> None:
        worker = self._load_worker(self._environment())
        settings = MagicMock(
            provider="the_odds_api",
            sport_key="baseball_mlb",
            regions=("us",),
            markets=("h2h",),
        )
        client = MagicMock()
        accepted = OddsApiFetch(
            quotes=[],
            requests_remaining=499,
            requests_used=1,
            request_cost=1,
            skipped_live_events=0,
            raw_payload=b"[]",
            request_scope=OddsApiRequestScope(
                sport_key="baseball_mlb",
                regions=("us",),
                markets=("h2h",),
            ),
        )
        client.fetch_pregame_odds.return_value = accepted

        self.assertIs(worker._fetch_once(client, settings), accepted)
        client.fetch_pregame_odds.assert_called_once_with(
            "baseball_mlb",
            regions="us",
            markets=("h2h",),
        )

        zero_cost = OddsApiFetch(
            quotes=[],
            requests_remaining=500,
            requests_used=0,
            request_cost=0,
            skipped_live_events=0,
            raw_payload=b"[]",
            request_scope=accepted.request_scope,
        )
        zero_cost_client = MagicMock()
        zero_cost_client.fetch_pregame_odds.return_value = zero_cost
        self.assertIs(worker._fetch_once(zero_cost_client, settings), zero_cost)

        for quota in (
            (None, 1, 1),
            (499, None, 1),
            (499, 1, None),
            (-1, 1, 1),
            (499, 1, 2),
            (499, 1, True),
        ):
            with self.subTest(quota=quota):
                client = MagicMock()
                client.fetch_pregame_odds.return_value = OddsApiFetch(
                    quotes=[],
                    requests_remaining=quota[0],
                    requests_used=quota[1],
                    request_cost=quota[2],
                    skipped_live_events=0,
                    raw_payload=b"[]",
                    request_scope=OddsApiRequestScope(
                        sport_key="baseball_mlb",
                        regions=("us",),
                        markets=("h2h",),
                    ),
                )
                with self.assertRaises(ProviderShadowFetchFailure) as raised:
                    worker._fetch_once(client, settings)
                self.assertEqual(
                    raised.exception.failure_code,
                    IngestionFailureCode.PROVIDER_RESPONSE_INVALID,
                )

        for request_scope in (
            None,
            OddsApiRequestScope(
                sport_key="basketball_nba",
                regions=("us",),
                markets=("h2h",),
            ),
            OddsApiRequestScope(
                sport_key="baseball_mlb",
                regions=("us", "eu"),
                markets=("h2h",),
            ),
            OddsApiRequestScope(
                sport_key="baseball_mlb",
                regions=("us",),
                markets=("h2h",),
                bookmakers=("book",),
            ),
        ):
            with self.subTest(request_scope=request_scope):
                client = MagicMock()
                client.fetch_pregame_odds.return_value = OddsApiFetch(
                    quotes=[],
                    requests_remaining=499,
                    requests_used=1,
                    request_cost=1,
                    skipped_live_events=0,
                    raw_payload=b"[]",
                    request_scope=request_scope,
                )
                with self.assertRaises(ProviderShadowFetchFailure):
                    worker._fetch_once(client, settings)

    def test_response_quotes_stay_inside_the_admitted_normalized_scope(self) -> None:
        worker = self._load_worker(self._environment())
        settings = MagicMock(
            provider="the_odds_api",
            sport_key="baseball_mlb",
            regions=("us",),
            markets=("h2h",),
        )
        request_scope = OddsApiRequestScope(
            sport_key="baseball_mlb",
            regions=("us",),
            markets=("h2h",),
        )
        admitted = _quote()
        provider_added = _quote(market="h2h_lay", quote_id="lay-quote")
        raw_payload = b'[{"provider_added_market":"h2h_lay"}]'
        fetched = OddsApiFetch(
            quotes=[admitted, provider_added],
            requests_remaining=499,
            requests_used=1,
            request_cost=1,
            skipped_live_events=0,
            raw_payload=raw_payload,
            request_scope=request_scope,
        )
        client = MagicMock()
        client.fetch_pregame_odds.return_value = fetched

        filtered = worker._fetch_once(client, settings)

        self.assertEqual(filtered.quotes, [admitted])
        self.assertEqual(filtered.raw_payload, raw_payload)
        self.assertEqual(filtered.request_scope, request_scope)

        for quote in (
            _quote(provider="other_provider"),
            _quote(sport="basketball_nba"),
            _quote(market="spreads"),
        ):
            with self.subTest(quote=quote):
                rejected_client = MagicMock()
                rejected_client.fetch_pregame_odds.return_value = OddsApiFetch(
                    quotes=[quote],
                    requests_remaining=499,
                    requests_used=1,
                    request_cost=1,
                    skipped_live_events=0,
                    raw_payload=b"[]",
                    request_scope=request_scope,
                )
                with self.assertRaises(ProviderShadowFetchFailure) as raised:
                    worker._fetch_once(rejected_client, settings)
                self.assertEqual(
                    raised.exception.failure_code,
                    IngestionFailureCode.PROVIDER_RESPONSE_INVALID,
                )


def _quote(
    *,
    provider: str = "the_odds_api",
    sport: str = "baseball_mlb",
    market: str = "h2h",
    quote_id: str = "quote-1",
) -> RawOddsQuote:
    now = datetime(2026, 9, 6, 12, tzinfo=UTC)
    return RawOddsQuote(
        provider=provider,
        provider_quote_id=quote_id,
        event_id="event-1",
        sport=sport,
        market=market,
        selection="Home",
        american_odds=-110.0,
        line=None,
        captured_at=now - timedelta(seconds=10),
        starts_at=now + timedelta(hours=2),
        bookmaker="example_book",
        league="MLB",
        home_team="Home",
        away_team="Away",
    )


if __name__ == "__main__":
    unittest.main()
