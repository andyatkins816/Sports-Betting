"""Tests for the bounded The Odds API provider-shadow worker."""

from __future__ import annotations

import importlib.util
import io
import inspect
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch
from uuid import UUID, uuid4

from sam_analytics.ingestion import RawOddsQuote
from sam_analytics.ingestion_runs import IngestionFailureCode, IngestionRunState
from sam_analytics.modeling import (
    PREGAME_H2H_FEATURE_SCHEMA,
    FittedProbabilityModel,
    ModelCandidate,
    ProbabilityMetrics,
    PromotionDecision,
)
from sam_analytics.odds_ledger import ResultsLedgerWriteResult
from sam_analytics.provider_shadow import ProviderShadowFetchFailure
from sam_analytics.providers.the_odds_api import (
    CompletedScore,
    OddsApiFetch,
    OddsApiRequestScope,
    ScoresApiFetch,
    ScoresApiRequestScope,
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

    def test_worker_has_one_queue_and_bounded_fixed_schedules(self) -> None:
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
        self.assertEqual(
            app.conf.beat_schedule_filename,
            os.path.join(tempfile.gettempdir(), "sam-provider-shadow-celerybeat-schedule"),
        )
        self.assertEqual(
            app.conf.beat_schedule,
            {
                "ingest-the-odds-api-shadow-every-five-minutes": {
                    "task": "sam_analytics.ingest_the_odds_api_shadow",
                    "schedule": 300,
                    "options": {"queue": "sam_provider_shadow", "expires": 240},
                },
                "settle-the-odds-api-scores-hourly": {
                    "task": "sam_analytics.settle_the_odds_api_scores",
                    "schedule": 3600,
                    "options": {"queue": "sam_provider_shadow", "expires": 3300},
                },
            },
        )
        self.assertFalse(app.conf.task_send_sent_event)
        self.assertFalse(app.conf.worker_send_task_events)
        self.assertFalse(app.conf.worker_enable_remote_control)
        self.assertEqual(
            app.conf.task_routes["sam_analytics.ingest_the_odds_api_shadow"]["queue"],
            "sam_provider_shadow",
        )
        self.assertEqual(
            app.conf.task_routes["sam_analytics.settle_the_odds_api_scores"]["queue"],
            "sam_provider_shadow",
        )
        self.assertEqual(
            app.conf.task_routes["sam_analytics.train_model_candidate"]["queue"],
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

    def test_score_task_is_zero_argument_no_retry_and_returns_nothing(self) -> None:
        worker = self._load_worker(self._environment())
        task_id = str(uuid4())

        self.assertEqual(
            tuple(inspect.signature(worker.settle_the_odds_api_scores.run).parameters),
            (),
        )
        self.assertFalse(worker.settle_the_odds_api_scores.acks_late)
        self.assertFalse(worker.settle_the_odds_api_scores.reject_on_worker_lost)
        self.assertEqual(worker.settle_the_odds_api_scores.max_retries, 0)

        with patch.object(worker, "_execute_score_settlement", return_value=None) as execute:
            result = worker.settle_the_odds_api_scores.apply(
                task_id=task_id,
                throw=True,
            )

        self.assertTrue(result.successful())
        self.assertIsNone(result.result)
        execute.assert_called_once_with(task_id=task_id)

    def test_model_task_is_zero_argument_no_retry_and_returns_nothing(self) -> None:
        worker = self._load_worker(self._environment())

        self.assertEqual(
            tuple(inspect.signature(worker.train_model_candidate.run).parameters),
            (),
        )
        self.assertFalse(worker.train_model_candidate.acks_late)
        self.assertFalse(worker.train_model_candidate.reject_on_worker_lost)
        self.assertEqual(worker.train_model_candidate.max_retries, 0)
        self.assertEqual(worker.train_model_candidate.soft_time_limit, 600)
        self.assertEqual(worker.train_model_candidate.time_limit, 720)

        with patch.object(worker, "_execute_model_training", return_value=None) as execute:
            result = worker.train_model_candidate.apply(throw=True)

        self.assertTrue(result.successful())
        self.assertIsNone(result.result)
        execute.assert_called_once_with()

    def test_execution_revalidates_before_constructing_dependencies(self) -> None:
        worker = self._load_worker(self._environment())
        environment = self._environment()
        environment["APP_ENV"] = "production"

        for executor in (
            worker._execute_provider_shadow,
            worker._execute_score_settlement,
        ):
            with self.subTest(executor=executor.__name__):
                with patch.object(worker, "TheOddsApiClient") as client_factory:
                    with self.assertRaises(worker.WorkerConfigurationError):
                        executor(task_id=str(uuid4()), environ=environment)
                client_factory.assert_not_called()

        with patch.object(worker, "load_h2h_market_training_rows") as load_rows:
            with self.assertRaises(worker.WorkerConfigurationError):
                worker._execute_model_training(environ=environment)
        load_rows.assert_not_called()

    def test_model_training_waits_for_required_settled_history(self) -> None:
        environment = self._environment()
        worker = self._load_worker(environment)

        with (
            patch.object(
                worker,
                "load_h2h_market_training_rows",
                return_value=tuple(range(749)),
            ) as load_rows,
            patch.object(worker, "ProbabilityModelEvaluator") as evaluator,
            patch.object(worker, "_persist_model_candidate") as persist,
        ):
            result = worker._execute_model_training(environ=environment)

        self.assertIsNone(result)
        load_rows.assert_called_once_with(
            environment["DATABASE_URL"],
            sport="baseball_mlb",
            training_cutoff=ANY,
        )
        evaluator.assert_not_called()
        persist.assert_not_called()

    def test_model_training_refuses_to_register_when_no_candidate_passes(self) -> None:
        environment = self._environment()
        worker = self._load_worker(environment)
        evaluation = _evaluation()
        evaluator = MagicMock()
        evaluator.evaluate_many.return_value = (evaluation,)

        with (
            patch.object(
                worker,
                "load_h2h_market_training_rows",
                return_value=tuple(range(750)),
            ),
            patch.object(worker, "ProbabilityModelEvaluator", return_value=evaluator),
            patch.object(
                worker,
                "evaluate_candidate_promotion",
                return_value=PromotionDecision(False, ("quality gate failed",)),
            ),
            patch.object(worker, "_persist_model_candidate") as persist,
        ):
            with self.assertRaises(worker.WorkerConfigurationError):
                worker._execute_model_training(environ=environment)

        persist.assert_not_called()

    def test_model_training_registers_only_a_gated_candidate(self) -> None:
        environment = self._environment()
        worker = self._load_worker(environment)
        evaluation = _evaluation()
        evaluator = MagicMock()
        evaluator.evaluate_many.return_value = (evaluation,)
        fitted = FittedProbabilityModel(
            candidate=evaluation.candidate,
            schema=PREGAME_H2H_FEATURE_SCHEMA,
            estimator={"fitted": True},
            calibrator=None,
            trained_at=datetime(2026, 9, 6, 12, tzinfo=UTC),
            released_at=datetime(2026, 9, 6, 12, tzinfo=UTC),
            training_rows=750,
        )

        with (
            patch.object(
                worker,
                "load_h2h_market_training_rows",
                return_value=tuple(range(750)),
            ),
            patch.object(worker, "ProbabilityModelEvaluator", return_value=evaluator),
            patch.object(
                worker,
                "evaluate_candidate_promotion",
                return_value=PromotionDecision(True, ()),
            ),
            patch.object(worker, "select_best_candidate", return_value=evaluation),
            patch.object(worker, "fit_approved_model", return_value=fitted) as fit,
            patch.object(worker, "_persist_model_candidate", return_value=True) as persist,
        ):
            result = worker._execute_model_training(environ=environment)

        self.assertIsNone(result)
        fit.assert_called_once()
        registered = persist.call_args.kwargs
        self.assertEqual(registered["sport"], "baseball_mlb")
        self.assertEqual(len(registered["schema_sha256"]), 64)
        self.assertEqual(len(registered["artifact_sha256"]), 64)
        self.assertEqual(registered["artifact_sha256"], worker.hashlib.sha256(registered["artifact"]).hexdigest())
        self.assertEqual(
            registered["validation_report"]["governance_status"],
            "candidate_requires_independent_approval",
        )
        self.assertTrue(
            registered["validation_report"]["candidate_evaluations"][0][
                "promotion_gates_passed"
            ]
        )

    def test_candidate_artifact_omits_release_authority_and_round_trips(self) -> None:
        worker = self._load_worker(self._environment())
        model = FittedProbabilityModel(
            candidate=ModelCandidate("logistic_baseline", "logistic_regression"),
            schema=PREGAME_H2H_FEATURE_SCHEMA,
            estimator={"coefficient": 0.25},
            calibrator=None,
            trained_at=datetime(2026, 9, 6, 12, tzinfo=UTC),
            released_at=datetime(2026, 9, 7, 12, tzinfo=UTC),
            training_rows=750,
        )

        artifact, digest = worker._serialize_model_candidate(model)
        restored = worker.joblib.load(io.BytesIO(artifact))

        self.assertEqual(digest, worker.hashlib.sha256(artifact).hexdigest())
        self.assertEqual(restored["candidate"]["name"], "logistic_baseline")
        self.assertEqual(restored["training_rows"], 750)
        self.assertNotIn("released_at", restored)

    def test_model_candidate_persistence_is_unapproved_and_idempotent(self) -> None:
        worker = self._load_worker(self._environment())
        connection = MagicMock()
        cursor = connection.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = (uuid4(),)
        artifact = b"immutable-model"
        digest = worker.hashlib.sha256(artifact).hexdigest()

        with patch.object(worker.psycopg, "connect", return_value=connection):
            inserted = worker._persist_model_candidate(
                database_url=self._environment()["DATABASE_URL"],
                version="sam-test-version",
                sport="baseball_mlb",
                schema_sha256="a" * 64,
                artifact=artifact,
                artifact_sha256=digest,
                training_cutoff=datetime(2026, 9, 6, 12, tzinfo=UTC),
                validation_report={"status": "passed"},
            )

        self.assertTrue(inserted)
        self.assertEqual(cursor.execute.call_count, 2)
        registry_sql, registry_params = cursor.execute.call_args_list[0].args
        self.assertIn("INSERT INTO model_registry", registry_sql)
        self.assertIn("'candidate'", registry_sql)
        self.assertIn("ON CONFLICT (version) DO NOTHING", registry_sql)
        self.assertEqual(registry_params[0], "sam-test-version")
        self.assertEqual(registry_params[5], digest)
        self.assertEqual(registry_params[6], "joblib-sklearn-v1")
        self.assertEqual(registry_params[7], artifact)
        signal_sql = cursor.execute.call_args_list[1].args[0]
        self.assertIn("INSERT INTO operational_signal", signal_sql)
        connection.commit.assert_called_once_with()
        connection.close.assert_called_once_with()

        replay_connection = MagicMock()
        replay_cursor = replay_connection.cursor.return_value.__enter__.return_value
        replay_cursor.fetchone.return_value = None
        with patch.object(worker.psycopg, "connect", return_value=replay_connection):
            replayed = worker._persist_model_candidate(
                database_url=self._environment()["DATABASE_URL"],
                version="sam-test-version",
                sport="baseball_mlb",
                schema_sha256="a" * 64,
                artifact=artifact,
                artifact_sha256=digest,
                training_cutoff=datetime(2026, 9, 6, 12, tzinfo=UTC),
                validation_report={"status": "passed"},
            )

        self.assertFalse(replayed)
        self.assertEqual(replay_cursor.execute.call_count, 1)

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
        self.assertEqual(
            ledger_factory.call_args.kwargs["provider_contracts"]
            .contracts[0]
            .permitted_source_types,
            frozenset({"odds", "result"}),
        )
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
            run_id=UUID(task_id),
        )

    def test_score_execution_records_one_attempt_fetch_and_persist(self) -> None:
        environment = self._environment()
        worker = self._load_worker(environment)
        task_id = str(uuid4())
        fetched = _scores_fetch()
        prepared = object()
        store = MagicMock()
        ledger = MagicMock()
        repository = MagicMock()
        repository.create_run.side_effect = lambda _run, queued: queued
        repository.append_transition.side_effect = (
            lambda _run, _previous, transition: transition
        )
        ledger.persist_results.return_value = ResultsLedgerWriteResult(
            status="accepted",
            receipt_sha256="a" * 64,
            provenance_sha256="b" * 64,
            events_created=1,
            results_created=1,
            results_replayed=0,
            provenance_links_created=1,
            incidents_created=0,
        )
        client = MagicMock()
        client.fetch_scores.return_value = fetched

        with (
            patch.object(worker, "TheOddsApiClient", return_value=client) as client_factory,
            patch.object(
                worker.S3CompatibleRawPayloadStore,
                "from_environment",
                return_value=store,
            ),
            patch.object(worker, "OddsLedger", return_value=ledger) as ledger_factory,
            patch.object(
                worker,
                "PostgresIngestionRunRepository",
                return_value=repository,
            ),
            patch.object(
                worker,
                "prepare_the_odds_api_results_payload",
                return_value=prepared,
            ) as prepare,
        ):
            result = worker._execute_score_settlement(
                task_id=task_id,
                environ=environment,
            )

        self.assertIsNone(result)
        client_factory.assert_called_once_with(
            environment["ODDS_PROVIDER_API_KEY"],
            max_response_bytes=10485760,
        )
        client.fetch_scores.assert_called_once_with("baseball_mlb", days_from=3)
        prepare.assert_called_once_with(
            fetched,
            license_scope="internal_analytics_only",
            license_version="terms-2026-08-31",
        )
        ledger.persist_results.assert_called_once_with(prepared, now=ANY)
        contract = ledger_factory.call_args.kwargs["provider_contracts"].contracts[0]
        self.assertEqual(contract.permitted_source_types, frozenset({"odds", "result"}))

        run, queued = repository.create_run.call_args.args
        self.assertEqual(run.id, UUID(task_id))
        self.assertEqual(run.job_identity, f"celery:{task_id}")
        self.assertEqual(run.provider, "the_odds_api")
        self.assertEqual(run.source_type, "result")
        self.assertEqual(run.max_attempts, 1)
        self.assertEqual(queued.state, IngestionRunState.QUEUED)
        self.assertEqual(repository.append_transition.call_count, 2)
        running = repository.append_transition.call_args_list[0].args[2]
        succeeded = repository.append_transition.call_args_list[1].args[2]
        self.assertEqual(running.state, IngestionRunState.RUNNING)
        self.assertEqual(running.attempt_count, 1)
        self.assertEqual(succeeded.state, IngestionRunState.SUCCEEDED)
        self.assertEqual(succeeded.attempt_count, 1)

    def test_only_canonical_uuid_task_ids_enter_the_audit_ledger(self) -> None:
        worker = self._load_worker(self._environment())
        task_id = str(uuid4())

        self.assertEqual(worker._canonical_task_uuid(task_id), UUID(task_id))
        self.assertEqual(worker._celery_job_identity(task_id), f"celery:{task_id}")
        for unsafe in (None, "", "manual-task", task_id + "?key=secret"):
            with self.subTest(unsafe=unsafe):
                with self.assertRaises(worker.WorkerConfigurationError):
                    worker._celery_job_identity(unsafe)

    def test_score_fetch_uses_one_exact_scope_and_sanitizes_provider_errors(self) -> None:
        worker = self._load_worker(self._environment())
        settings = MagicMock(sport_key="baseball_mlb")
        accepted = _scores_fetch()
        client = MagicMock()
        client.fetch_scores.return_value = accepted

        self.assertIs(worker._fetch_scores_once(client, settings), accepted)
        client.fetch_scores.assert_called_once_with("baseball_mlb", days_from=3)

        for malformed in (
            replace(accepted, requests_remaining=None),
            replace(accepted, request_cost=1),
            replace(
                accepted,
                request_scope=ScoresApiRequestScope(
                    sport_key="basketball_nba",
                    days_from=3,
                ),
            ),
            replace(
                accepted,
                request_scope=ScoresApiRequestScope(
                    sport_key="baseball_mlb",
                    days_from=2,
                ),
            ),
            replace(accepted, scores=[]),
        ):
            with self.subTest(malformed=malformed):
                malformed_client = MagicMock()
                malformed_client.fetch_scores.return_value = malformed
                with self.assertRaises(ProviderShadowFetchFailure) as raised:
                    worker._fetch_scores_once(malformed_client, settings)
                self.assertEqual(
                    raised.exception.failure_code,
                    IngestionFailureCode.PROVIDER_RESPONSE_INVALID,
                )

        for message, expected in (
            (
                "The Odds API returned HTTP 429",
                IngestionFailureCode.PROVIDER_RATE_LIMITED,
            ),
            (
                "The Odds API request failed",
                IngestionFailureCode.PROVIDER_TEMPORARY_UNAVAILABLE,
            ),
            (
                "The Odds API returned invalid JSON",
                IngestionFailureCode.PROVIDER_RESPONSE_INVALID,
            ),
        ):
            with self.subTest(message=message):
                failed_client = MagicMock()
                failed_client.fetch_scores.side_effect = TheOddsApiError(message)
                with self.assertRaises(ProviderShadowFetchFailure) as raised:
                    worker._fetch_scores_once(failed_client, settings)
                self.assertEqual(raised.exception.failure_code, expected)
                self.assertIsNone(raised.exception.__context__)

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


def _evaluation() -> MagicMock:
    metrics = ProbabilityMetrics(
        sample_size=500,
        brier=0.20,
        logloss=0.60,
        expected_calibration_error=0.04,
    )
    score = MagicMock(
        evaluated_rows=500,
        total_rows=750,
        fold_count=5,
        coverage=500 / 750,
        metrics=metrics,
    )
    return MagicMock(
        candidate=ModelCandidate("logistic_baseline", "logistic_regression"),
        data_fingerprint="d" * 64,
        score=score,
        raw_metrics=metrics,
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


def _scores_fetch() -> ScoresApiFetch:
    now = datetime(2026, 9, 6, 12, tzinfo=UTC)
    return ScoresApiFetch(
        scores=(
            CompletedScore(
                provider="the_odds_api",
                event_id="event-1",
                sport="baseball_mlb",
                league="MLB",
                commence_time=now - timedelta(hours=3),
                last_update=now - timedelta(minutes=5),
                home_team="Home",
                away_team="Away",
                home_score=4,
                away_score=2,
            ),
        ),
        requests_remaining=498,
        requests_used=2,
        request_cost=2,
        skipped_incomplete_events=0,
        raw_payload=b'[{"completed":true}]',
        received_at=now,
        request_scope=ScoresApiRequestScope(
            sport_key="baseball_mlb",
            days_from=3,
        ),
    )


if __name__ == "__main__":
    unittest.main()
