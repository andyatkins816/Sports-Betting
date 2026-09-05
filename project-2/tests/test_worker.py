"""Tests for the manual-only synthetic staging worker entry point."""

from __future__ import annotations

import importlib.util
import inspect
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4


_CELERY_AVAILABLE = importlib.util.find_spec("celery") is not None
_WORKER_PATH = Path(__file__).resolve().parents[1] / "worker.py"


@unittest.skipUnless(_CELERY_AVAILABLE, "Celery is installed in the deployment/CI dependency set")
class WorkerTests(unittest.TestCase):
    @staticmethod
    def _environment() -> dict[str, str]:
        return {
            "APP_ENV": "staging",
            "DATABASE_URL": "postgresql://sam:opaque@dpg-sam-test-a/sam",
            "REDIS_URL": "redis://red-sam-test:6379/0",
            "SAM_WORKER_ROLE": "private_ingestion",
            "SAM_WORKER_MODE": "synthetic_storage_probe",
            "SAM_INGESTION_ENABLED": "false",
            "SAM_RAW_EVIDENCE_STORE_BACKEND": "cloudflare_r2",
            "SAM_RAW_EVIDENCE_STORE_URI": "s3://sam-raw-evidence-staging/raw/synthetic",
            "SAM_RAW_EVIDENCE_S3_REGION": "auto",
            "SAM_RAW_EVIDENCE_S3_ENDPOINT_URL": (
                "https://0123456789abcdef0123456789abcdef.r2.cloudflarestorage.com"
            ),
            "SAM_RAW_EVIDENCE_MAX_BYTES": "1048576",
            "SAM_RAW_EVIDENCE_S3_ACCESS_KEY_ID": "scoped-r2-access-id",
            "SAM_RAW_EVIDENCE_S3_SECRET_ACCESS_KEY": "scoped-r2-secret",
        }

    def _load_worker(self, environment: dict[str, str]):
        module_name = f"worker_test_{uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, _WORKER_PATH)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        self.addCleanup(sys.modules.pop, module_name, None)
        with patch.dict(os.environ, environment, clear=True):
            spec.loader.exec_module(module)
        return module

    def test_module_import_requires_the_exact_private_staging_boundary(self) -> None:
        valid = self._environment()
        self._load_worker(valid)

        changes = (
            ("APP_ENV", "production"),
            ("SAM_WORKER_MODE", "provider_shadow"),
            ("SAM_RAW_EVIDENCE_STORE_URI", "s3://sam-raw-evidence-prod/raw/synthetic"),
        )
        for name, value in changes:
            with self.subTest(name=name):
                environment = self._environment()
                environment[name] = value
                with self.assertRaises(RuntimeError):
                    self._load_worker(environment)

        for missing_name in ("DATABASE_URL", "REDIS_URL", "SAM_WORKER_ROLE"):
            with self.subTest(missing_name=missing_name):
                environment = self._environment()
                environment.pop(missing_name)
                with self.assertRaises(RuntimeError):
                    self._load_worker(environment)

    def test_worker_has_one_manual_queue_and_no_automatic_delivery_path(self) -> None:
        worker = self._load_worker(self._environment())
        app = worker.create_celery_app(self._environment())

        self.assertEqual(app.conf.broker_url, self._environment()["REDIS_URL"])
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
            app.conf.task_routes["sam_analytics.verify_staging_raw_evidence"]["queue"],
            "sam_manual_shadow",
        )
        self.assertEqual(
            app.conf.task_routes["sam_analytics.ingest_quotes"]["queue"],
            "sam_manual_shadow",
        )
        self.assertEqual(
            app.conf.task_routes["sam_analytics.settle_events"]["queue"],
            "sam_manual_shadow",
        )
        self.assertEqual(
            {queue.name for queue in app.conf.task_queues},
            {"sam_manual_shadow"},
        )

    def test_probe_task_is_zero_argument_no_retry_and_returns_nothing(self) -> None:
        worker = self._load_worker(self._environment())
        task_id = str(uuid4())

        self.assertEqual(tuple(inspect.signature(worker.verify_staging_raw_evidence.run).parameters), ())
        self.assertFalse(worker.verify_staging_raw_evidence.acks_late)
        self.assertFalse(worker.verify_staging_raw_evidence.reject_on_worker_lost)
        self.assertEqual(worker.verify_staging_raw_evidence.max_retries, 0)

        with patch.object(worker, "_execute_synthetic_storage_probe", return_value=None) as execute:
            result = worker.verify_staging_raw_evidence.apply(task_id=task_id, throw=True)

        self.assertTrue(result.successful())
        self.assertIsNone(result.result)
        execute.assert_called_once_with(task_id=task_id)

    def test_execution_revalidates_admission_before_constructing_dependencies(self) -> None:
        worker = self._load_worker(self._environment())
        environment = self._environment()
        environment["APP_ENV"] = "production"

        with patch.object(
            worker.S3CompatibleRawPayloadStore,
            "from_environment",
        ) as store_factory:
            with self.assertRaises(worker.WorkerConfigurationError):
                worker._execute_synthetic_storage_probe(
                    task_id=str(uuid4()),
                    environ=environment,
                )

        store_factory.assert_not_called()

    def test_execution_injects_only_the_private_store_and_audit_repository(self) -> None:
        worker = self._load_worker(self._environment())
        environment = self._environment()
        task_id = str(uuid4())
        store = MagicMock()
        repository = MagicMock()
        probe = MagicMock()

        with patch.object(
            worker.S3CompatibleRawPayloadStore,
            "from_environment",
            return_value=store,
        ) as store_factory, patch.object(
            worker,
            "PostgresIngestionRunRepository",
            return_value=repository,
        ) as repository_factory, patch.object(
            worker,
            "SyntheticEvidenceProbe",
            return_value=probe,
        ) as probe_factory:
            result = worker._execute_synthetic_storage_probe(
                task_id=task_id,
                environ=environment,
            )

        self.assertIsNone(result)
        store_factory.assert_called_once_with(environment)
        repository_factory.assert_called_once_with(environment["DATABASE_URL"])
        probe_factory.assert_called_once_with(
            raw_payload_store=store,
            ingestion_run_repository=repository,
        )
        probe.run.assert_called_once_with(job_identity=f"celery:{task_id}")

    def test_only_canonical_uuid_task_ids_enter_the_audit_ledger(self) -> None:
        worker = self._load_worker(self._environment())
        task_id = str(uuid4())

        self.assertEqual(worker._celery_job_identity(task_id), f"celery:{task_id}")
        for unsafe_id in (None, "", "custom-task", f"token:{task_id}", task_id + "?secret=x"):
            with self.subTest(unsafe_id=unsafe_id):
                with self.assertRaises(worker.WorkerConfigurationError) as caught:
                    worker._celery_job_identity(unsafe_id)
                self.assertNotIn(str(unsafe_id), str(caught.exception))

    def test_provider_tasks_remain_inert_without_making_a_network_request(self) -> None:
        worker = self._load_worker(self._environment())

        with patch.dict(os.environ, {"SAM_INGESTION_ENABLED": "false"}, clear=False):
            with self.assertRaisesRegex(worker.IngestionNotImplementedError, "is false"):
                worker.ingest_quotes.run()

        with patch.dict(os.environ, {"SAM_INGESTION_ENABLED": "true"}, clear=False):
            with patch(
                "socket.create_connection",
                side_effect=AssertionError("network must not run"),
            ):
                with self.assertRaisesRegex(worker.IngestionNotImplementedError, "not implemented"):
                    worker.ingest_quotes.run()
                with self.assertRaisesRegex(
                    worker.IngestionNotImplementedError,
                    "results ingestion",
                ):
                    worker.settle_events.run()

        self.assertTrue(worker.ingestion_enabled({"SAM_INGESTION_ENABLED": "true"}))
        self.assertFalse(worker.ingestion_enabled({"SAM_INGESTION_ENABLED": "yes"}))


if __name__ == "__main__":
    unittest.main()
