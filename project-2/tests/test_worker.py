"""Tests for the deliberately inert future-ingestion Celery entry point."""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4


_CELERY_AVAILABLE = importlib.util.find_spec("celery") is not None
_WORKER_PATH = Path(__file__).resolve().parents[1] / "worker.py"


@unittest.skipUnless(_CELERY_AVAILABLE, "Celery is installed in the deployment/CI dependency set")
class WorkerTests(unittest.TestCase):
    @staticmethod
    def _environment() -> dict[str, str]:
        return {
            "DATABASE_URL": "postgresql://sam:opaque@db.example/sam",
            "REDIS_URL": "redis://cache.example:6379/0",
            "SAM_WORKER_ROLE": "private_ingestion",
            "SAM_RAW_EVIDENCE_STORE_URI": "s3://sam-private-evidence/odds",
            "SAM_INGESTION_ENABLED": "false",
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

    def test_worker_requires_private_database_and_redis_urls_without_defaults(self):
        worker = self._load_worker(self._environment())

        with self.assertRaisesRegex(worker.WorkerConfigurationError, "DATABASE_URL"):
            worker.create_celery_app(
                {
                    "REDIS_URL": "redis://cache.example:6379/0",
                    "SAM_WORKER_ROLE": "private_ingestion",
                    "SAM_RAW_EVIDENCE_STORE_URI": "s3://sam-private-evidence/odds",
                }
            )
        with self.assertRaisesRegex(worker.WorkerConfigurationError, "REDIS_URL"):
            worker.create_celery_app(
                {
                    "DATABASE_URL": "postgresql://sam:opaque@db.example/sam",
                    "SAM_WORKER_ROLE": "private_ingestion",
                    "SAM_RAW_EVIDENCE_STORE_URI": "s3://sam-private-evidence/odds",
                }
            )
        with self.assertRaisesRegex(worker.WorkerConfigurationError, "REDIS_URL"):
            worker.create_celery_app(
                {
                    "DATABASE_URL": "postgresql://sam:opaque@db.example/sam",
                    "REDIS_URL": "amqp://guest@queue.example//",
                    "SAM_WORKER_ROLE": "private_ingestion",
                    "SAM_RAW_EVIDENCE_STORE_URI": "s3://sam-private-evidence/odds",
                }
            )

    def test_module_import_refuses_to_fall_back_to_a_default_broker(self):
        module_name = f"worker_missing_redis_{uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, _WORKER_PATH)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        self.addCleanup(sys.modules.pop, module_name, None)

        with patch.dict(
            os.environ,
            {
                "DATABASE_URL": "postgresql://sam:opaque@db.example/sam",
                "SAM_WORKER_ROLE": "private_ingestion",
                "SAM_RAW_EVIDENCE_STORE_URI": "s3://sam-private-evidence/odds",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "REDIS_URL"):
                spec.loader.exec_module(module)

    def test_module_import_refuses_to_start_without_private_worker_admission(self):
        module_name = f"worker_missing_role_{uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, _WORKER_PATH)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        self.addCleanup(sys.modules.pop, module_name, None)

        with patch.dict(
            os.environ,
            {
                "DATABASE_URL": "postgresql://sam:opaque@db.example/sam",
                "REDIS_URL": "redis://cache.example:6379/0",
                "SAM_RAW_EVIDENCE_STORE_URI": "s3://sam-private-evidence/odds",
                "SAM_INGESTION_ENABLED": "false",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "SAM_WORKER_ROLE"):
                spec.loader.exec_module(module)

    def test_worker_requires_private_role_and_private_object_store_prefix(self):
        worker = self._load_worker(self._environment())

        missing_role = self._environment()
        missing_role.pop("SAM_WORKER_ROLE")
        with self.assertRaisesRegex(worker.WorkerConfigurationError, "SAM_WORKER_ROLE"):
            worker.create_celery_app(missing_role)

        for unsafe_uri in (
            "",
            "https://storage.example/sam-evidence",
            "s3://sam-private-evidence",
            "s3://sam-private-evidence/odds?token=not-a-secret-store",
            "s3://access:credential@sam-private-evidence/odds",
            "s3://sam-private-evidence/odds/../other",
            "s3://sam-private-evidence/sha256/odds",
            "s3://ab/odds",
        ):
            with self.subTest(unsafe_uri=unsafe_uri):
                environment = self._environment()
                environment["SAM_RAW_EVIDENCE_STORE_URI"] = unsafe_uri
                with self.assertRaisesRegex(
                    worker.WorkerConfigurationError, "SAM_RAW_EVIDENCE_STORE_URI"
                ):
                    worker.create_celery_app(environment)

    def test_inert_worker_rejects_unneeded_credentials_without_echoing_them(self):
        worker = self._load_worker(self._environment())
        sentinel = "not-a-real-provider-credential"
        for secret_name in (
            "SESSION_SECRET",
            "SAM_API_KEY",
            "SAM_STATUS_API_KEY",
            "ODDS_PROVIDER_API_KEY",
            "RESULTS_PROVIDER_API_KEY",
            "SAM_RAW_EVIDENCE_S3_ACCESS_KEY_ID",
            "SAM_RAW_EVIDENCE_S3_SECRET_ACCESS_KEY",
        ):
            with self.subTest(secret_name=secret_name):
                environment = self._environment()
                environment[secret_name] = sentinel

                with self.assertRaises(worker.WorkerConfigurationError) as raised:
                    worker.create_celery_app(environment)
                self.assertNotIn(sentinel, str(raised.exception))
                self.assertIn("credentials", str(raised.exception))

    def test_inert_worker_refuses_an_enabled_ingestion_flag_at_startup(self):
        worker = self._load_worker(self._environment())
        environment = self._environment()
        environment["SAM_INGESTION_ENABLED"] = "true"

        with self.assertRaisesRegex(worker.WorkerConfigurationError, "SAM_INGESTION_ENABLED"):
            worker.create_celery_app(environment)

    def test_worker_disables_result_storage_and_has_no_automatic_beat_schedule(self):
        worker = self._load_worker(self._environment())
        app = worker.create_celery_app(self._environment())

        self.assertEqual(app.conf.broker_url, self._environment()["REDIS_URL"])
        self.assertEqual(app.conf.result_backend, "disabled://")
        self.assertTrue(app.conf.task_ignore_result)
        self.assertFalse(app.conf.task_store_errors_even_if_ignored)
        self.assertEqual(app.conf.beat_schedule, {})
        self.assertTrue(app.conf.task_reject_on_worker_lost)
        self.assertFalse(app.conf.task_send_sent_event)
        self.assertFalse(app.conf.worker_send_task_events)
        self.assertFalse(app.conf.worker_enable_remote_control)
        self.assertEqual(
            app.conf.task_routes["sam_analytics.ingest_quotes"]["queue"], "sam_ingestion"
        )

    def test_inert_tasks_fail_closed_without_making_a_network_request(self):
        worker = self._load_worker(self._environment())

        with patch.dict(os.environ, {"SAM_INGESTION_ENABLED": "false"}, clear=False):
            with self.assertRaisesRegex(worker.IngestionNotImplementedError, "is false"):
                worker.ingest_quotes.run()

        with patch.dict(os.environ, {"SAM_INGESTION_ENABLED": "true"}, clear=False):
            with patch(
                "socket.create_connection", side_effect=AssertionError("network must not run")
            ):
                with self.assertRaisesRegex(worker.IngestionNotImplementedError, "not implemented"):
                    worker.ingest_quotes.run()
                with self.assertRaisesRegex(
                    worker.IngestionNotImplementedError, "results ingestion"
                ):
                    worker.settle_events.run()
        self.assertTrue(worker.ingestion_enabled({"SAM_INGESTION_ENABLED": "true"}))
        self.assertFalse(worker.ingestion_enabled({"SAM_INGESTION_ENABLED": "yes"}))


if __name__ == "__main__":
    unittest.main()
