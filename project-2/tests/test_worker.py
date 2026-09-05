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
            worker.create_celery_app({"REDIS_URL": "redis://cache.example:6379/0"})
        with self.assertRaisesRegex(worker.WorkerConfigurationError, "REDIS_URL"):
            worker.create_celery_app({"DATABASE_URL": "postgresql://sam:opaque@db.example/sam"})
        with self.assertRaisesRegex(worker.WorkerConfigurationError, "REDIS_URL"):
            worker.create_celery_app(
                {
                    "DATABASE_URL": "postgresql://sam:opaque@db.example/sam",
                    "REDIS_URL": "amqp://guest@queue.example//",
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
            {"DATABASE_URL": "postgresql://sam:opaque@db.example/sam"},
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "REDIS_URL"):
                spec.loader.exec_module(module)

    def test_worker_disables_result_storage_and_has_no_automatic_beat_schedule(self):
        worker = self._load_worker(self._environment())
        app = worker.create_celery_app(self._environment())

        self.assertEqual(app.conf.broker_url, self._environment()["REDIS_URL"])
        self.assertEqual(app.conf.result_backend, "disabled://")
        self.assertTrue(app.conf.task_ignore_result)
        self.assertFalse(app.conf.task_store_errors_even_if_ignored)
        self.assertEqual(app.conf.beat_schedule, {})
        self.assertTrue(app.conf.task_reject_on_worker_lost)

    def test_ingestion_is_inert_until_explicitly_enabled_and_still_makes_no_request(self):
        worker = self._load_worker(self._environment())

        with patch.dict(os.environ, {"SAM_INGESTION_ENABLED": "false"}, clear=False):
            disabled = worker.ingest_quotes.run()
        self.assertEqual(disabled["status"], "disabled")

        with patch.dict(os.environ, {"SAM_INGESTION_ENABLED": "true"}, clear=False):
            with patch("socket.create_connection", side_effect=AssertionError("network must not run")):
                enabled = worker.ingest_quotes.run()
        self.assertEqual(enabled["status"], "not_configured")
        self.assertTrue(worker.ingestion_enabled({"SAM_INGESTION_ENABLED": "true"}))
        self.assertFalse(worker.ingestion_enabled({"SAM_INGESTION_ENABLED": "yes"}))


if __name__ == "__main__":
    unittest.main()
