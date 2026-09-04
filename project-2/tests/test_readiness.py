import unittest

from sam_analytics.readiness import DatabaseReadiness, check_dependencies


class ReadinessTests(unittest.TestCase):
    def test_missing_urls_fail_closed_without_invoking_probes(self):
        result = check_dependencies(
            None,
            None,
            database_probe=lambda _: self.fail("database probe should not run"),
            queue_probe=lambda _: self.fail("queue probe should not run"),
        )
        self.assertFalse(result.ready)
        self.assertFalse(result.database_reachable)
        self.assertFalse(result.migrations_current)
        self.assertFalse(result.queue_reachable)

    def test_ready_requires_database_ledger_and_queue(self):
        result = check_dependencies(
            "postgresql://opaque",
            "redis://opaque",
            database_probe=lambda _: DatabaseReadiness(reachable=True, migrations_current=True),
            queue_probe=lambda _: True,
        )
        self.assertTrue(result.ready)

    def test_failed_or_throwing_probe_never_becomes_ready(self):
        result = check_dependencies(
            "postgresql://opaque",
            "redis://opaque",
            database_probe=lambda _: DatabaseReadiness(reachable=True, migrations_current=False),
            queue_probe=lambda _: (_ for _ in ()).throw(RuntimeError("unavailable")),
        )
        self.assertFalse(result.ready)
        self.assertTrue(result.database_reachable)
        self.assertFalse(result.migrations_current)
        self.assertFalse(result.queue_reachable)
