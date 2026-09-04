"""Bounded, credential-safe dependency checks for SAM deployment readiness.

The public liveness endpoint deliberately answers only whether the Flask
process is running.  This module is for the separate readiness endpoint: it
verifies that a configured PostgreSQL instance is reachable, has exactly the
checked-in migration ledger, and that the configured Redis-compatible queue
can answer a ping.  It never returns a connection URL, password, host, or
underlying exception to a caller.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sam_analytics.migrate import discover_migrations


@dataclass(frozen=True)
class DatabaseReadiness:
    """Non-secret outcome of a PostgreSQL reachability and ledger check."""

    reachable: bool
    migrations_current: bool


@dataclass(frozen=True)
class DependencyReadiness:
    """Non-secret outcome used by ``/api/readyz``."""

    database_reachable: bool
    migrations_current: bool
    queue_reachable: bool

    @property
    def ready(self) -> bool:
        return self.database_reachable and self.migrations_current and self.queue_reachable


DatabaseProbe = Callable[[str], DatabaseReadiness]
QueueProbe = Callable[[str], bool]


def check_dependencies(
    database_url: str | None,
    redis_url: str | None,
    *,
    database_probe: DatabaseProbe | None = None,
    queue_probe: QueueProbe | None = None,
) -> DependencyReadiness:
    """Check configured dependencies without allowing probe errors to escape.

    Dependency URLs are deliberately accepted only as opaque strings and are
    never included in a return value or error.  Injected probes make the
    decision logic independently testable without a live datastore.
    """

    database = _database_not_ready()
    if isinstance(database_url, str) and database_url.strip():
        try:
            database = (database_probe or probe_postgres)(database_url)
        except Exception:
            database = _database_not_ready()

    queue_reachable = False
    if isinstance(redis_url, str) and redis_url.strip():
        try:
            queue_reachable = bool((queue_probe or probe_redis)(redis_url))
        except Exception:
            queue_reachable = False

    return DependencyReadiness(
        database_reachable=database.reachable is True,
        migrations_current=database.migrations_current is True,
        queue_reachable=queue_reachable,
    )


def probe_postgres(database_url: str) -> DatabaseReadiness:
    """Return whether PostgreSQL responds and its migration ledger is current."""

    connection: Any | None = None
    try:
        import psycopg

        expected = tuple(
            (migration.version, migration.filename, migration.checksum_sha256)
            for migration in discover_migrations()
        )
        connection = psycopg.connect(
            database_url,
            application_name="sam-analytics-readyz",
            connect_timeout=2,
            options="-c statement_timeout=1500",
        )
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT version, filename, checksum_sha256 "
                "FROM sam_schema_migrations ORDER BY version"
            )
            actual = tuple(tuple(str(value) for value in row) for row in cursor.fetchall())
        return DatabaseReadiness(reachable=True, migrations_current=actual == expected)
    except Exception:
        return _database_not_ready()
    finally:
        if connection is not None:
            # A close failure is deliberately allowed to reach
            # ``check_dependencies``. It then reports the dependency as not
            # ready instead of hiding an indeterminate connection state.
            connection.close()


def probe_redis(redis_url: str) -> bool:
    """Return whether a Redis-compatible queue responds before a short timeout."""

    client: Any | None = None
    try:
        import redis

        client = redis.Redis.from_url(
            redis_url,
            socket_connect_timeout=2,
            socket_timeout=2,
            health_check_interval=0,
        )
        return bool(client.ping())
    except Exception:
        return False
    finally:
        if client is not None:
            # As with PostgreSQL, a close failure fails closed through the
            # bounded caller instead of being silently discarded.
            client.close()


def _database_not_ready() -> DatabaseReadiness:
    return DatabaseReadiness(reachable=False, migrations_current=False)
