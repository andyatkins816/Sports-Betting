"""Fail-closed PostgreSQL migration runner for SAM Analytics.

The runner owns the transaction boundary for every numbered SQL file. It
records an immutable checksum in PostgreSQL and refuses to run when an already
applied file has been changed, when two runners contend for the same database,
or when a migration attempts to manage its own transaction.

Credentials are read only from DATABASE_URL and never included in command
output or exception text emitted by the command-line entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


_MIGRATION_FILENAME_RE = re.compile(r"(?P<version>[0-9]+)_(?P<name>[a-z][a-z0-9_]*)\.sql\Z")
_DOLLAR_QUOTE_RE = re.compile(r"\$(?:[A-Za-z_][A-Za-z0-9_]*)?\$")
_TRANSACTION_CONTROL_RE = re.compile(
    r"^(?:"
    r"BEGIN(?:\s+(?:WORK|TRANSACTION))?"
    r"|START\s+TRANSACTION"
    r"|COMMIT(?:\s+(?:WORK|TRANSACTION|PREPARED))?"
    r"|ROLLBACK(?:\s+(?:WORK|TRANSACTION|PREPARED))?"
    r"|ABORT(?:\s+(?:WORK|TRANSACTION))?"
    r"|SAVEPOINT"
    r"|RELEASE(?:\s+SAVEPOINT)?"
    r"|PREPARE\s+TRANSACTION"
    r"|SET\s+(?:LOCAL\s+)?TRANSACTION"
    r")\b",
    re.IGNORECASE,
)

# A stable, application-specific PostgreSQL advisory-lock key. The lock is
# session scoped and is released even if the migration process is terminated.
_MIGRATION_LOCK_KEY = 6_141_889_114_257_901
_CREATE_LEDGER_SQL = """
CREATE TABLE IF NOT EXISTS sam_schema_migrations (
    version TEXT PRIMARY KEY,
    filename TEXT NOT NULL UNIQUE,
    checksum_sha256 TEXT NOT NULL CHECK (checksum_sha256 ~ '^[0-9a-f]{64}$'),
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""
_READ_APPLIED_SQL = """
SELECT version, filename, checksum_sha256
FROM sam_schema_migrations
ORDER BY version
"""
_INSERT_APPLIED_SQL = """
INSERT INTO sam_schema_migrations (version, filename, checksum_sha256)
VALUES (%s, %s, %s)
"""


class MigrationError(RuntimeError):
    """Base class for errors that are safe to surface from the command line."""


class MigrationConfigurationError(MigrationError):
    """Raised when the runner cannot locate required local configuration."""


class MigrationIntegrityError(MigrationError):
    """Raised when local migration history disagrees with the database ledger."""


class MigrationLockError(MigrationError):
    """Raised when another process is already migrating the same database."""


class MigrationTransactionControlError(MigrationError):
    """Raised when a SQL file tries to own a transaction boundary."""


class DatabaseMigrationError(MigrationError):
    """Raised with deliberately non-sensitive database failure text."""


@dataclass(frozen=True)
class Migration:
    """A discovered numbered SQL migration and its exact byte checksum."""

    version: str
    filename: str
    sql: str
    checksum_sha256: str


@dataclass(frozen=True)
class AppliedMigration:
    """One immutable row read from the database ledger."""

    version: str
    filename: str
    checksum_sha256: str


@dataclass(frozen=True)
class MigrationRunResult:
    """The safe, credential-free result of a migration run."""

    applied_versions: tuple[str, ...]
    current_versions: tuple[str, ...]


class _Cursor(Protocol):
    def execute(self, query: str, params: Sequence[Any] | None = None) -> Any:
        ...

    def fetchall(self) -> Sequence[Sequence[Any]]:
        ...

    def fetchone(self) -> Sequence[Any] | None:
        ...

    def __enter__(self) -> "_Cursor":
        ...

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        ...


class _Connection(Protocol):
    def cursor(self) -> _Cursor:
        ...

    def transaction(self) -> Any:
        ...

    def commit(self) -> None:
        ...

    def rollback(self) -> None:
        ...

    def close(self) -> None:
        ...


ConnectionFactory = Callable[[str], _Connection]


def default_migrations_dir() -> Path:
    """Return the checked-in migration directory in source and container builds."""

    return Path(__file__).resolve().parent.parent / "migrations"


def discover_migrations(migrations_dir: Path | str | None = None) -> tuple[Migration, ...]:
    """Load numbered UTF-8 SQL files in numeric order without executing them."""

    directory = Path(migrations_dir) if migrations_dir is not None else default_migrations_dir()
    if not directory.is_dir():
        raise MigrationConfigurationError("migration directory is unavailable")

    migrations: list[Migration] = []
    numeric_versions: set[int] = set()
    for path in sorted(directory.iterdir(), key=lambda candidate: candidate.name):
        if not path.is_file() or path.suffix != ".sql":
            continue
        match = _MIGRATION_FILENAME_RE.fullmatch(path.name)
        if match is None:
            raise MigrationConfigurationError("all SQL migrations must use the numbered filename convention")
        version = match.group("version")
        numeric_version = int(version)
        if numeric_version in numeric_versions:
            raise MigrationConfigurationError("migration version numbers must be unique")
        numeric_versions.add(numeric_version)
        try:
            raw_sql = path.read_bytes()
            sql = raw_sql.decode("utf-8")
        except (OSError, UnicodeDecodeError) as error:
            raise MigrationConfigurationError("a migration file could not be read as UTF-8") from error
        if "\x00" in sql:
            raise MigrationConfigurationError("migration files cannot contain NUL bytes")
        _reject_transaction_control(sql)
        migrations.append(
            Migration(
                version=version,
                filename=path.name,
                sql=sql,
                checksum_sha256=hashlib.sha256(raw_sql).hexdigest(),
            )
        )

    if not migrations:
        raise MigrationConfigurationError("no numbered SQL migrations were found")
    return tuple(sorted(migrations, key=lambda migration: (int(migration.version), migration.filename)))


def _top_level_sql_statements(sql: str) -> Iterator[str]:
    """Yield SQL statements while ignoring comments and quoted bodies.

    This is intentionally a narrow guard, not a general SQL parser. It is
    sufficient to distinguish a top-level transaction command from PL/pgSQL's
    BEGIN inside a dollar-quoted function body.
    """

    statement: list[str] = []
    index = 0
    length = len(sql)
    in_single_quote = False
    in_double_quote = False
    in_line_comment = False
    block_comment_depth = 0
    dollar_quote: str | None = None

    while index < length:
        character = sql[index]
        following = sql[index + 1] if index + 1 < length else ""

        if dollar_quote is not None:
            if sql.startswith(dollar_quote, index):
                tag = dollar_quote
                dollar_quote = None
                statement.append(" ")
                index += len(tag)
            else:
                index += 1
            continue

        if in_line_comment:
            if character in "\r\n":
                in_line_comment = False
                statement.append(" ")
            index += 1
            continue

        if block_comment_depth:
            if character == "/" and following == "*":
                block_comment_depth += 1
                index += 2
            elif character == "*" and following == "/":
                block_comment_depth -= 1
                if block_comment_depth == 0:
                    statement.append(" ")
                index += 2
            else:
                index += 1
            continue

        if in_single_quote:
            if character == "'" and following == "'":
                index += 2
            elif character == "'":
                in_single_quote = False
                statement.append(" ")
                index += 1
            elif character == "\\":
                # Supports E'' strings without making their contents look like
                # top-level SQL. PostgreSQL validates the actual string later.
                index += 2
            else:
                index += 1
            continue

        if in_double_quote:
            if character == '"' and following == '"':
                index += 2
            elif character == '"':
                in_double_quote = False
                statement.append(" ")
                index += 1
            else:
                index += 1
            continue

        if character == "-" and following == "-":
            in_line_comment = True
            index += 2
            continue
        if character == "/" and following == "*":
            block_comment_depth = 1
            index += 2
            continue
        if character == "'":
            in_single_quote = True
            statement.append(" ")
            index += 1
            continue
        if character == '"':
            in_double_quote = True
            statement.append(" ")
            index += 1
            continue
        if character == "$":
            match = _DOLLAR_QUOTE_RE.match(sql, index)
            if match is not None:
                dollar_quote = match.group(0)
                statement.append(" ")
                index = match.end()
                continue
        if character == ";":
            yield "".join(statement)
            statement = []
            index += 1
            continue
        statement.append(character)
        index += 1

    if statement:
        yield "".join(statement)


def _reject_transaction_control(sql: str) -> None:
    for statement in _top_level_sql_statements(sql):
        normalized = " ".join(statement.split())
        if _TRANSACTION_CONTROL_RE.match(normalized):
            raise MigrationTransactionControlError(
                "a migration contains transaction-control SQL; the migration runner owns transactions"
            )


@contextmanager
def _transaction(connection: _Connection) -> Iterator[None]:
    """Use psycopg's transaction context, with a simple fallback for tests."""

    transaction = getattr(connection, "transaction", None)
    if callable(transaction):
        with transaction():
            yield
        return

    try:
        yield
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def _ensure_ledger(connection: _Connection) -> None:
    with _transaction(connection):
        with connection.cursor() as cursor:
            cursor.execute(_CREATE_LEDGER_SQL)


def _read_applied_migrations(connection: _Connection) -> tuple[AppliedMigration, ...]:
    # A PostgreSQL SELECT also starts a transaction when autocommit is off.
    # Keep it bounded so each subsequent migration gets its own transaction,
    # rather than becoming a savepoint inside an accidentally open one.
    with _transaction(connection):
        with connection.cursor() as cursor:
            cursor.execute(_READ_APPLIED_SQL)
            rows = cursor.fetchall()
    return tuple(
        AppliedMigration(version=str(row[0]), filename=str(row[1]), checksum_sha256=str(row[2]))
        for row in rows
    )


def _assert_ledger_matches_local(
    applied: Sequence[AppliedMigration], migrations: Sequence[Migration]
) -> None:
    local_by_version = {migration.version: migration for migration in migrations}
    if len(local_by_version) != len(migrations):
        raise MigrationIntegrityError("local migration versions are ambiguous")
    if len({row.version for row in applied}) != len(applied):
        raise MigrationIntegrityError("database migration ledger contains duplicate versions")

    for row in applied:
        local = local_by_version.get(row.version)
        if (
            local is None
            or local.filename != row.filename
            or local.checksum_sha256 != row.checksum_sha256
        ):
            raise MigrationIntegrityError(
                "database migration history does not match the checked-in migration set"
            )

    # A database that claims a later migration without every earlier checked-in
    # migration is unsafe to repair by applying old DDL after newer DDL.
    applied_versions = {row.version for row in applied}
    encountered_pending = False
    for migration in migrations:
        if migration.version in applied_versions:
            if encountered_pending:
                raise MigrationIntegrityError(
                    "database migration history is not an ordered prefix of the checked-in migration set"
                )
        else:
            encountered_pending = True


def _acquire_lock(connection: _Connection) -> None:
    with _transaction(connection):
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_try_advisory_lock(%s)", (_MIGRATION_LOCK_KEY,))
            row = cursor.fetchone()
    if not row or not bool(row[0]):
        raise MigrationLockError("another database migration process is already running")


def _release_lock(connection: _Connection) -> None:
    with _transaction(connection):
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_unlock(%s)", (_MIGRATION_LOCK_KEY,))


def _release_lock_safely(connection: _Connection) -> None:
    """Release on a best-effort basis; closing PostgreSQL also releases it."""

    try:
        _release_lock(connection)
    except Exception:
        # Do not hide a prior migration failure with cleanup. This function is
        # intentionally only called immediately before the connection closes.
        return


def _apply_one(connection: _Connection, migration: Migration) -> None:
    try:
        with _transaction(connection):
            with connection.cursor() as cursor:
                # The SQL text is checked-in repository content, never request
                # data. Its transaction control was rejected during discovery.
                cursor.execute(migration.sql)
                cursor.execute(
                    _INSERT_APPLIED_SQL,
                    (migration.version, migration.filename, migration.checksum_sha256),
                )
    except MigrationError:
        raise
    except Exception as error:
        # The numbered filename/version is checked-in source, so it is safe to
        # name; the underlying database exception may include connection data.
        raise DatabaseMigrationError(
            f"checked-in migration {migration.version} could not be applied"
        ) from error


def apply_migrations(
    connection: _Connection, migrations_dir: Path | str | None = None
) -> MigrationRunResult:
    """Apply every pending checked-in migration exactly once on one connection."""

    migrations = discover_migrations(migrations_dir)
    lock_acquired = False
    try:
        _acquire_lock(connection)
        lock_acquired = True
        _ensure_ledger(connection)
        applied = _read_applied_migrations(connection)
        _assert_ledger_matches_local(applied, migrations)
        applied_versions = {row.version for row in applied}

        newly_applied: list[str] = []
        for migration in migrations:
            if migration.version in applied_versions:
                continue
            _apply_one(connection, migration)
            newly_applied.append(migration.version)

        return MigrationRunResult(
            applied_versions=tuple(newly_applied),
            current_versions=tuple(migration.version for migration in migrations),
        )
    finally:
        if lock_acquired:
            _release_lock_safely(connection)


def _default_connection_factory(database_url: str) -> _Connection:
    try:
        import psycopg
    except ImportError as error:  # pragma: no cover - exercised in deployment image/CI.
        raise DatabaseMigrationError("the PostgreSQL migration dependency is unavailable") from error
    try:
        return psycopg.connect(
            database_url,
            autocommit=False,
            application_name="sam-analytics-migrate",
        )
    except Exception as error:  # Do not leak a DSN, host, or credentials.
        raise DatabaseMigrationError("database connection could not be established") from error


def run_migrations(
    database_url: str,
    migrations_dir: Path | str | None = None,
    *,
    connection_factory: ConnectionFactory | None = None,
) -> MigrationRunResult:
    """Connect, migrate, and close without ever logging the supplied DSN."""

    if not isinstance(database_url, str) or not database_url.strip():
        raise MigrationConfigurationError("DATABASE_URL is required for database migrations")
    factory = connection_factory or _default_connection_factory
    connection: _Connection | None = None
    try:
        connection = factory(database_url)
        return apply_migrations(connection, migrations_dir)
    except MigrationError:
        raise
    except Exception as error:
        raise DatabaseMigrationError(
            "database migration failed; inspect protected database logs for details"
        ) from error
    finally:
        if connection is not None:
            _close_connection_safely(connection)


def _close_connection_safely(connection: _Connection) -> None:
    """Close without changing the outcome of a completed/failed migration."""

    try:
        connection.close()
    except Exception:
        return


def run_migrations_from_environment(
    *,
    migrations_dir: Path | str | None = None,
    environ: Mapping[str, str] | None = None,
    connection_factory: ConnectionFactory | None = None,
) -> MigrationRunResult:
    """Run using the host secret store's DATABASE_URL environment variable."""

    environment = os.environ if environ is None else environ
    database_url = environment.get("DATABASE_URL", "")
    return run_migrations(
        database_url,
        migrations_dir,
        connection_factory=connection_factory,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point that keeps credential-bearing errors private."""

    parser = argparse.ArgumentParser(description="Apply checked-in SAM PostgreSQL migrations.")
    parser.add_argument(
        "--migrations-dir",
        type=Path,
        default=None,
        help="override the checked-in migration directory (no database credentials accepted)",
    )
    args = parser.parse_args(argv)
    try:
        result = run_migrations_from_environment(migrations_dir=args.migrations_dir)
    except MigrationError as error:
        print(f"database migration failed: {error}", file=sys.stderr)
        return 1
    print(
        "database migrations complete: "
        f"{len(result.applied_versions)} applied; {len(result.current_versions)} current"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the deployment command.
    raise SystemExit(main())
