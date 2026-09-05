"""PostgreSQL persistence for sanitized manual-ingestion run facts.

The domain model in :mod:`sam_analytics.ingestion_runs` constructs valid run
identities and state transitions.  This module supplies the narrow database
boundary that appends those facts to the tables created by migration 005.

Each public write uses its own transaction.  A queued/running fact therefore
survives a later evidence-store or provider failure instead of being rolled
back with that separate operation.  PostgreSQL's transition trigger remains
the authoritative concurrency guard; this adapter also locks the run identity
and rejects a caller whose view of the latest state is stale.

Database URLs, driver exceptions, SQL text, and arbitrary failure messages are
never returned from this boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol
from uuid import UUID

from sam_analytics.ingestion_runs import (
    IngestionFailure,
    IngestionFailureCode,
    IngestionRun,
    IngestionRunState,
    IngestionRunStateTransition,
    IngestionRunTransitionError,
    IngestionRunValidationError,
)


class IngestionRunRepositoryError(RuntimeError):
    """Base error whose message is safe to expose to a private dispatcher."""


class IngestionRunRepositoryUnavailable(IngestionRunRepositoryError):
    """The append-only audit database could not be used safely."""


class IngestionRunRepositoryConflict(IngestionRunRepositoryError):
    """The requested identity or transition conflicts with stored audit facts."""


class IngestionRunRepositoryNotFound(IngestionRunRepositoryError):
    """No append-only audit state exists for the requested run."""


class IngestionRunRepository(Protocol):
    """Minimal persistence boundary for one manual shadow-ingestion run."""

    def create_run(
        self,
        run: IngestionRun,
        initial_transition: IngestionRunStateTransition,
    ) -> IngestionRunStateTransition:
        """Atomically append a run identity and its mandatory queued state."""

        ...

    def append_transition(
        self,
        run: IngestionRun,
        previous: IngestionRunStateTransition,
        transition: IngestionRunStateTransition,
    ) -> IngestionRunStateTransition:
        """Append a transition if ``previous`` is still the stored latest state."""

        ...

    def latest_transition(self, run_id: UUID) -> IngestionRunStateTransition:
        """Return the latest immutable state for a run."""

        ...


class _Cursor(Protocol):
    def execute(self, query: str, params: Sequence[Any] | None = None) -> Any:
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

    def close(self) -> None:
        ...


ConnectionFactory = Callable[[str], _Connection]


class PostgresIngestionRunRepository:
    """Append sanitized ingestion-run facts under short PostgreSQL transactions."""

    def __init__(
        self,
        database_url: str,
        *,
        connection_factory: ConnectionFactory | None = None,
    ) -> None:
        if not isinstance(database_url, str) or not database_url.strip():
            raise IngestionRunValidationError(
                "a database connection must be configured for ingestion run auditing"
            )
        self._database_url = database_url
        self._connection_factory = connection_factory or _connect_postgres

    def __repr__(self) -> str:
        return "PostgresIngestionRunRepository()"

    def create_run(
        self,
        run: IngestionRun,
        initial_transition: IngestionRunStateTransition,
    ) -> IngestionRunStateTransition:
        """Atomically write a new run and its initial queued transition."""

        _validate_initial_transition(run, initial_transition)
        connection: _Connection | None = None
        failure_kind: str | None = None
        try:
            connection = self._connection_factory(self._database_url)
            with connection.transaction():
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO ingestion_run (
                            id, provider, job_identity, source_type, run_mode,
                            max_attempts, created_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            run.id,
                            run.provider,
                            run.job_identity,
                            run.source_type,
                            run.run_mode,
                            run.max_attempts,
                            run.created_at,
                        ),
                    )
                    _insert_transition(cursor, initial_transition)
        except IngestionRunRepositoryError:
            raise
        except Exception as error:
            failure_kind = _write_failure_kind(error)
        finally:
            _close_safely(connection)

        if failure_kind == "conflict":
            raise IngestionRunRepositoryConflict(
                "ingestion run audit identity or initial state already exists"
            )
        if failure_kind is not None:
            raise IngestionRunRepositoryUnavailable(
                "ingestion run audit creation failed"
            )
        return initial_transition

    def append_transition(
        self,
        run: IngestionRun,
        previous: IngestionRunStateTransition,
        transition: IngestionRunStateTransition,
    ) -> IngestionRunStateTransition:
        """Append exactly one state transition from the caller's latest view."""

        _validate_append_inputs(run, previous, transition)
        connection: _Connection | None = None
        failure_kind: str | None = None
        try:
            connection = self._connection_factory(self._database_url)
            with connection.transaction():
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT provider, job_identity, source_type, run_mode,
                               max_attempts, created_at
                        FROM ingestion_run
                        WHERE id = %s
                        FOR UPDATE
                        """,
                        (run.id,),
                    )
                    stored_run_row = cursor.fetchone()
                    if stored_run_row is None:
                        raise IngestionRunRepositoryNotFound(
                            "ingestion run audit identity was not found"
                        )
                    if _decode_run(run.id, stored_run_row) != run:
                        raise IngestionRunRepositoryConflict(
                            "ingestion run audit identity does not match stored facts"
                        )

                    cursor.execute(
                        """
                        SELECT state_sequence, state, attempt_count, occurred_at,
                               failure_class, failure_code
                        FROM ingestion_run_state_transition
                        WHERE ingestion_run_id = %s
                        ORDER BY state_sequence DESC
                        LIMIT 1
                        """,
                        (run.id,),
                    )
                    latest_row = cursor.fetchone()
                    if latest_row is None:
                        raise IngestionRunRepositoryNotFound(
                            "ingestion run audit state was not found"
                        )
                    if _decode_transition(run.id, latest_row) != previous:
                        raise IngestionRunRepositoryConflict(
                            "ingestion run audit state has changed"
                        )
                    _insert_transition(cursor, transition)
        except IngestionRunRepositoryError:
            raise
        except Exception as error:
            failure_kind = _write_failure_kind(error)
        finally:
            _close_safely(connection)

        if failure_kind == "conflict":
            raise IngestionRunRepositoryConflict(
                "ingestion run audit transition conflicts with stored facts"
            )
        if failure_kind is not None:
            raise IngestionRunRepositoryUnavailable(
                "ingestion run audit append failed"
            )
        return transition

    def latest_transition(self, run_id: UUID) -> IngestionRunStateTransition:
        """Read the latest transition without returning free-form database data."""

        if not isinstance(run_id, UUID):
            raise IngestionRunValidationError("run id must be a UUID")
        connection: _Connection | None = None
        row: Sequence[Any] | None = None
        read_failed = False
        try:
            connection = self._connection_factory(self._database_url)
            with connection.transaction():
                with connection.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT state_sequence, state, attempt_count, occurred_at,
                               failure_class, failure_code
                        FROM ingestion_run_state_transition
                        WHERE ingestion_run_id = %s
                        ORDER BY state_sequence DESC
                        LIMIT 1
                        """,
                        (run_id,),
                    )
                    row = cursor.fetchone()
        except IngestionRunRepositoryError:
            raise
        except Exception:
            read_failed = True
        finally:
            _close_safely(connection)

        if read_failed:
            raise IngestionRunRepositoryUnavailable(
                "ingestion run audit read failed"
            )
        if row is None:
            raise IngestionRunRepositoryNotFound(
                "ingestion run audit state was not found"
            )
        return _decode_transition(run_id, row)


def _validate_initial_transition(
    run: IngestionRun,
    initial: IngestionRunStateTransition,
) -> None:
    if not isinstance(run, IngestionRun):
        raise IngestionRunValidationError("run must be an IngestionRun")
    if not isinstance(initial, IngestionRunStateTransition):
        raise IngestionRunValidationError(
            "initial state must be an ingestion run state transition"
        )
    if initial.ingestion_run_id != run.id:
        raise IngestionRunTransitionError(
            "initial state belongs to another ingestion run"
        )
    if (
        initial.state_sequence != 1
        or initial.state != IngestionRunState.QUEUED
        or initial.attempt_count != 0
        or initial.failure is not None
        or initial.occurred_at != run.created_at
    ):
        raise IngestionRunTransitionError(
            "an ingestion run must be created with its initial queued state"
        )


def _validate_append_inputs(
    run: IngestionRun,
    previous: IngestionRunStateTransition,
    transition: IngestionRunStateTransition,
) -> None:
    if not isinstance(run, IngestionRun):
        raise IngestionRunValidationError("run must be an IngestionRun")
    if not isinstance(previous, IngestionRunStateTransition):
        raise IngestionRunValidationError(
            "previous state must be an ingestion run state transition"
        )
    if not isinstance(transition, IngestionRunStateTransition):
        raise IngestionRunValidationError(
            "transition must be an ingestion run state transition"
        )
    if previous.ingestion_run_id != run.id or transition.ingestion_run_id != run.id:
        raise IngestionRunTransitionError(
            "ingestion run transitions must belong to the supplied run"
        )
    if transition.state_sequence != previous.state_sequence + 1:
        raise IngestionRunTransitionError(
            "ingestion run state sequence must be contiguous"
        )
    if transition.occurred_at < previous.occurred_at:
        raise IngestionRunTransitionError(
            "ingestion run state transition time cannot move backwards"
        )


def _insert_transition(
    cursor: _Cursor,
    transition: IngestionRunStateTransition,
) -> None:
    failure_class = None
    failure_code = None
    if transition.failure is not None:
        failure_class = transition.failure.classification.value
        failure_code = transition.failure.code.value
    cursor.execute(
        """
        INSERT INTO ingestion_run_state_transition (
            ingestion_run_id, state_sequence, state, attempt_count,
            failure_class, failure_code, occurred_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (
            transition.ingestion_run_id,
            transition.state_sequence,
            transition.state.value,
            transition.attempt_count,
            failure_class,
            failure_code,
            transition.occurred_at,
        ),
    )


def _decode_run(run_id: UUID, row: Sequence[Any]) -> IngestionRun:
    try:
        provider, job_identity, source_type, run_mode, max_attempts, created_at = row
        return IngestionRun(
            id=run_id,
            provider=provider,
            job_identity=job_identity,
            source_type=source_type,
            run_mode=run_mode,
            max_attempts=max_attempts,
            created_at=created_at,
        )
    except (TypeError, ValueError):
        raise IngestionRunRepositoryError(
            "stored ingestion run audit identity is invalid"
        ) from None


def _decode_transition(
    run_id: UUID,
    row: Sequence[Any],
) -> IngestionRunStateTransition:
    try:
        (
            state_sequence,
            state_value,
            attempt_count,
            occurred_at,
            failure_class_value,
            failure_code_value,
        ) = row
        failure = None
        if failure_code_value is not None:
            failure = IngestionFailure(IngestionFailureCode(failure_code_value))
            if failure_class_value != failure.classification.value:
                raise ValueError("failure classification mismatch")
        elif failure_class_value is not None:
            raise ValueError("failure code is missing")
        return IngestionRunStateTransition(
            ingestion_run_id=run_id,
            state_sequence=state_sequence,
            state=IngestionRunState(state_value),
            attempt_count=attempt_count,
            occurred_at=occurred_at,
            failure=failure,
        )
    except (TypeError, ValueError):
        raise IngestionRunRepositoryError(
            "stored ingestion run audit state is invalid"
        ) from None


def _write_failure_kind(error: Exception) -> str:
    """Classify integrity failures without inspecting database error text."""

    sqlstate = getattr(error, "sqlstate", None)
    if isinstance(sqlstate, str) and (
        sqlstate.startswith("23") or sqlstate == "P0001"
    ):
        return "conflict"
    return "unavailable"


def _close_safely(connection: _Connection | None) -> None:
    if connection is None:
        return
    try:
        connection.close()
    except Exception:
        # A completed transaction has already committed or rolled back.  A
        # transport-level cleanup error cannot safely change that known result.
        return


def _connect_postgres(database_url: str) -> _Connection:
    psycopg_module: Any = None
    import_failed = False
    try:
        import psycopg

        psycopg_module = psycopg
    except ImportError:
        import_failed = True
    if import_failed or psycopg_module is None:
        raise IngestionRunRepositoryUnavailable(
            "ingestion run audit database dependency is unavailable"
        )

    connection: _Connection | None = None
    connection_failed = False
    try:
        connection = psycopg_module.connect(
            database_url,
            application_name="sam-analytics-ingestion-run-audit",
            connect_timeout=5,
            options="-c statement_timeout=10000",
        )
    except Exception:
        connection_failed = True
    if connection_failed or connection is None:
        raise IngestionRunRepositoryUnavailable(
            "ingestion run audit database is unavailable"
        )
    return connection
