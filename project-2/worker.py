"""Fail-closed Celery entry point for future isolated ingestion jobs.

This module intentionally has no provider client import and no periodic task
schedule.  Starting a worker requires explicit database and Redis URLs, and a
queued ingestion task remains a no-op until a future audited implementation is
deliberately enabled.  Provider credentials must never be present in the web
process or in this module's logs.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from urllib.parse import urlsplit

from celery import Celery


class WorkerConfigurationError(RuntimeError):
    """Raised without exposing connection strings or provider credentials."""


def _required_url(
    name: str,
    allowed_schemes: frozenset[str],
    environ: Mapping[str, str],
) -> str:
    """Return a required service URL after a credential-safe structural check."""

    value = environ.get(name)
    if not isinstance(value, str) or not value.strip():
        raise WorkerConfigurationError(f"{name} must be configured before the worker can start")
    value = value.strip()
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
    except ValueError:
        raise WorkerConfigurationError(f"{name} must be a valid service URL") from None
    if parsed.scheme not in allowed_schemes or not hostname:
        raise WorkerConfigurationError(f"{name} must be a valid service URL")
    return value


def create_celery_app(environ: Mapping[str, str] | None = None) -> Celery:
    """Build a worker app only when its private dependencies are configured.

    The database URL is validated even though this placeholder task does not
    use it yet: no future ingestion worker should start with a queue but no
    durable audit store.  Celery result storage is disabled so task return
    values cannot fill the shared, ``noeviction`` Valkey instance.
    """

    source = os.environ if environ is None else environ
    _required_url("DATABASE_URL", frozenset({"postgresql", "postgres"}), source)
    redis_url = _required_url("REDIS_URL", frozenset({"redis", "rediss"}), source)
    app = Celery("sam_analytics", broker=redis_url, backend="disabled://")
    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        task_ignore_result=True,
        task_store_errors_even_if_ignored=False,
        task_track_started=False,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        task_soft_time_limit=75,
        task_time_limit=90,
        task_default_queue="sam_ingestion",
        # An explicit empty mapping prevents Celery Beat from autonomously
        # polling a provider.  A later audited dispatcher may enqueue bounded
        # work only after a human deliberately enables it.
        beat_schedule={},
    )
    return app


def ingestion_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return true only for an explicit operator opt-in.

    Anything other than the literal word ``true`` is disabled, including a
    missing variable, an accidental provider key, or a copied development
    configuration.  Enabling this flag alone still does not authorize network
    calls; the task remains intentionally unimplemented below.
    """

    source = os.environ if environ is None else environ
    value = source.get("SAM_INGESTION_ENABLED", "")
    return isinstance(value, str) and value.strip().lower() == "true"


celery_app = create_celery_app()


@celery_app.task(name="sam_analytics.ingest_quotes")
def ingest_quotes() -> dict:
    """Deliberately inert queue hook for a future licensed provider adapter.

    This task does not import an adapter, issue HTTP requests, or write to
    PostgreSQL.  It exists only to make the disabled/explicit-enabled boundary
    testable before the complete payload-receipt ledger is wired in.
    """
    if not ingestion_enabled():
        return {
            "status": "disabled",
            "reason": "SAM_INGESTION_ENABLED must be explicitly true before ingestion can be considered",
        }
    return {
        "status": "not_configured",
        "reason": "audited payload storage and persistence are not configured; no provider request was made",
    }


@celery_app.task(name="sam_analytics.settle_events")
def settle_events() -> dict:
    """Deliberately inert hook for a future licensed results provider."""
    return {
        "status": "not_configured",
        "reason": "results ingestion is not implemented; no provider request was made",
    }
