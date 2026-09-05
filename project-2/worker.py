"""Fail-closed Celery entry point for one synthetic staging storage probe.

The worker is private, manual-only, and intentionally incapable of contacting
an odds or results provider. Its sole admitted operation writes deterministic
synthetic bytes through the reviewed R2 adapter and records an append-only
PostgreSQL lifecycle. The real ingestion and settlement task names remain
registered only so an accidental dispatch fails visibly.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from uuid import UUID

from celery import Celery
from kombu import Queue

from sam_analytics.ingestion_run_repository import PostgresIngestionRunRepository
from sam_analytics.s3_payload_store import S3CompatibleRawPayloadStore
from sam_analytics.synthetic_evidence_probe import SyntheticEvidenceProbe
from sam_analytics.worker_settings import (
    PrivateWorkerConfigurationError,
    PrivateWorkerSettings,
)


WorkerConfigurationError = PrivateWorkerConfigurationError


class IngestionNotImplementedError(RuntimeError):
    """Raised when an intentionally inert provider task is dispatched."""


_MANUAL_SHADOW_QUEUE = "sam_manual_shadow"
_SYNTHETIC_PROBE_TASK = "sam_analytics.verify_staging_raw_evidence"


def create_celery_app(environ: Mapping[str, str] | None = None) -> Celery:
    """Build the worker only inside the exact synthetic staging boundary."""

    source = os.environ if environ is None else environ
    PrivateWorkerSettings.from_environment(source)
    broker_url = source["REDIS_URL"]
    app = Celery("sam_analytics", broker=broker_url, backend="disabled://")
    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        task_ignore_result=True,
        task_store_errors_even_if_ignored=False,
        task_track_started=False,
        # One deliberate dispatch must never become an automatic retry or a
        # worker-lost redelivery. A human may inspect the append-only outcome
        # before choosing whether to dispatch a new task.
        task_acks_late=False,
        task_reject_on_worker_lost=False,
        worker_prefetch_multiplier=1,
        worker_concurrency=1,
        task_soft_time_limit=75,
        task_time_limit=90,
        task_default_queue=_MANUAL_SHADOW_QUEUE,
        task_queues=(Queue(_MANUAL_SHADOW_QUEUE),),
        task_create_missing_queues=False,
        task_routes={
            _SYNTHETIC_PROBE_TASK: {"queue": _MANUAL_SHADOW_QUEUE},
            # Known-but-disabled task names use the consumed queue so an
            # accidental dispatch fails visibly instead of accumulating in an
            # unconsumed Redis list.
            "sam_analytics.ingest_quotes": {"queue": _MANUAL_SHADOW_QUEUE},
            "sam_analytics.settle_events": {"queue": _MANUAL_SHADOW_QUEUE},
        },
        task_send_sent_event=False,
        worker_send_task_events=False,
        worker_enable_remote_control=False,
        beat_schedule={},
    )
    return app


def ingestion_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return true only for an explicit operator switch.

    The admitted worker configuration rejects this value, so it cannot make a
    provider request. Keeping the parser separate makes the inert task's
    failure behavior explicit and testable.
    """

    source = os.environ if environ is None else environ
    value = source.get("SAM_INGESTION_ENABLED", "")
    return isinstance(value, str) and value.strip().lower() == "true"


def _celery_job_identity(task_id: object) -> str:
    """Convert only a canonical Celery UUID into a safe audit identity."""

    if not isinstance(task_id, str):
        raise WorkerConfigurationError("the manual probe task identity is invalid")
    parsed: UUID | None = None
    parse_failed = False
    try:
        parsed = UUID(task_id)
    except (AttributeError, TypeError, ValueError):
        parse_failed = True
    if parse_failed or parsed is None or str(parsed) != task_id.lower():
        raise WorkerConfigurationError("the manual probe task identity is invalid")
    return f"celery:{parsed}"


def _execute_synthetic_storage_probe(
    *,
    task_id: object,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Revalidate admission and execute the injected synthetic probe once."""

    source = os.environ if environ is None else environ
    PrivateWorkerSettings.from_environment(source)
    job_identity = _celery_job_identity(task_id)
    raw_payload_store = S3CompatibleRawPayloadStore.from_environment(source)
    ingestion_run_repository = PostgresIngestionRunRepository(source["DATABASE_URL"])
    probe = SyntheticEvidenceProbe(
        raw_payload_store=raw_payload_store,
        ingestion_run_repository=ingestion_run_repository,
    )
    # Deliberately discard the internal receipt summary. Celery result storage
    # is disabled and the task returns no digest, object reference, or bytes.
    probe.run(job_identity=job_identity)


celery_app = create_celery_app()


@celery_app.task(
    bind=True,
    name=_SYNTHETIC_PROBE_TASK,
    queue=_MANUAL_SHADOW_QUEUE,
    ignore_result=True,
    acks_late=False,
    reject_on_worker_lost=False,
    max_retries=0,
)
def verify_staging_raw_evidence(task) -> None:
    """Manually verify fixed synthetic bytes against staging R2 and audit."""

    _execute_synthetic_storage_probe(task_id=task.request.id)


@celery_app.task(
    name="sam_analytics.ingest_quotes",
    queue=_MANUAL_SHADOW_QUEUE,
    ignore_result=True,
    max_retries=0,
)
def ingest_quotes() -> None:
    """Fail closed; licensed provider ingestion is not implemented."""

    if not ingestion_enabled():
        raise IngestionNotImplementedError(
            "SAM_INGESTION_ENABLED is false; no provider request was made"
        )
    raise IngestionNotImplementedError(
        "audited provider ingestion is not implemented; no provider request was made"
    )


@celery_app.task(
    name="sam_analytics.settle_events",
    queue=_MANUAL_SHADOW_QUEUE,
    ignore_result=True,
    max_retries=0,
)
def settle_events() -> None:
    """Fail closed; licensed results ingestion is not implemented."""

    raise IngestionNotImplementedError(
        "results ingestion is not implemented; no provider request was made"
    )
