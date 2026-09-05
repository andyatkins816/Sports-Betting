"""Fail-closed Celery entry point for future isolated ingestion jobs.

This module intentionally has no provider client import and no periodic task
schedule.  It may only start after an operator has declared a *private* worker
role and a private, secret-free raw-evidence object-store prefix. It still
does not implement a provider request: that requires a reviewed integration
with the evidence store, provider contract, and bounded dispatcher.

Web/API, provider, and object-store credentials are deliberately rejected by
this placeholder entry point. Keeping credentials out of an inert worker
prevents a future config change from accidentally making a live request before
the complete evidence path exists.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from urllib.parse import urlsplit

from celery import Celery


class WorkerConfigurationError(RuntimeError):
    """Raised without exposing connection strings or provider credentials."""


class IngestionNotImplementedError(RuntimeError):
    """Raised when someone attempts to execute an intentionally inert task."""


_PRIVATE_WORKER_ROLE = "private_ingestion"
# The concrete immutable-store boundary currently supports AWS S3 and
# Cloudflare R2 through the S3-compatible ``s3://`` object-reference form.
_PRIVATE_EVIDENCE_STORE_SCHEMES = frozenset({"s3"})
# These shapes intentionally match the durable object-reference contract in
# ``raw_payload_store``.  A configured prefix must be able to become a valid
# content-addressed receipt URI once the store is wired into a worker.
_PRIVATE_OBJECT_NAMESPACE_RE = re.compile(
    r"^[a-z0-9](?:[a-z0-9.-]{1,61}[a-z0-9])$"
)
_PRIVATE_OBJECT_PATH_PART_RE = re.compile(r"^[a-z0-9][a-z0-9._=-]{0,63}$")
_IPV4_LIKE_NAMESPACE_RE = re.compile(r"^\d+\.\d+\.\d+\.\d+$")
_INERT_WORKER_FORBIDDEN_SECRET_NAMES = (
    "SESSION_SECRET",
    "SAM_API_KEY",
    "SAM_STATUS_API_KEY",
    "ODDS_PROVIDER_API_KEY",
    "RESULTS_PROVIDER_API_KEY",
    "SAM_RAW_EVIDENCE_S3_ACCESS_KEY_ID",
    "SAM_RAW_EVIDENCE_S3_SECRET_ACCESS_KEY",
)


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


def _required_private_evidence_store_uri(environ: Mapping[str, str]) -> None:
    """Require a secret-free private object-store prefix for a worker.

    This verifies only the deployment contract, not a cloud-provider policy:
    operators must still block public access and grant this worker the minimum
    bucket/prefix permissions. A concrete storage adapter exists separately,
    but is intentionally not wired into this inert worker yet.
    """

    value = environ.get("SAM_RAW_EVIDENCE_STORE_URI")
    if not isinstance(value, str) or not value.strip():
        raise WorkerConfigurationError(
            "SAM_RAW_EVIDENCE_STORE_URI must be configured before the worker can start"
        )
    try:
        parsed = urlsplit(value.strip())
        port = parsed.port
    except ValueError:
        raise WorkerConfigurationError(
            "SAM_RAW_EVIDENCE_STORE_URI must be a valid private URI"
        ) from None

    raw_path_parts = tuple(parsed.path.split("/"))
    path_parts = tuple(part for part in raw_path_parts if part)
    namespace = parsed.hostname
    if (
        parsed.scheme not in _PRIVATE_EVIDENCE_STORE_SCHEMES
        or not namespace
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
        or not path_parts
        or any(not part for part in raw_path_parts[1:])
        or any(part in {".", "..", "sha256"} for part in path_parts)
        or not _PRIVATE_OBJECT_NAMESPACE_RE.fullmatch(namespace)
        or ".." in namespace
        or ".-" in namespace
        or "-." in namespace
        or _IPV4_LIKE_NAMESPACE_RE.fullmatch(namespace)
        or namespace.startswith("xn--")
        or not all(_PRIVATE_OBJECT_PATH_PART_RE.fullmatch(part) for part in path_parts)
    ):
        raise WorkerConfigurationError(
            "SAM_RAW_EVIDENCE_STORE_URI must be a valid private URI"
        )


def _require_private_worker_role(environ: Mapping[str, str]) -> None:
    """Reject a worker that was not explicitly designated private."""

    value = environ.get("SAM_WORKER_ROLE", "")
    if not isinstance(value, str) or value.strip() != _PRIVATE_WORKER_ROLE:
        raise WorkerConfigurationError(
            "SAM_WORKER_ROLE must be private_ingestion before the worker can start"
        )


def _reject_inert_worker_secrets(environ: Mapping[str, str]) -> None:
    """Do not let a currently inert worker accumulate usable credentials."""

    if any(
        isinstance(environ.get(name), str) and environ[name].strip()
        for name in _INERT_WORKER_FORBIDDEN_SECRET_NAMES
    ):
        raise WorkerConfigurationError(
            "web, provider, or object-store credentials cannot be configured "
            "until audited ingestion is implemented"
        )


def _validate_private_worker_admission(environ: Mapping[str, str]) -> str:
    """Validate the minimum safe boundary for launching the worker process.

    The returned broker URL is used only to configure Celery.  Provider keys
    are neither read nor accepted here.  This boundary intentionally requires
    an object-store *prefix* before the process can run, but it makes no
    network call and cannot prove a bucket policy by itself.
    """

    _require_private_worker_role(environ)
    _required_url("DATABASE_URL", frozenset({"postgresql", "postgres"}), environ)
    redis_url = _required_url("REDIS_URL", frozenset({"redis", "rediss"}), environ)
    _required_private_evidence_store_uri(environ)
    _reject_inert_worker_secrets(environ)
    if ingestion_enabled(environ):
        raise WorkerConfigurationError(
            "SAM_INGESTION_ENABLED cannot be true until audited ingestion is implemented"
        )
    return redis_url


def create_celery_app(environ: Mapping[str, str] | None = None) -> Celery:
    """Build a worker app only when its private dependencies are configured.

    This entry point will not run as a generic worker or a public web process.
    It needs an explicit private worker role, internal database/broker URLs,
    and a private object-store prefix even though its task is still inert.
    Celery result storage is disabled so task return values cannot fill the
    shared, ``noeviction`` Valkey instance.
    """

    source = os.environ if environ is None else environ
    redis_url = _validate_private_worker_admission(source)
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
        task_routes={
            "sam_analytics.ingest_quotes": {"queue": "sam_ingestion"},
            "sam_analytics.settle_events": {"queue": "sam_ingestion"},
        },
        task_send_sent_event=False,
        worker_send_task_events=False,
        worker_enable_remote_control=False,
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
def ingest_quotes() -> None:
    """Deliberately inert queue hook for a future licensed provider adapter.

    This task does not import an adapter, issue HTTP requests, or write to
    PostgreSQL.  It exists only to make the disabled/explicit-enabled boundary
    testable before the complete payload-receipt ledger is wired in.
    """
    # This is intentionally an error rather than a successful no-op. An
    # accidental dispatcher should be observable as blocked, never mistaken
    # for a completed provider pull. No provider module is imported first.
    if not ingestion_enabled():
        raise IngestionNotImplementedError(
            "SAM_INGESTION_ENABLED is false; no provider request was made"
        )
    raise IngestionNotImplementedError(
        "audited provider ingestion is not implemented; no provider request was made"
    )


@celery_app.task(name="sam_analytics.settle_events")
def settle_events() -> None:
    """Deliberately inert hook for a future licensed results provider."""
    raise IngestionNotImplementedError(
        "results ingestion is not implemented; no provider request was made"
    )
