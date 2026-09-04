"""Secrets-safe operational status contract for trusted UI integrations.

The public web process must never manufacture a healthy feed or an approved
model merely because a UI needs something to render.  Workers or a repository
adapter may supply :class:`OperationalSignals`; when a signal is absent or
invalid, this module reports that fact and keeps prediction delivery disabled.

This is intentionally a contract layer, not a database client.  It makes the
current safe default useful to Base44 (or another UI) while leaving the future
PostgreSQL health repository independently testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


CONTRACT_VERSION = "v1"


@dataclass(frozen=True)
class OperationalSignals:
    """Non-secret signals supplied by a trusted worker or repository adapter.

    All timestamps refer to when the source fact became available to SAM, not
    when an event happened.  A missing value means "unknown" rather than a
    healthy default.  `model_evaluated_at` is the timestamp of the approved
    validation/monitoring report, not a claimed profitability timestamp.
    """

    provider: str | None = None
    latest_quote_at: datetime | None = None
    model_version: str | None = None
    model_approved: bool = False
    model_artifact_verified: bool = False
    model_evaluated_at: datetime | None = None
    # A configured URL is not proof that the dependency is reachable. These
    # signals may be set only by a trusted repository/worker health probe.
    audit_repository_healthy: bool = False
    worker_queue_healthy: bool = False


def build_integration_status(
    *,
    database_configured: bool,
    redis_configured: bool,
    quote_max_age_seconds: int,
    model_evaluation_max_age_seconds: int = 86_400,
    signals: OperationalSignals | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build the versioned status payload consumed by trusted UI gateways.

    Configuration flags establish only that an endpoint has been supplied;
    they are not connectivity probes.  Consequently, a "ready" response also
    requires a fresh source observation and a verified, approved model health
    report.  No predictions, odds, database URLs, or credentials are included.
    """

    if quote_max_age_seconds <= 0:
        raise ValueError("quote_max_age_seconds must be positive")
    if model_evaluation_max_age_seconds <= 0:
        raise ValueError("model_evaluation_max_age_seconds must be positive")
    current_time = _as_utc(now or datetime.now(timezone.utc))
    if current_time is None:
        raise ValueError("now must be timezone-aware")

    signals = signals or OperationalSignals()
    data = _data_status(signals, quote_max_age_seconds, current_time)
    model = _model_status(signals, current_time, model_evaluation_max_age_seconds)

    blockers: list[str] = []
    if not database_configured:
        blockers.append("audit database is not configured")
    if not redis_configured:
        blockers.append("worker queue is not configured")
    if signals.audit_repository_healthy is not True:
        blockers.append("audit database health is unverified")
    if signals.worker_queue_healthy is not True:
        blockers.append("worker queue health is unverified")
    if data["status"] != "fresh":
        blockers.append(f"data freshness is {data['status']}")
    if model["status"] != "healthy":
        blockers.append(f"model health is {model['status']}")

    prediction_delivery_enabled = not blockers
    return {
        "status": "ready" if prediction_delivery_enabled else "blocked",
        "generated_at": _timestamp(current_time),
        "data_freshness": data,
        "model_health": model,
        "risk_status": {
            "status": "research_only",
            "wager_submission": "unsupported",
        },
        "deployment": {
            "status": "ready" if prediction_delivery_enabled else "blocked",
            "contract_version": CONTRACT_VERSION,
            "audit_database": _dependency_status(
                database_configured, signals.audit_repository_healthy
            ),
            "worker_queue": _dependency_status(redis_configured, signals.worker_queue_healthy),
            "prediction_delivery": "available" if prediction_delivery_enabled else "disabled",
            "blockers": blockers,
        },
    }


def _data_status(
    signals: OperationalSignals, quote_max_age_seconds: int, now: datetime
) -> dict[str, Any]:
    observed_at = _as_utc(signals.latest_quote_at)
    provider = signals.provider if isinstance(signals.provider, str) and signals.provider.strip() else None
    if signals.latest_quote_at is None:
        return {
            "status": "unavailable",
            "provider": provider,
            "latest_quote_at": None,
            "age_seconds": None,
            "max_age_seconds": quote_max_age_seconds,
        }
    if observed_at is None or observed_at > now:
        return {
            "status": "invalid",
            "provider": provider,
            "latest_quote_at": _timestamp_or_none(observed_at),
            "age_seconds": None,
            "max_age_seconds": quote_max_age_seconds,
        }

    age_seconds = round((now - observed_at).total_seconds(), 3)
    return {
        "status": "fresh" if provider and age_seconds <= quote_max_age_seconds else (
            "stale" if provider else "invalid"
        ),
        "provider": provider,
        "latest_quote_at": _timestamp(observed_at),
        "age_seconds": age_seconds,
        "max_age_seconds": quote_max_age_seconds,
    }


def _model_status(
    signals: OperationalSignals, now: datetime, model_evaluation_max_age_seconds: int
) -> dict[str, Any]:
    evaluated_at = _as_utc(signals.model_evaluated_at)
    version = signals.model_version if isinstance(signals.model_version, str) and signals.model_version.strip() else None
    payload: dict[str, Any] = {
        "status": "unavailable",
        "version": version,
        "approved": signals.model_approved is True,
        "artifact_verified": signals.model_artifact_verified is True,
        "last_evaluated_at": _timestamp_or_none(evaluated_at),
        "serving_allowed": False,
    }
    if not version:
        return payload
    if signals.model_approved is not True:
        payload["status"] = "unapproved"
        return payload
    if signals.model_artifact_verified is not True:
        payload["status"] = "artifact_unverified"
        return payload
    if signals.model_evaluated_at is None:
        payload["status"] = "monitoring_unavailable"
        return payload
    if evaluated_at is None or evaluated_at > now:
        payload["status"] = "invalid"
        return payload
    if (now - evaluated_at).total_seconds() > model_evaluation_max_age_seconds:
        payload["status"] = "monitoring_stale"
        return payload
    payload["status"] = "healthy"
    payload["serving_allowed"] = True
    return payload


def _as_utc(value: datetime | None) -> datetime | None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        return None
    return value.astimezone(timezone.utc)


def _timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _timestamp_or_none(value: datetime | None) -> str | None:
    return _timestamp(value) if value is not None else None


def _dependency_status(configured: bool, healthy: object) -> str:
    if configured is not True:
        return "unconfigured"
    return "healthy" if healthy is True else "unverified"
