"""Pure, secrets-safe health evaluation for durable ingestion observations.

This module deliberately does not inspect a database, broker, worker process,
provider, environment, or network.  A trusted adapter may supply already
sanitized facts from those private boundaries.  Missing or malformed facts do
not become healthy defaults.

Worker status is named *activity* because a recent durable transition proves
that work happened, not that a process is currently alive.  A future heartbeat
or lease contract is required before SAM can truthfully report process liveness.
"""

from __future__ import annotations

import math
import re
from dataclasses import InitVar, dataclass, fields
from datetime import UTC, datetime, timedelta
from typing import Any

ALERT_CODES = frozenset(
    {
        "dead_letter_present",
        "dead_letter_unavailable",
        "feed_stale",
        "feed_unavailable",
        "monitoring_invalid",
        "queue_backlogged",
        "queue_stalled",
        "queue_unavailable",
        "quota_exhausted",
        "quota_low",
        "quota_stale",
        "quota_unavailable",
        "retry_wait_present",
        "worker_activity_stale",
        "worker_activity_unavailable",
        "worker_stalled",
    }
)

_ALERT_ORDER = (
    "monitoring_invalid",
    "feed_unavailable",
    "feed_stale",
    "worker_activity_unavailable",
    "worker_activity_stale",
    "worker_stalled",
    "queue_unavailable",
    "queue_backlogged",
    "queue_stalled",
    "retry_wait_present",
    "quota_unavailable",
    "quota_stale",
    "quota_low",
    "quota_exhausted",
    "dead_letter_unavailable",
    "dead_letter_present",
)

# Construction is deliberately limited to ``evaluate_ingestion_health``.  The
# value object crosses a publication gate, so accepting an arbitrary hand-built
# instance would let a caller claim a healthy top-level status without running
# the fail-closed evaluator.  InitVar keeps the marker out of the stored/public
# object and also makes ``dataclasses.replace`` revalidate through the factory.
_EVALUATED_HEALTH_TOKEN = object()
_SAFE_PROVIDER_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
INGESTION_HEALTH_MAX_REUSE_SECONDS = 5


@dataclass(frozen=True)
class IngestionHealthPolicy:
    """Reviewed thresholds used to interpret sanitized observations.

    Age and depth comparisons are inclusive: an observation exactly at its
    maximum age, or a queue exactly at its warning depth, has not crossed the
    corresponding threshold.  Quota is low at the low-watermark because that
    value represents the minimum reserve that must remain untouched.
    """

    quote_max_age_seconds: int
    worker_activity_max_age_seconds: int
    quota_max_age_seconds: int
    queue_oldest_max_age_seconds: int
    queue_depth_warning: int
    quota_low_watermark: int

    def __post_init__(self) -> None:
        _require_positive_integer(self.quote_max_age_seconds, "quote maximum age")
        _require_positive_integer(
            self.worker_activity_max_age_seconds,
            "worker activity maximum age",
        )
        _require_positive_integer(self.quota_max_age_seconds, "quota maximum age")
        _require_positive_integer(
            self.queue_oldest_max_age_seconds,
            "queue oldest maximum age",
        )
        _require_nonnegative_integer(self.queue_depth_warning, "queue depth warning")
        _require_nonnegative_integer(self.quota_low_watermark, "quota low watermark")


@dataclass(frozen=True)
class IngestionHealthFacts:
    """Sanitized facts loaded from trusted, private persistence adapters.

    Fields use ``object`` intentionally.  Type hints at an integration boundary
    are not runtime validation, so :func:`evaluate_ingestion_health` validates
    every value and fails closed if an adapter returns corrupt data.

    ``backlog_count`` covers dispatches durably pending publication or queued.
    ``retry_wait_count`` is separate, while ``oldest_outstanding_at`` covers the
    union of both sets.  ``latest_worker_activity_at`` is the latest durable
    worker-owned transition, never a broker ping or inferred heartbeat.
    ``provider`` binds all of these facts to the same internal source identity;
    it is validated here but omitted from the nested public projection.
    """

    provider: object = None
    latest_quote_received_at: object = None
    latest_worker_activity_at: object = None
    quota_remaining: object = None
    quota_reserved: object = None
    quota_observed_at: object = None
    backlog_count: object = None
    oldest_outstanding_at: object = None
    retry_wait_count: object = None
    dead_letter_count: object = None


@dataclass(frozen=True)
class IngestionHealth:
    """Immutable result that exposes only the reviewed, finite public shape."""

    provider: str | None
    evaluated_at: datetime
    valid_until: datetime
    status: str
    quote_freshness_status: str
    quote_age_seconds: float | None
    quote_max_age_seconds: int
    worker_activity_status: str
    worker_activity_age_seconds: float | None
    worker_activity_max_age_seconds: int
    queue_status: str
    queue_depth_band: str
    queue_oldest_age_seconds: float | None
    queue_max_oldest_age_seconds: int
    retry_wait_status: str
    quota_status: str
    quota_remaining_band: str
    quota_age_seconds: float | None
    quota_max_age_seconds: int
    dead_letter_status: str
    dead_letter_count_band: str
    alert_codes: tuple[str, ...]
    _evaluation_token: InitVar[object] = None

    def __post_init__(self, _evaluation_token: object) -> None:
        if _evaluation_token is not _EVALUATED_HEALTH_TOKEN:
            raise ValueError("ingestion health must be created by the evaluator")
        if self.provider is not None and _safe_provider(self.provider) is None:
            raise ValueError("ingestion health provider must be a safe identifier")
        evaluated_at = _require_aware_datetime(
            self.evaluated_at,
            "ingestion health evaluation time",
        )
        valid_until = _require_aware_datetime(
            self.valid_until,
            "ingestion health validity time",
        )
        if not (
            evaluated_at
            <= valid_until
            <= evaluated_at + timedelta(seconds=INGESTION_HEALTH_MAX_REUSE_SECONDS)
        ):
            raise ValueError("ingestion health validity must use the reviewed reuse window")
        _require_member(self.status, {"healthy", "degraded", "blocked", "unavailable"})
        _require_member(
            self.quote_freshness_status,
            {"fresh", "stale", "unavailable", "invalid"},
        )
        _require_member(
            self.worker_activity_status,
            {"active", "stalled", "idle_unverified", "unavailable", "invalid"},
        )
        _require_member(
            self.queue_status,
            {"clear", "backlogged", "stalled", "unavailable", "invalid"},
        )
        _require_member(
            self.queue_depth_band,
            {"empty", "normal", "high", "unavailable", "invalid"},
        )
        _require_member(
            self.retry_wait_status,
            {"clear", "present", "unavailable", "invalid"},
        )
        _require_member(
            self.quota_status,
            {"healthy", "low", "exhausted", "stale", "unavailable", "invalid"},
        )
        _require_member(
            self.quota_remaining_band,
            {"adequate", "low", "zero", "unknown", "unavailable", "invalid"},
        )
        _require_member(
            self.dead_letter_status,
            {"clear", "present", "unavailable", "invalid"},
        )
        _require_member(
            self.dead_letter_count_band,
            {"zero", "one_or_more", "unavailable", "invalid"},
        )
        for age in (
            self.quote_age_seconds,
            self.worker_activity_age_seconds,
            self.queue_oldest_age_seconds,
            self.quota_age_seconds,
        ):
            _require_optional_nonnegative_number(age)
        for maximum in (
            self.quote_max_age_seconds,
            self.worker_activity_max_age_seconds,
            self.queue_max_oldest_age_seconds,
            self.quota_max_age_seconds,
        ):
            _require_positive_integer(maximum, "public maximum age")
        if (
            not isinstance(self.alert_codes, tuple)
            or len(self.alert_codes) != len(set(self.alert_codes))
            or any(code not in ALERT_CODES for code in self.alert_codes)
        ):
            raise ValueError("alert codes must be unique approved finite codes")

    def to_public_dict(self) -> dict[str, Any]:
        """Return a new credential-safe status payload for the trusted gateway."""

        return {
            "status": self.status,
            "quote_freshness": {
                "status": self.quote_freshness_status,
                "age_seconds": self.quote_age_seconds,
                "max_age_seconds": self.quote_max_age_seconds,
            },
            "worker_activity": {
                "status": self.worker_activity_status,
                "age_seconds": self.worker_activity_age_seconds,
                "max_age_seconds": self.worker_activity_max_age_seconds,
                "basis": "durable_activity_only",
            },
            "queue": {
                "status": self.queue_status,
                "depth_band": self.queue_depth_band,
                "oldest_age_seconds": self.queue_oldest_age_seconds,
                "max_oldest_age_seconds": self.queue_max_oldest_age_seconds,
                "retry_wait": self.retry_wait_status,
            },
            "quota": {
                "status": self.quota_status,
                "remaining_band": self.quota_remaining_band,
                "age_seconds": self.quota_age_seconds,
                "max_age_seconds": self.quota_max_age_seconds,
            },
            "dead_letter": {
                "status": self.dead_letter_status,
                "count_band": self.dead_letter_count_band,
            },
            "alert_codes": list(self.alert_codes),
        }


def evaluate_ingestion_health(
    *,
    policy: IngestionHealthPolicy,
    facts: IngestionHealthFacts | None = None,
    now: datetime,
) -> IngestionHealth:
    """Return finite status bands from sanitized observations.

    Only the top-level ``healthy`` state is a passing result.  ``degraded`` is
    useful for operator display but remains fail-closed to a publication gate.
    The payload contains no provider content, request identity, object location,
    credential, queue name, or exact quota/dead-letter count.
    """

    if not isinstance(policy, IngestionHealthPolicy):
        raise ValueError("policy must be an IngestionHealthPolicy")
    current_time = _require_aware_datetime(now, "now")
    observations = IngestionHealthFacts() if facts is None else facts
    if not isinstance(observations, IngestionHealthFacts):
        raise ValueError("facts must be IngestionHealthFacts or None")

    alerts: set[str] = set()
    provider = _safe_provider(observations.provider)
    provider_invalid = provider is None and not _all_signals_missing(observations)
    if provider_invalid:
        alerts.add("monitoring_invalid")
    quote = _quote_status(observations, policy, current_time, alerts)
    queue = _queue_status(observations, policy, current_time, alerts)
    worker = _worker_activity_status(observations, policy, current_time, queue, alerts)
    quota = _quota_status(observations, policy, current_time, alerts)
    dead_letter = _dead_letter_status(observations, alerts)

    components = (quote, worker, queue, quota, dead_letter)
    if _all_signals_missing(observations):
        overall = "unavailable"
    elif provider_invalid:
        overall = "blocked"
    elif _has_blocking_component(quote, worker, queue, quota, dead_letter):
        overall = "blocked"
    elif (
        worker["status"] == "idle_unverified"
        or queue["status"] == "backlogged"
        or queue["retry_wait"] == "present"
        or quota["status"] == "low"
    ):
        overall = "degraded"
    elif all(component["status"] in {"fresh", "active", "clear", "healthy"} for component in components):
        overall = "healthy"
    else:
        # Future status additions cannot accidentally turn into a passing state.
        overall = "blocked"

    return IngestionHealth(
        provider=provider,
        evaluated_at=current_time,
        valid_until=_health_valid_until(
            now=current_time,
            quote=quote,
            worker=worker,
            queue=queue,
            quota=quota,
        ),
        status=overall,
        quote_freshness_status=quote["status"],
        quote_age_seconds=quote["age_seconds"],
        quote_max_age_seconds=quote["max_age_seconds"],
        worker_activity_status=worker["status"],
        worker_activity_age_seconds=worker["age_seconds"],
        worker_activity_max_age_seconds=worker["max_age_seconds"],
        queue_status=queue["status"],
        queue_depth_band=queue["depth_band"],
        queue_oldest_age_seconds=queue["oldest_age_seconds"],
        queue_max_oldest_age_seconds=queue["max_oldest_age_seconds"],
        retry_wait_status=queue["retry_wait"],
        quota_status=quota["status"],
        quota_remaining_band=quota["remaining_band"],
        quota_age_seconds=quota["age_seconds"],
        quota_max_age_seconds=quota["max_age_seconds"],
        dead_letter_status=dead_letter["status"],
        dead_letter_count_band=dead_letter["count_band"],
        alert_codes=tuple(code for code in _ALERT_ORDER if code in alerts),
        _evaluation_token=_EVALUATED_HEALTH_TOKEN,
    )


def _quote_status(
    signals: IngestionHealthFacts,
    policy: IngestionHealthPolicy,
    now: datetime,
    alerts: set[str],
) -> dict[str, Any]:
    age_state, age = _observation_age(signals.latest_quote_received_at, now)
    payload = {
        "status": "unavailable",
        "age_seconds": None,
        "max_age_seconds": policy.quote_max_age_seconds,
    }
    if age_state == "missing":
        alerts.add("feed_unavailable")
        return payload
    if age_state == "invalid":
        payload["status"] = "invalid"
        alerts.add("monitoring_invalid")
        return payload
    payload["age_seconds"] = _rounded_age(age)
    if age is not None and age <= policy.quote_max_age_seconds:
        payload["status"] = "fresh"
    else:
        payload["status"] = "stale"
        alerts.add("feed_stale")
    return payload


def _queue_status(
    signals: IngestionHealthFacts,
    policy: IngestionHealthPolicy,
    now: datetime,
    alerts: set[str],
) -> dict[str, Any]:
    payload = {
        "status": "unavailable",
        "depth_band": "unavailable",
        "oldest_age_seconds": None,
        "max_oldest_age_seconds": policy.queue_oldest_max_age_seconds,
        "retry_wait": "unavailable",
    }
    backlog = signals.backlog_count
    retry_wait = signals.retry_wait_count
    if backlog is None and retry_wait is None and signals.oldest_outstanding_at is None:
        alerts.add("queue_unavailable")
        return payload
    if not _is_nonnegative_integer(backlog) or not _is_nonnegative_integer(retry_wait):
        payload["status"] = "invalid"
        payload["depth_band"] = "invalid"
        payload["retry_wait"] = "invalid"
        alerts.add("monitoring_invalid")
        return payload

    payload["depth_band"] = (
        "empty"
        if backlog == 0
        else "normal"
        if backlog <= policy.queue_depth_warning
        else "high"
    )
    payload["retry_wait"] = "clear" if retry_wait == 0 else "present"
    if retry_wait > 0:
        alerts.add("retry_wait_present")

    outstanding = backlog + retry_wait
    if outstanding == 0:
        if signals.oldest_outstanding_at is not None:
            payload["status"] = "invalid"
            payload["depth_band"] = "invalid"
            payload["retry_wait"] = "invalid"
            alerts.add("monitoring_invalid")
            return payload
        payload["status"] = "clear"
        return payload

    age_state, age = _observation_age(signals.oldest_outstanding_at, now)
    if age_state == "missing":
        alerts.add("queue_unavailable")
        return payload
    if age_state == "invalid":
        payload["status"] = "invalid"
        alerts.add("monitoring_invalid")
        return payload
    payload["oldest_age_seconds"] = _rounded_age(age)
    if age is not None and age > policy.queue_oldest_max_age_seconds:
        payload["status"] = "stalled"
        alerts.add("queue_stalled")
    elif backlog > policy.queue_depth_warning:
        payload["status"] = "backlogged"
        alerts.add("queue_backlogged")
    else:
        payload["status"] = "clear"
    return payload


def _worker_activity_status(
    signals: IngestionHealthFacts,
    policy: IngestionHealthPolicy,
    now: datetime,
    queue: dict[str, Any],
    alerts: set[str],
) -> dict[str, Any]:
    payload = {
        "status": "unavailable",
        "age_seconds": None,
        "max_age_seconds": policy.worker_activity_max_age_seconds,
        "basis": "durable_activity_only",
    }
    age_state, age = _observation_age(signals.latest_worker_activity_at, now)
    if age_state == "invalid":
        payload["status"] = "invalid"
        alerts.add("monitoring_invalid")
        return payload
    if age_state == "missing":
        if _known_no_outstanding_work(signals):
            payload["status"] = "idle_unverified"
        elif queue["status"] == "stalled":
            payload["status"] = "stalled"
            alerts.add("worker_stalled")
        else:
            alerts.add("worker_activity_unavailable")
        return payload

    payload["age_seconds"] = _rounded_age(age)
    if age is not None and age <= policy.worker_activity_max_age_seconds:
        payload["status"] = "active"
        return payload

    alerts.add("worker_activity_stale")
    if _known_outstanding_work(signals):
        payload["status"] = "stalled"
        alerts.add("worker_stalled")
    elif _known_no_outstanding_work(signals):
        payload["status"] = "idle_unverified"
    else:
        alerts.add("worker_activity_unavailable")
    return payload


def _quota_status(
    signals: IngestionHealthFacts,
    policy: IngestionHealthPolicy,
    now: datetime,
    alerts: set[str],
) -> dict[str, Any]:
    payload = {
        "status": "unavailable",
        "remaining_band": "unavailable",
        "age_seconds": None,
        "max_age_seconds": policy.quota_max_age_seconds,
    }
    values = (
        signals.quota_remaining,
        signals.quota_reserved,
        signals.quota_observed_at,
    )
    if all(value is None for value in values):
        alerts.add("quota_unavailable")
        return payload
    if (
        not _is_nonnegative_integer(signals.quota_remaining)
        or not _is_nonnegative_integer(signals.quota_reserved)
        or signals.quota_reserved > signals.quota_remaining
    ):
        payload["status"] = "invalid"
        payload["remaining_band"] = "invalid"
        alerts.add("monitoring_invalid")
        return payload

    age_state, age = _observation_age(signals.quota_observed_at, now)
    if age_state != "valid":
        payload["status"] = "invalid"
        payload["remaining_band"] = "invalid"
        alerts.add("monitoring_invalid")
        return payload
    payload["age_seconds"] = _rounded_age(age)
    if age is not None and age > policy.quota_max_age_seconds:
        payload["status"] = "stale"
        payload["remaining_band"] = "unknown"
        alerts.add("quota_stale")
        return payload

    effective_remaining = signals.quota_remaining - signals.quota_reserved
    if effective_remaining == 0:
        payload["status"] = "exhausted"
        payload["remaining_band"] = "zero"
        alerts.add("quota_exhausted")
    elif effective_remaining <= policy.quota_low_watermark:
        payload["status"] = "low"
        payload["remaining_band"] = "low"
        alerts.add("quota_low")
    else:
        payload["status"] = "healthy"
        payload["remaining_band"] = "adequate"
    return payload


def _dead_letter_status(
    signals: IngestionHealthFacts,
    alerts: set[str],
) -> dict[str, str]:
    value = signals.dead_letter_count
    if value is None:
        alerts.add("dead_letter_unavailable")
        return {"status": "unavailable", "count_band": "unavailable"}
    if not _is_nonnegative_integer(value):
        alerts.add("monitoring_invalid")
        return {"status": "invalid", "count_band": "invalid"}
    if value > 0:
        alerts.add("dead_letter_present")
        return {"status": "present", "count_band": "one_or_more"}
    return {"status": "clear", "count_band": "zero"}


def _has_blocking_component(
    quote: dict[str, Any],
    worker: dict[str, Any],
    queue: dict[str, Any],
    quota: dict[str, Any],
    dead_letter: dict[str, Any],
) -> bool:
    return (
        quote["status"] in {"unavailable", "invalid", "stale"}
        or worker["status"] in {"unavailable", "invalid", "stalled"}
        or queue["status"] in {"unavailable", "invalid", "stalled"}
        or quota["status"] in {"unavailable", "invalid", "stale", "exhausted"}
        or dead_letter["status"] in {"unavailable", "invalid", "present"}
    )


def _health_valid_until(
    *,
    now: datetime,
    quote: dict[str, Any],
    worker: dict[str, Any],
    queue: dict[str, Any],
    quota: dict[str, Any],
) -> datetime:
    """Bound reuse by both the snapshot TTL and the next known age boundary."""

    remaining_seconds = [float(INGESTION_HEALTH_MAX_REUSE_SECONDS)]
    age_limits = (
        (quote, "age_seconds", "max_age_seconds"),
        (worker, "age_seconds", "max_age_seconds"),
        (queue, "oldest_age_seconds", "max_oldest_age_seconds"),
        (quota, "age_seconds", "max_age_seconds"),
    )
    for component, age_key, maximum_key in age_limits:
        age = component.get(age_key)
        maximum = component.get(maximum_key)
        if age is not None and maximum is not None and age <= maximum:
            remaining_seconds.append(float(maximum) - float(age))
    return now + timedelta(seconds=max(0.0, min(remaining_seconds)))


def _known_outstanding_work(signals: IngestionHealthFacts) -> bool:
    return (
        _is_nonnegative_integer(signals.backlog_count)
        and _is_nonnegative_integer(signals.retry_wait_count)
        and signals.backlog_count + signals.retry_wait_count > 0
    )


def _known_no_outstanding_work(signals: IngestionHealthFacts) -> bool:
    return (
        _is_nonnegative_integer(signals.backlog_count)
        and _is_nonnegative_integer(signals.retry_wait_count)
        and signals.backlog_count == 0
        and signals.retry_wait_count == 0
    )


def _observation_age(value: object, now: datetime) -> tuple[str, float | None]:
    if value is None:
        return "missing", None
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        return "invalid", None
    try:
        observed_at = value.astimezone(UTC)
    except (OverflowError, ValueError):
        return "invalid", None
    if observed_at > now:
        return "invalid", None
    return "valid", (now - observed_at).total_seconds()


def _rounded_age(value: float | None) -> float | None:
    # Datetime observations have microsecond precision. Preserve it so a value
    # just beyond a threshold cannot be rendered as if it were exactly on it.
    return round(value, 6) if value is not None else None


def _all_signals_missing(signals: IngestionHealthFacts) -> bool:
    return all(getattr(signals, field.name) is None for field in fields(signals))


def _safe_provider(value: object) -> str | None:
    if isinstance(value, str) and _SAFE_PROVIDER_RE.fullmatch(value) is not None:
        return value
    return None


def _require_aware_datetime(value: object, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    try:
        return value.astimezone(UTC)
    except (OverflowError, ValueError):
        raise ValueError(f"{label} must be a valid timezone-aware timestamp") from None


def _is_nonnegative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _require_nonnegative_integer(value: object, label: str) -> None:
    if not _is_nonnegative_integer(value):
        raise ValueError(f"{label} must be a non-negative integer")


def _require_positive_integer(value: object, label: str) -> None:
    if not _is_nonnegative_integer(value) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")


def _require_member(value: object, allowed: set[str]) -> None:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError("ingestion health contains an unsupported public status")


def _require_optional_nonnegative_number(value: object) -> None:
    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ValueError("ingestion health age must be a finite non-negative number")
