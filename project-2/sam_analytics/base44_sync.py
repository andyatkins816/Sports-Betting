"""One-way, secrets-safe evidence publishing to the Base44 control plane.

Base44 is deliberately an evidence presentation surface for this backend. It
may receive freshness, governance, calibration, and incident facts after a
trusted Python worker has created them; it never receives provider credentials,
raw payloads, feature vectors, or a synthetic model prediction.

The module uses only the standard library so a queue worker can publish an
operational fact without adding a second HTTP client or placing a token in a
browser-facing application.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import hmac
import json
import math
import os
import re
from numbers import Real
from typing import Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class EvidencePublishError(RuntimeError):
    """Raised for a local validation or remote evidence-publication failure."""


EVIDENCE_TYPES = frozenset(
    {
        "data_freshness",
        "model_release",
        "backtest",
        "calibration",
        "risk_gate",
        "provider_notice",
        "incident",
    }
)
SEVERITIES = frozenset({"info", "warning", "critical"})
SPORTS = frozenset({"MLB", "NHL", "NBA", "NFL", "ALL"})
DECISION_STATES = frozenset({"hold", "published", "retired", "investigating"})
_METRIC_KEY = re.compile(r"^[A-Za-z0-9_.-]{1,80}$")


class _NoRedirect(HTTPRedirectHandler):
    """Reject redirects so a bearer token cannot travel to another host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


@dataclass(frozen=True)
class EvidenceRecord:
    """The restricted fact shape accepted by Base44's ingest function."""

    evidence_type: str
    severity: str
    observed_at: datetime
    source: str
    provenance_id: str
    decision_state: str
    summary: str
    sport: str = "ALL"
    source_reference: str | None = None
    model_version: str | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)
    expires_at: datetime | None = None

    def to_payload(self) -> dict[str, object]:
        """Validate and serialize without returning or logging a credential."""

        _choice("evidence_type", self.evidence_type, EVIDENCE_TYPES)
        _choice("severity", self.severity, SEVERITIES)
        _choice("sport", self.sport, SPORTS)
        _choice("decision_state", self.decision_state, DECISION_STATES)
        observed_at = _timestamp("observed_at", self.observed_at)
        expires_at = _timestamp("expires_at", self.expires_at) if self.expires_at else None
        if self.expires_at and self.expires_at <= self.observed_at:
            raise EvidencePublishError("expires_at must be later than observed_at")

        payload: dict[str, object] = {
            "evidence_type": self.evidence_type,
            "severity": self.severity,
            "sport": self.sport,
            "observed_at": observed_at,
            "source": _required_text("source", self.source, 120),
            "provenance_id": _required_text("provenance_id", self.provenance_id, 180),
            "decision_state": self.decision_state,
            "summary": _required_text("summary", self.summary, 2_000),
            "metrics": _metrics(self.metrics),
        }
        if self.source_reference is not None:
            payload["source_reference"] = _required_text(
                "source_reference", self.source_reference, 500
            )
        if self.model_version is not None:
            payload["model_version"] = _required_text("model_version", self.model_version, 128)
        if expires_at is not None:
            payload["expires_at"] = expires_at
        return payload


@dataclass(frozen=True)
class EvidencePublishReceipt:
    """A minimal acknowledgement that does not expose Base44 data."""

    status_code: int


@dataclass(frozen=True)
class Base44EvidencePublisher:
    """Publish validated evidence over HTTPS using a server-side bearer token."""

    webhook_url: str
    token: str
    timeout_seconds: float = 5.0
    expected_host: str | None = None

    def __post_init__(self) -> None:
        _validate_webhook_url(self.webhook_url, self.expected_host)
        if not isinstance(self.token, str) or len(self.token.strip()) < 32:
            raise EvidencePublishError("Base44 evidence token must be at least 32 characters")
        if not isinstance(self.timeout_seconds, Real) or isinstance(self.timeout_seconds, bool):
            raise EvidencePublishError("evidence timeout must be numeric")
        if not 0 < float(self.timeout_seconds) <= 30:
            raise EvidencePublishError("evidence timeout must be between 0 and 30 seconds")

    @classmethod
    def from_environment(cls) -> "Base44EvidencePublisher | None":
        """Return no publisher when deliberately unconfigured; reject half-configurations."""

        webhook_url = os.getenv("BASE44_EVIDENCE_WEBHOOK_URL", "").strip()
        token = os.getenv("BASE44_EVIDENCE_WEBHOOK_TOKEN", "").strip()
        expected_host = os.getenv("BASE44_EVIDENCE_WEBHOOK_HOST", "").strip()
        if not webhook_url and not token:
            return None
        if not webhook_url or not token:
            raise EvidencePublishError(
                "BASE44_EVIDENCE_WEBHOOK_URL and BASE44_EVIDENCE_WEBHOOK_TOKEN must be set together"
            )
        if not expected_host:
            raise EvidencePublishError(
                "BASE44_EVIDENCE_WEBHOOK_HOST must pin the expected Base44 webhook host"
            )
        return cls(webhook_url=webhook_url, token=token, expected_host=expected_host)

    def publish(self, record: EvidenceRecord) -> EvidencePublishReceipt:
        """Send one evidence fact and fail without divulging payloads or tokens."""

        if not isinstance(record, EvidenceRecord):
            raise EvidencePublishError("record must be an EvidenceRecord")
        body = json.dumps(
            record.to_payload(), allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        timestamp = _timestamp("published_at", datetime.now(timezone.utc))
        signature = hmac.new(
            self.token.encode("utf-8"),
            timestamp.encode("utf-8") + b"." + body,
            hashlib.sha256,
        ).hexdigest()
        request = Request(
            self.webhook_url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-SAM-Evidence-Timestamp": timestamp,
                "X-SAM-Evidence-Signature": "sha256=" + signature,
            },
            method="POST",
        )
        opener = build_opener(_NoRedirect())
        try:
            response = opener.open(request, timeout=float(self.timeout_seconds))
            try:
                status_code = int(response.getcode())
            finally:
                response.close()
        except HTTPError as error:
            # Do not read an error body: a proxy can echo request data or reveal
            # a provider implementation detail.
            raise EvidencePublishError(
                f"Base44 evidence endpoint rejected the request (HTTP {error.code})"
            ) from error
        except (URLError, OSError, TimeoutError) as error:
            raise EvidencePublishError("Base44 evidence endpoint is unavailable") from error

        if not 200 <= status_code < 300:
            raise EvidencePublishError(
                f"Base44 evidence endpoint returned unexpected HTTP status {status_code}"
            )
        return EvidencePublishReceipt(status_code=status_code)


def _validate_webhook_url(value: str, expected_host: str | None) -> None:
    if not isinstance(value, str) or not value.strip():
        raise EvidencePublishError("Base44 evidence webhook URL is required")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise EvidencePublishError(
            "Base44 evidence webhook URL must be an HTTPS URL without credentials, query, or fragment"
        )
    if expected_host is not None:
        normalized_host = expected_host.strip().lower()
        if not normalized_host or parsed.hostname is None or parsed.hostname.lower() != normalized_host:
            raise EvidencePublishError("Base44 evidence webhook URL host does not match the configured pin")


def _choice(label: str, value: object, allowed: frozenset[str]) -> None:
    if not isinstance(value, str) or value not in allowed:
        raise EvidencePublishError(f"{label} is invalid")


def _required_text(label: str, value: object, maximum: int) -> str:
    if not isinstance(value, str):
        raise EvidencePublishError(f"{label} must be text")
    normalized = value.strip()
    if not normalized or len(normalized) > maximum:
        raise EvidencePublishError(f"{label} must contain 1 to {maximum} characters")
    return normalized


def _timestamp(label: str, value: object) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise EvidencePublishError(f"{label} must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _metrics(values: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(values, Mapping):
        raise EvidencePublishError("metrics must be a mapping")
    if len(values) > 30:
        raise EvidencePublishError("metrics may contain at most 30 numeric values")

    cleaned: dict[str, float] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not _METRIC_KEY.fullmatch(key):
            raise EvidencePublishError("metric names contain unsupported characters")
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
            raise EvidencePublishError("metrics must contain finite numeric values")
        cleaned[key] = float(value)
    return cleaned
