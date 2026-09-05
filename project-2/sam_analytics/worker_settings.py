"""Fail-closed admission settings for the synthetic staging worker.

This module is intentionally narrower than the public web-process settings.
The first private-worker release may verify deterministic synthetic bytes in a
dedicated staging R2 bucket, but it may not receive provider, web, Base44, or
ambient cloud credentials.  Database, broker, and R2 credential values are
validated for presence and shape without being retained on the returned
settings object, so its representation remains safe for ordinary diagnostics.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit


SYNTHETIC_STAGING_EVIDENCE_URI = (
    "s3://sam-raw-evidence-staging/raw/synthetic"
)
SYNTHETIC_STAGING_MAX_BYTES = 1024 * 1024

_STAGING_ENVIRONMENT = "staging"
_PRIVATE_WORKER_ROLE = "private_ingestion"
_SYNTHETIC_WORKER_MODE = "synthetic_storage_probe"
_R2_BACKEND = "cloudflare_r2"
_R2_REGION = "auto"
_R2_ENDPOINT_HOST_RE = re.compile(
    r"^[a-f0-9]{32}(?:\.(?:eu|us|fedramp))?\.r2\.cloudflarestorage\.com$"
)
_RENDER_POSTGRES_HOST_RE = re.compile(r"^dpg-[a-z0-9][a-z0-9-]{1,62}$")
_RENDER_KEY_VALUE_HOST_RE = re.compile(r"^red-[a-z0-9][a-z0-9-]{1,62}$")
_LOCAL_SERVICE_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

_R2_CREDENTIAL_NAMES = (
    "SAM_RAW_EVIDENCE_S3_ACCESS_KEY_ID",
    "SAM_RAW_EVIDENCE_S3_SECRET_ACCESS_KEY",
)

# Defense in depth for environment groups: the worker may receive only the two
# explicitly named R2 credentials above.  A newly introduced provider or admin
# secret must fail admission even before this exact-name list is updated.
_SENSITIVE_SETTING_NAME_RE = re.compile(
    r"api_?key|(?:^|_)(?:token|secret|password|authorization|credentials?|cookie|bearer)(?:_|$)",
    re.IGNORECASE,
)

# A synthetic storage probe has no lawful reason to receive any of these.
# Rejecting the settings at admission also catches accidental reuse of the
# public API's Render environment group.
_FORBIDDEN_SETTING_NAMES = (
    "ODDS_PROVIDER",
    "ODDS_PROVIDER_API_KEY",
    "NFL_API_KEY",
    "OPENAI_API_KEY",
    "RESULTS_PROVIDER",
    "RESULTS_PROVIDER_API_KEY",
    "SESSION_SECRET",
    "SAM_API_KEY",
    "SAM_STATUS_API_KEY",
    "ALLOWED_ORIGINS",
    "BASE44_EVIDENCE_WEBHOOK_URL",
    "BASE44_EVIDENCE_WEBHOOK_TOKEN",
    "BASE44_EVIDENCE_WEBHOOK_HOST",
    # The adapter must use only its explicit, staging-bucket-scoped R2 pair.
    # Ambient SDK or account-admin credentials create a second, unreviewed
    # authority.
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_PROFILE",
    "AWS_DEFAULT_PROFILE",
    "AWS_CONFIG_FILE",
    "AWS_SHARED_CREDENTIALS_FILE",
    "AWS_ROLE_ARN",
    "AWS_ROLE_SESSION_NAME",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
    "AWS_CONTAINER_AUTHORIZATION_TOKEN",
    "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE",
    "CLOUDFLARE_API_TOKEN",
    "CLOUDFLARE_API_KEY",
    "CLOUDFLARE_API_USER_SERVICE_KEY",
    "CLOUDFLARE_EMAIL",
    "CF_API_TOKEN",
    "CF_API_KEY",
    "CF_API_USER_SERVICE_KEY",
    "CF_API_EMAIL",
)


class PrivateWorkerConfigurationError(RuntimeError):
    """Raised with configuration-category text, never a supplied value."""


@dataclass(frozen=True)
class PrivateWorkerSettings:
    """Credential-free description of the only currently admitted worker.

    Connection strings and access keys are deliberately absent.  Their shape
    and presence are checked by :meth:`from_environment`, while the concrete
    database, Celery, and object-store clients remain responsible for reading
    their own values at the point of use.
    """

    environment: str
    role: str
    mode: str
    raw_evidence_store_backend: str
    raw_evidence_store_uri: str
    raw_evidence_s3_region: str
    raw_evidence_s3_endpoint_url: str
    raw_evidence_max_bytes: int

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> "PrivateWorkerSettings":
        """Validate the exact synthetic staging boundary without doing I/O."""

        values = os.environ if environ is None else environ
        _reject_forbidden_settings(values)
        _reject_live_ingestion_switch(values)
        _require_service_url(
            values,
            name="DATABASE_URL",
            allowed_schemes=frozenset({"postgres", "postgresql"}),
        )
        _require_service_url(
            values,
            name="REDIS_URL",
            allowed_schemes=frozenset({"redis", "rediss"}),
        )
        _require_r2_credentials(values)

        environment = _required_exact(
            values, "APP_ENV", _STAGING_ENVIRONMENT
        )
        role = _required_exact(
            values, "SAM_WORKER_ROLE", _PRIVATE_WORKER_ROLE
        )
        mode = _required_exact(
            values, "SAM_WORKER_MODE", _SYNTHETIC_WORKER_MODE
        )
        backend = _required_exact(
            values, "SAM_RAW_EVIDENCE_STORE_BACKEND", _R2_BACKEND
        )
        store_uri = _required_exact(
            values,
            "SAM_RAW_EVIDENCE_STORE_URI",
            SYNTHETIC_STAGING_EVIDENCE_URI,
        )
        region = _required_exact(
            values, "SAM_RAW_EVIDENCE_S3_REGION", _R2_REGION
        )
        endpoint_url = _required_text(
            values, "SAM_RAW_EVIDENCE_S3_ENDPOINT_URL"
        )
        _validate_r2_endpoint(endpoint_url)
        _required_exact(
            values,
            "SAM_RAW_EVIDENCE_MAX_BYTES",
            str(SYNTHETIC_STAGING_MAX_BYTES),
        )
        max_payload_bytes = SYNTHETIC_STAGING_MAX_BYTES

        return cls(
            environment=environment,
            role=role,
            mode=mode,
            raw_evidence_store_backend=backend,
            raw_evidence_store_uri=store_uri,
            raw_evidence_s3_region=region,
            raw_evidence_s3_endpoint_url=endpoint_url,
            raw_evidence_max_bytes=max_payload_bytes,
        )


def _reject_forbidden_settings(values: Mapping[str, str]) -> None:
    for name, value in values.items():
        if not isinstance(name, str) or not _is_configured_value(value):
            continue
        if name in _R2_CREDENTIAL_NAMES:
            continue
        if (
            name.upper().startswith("PG")
            or name.upper().startswith("CELERY_")
            or name in _FORBIDDEN_SETTING_NAMES
            or _SENSITIVE_SETTING_NAME_RE.search(name)
        ):
            raise PrivateWorkerConfigurationError(
                "provider, public-service, Base44, or ambient connection settings "
                "cannot be configured in the synthetic staging worker"
            )


def _reject_live_ingestion_switch(values: Mapping[str, str]) -> None:
    value = values.get("SAM_INGESTION_ENABLED")
    if value is None:
        return
    if not isinstance(value, str) or value != "false":
        raise PrivateWorkerConfigurationError(
            "live ingestion cannot be enabled in the synthetic staging worker"
        )


def _require_r2_credentials(values: Mapping[str, str]) -> None:
    for name in _R2_CREDENTIAL_NAMES:
        value = values.get(name)
        if (
            not isinstance(value, str)
            or not value
            or len(value) > 1024
            or value != value.strip()
            or _contains_control_character(value)
        ):
            raise PrivateWorkerConfigurationError(
                "the scoped R2 credential pair is missing or invalid"
            )


def _require_service_url(
    values: Mapping[str, str],
    *,
    name: str,
    allowed_schemes: frozenset[str],
) -> None:
    value = values.get(name)
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 2048
        or value != value.strip()
        or _contains_control_character(value)
    ):
        raise PrivateWorkerConfigurationError(
            "private database and broker URLs are required"
        )
    parsed = None
    hostname = None
    port = None
    parse_failed = False
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except (TypeError, ValueError):
        parse_failed = True
    if parse_failed or parsed is None:
        raise PrivateWorkerConfigurationError(
            "private database or broker URL is invalid"
        )
    if (
        parsed.scheme not in allowed_schemes
        or hostname is None
        or not _is_admitted_private_service_host(name, hostname)
        or port is not None and not 1 <= port <= 65535
        or parsed.query
        or parsed.fragment
    ):
        raise PrivateWorkerConfigurationError(
            "private database or broker URL is invalid"
        )


def _is_admitted_private_service_host(name: str, hostname: str) -> bool:
    """Admit Render's private host forms plus explicit local test endpoints."""

    normalized = hostname.lower()
    if normalized in _LOCAL_SERVICE_HOSTS:
        return True
    if name == "DATABASE_URL":
        return normalized == "postgres" or bool(
            _RENDER_POSTGRES_HOST_RE.fullmatch(normalized)
        )
    if name == "REDIS_URL":
        return normalized == "redis" or bool(
            _RENDER_KEY_VALUE_HOST_RE.fullmatch(normalized)
        )
    return False


def _required_exact(
    values: Mapping[str, str], name: str, expected: str
) -> str:
    value = values.get(name)
    if not isinstance(value, str) or value != expected:
        raise PrivateWorkerConfigurationError(
            "synthetic staging worker configuration is not admitted"
        )
    return value


def _required_text(values: Mapping[str, str], name: str) -> str:
    value = values.get(name)
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or _contains_control_character(value)
    ):
        raise PrivateWorkerConfigurationError(
            "synthetic staging worker configuration is invalid"
        )
    return value


def _validate_r2_endpoint(value: str) -> None:
    parsed = None
    hostname = None
    port = None
    parse_failed = False
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except (TypeError, ValueError):
        parse_failed = True
    if parse_failed or parsed is None:
        raise PrivateWorkerConfigurationError(
            "Cloudflare R2 endpoint is invalid"
        )
    if (
        parsed.scheme != "https"
        or hostname is None
        or not _R2_ENDPOINT_HOST_RE.fullmatch(hostname.lower())
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
        or "%" in value
    ):
        raise PrivateWorkerConfigurationError(
            "Cloudflare R2 endpoint is invalid"
        )


def _is_configured_value(value: object) -> bool:
    if value is None:
        return False
    if not isinstance(value, str):
        return True
    return bool(value.strip())


def _contains_control_character(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)
