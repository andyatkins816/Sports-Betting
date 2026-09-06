"""Fail-closed settings for the bounded The Odds API shadow worker.

This module defines configuration admission only.  Importing it does not create
a provider client, contact a service, register a Celery task, or start a
scheduler.  The boundary is deliberately separate from the synthetic evidence
probe so that relaxing one worker cannot silently grant authority to the other.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit


PROVIDER_SHADOW_EVIDENCE_URI = "s3://sam-raw-evidence-staging/raw/the_odds_api"
PROVIDER_SHADOW_MAX_BYTES = 10 * 1024 * 1024
PROVIDER_SHADOW_ALLOWED_SPORT_KEYS = frozenset(
    {"americanfootball_nfl", "baseball_mlb", "basketball_nba"}
)
PROVIDER_SHADOW_REGIONS = "us"
PROVIDER_SHADOW_MARKETS = "h2h"
PROVIDER_SHADOW_LICENSE_SCOPE = "internal_analytics_only"
PROVIDER_SHADOW_LICENSE_VERSION = "terms-2026-08-31"

_STAGING_ENVIRONMENT = "staging"
_PRIVATE_WORKER_ROLE = "private_ingestion"
_PROVIDER_SHADOW_MODE = "provider_shadow"
_ODDS_PROVIDER = "the_odds_api"
_R2_BACKEND = "cloudflare_r2"
_R2_REGION = "auto"
_R2_ENDPOINT_HOST_RE = re.compile(
    r"^[a-f0-9]{32}(?:\.(?:eu|us|fedramp))?\.r2\.cloudflarestorage\.com$"
)
_RENDER_POSTGRES_HOST_RE = re.compile(r"^dpg-[a-z0-9][a-z0-9-]{1,62}$")
_RENDER_KEY_VALUE_HOST_RE = re.compile(r"^red-[a-z0-9][a-z0-9-]{1,62}$")

_ADMITTED_SECRET_NAMES = frozenset(
    {
        "ODDS_PROVIDER_API_KEY",
        "SAM_RAW_EVIDENCE_S3_ACCESS_KEY_ID",
        "SAM_RAW_EVIDENCE_S3_SECRET_ACCESS_KEY",
    }
)

# Celery creates these exact values while the reviewed worker command starts.
# Any other Celery environment configuration remains outside this boundary.
_CELERY_RUNTIME_MARKERS = frozenset(
    {
        ("CELERY_LOG_LEVEL", "20"),
        ("CELERY_LOG_REDIRECT", "1"),
        ("CELERY_LOG_REDIRECT_LEVEL", "WARNING"),
        ("celery_dummy_proxy", "set_by_celeryd"),
    }
)

_SENSITIVE_SETTING_NAME_RE = re.compile(
    r"api_?key|access_?key(?:_?id)?|private_?key|"
    r"(?:^|_)(?:auth|token|secret|password|authorization|credentials?|cookie|"
    r"bearer|pat)(?:_|$)",
    re.IGNORECASE,
)
_AUTOMATION_SETTING_NAME_RE = re.compile(
    r"(?:^|_)(?:scheduler|schedule|cron|beat)(?:_|$)", re.IGNORECASE
)
_SEPARATE_AUTHORITY_SETTING_NAME_RE = re.compile(
    r"(?:^|_)(?:admin|public|results?|session)(?:_|$)", re.IGNORECASE
)

_FORBIDDEN_EXACT_NAMES = frozenset(
    {
        # Public web-process authority.
        "SESSION_SECRET",
        "SAM_API_KEY",
        "SAM_STATUS_API_KEY",
        "ALLOWED_ORIGINS",
        # Results, settlement, and automatic execution are separate releases.
        "RESULTS_PROVIDER",
        "RESULTS_PROVIDER_API_KEY",
        "SAM_RESULTS_ENABLED",
        "SAM_SETTLEMENT_ENABLED",
        "SAM_SCHEDULER_ENABLED",
        "SAM_CRON_ENABLED",
        # Unrelated provider or administrative authority.
        "NFL_API_KEY",
        "OPENAI_API_KEY",
    }
)


class ProviderShadowConfigurationError(RuntimeError):
    """Raised with category-only text that is safe for deployment logs."""


@dataclass(frozen=True)
class ProviderShadowSettings:
    """Credential-free description of the admitted provider-shadow boundary."""

    environment: str
    role: str
    mode: str
    provider: str
    sport_key: str
    regions: tuple[str, ...]
    markets: tuple[str, ...]
    license_scope: str
    license_version: str
    raw_evidence_store_backend: str
    raw_evidence_store_uri: str
    raw_evidence_s3_region: str
    raw_evidence_s3_endpoint_url: str
    raw_evidence_max_bytes: int

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> ProviderShadowSettings:
        """Validate the exact staging-only provider-shadow configuration."""

        values = os.environ if environ is None else environ
        _reject_forbidden_settings(values)
        _require_service_url(values, name="DATABASE_URL")
        _require_service_url(values, name="REDIS_URL")
        _require_secrets(values)

        environment = _required_exact(values, "APP_ENV", _STAGING_ENVIRONMENT)
        role = _required_exact(values, "SAM_WORKER_ROLE", _PRIVATE_WORKER_ROLE)
        mode = _required_exact(values, "SAM_WORKER_MODE", _PROVIDER_SHADOW_MODE)
        _required_exact(values, "SAM_INGESTION_ENABLED", "true")
        provider = _required_exact(values, "ODDS_PROVIDER", _ODDS_PROVIDER)
        sport_key = _required_allowed(
            values,
            "SAM_ODDS_SPORT_KEY",
            PROVIDER_SHADOW_ALLOWED_SPORT_KEYS,
        )
        regions = _required_exact(values, "SAM_ODDS_REGIONS", PROVIDER_SHADOW_REGIONS)
        markets = _required_exact(values, "SAM_ODDS_MARKETS", PROVIDER_SHADOW_MARKETS)
        license_scope = _required_exact(
            values,
            "SAM_PROVIDER_LICENSE_SCOPE",
            PROVIDER_SHADOW_LICENSE_SCOPE,
        )
        license_version = _required_exact(
            values,
            "SAM_PROVIDER_LICENSE_VERSION",
            PROVIDER_SHADOW_LICENSE_VERSION,
        )
        backend = _required_exact(values, "SAM_RAW_EVIDENCE_STORE_BACKEND", _R2_BACKEND)
        store_uri = _required_exact(
            values, "SAM_RAW_EVIDENCE_STORE_URI", PROVIDER_SHADOW_EVIDENCE_URI
        )
        region = _required_exact(values, "SAM_RAW_EVIDENCE_S3_REGION", _R2_REGION)
        endpoint_url = _required_text(values, "SAM_RAW_EVIDENCE_S3_ENDPOINT_URL")
        _validate_r2_endpoint(endpoint_url)
        _required_exact(
            values,
            "SAM_RAW_EVIDENCE_MAX_BYTES",
            str(PROVIDER_SHADOW_MAX_BYTES),
        )

        return cls(
            environment=environment,
            role=role,
            mode=mode,
            provider=provider,
            sport_key=sport_key,
            regions=(regions,),
            markets=(markets,),
            license_scope=license_scope,
            license_version=license_version,
            raw_evidence_store_backend=backend,
            raw_evidence_store_uri=store_uri,
            raw_evidence_s3_region=region,
            raw_evidence_s3_endpoint_url=endpoint_url,
            raw_evidence_max_bytes=PROVIDER_SHADOW_MAX_BYTES,
        )


def _reject_forbidden_settings(values: Mapping[str, str]) -> None:
    for name, value in values.items():
        if not isinstance(name, str) or not _is_configured_value(value):
            continue
        if name in _ADMITTED_SECRET_NAMES or (name, value) in _CELERY_RUNTIME_MARKERS:
            continue
        upper_name = name.upper()
        if (
            upper_name.startswith("PG")
            or upper_name.startswith("CELERY_")
            or upper_name.startswith("CELERYBEAT_")
            or upper_name.startswith("BASE44_")
            or upper_name.startswith("RESULTS_")
            or upper_name.startswith("AWS_")
            or upper_name.startswith("CLOUDFLARE_")
            or upper_name.startswith("CF_")
            or upper_name.startswith("SCHEDULER_")
            or upper_name.startswith("CRON_")
            or upper_name in _FORBIDDEN_EXACT_NAMES
            or _AUTOMATION_SETTING_NAME_RE.search(name)
            or _SEPARATE_AUTHORITY_SETTING_NAME_RE.search(name)
            or _SENSITIVE_SETTING_NAME_RE.search(name)
        ):
            raise ProviderShadowConfigurationError(
                "public-service, Base44, results, scheduler, or administrative "
                "settings cannot be configured in the provider-shadow worker"
            )


def _require_secrets(values: Mapping[str, str]) -> None:
    for name in _ADMITTED_SECRET_NAMES:
        value = values.get(name)
        if (
            not isinstance(value, str)
            or not value
            or len(value) > 1024
            or value != value.strip()
            or _contains_control_character(value)
        ):
            raise ProviderShadowConfigurationError(
                "the provider-shadow credential set is missing or invalid"
            )


def _require_service_url(values: Mapping[str, str], *, name: str) -> None:
    value = values.get(name)
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 2048
        or value != value.strip()
        or _contains_control_character(value)
    ):
        raise ProviderShadowConfigurationError(
            "private Render database and broker URLs are required"
        )

    parsed = None
    parse_failed = False
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except (TypeError, ValueError):
        parse_failed = True
        hostname = None
        port = None
    expected_schemes = (
        frozenset({"postgres", "postgresql"})
        if name == "DATABASE_URL"
        else frozenset({"redis", "rediss"})
    )
    host_pattern = _RENDER_POSTGRES_HOST_RE if name == "DATABASE_URL" else _RENDER_KEY_VALUE_HOST_RE
    if (
        parse_failed
        or parsed is None
        or parsed.scheme not in expected_schemes
        or hostname is None
        or not host_pattern.fullmatch(hostname.lower())
        or (port is not None and not 1 <= port <= 65535)
        or parsed.query
        or parsed.fragment
    ):
        raise ProviderShadowConfigurationError("private Render database or broker URL is invalid")


def _required_exact(values: Mapping[str, str], name: str, expected: str) -> str:
    value = values.get(name)
    if not isinstance(value, str) or value != expected:
        raise ProviderShadowConfigurationError(
            "provider-shadow worker configuration is not admitted"
        )
    return value


def _required_allowed(values: Mapping[str, str], name: str, allowed: frozenset[str]) -> str:
    value = values.get(name)
    if not isinstance(value, str) or value not in allowed:
        raise ProviderShadowConfigurationError(
            "provider-shadow worker configuration is not admitted"
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
        raise ProviderShadowConfigurationError("provider-shadow worker configuration is invalid")
    return value


def _validate_r2_endpoint(value: str) -> None:
    parsed = None
    parse_failed = False
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except (TypeError, ValueError):
        parse_failed = True
        hostname = None
        port = None
    if (
        parse_failed
        or parsed is None
        or parsed.scheme != "https"
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
        raise ProviderShadowConfigurationError("Cloudflare R2 endpoint is invalid")


def _is_configured_value(value: object) -> bool:
    if value is None:
        return False
    if not isinstance(value, str):
        return True
    return bool(value.strip())


def _contains_control_character(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)
