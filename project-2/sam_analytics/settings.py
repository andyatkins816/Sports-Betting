"""Minimal, fail-closed configuration for the web process."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    environment: str
    secret_key: str
    api_key: str | None
    database_url: str | None
    redis_url: str | None
    allowed_origins: tuple[str, ...]
    quote_max_age_seconds: int
    approved_model_versions: tuple[str, ...] = ()
    status_api_key: str | None = None

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @classmethod
    def from_environment(cls) -> "Settings":
        environment = os.getenv("APP_ENV", "development").lower()
        if environment not in {"development", "test", "staging", "production"}:
            raise ValueError("APP_ENV must be development, test, staging, or production")
        secret_key = os.getenv("SESSION_SECRET", "")
        api_key = os.getenv("SAM_API_KEY") or None
        status_api_key = os.getenv("SAM_STATUS_API_KEY") or None
        database_url = os.getenv("DATABASE_URL") or None
        redis_url = os.getenv("REDIS_URL") or None
        origins = tuple(origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "").split(",") if origin.strip())
        approved_versions = tuple(
            version.strip()
            for version in os.getenv("SAM_APPROVED_MODEL_VERSIONS", "").split(",")
            if version.strip()
        )
        settings = cls(
            environment=environment,
            secret_key=secret_key,
            api_key=api_key,
            database_url=database_url,
            redis_url=redis_url,
            allowed_origins=origins,
            quote_max_age_seconds=int(os.getenv("QUOTE_MAX_AGE_SECONDS", "300")),
            approved_model_versions=approved_versions,
            status_api_key=status_api_key,
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.quote_max_age_seconds <= 0:
            raise ValueError("QUOTE_MAX_AGE_SECONDS must be positive")
        if len(self.approved_model_versions) != len(set(self.approved_model_versions)):
            raise ValueError("SAM_APPROVED_MODEL_VERSIONS cannot contain duplicates")
        if self.status_api_key and self.api_key and self.status_api_key == self.api_key:
            raise ValueError("SAM_STATUS_API_KEY must be distinct from SAM_API_KEY")
        if self.is_production:
            if len(self.secret_key) < 32:
                raise ValueError("SESSION_SECRET must be at least 32 characters in production")
            if not self.api_key:
                raise ValueError("SAM_API_KEY is required in production")
            if not self.database_url or not self.redis_url:
                raise ValueError("DATABASE_URL and REDIS_URL are required in production")
