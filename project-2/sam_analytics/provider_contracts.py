"""Fail-closed authorization contracts for licensed data providers.

Provider credentials only prove that a request can be made.  They do not prove
that a particular response may be retained, trained on, or shown to users.
This module binds an ingestion run to an explicitly approved provider, license
scope, license version, and source type.  Raw provider content remains private
evidence; only derived analytics may cross SAM's application boundary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Literal

from .raw_payload_store import RawPayloadMetadata, validate_raw_payload_metadata


_PROVIDER_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_SOURCE_TYPE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_LICENSE_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_EXPOSURES = frozenset({"private_raw", "derived"})


class ProviderContractViolation(ValueError):
    """Raised when a provider use is absent from the approved license policy."""


def _safe_provider(value: object) -> bool:
    return isinstance(value, str) and bool(_PROVIDER_RE.fullmatch(value))


def _safe_source_type(value: object) -> bool:
    return isinstance(value, str) and bool(_SOURCE_TYPE_RE.fullmatch(value))


def _safe_license_token(value: object) -> bool:
    return isinstance(value, str) and bool(_LICENSE_TOKEN_RE.fullmatch(value))


@dataclass(frozen=True)
class ProviderUse:
    """A specific authorized use of an approved provider response.

    ``private_raw`` is solely for immutable evidence storage and normalized
    internal processing.  ``derived`` means a non-reconstructable aggregate,
    status, model input, or other derived analytical result.  There is no
    public-raw mode by design.
    """

    provider: str
    license_scope: str
    license_version: str
    source_type: str
    exposure: Literal["private_raw", "derived"] = "private_raw"


def validate_provider_use(use: ProviderUse) -> None:
    """Validate syntax only; a registry performs the approval lookup."""

    if not isinstance(use, ProviderUse):
        raise ProviderContractViolation("provider use has an invalid type")
    if not _safe_provider(use.provider):
        raise ProviderContractViolation("provider must be a lowercase safe identifier")
    if not _safe_license_token(use.license_scope):
        raise ProviderContractViolation("license_scope must be a safe non-secret token")
    if not _safe_license_token(use.license_version):
        raise ProviderContractViolation("license_version must be a safe non-secret token")
    if not _safe_source_type(use.source_type):
        raise ProviderContractViolation("source_type must be a lowercase safe identifier")
    if use.exposure not in _EXPOSURES:
        raise ProviderContractViolation("provider use exposure must be private_raw or derived")


def validate_private_payload_metadata_for_use(
    metadata: RawPayloadMetadata, use: ProviderUse
) -> None:
    """Bind an evidence receipt to the exact private ingestion authorization.

    The caller must invoke this before writing a normalized row or provenance
    ledger entry.  A derived-output authorization cannot be reused to store raw
    provider bytes, and a valid receipt from a different license version is
    deliberately not interchangeable.
    """

    validate_raw_payload_metadata(metadata)
    validate_provider_use(use)
    if use.exposure != "private_raw":
        raise ProviderContractViolation("only private_raw provider use may store a raw payload")
    if (
        metadata.provider,
        metadata.license_scope,
        metadata.license_version,
        metadata.source_type,
    ) != (
        use.provider,
        use.license_scope,
        use.license_version,
        use.source_type,
    ):
        raise ProviderContractViolation("raw payload metadata does not match its approved provider use")


@dataclass(frozen=True)
class ApprovedProviderContract:
    """A locally reviewed license policy for exactly one provider agreement.

    A contract must remain derived-only.  It allows immutable raw payloads to
    be stored privately for auditability, but it never authorizes a raw odds
    feed, provider payload, bookmaker table, or other reconstructable provider
    material to be exposed from SAM.
    """

    provider: str
    license_scope: str
    license_version: str
    permitted_source_types: frozenset[str]
    derived_only: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.permitted_source_types, str):
            raise ProviderContractViolation("permitted_source_types must be a collection, not a string")
        try:
            source_types = frozenset(self.permitted_source_types)
        except TypeError as error:
            raise ProviderContractViolation("permitted_source_types must be an immutable-safe collection") from error
        object.__setattr__(self, "permitted_source_types", source_types)
        validate_approved_provider_contract(self)

    def authorize_ingestion(self, source_type: str) -> ProviderUse:
        """Authorize one internal raw-payload and normalization operation."""

        return self._authorize(source_type=source_type, exposure="private_raw")

    def authorize_derived_exposure(self, source_type: str) -> ProviderUse:
        """Authorize a derived-only analytical result for this source type."""

        return self._authorize(source_type=source_type, exposure="derived")

    def _authorize(
        self, *, source_type: str, exposure: Literal["private_raw", "derived"]
    ) -> ProviderUse:
        if not _safe_source_type(source_type) or source_type not in self.permitted_source_types:
            raise ProviderContractViolation("provider/source type is not approved for this license contract")
        use = ProviderUse(
            provider=self.provider,
            license_scope=self.license_scope,
            license_version=self.license_version,
            source_type=source_type,
            exposure=exposure,
        )
        validate_provider_use(use)
        return use


def validate_approved_provider_contract(contract: ApprovedProviderContract) -> None:
    """Fail closed on malformed or non-derived-only license policy."""

    if not isinstance(contract, ApprovedProviderContract):
        raise ProviderContractViolation("approved provider contract has an invalid type")
    if not _safe_provider(contract.provider):
        raise ProviderContractViolation("provider must be a lowercase safe identifier")
    if not _safe_license_token(contract.license_scope):
        raise ProviderContractViolation("license_scope must be a safe non-secret token")
    if not _safe_license_token(contract.license_version):
        raise ProviderContractViolation("license_version must be a safe non-secret token")
    if contract.derived_only is not True:
        raise ProviderContractViolation("SAM provider contracts must enforce derived-only exposure")
    if not isinstance(contract.permitted_source_types, frozenset) or not contract.permitted_source_types:
        raise ProviderContractViolation("permitted_source_types must be a non-empty immutable collection")
    if not all(_safe_source_type(source_type) for source_type in contract.permitted_source_types):
        raise ProviderContractViolation("permitted_source_types contains an unsafe source type")


class ProviderContractRegistry:
    """Immutable lookup of all local provider contracts approved for SAM."""

    def __init__(self, contracts: Iterable[ApprovedProviderContract]) -> None:
        try:
            entries = tuple(contracts)
        except TypeError as error:
            raise ProviderContractViolation("contracts must be an iterable of approved contracts") from error
        if not entries:
            raise ProviderContractViolation("at least one approved provider contract is required")

        by_key: dict[tuple[str, str, str], ApprovedProviderContract] = {}
        for contract in entries:
            validate_approved_provider_contract(contract)
            key = (contract.provider, contract.license_scope, contract.license_version)
            if key in by_key:
                raise ProviderContractViolation("duplicate approved provider/license contract")
            by_key[key] = contract
        self._contracts = by_key

    @property
    def contracts(self) -> tuple[ApprovedProviderContract, ...]:
        """Return contracts in a stable order without exposing mutable state."""

        return tuple(self._contracts[key] for key in sorted(self._contracts))

    def authorize_ingestion(
        self,
        *,
        provider: str,
        license_scope: str,
        license_version: str,
        source_type: str,
    ) -> ProviderUse:
        """Resolve an exact reviewed contract for private raw-data ingestion."""

        return self._resolve(
            provider=provider,
            license_scope=license_scope,
            license_version=license_version,
        ).authorize_ingestion(source_type)

    def authorize_derived_exposure(
        self,
        *,
        provider: str,
        license_scope: str,
        license_version: str,
        source_type: str,
    ) -> ProviderUse:
        """Resolve an exact reviewed contract for derived-only output."""

        return self._resolve(
            provider=provider,
            license_scope=license_scope,
            license_version=license_version,
        ).authorize_derived_exposure(source_type)

    def _resolve(
        self,
        *,
        provider: str,
        license_scope: str,
        license_version: str,
    ) -> ApprovedProviderContract:
        # Validate shape before lookup so malformed values cannot accidentally
        # become a distinct policy key or make it into an exception/log line.
        probe = ProviderUse(
            provider=provider,
            license_scope=license_scope,
            license_version=license_version,
            source_type="odds",
        )
        validate_provider_use(probe)
        try:
            return self._contracts[(provider, license_scope, license_version)]
        except KeyError as error:
            raise ProviderContractViolation("provider/license contract has not been approved") from error
