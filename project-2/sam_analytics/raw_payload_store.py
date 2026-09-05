"""Private, content-addressed storage contracts for raw provider payloads.

SAM's normalized odds rows are useful only when they can be traced back to the
exact provider response that produced them. This module supplies the narrow
contract and deterministic in-memory fake used by the ledger. The concrete
AWS S3/Cloudflare R2 implementation lives in ``s3_payload_store.py``; it is
deliberately not constructed by the public API or the current inert worker.

Raw objects are addressed by the SHA-256 digest of their *unmodified bytes*.
The returned receipt intentionally never contains the payload itself, request
URLs, response headers, or credentials.  Those fields are easy ways to leak a
provider key into logs or a public analytics response.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol, runtime_checkable
from urllib.parse import urlsplit


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_RECORD_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")
_URI_NAMESPACE_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{0,127}$")
_URI_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._=-]{0,255}$")
_CONTENT_TYPE_RE = re.compile(
    r"^[a-z0-9][a-z0-9!#$&^_.+-]{0,63}/[a-z0-9][a-z0-9!#$&^_.+-]{0,63}"
    r"(?:; charset=(?:utf-8|us-ascii))?$"
)
_PRIVATE_OBJECT_URI_SCHEMES = frozenset({"s3", "gs", "memory"})
_MAX_PROVIDER_CLOCK_SKEW = timedelta(minutes=5)
_DEFAULT_MAX_PAYLOAD_BYTES = 20 * 1024 * 1024


class RawPayloadStoreViolation(ValueError):
    """Raised when evidence storage would be unsafe or non-auditable."""


def _is_aware(value: object) -> bool:
    return isinstance(value, datetime) and value.tzinfo is not None and value.utcoffset() is not None


def _safe_text(value: object, pattern: re.Pattern[str]) -> bool:
    return isinstance(value, str) and bool(pattern.fullmatch(value))


@dataclass(frozen=True)
class RawPayloadMetadata:
    """Small, allow-listed metadata bound to one provider response.

    There is intentionally no arbitrary metadata mapping.  Request URLs,
    response headers, authorization values, and arbitrary provider fields must
    not be copied into an evidence receipt where they can later leak through a
    database export or log statement.
    """

    provider: str
    provider_record_id: str
    source_type: str
    captured_at: datetime
    received_at: datetime
    schema_version: str
    license_scope: str
    license_version: str
    content_type: str = "application/json"


def validate_raw_payload_metadata(metadata: RawPayloadMetadata) -> None:
    """Fail closed unless a raw-payload receipt has safe, usable metadata."""

    if not isinstance(metadata, RawPayloadMetadata):
        raise RawPayloadStoreViolation("raw payload metadata has an invalid type")
    if not _safe_text(metadata.provider, _IDENTIFIER_RE):
        raise RawPayloadStoreViolation("provider must be a lowercase safe identifier")
    if not _safe_text(metadata.provider_record_id, _RECORD_ID_RE):
        raise RawPayloadStoreViolation("provider_record_id must be a safe opaque identifier")
    if not _safe_text(metadata.source_type, _IDENTIFIER_RE):
        raise RawPayloadStoreViolation("source_type must be a lowercase safe identifier")
    for field in ("schema_version", "license_scope", "license_version"):
        if not _safe_text(getattr(metadata, field), _SAFE_TOKEN_RE):
            raise RawPayloadStoreViolation(f"{field} must be a safe non-secret token")
    if not isinstance(metadata.content_type, str) or not _CONTENT_TYPE_RE.fullmatch(metadata.content_type):
        raise RawPayloadStoreViolation("content_type must be a safe, lowercase media type")
    if not _is_aware(metadata.captured_at) or not _is_aware(metadata.received_at):
        raise RawPayloadStoreViolation("captured_at and received_at must be timezone-aware")
    if _is_aware(metadata.captured_at) and _is_aware(metadata.received_at):
        if metadata.captured_at > metadata.received_at + _MAX_PROVIDER_CLOCK_SKEW:
            raise RawPayloadStoreViolation("captured_at is implausibly after local receipt")


def validate_private_payload_uri(payload_uri: str, *, payload_sha256: str | None = None) -> None:
    """Validate a stable, private, content-addressed object reference.

    ``s3://`` and ``gs://`` are object-store references, not browser URLs.
    They are accepted only as identifiers; deployment configuration must still
    keep the referenced bucket private.  ``memory://`` exists solely for the
    deterministic test double below.  Browser URLs, local files, signed URLs,
    fragments, credentials, and path traversal are all rejected.
    """

    if not isinstance(payload_uri, str) or not payload_uri or len(payload_uri) > 2048:
        raise RawPayloadStoreViolation("payload_uri must be a bounded, non-empty string")
    if any(ord(character) < 32 or ord(character) == 127 for character in payload_uri):
        raise RawPayloadStoreViolation("payload_uri cannot contain control characters")
    # Reject encoded query/fragment delimiters too.  A content-addressed object
    # key does not need either form, and allowing them makes accidental signed
    # URL persistence much easier.
    if "?" in payload_uri or "#" in payload_uri or "%3f" in payload_uri.lower() or "%23" in payload_uri.lower():
        raise RawPayloadStoreViolation("payload_uri cannot contain a query string or fragment")

    try:
        parsed = urlsplit(payload_uri)
        port = parsed.port
    except ValueError as error:
        raise RawPayloadStoreViolation("payload_uri is not a valid private object reference") from error
    if parsed.scheme not in _PRIVATE_OBJECT_URI_SCHEMES:
        raise RawPayloadStoreViolation("payload_uri must use an approved private object-store scheme")
    if not parsed.netloc or parsed.username is not None or parsed.password is not None or port is not None:
        raise RawPayloadStoreViolation("payload_uri cannot include a host credential or port")
    if not _URI_NAMESPACE_RE.fullmatch(parsed.netloc):
        raise RawPayloadStoreViolation("payload_uri object-store namespace is invalid")
    if parsed.query or parsed.fragment or not parsed.path.startswith("/"):
        raise RawPayloadStoreViolation("payload_uri must be a stable object reference")

    segments = parsed.path.split("/")[1:]
    if len(segments) < 2 or any(
        not segment or segment in {".", ".."} or not _URI_SEGMENT_RE.fullmatch(segment)
        for segment in segments
    ):
        raise RawPayloadStoreViolation("payload_uri contains an unsafe object key")
    if segments[-2] != "sha256" or not _SHA256_RE.fullmatch(segments[-1]):
        raise RawPayloadStoreViolation("payload_uri must end with /sha256/<lowercase-digest>")
    if payload_sha256 is not None:
        if not isinstance(payload_sha256, str) or not _SHA256_RE.fullmatch(payload_sha256):
            raise RawPayloadStoreViolation("payload_sha256 must be a lowercase SHA-256 digest")
        if segments[-1] != payload_sha256:
            raise RawPayloadStoreViolation("payload_uri digest does not match payload_sha256")


@dataclass(frozen=True)
class StoredRawPayload:
    """Evidence receipt returned after a private write.

    The raw bytes are deliberately absent.  Persist this receipt and use its
    URI/digest to create an immutable provenance record; never return payload
    bytes through a public API.
    """

    payload_uri: str
    payload_sha256: str
    byte_count: int
    metadata: RawPayloadMetadata
    stored_at: datetime

    def __post_init__(self) -> None:
        validate_private_payload_uri(self.payload_uri, payload_sha256=self.payload_sha256)
        validate_raw_payload_metadata(self.metadata)
        if isinstance(self.byte_count, bool) or not isinstance(self.byte_count, int) or self.byte_count <= 0:
            raise RawPayloadStoreViolation("byte_count must be a positive integer")
        if not _is_aware(self.stored_at):
            raise RawPayloadStoreViolation("stored_at must be timezone-aware")
        if self.stored_at < self.metadata.received_at:
            raise RawPayloadStoreViolation("stored_at cannot precede received_at")


@runtime_checkable
class RawPayloadStore(Protocol):
    """Private evidence-store boundary used by ingestion code.

    Implementations must write the exact bytes before the caller records any
    normalized row in the database.  They must be idempotent for identical
    bytes and return an address whose final digest matches those bytes.
    """

    def store(
        self,
        payload: bytes,
        *,
        metadata: RawPayloadMetadata,
        stored_at: datetime | None = None,
    ) -> StoredRawPayload:
        """Write an unmodified private payload and return only its receipt."""


class InMemoryRawPayloadStore:
    """Deterministic private-store fake for unit tests.

    It is intentionally in-memory and unsuitable for a worker or web service:
    restarts erase its data.  ``read_bytes`` exists only to make tests verify
    byte-for-byte retention; production code should depend on
    :class:`RawPayloadStore` and never expose raw bytes.
    """

    def __init__(
        self,
        *,
        namespace: str = "sam-private-test",
        max_payload_bytes: int = _DEFAULT_MAX_PAYLOAD_BYTES,
    ) -> None:
        if not isinstance(namespace, str) or not _URI_NAMESPACE_RE.fullmatch(namespace):
            raise RawPayloadStoreViolation("private-store namespace is invalid")
        if isinstance(max_payload_bytes, bool) or not isinstance(max_payload_bytes, int) or max_payload_bytes <= 0:
            raise RawPayloadStoreViolation("max_payload_bytes must be a positive integer")
        self._namespace = namespace
        self._max_payload_bytes = max_payload_bytes
        self._payloads: dict[str, bytes] = {}

    @property
    def stored_count(self) -> int:
        """Number of distinct byte payloads retained by this test fake."""

        return len(self._payloads)

    def store(
        self,
        payload: bytes,
        *,
        metadata: RawPayloadMetadata,
        stored_at: datetime | None = None,
    ) -> StoredRawPayload:
        validate_raw_payload_metadata(metadata)
        if not isinstance(payload, bytes) or not payload:
            raise RawPayloadStoreViolation("payload must be non-empty bytes")
        if len(payload) > self._max_payload_bytes:
            raise RawPayloadStoreViolation("payload exceeds the configured private-store byte limit")
        if stored_at is None:
            stored_at = metadata.received_at
        if not _is_aware(stored_at):
            raise RawPayloadStoreViolation("stored_at must be timezone-aware")
        if stored_at < metadata.received_at:
            raise RawPayloadStoreViolation("stored_at cannot precede received_at")

        payload_sha256 = hashlib.sha256(payload).hexdigest()
        payload_uri = f"memory://{self._namespace}/sha256/{payload_sha256}"
        existing = self._payloads.get(payload_uri)
        if existing is not None and existing != payload:
            # A cryptographic collision must never silently overwrite evidence.
            raise RawPayloadStoreViolation("content-addressed payload collision detected")
        self._payloads.setdefault(payload_uri, payload)
        return StoredRawPayload(
            payload_uri=payload_uri,
            payload_sha256=payload_sha256,
            byte_count=len(payload),
            metadata=metadata,
            stored_at=stored_at,
        )

    def read_bytes(self, payload_uri: str) -> bytes:
        """Return retained bytes for tests only; never wire this into an API."""

        validate_private_payload_uri(payload_uri)
        try:
            return self._payloads[payload_uri]
        except KeyError as error:
            raise RawPayloadStoreViolation("payload does not exist in the private test store") from error
