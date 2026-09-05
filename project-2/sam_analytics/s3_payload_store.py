"""Private S3-compatible evidence storage for raw provider payloads.

The raw-payload ledger needs an object store that is private by construction,
uses conditional creates rather than overwrite requests, and never turns an
object location into a public or presigned URL. This module implements that
boundary for either AWS S3 or Cloudflare R2. It deliberately has no scheduler,
provider client, or Flask route; a future *private* worker must opt in to
constructing this store. Provider-administered WORM/retention controls remain
an operational requirement.

Only ``PutObject``, ``HeadObject``, and a bounded ``GetObject`` integrity read
are used. The intended credentials must be scoped to the configured bucket and
prefix, and must not permit public ACLs, listing, deletion, or access to
unrelated objects. The object-store service identity, bucket policy, and
public-access settings are operational controls that cannot safely be inferred
from an application's credentials alone.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Protocol
from urllib.parse import urlsplit

from sam_analytics.raw_payload_store import (
    RawPayloadMetadata,
    RawPayloadStoreViolation,
    StoredRawPayload,
    validate_private_payload_uri,
    validate_raw_payload_metadata,
)


_BACKENDS = frozenset({"aws_s3", "cloudflare_r2"})
_BUCKET_RE = re.compile(r"^[a-z0-9](?:[a-z0-9.-]{1,61}[a-z0-9])$")
_IPV4_LIKE_BUCKET_RE = re.compile(r"^\d+\.\d+\.\d+\.\d+$")
_PREFIX_SEGMENT_RE = re.compile(r"^[a-z0-9][a-z0-9._=-]{0,63}$")
_AWS_REGION_RE = re.compile(r"^[a-z0-9-]{2,32}$")
_R2_ENDPOINT_RE = re.compile(
    r"^[a-f0-9]{32}(?:\.(?:eu|us|fedramp))?\.r2\.cloudflarestorage\.com$"
)
_MAX_PREFIX_LENGTH = 768
_MAX_PAYLOAD_BYTES_HARD_LIMIT = 100 * 1024 * 1024
_DEFAULT_MAX_PAYLOAD_BYTES = 20 * 1024 * 1024
_MAX_CONDITIONAL_WRITE_ATTEMPTS = 2
_VERIFY_READ_CHUNK_BYTES = 64 * 1024
_EVIDENCE_FORMAT = "raw-provider-payload-v1"
_OBJECT_CONTENT_TYPE = "application/octet-stream"


class RawPayloadStoreConfigurationError(RawPayloadStoreViolation):
    """Raised for an unsafe or incomplete object-store configuration.

    Messages intentionally identify only the invalid configuration category,
    never a supplied URL, access key, secret, or provider credential.
    """


class RawPayloadStoreUnavailable(RuntimeError):
    """Raised when a configured private object store cannot prove a safe write.

    The exception intentionally suppresses SDK error details, because S3
    failures can contain request URLs, signatures, account identifiers, or
    credential-adjacent data.
    """


class _S3ObjectClient(Protocol):
    """The intentionally tiny portion of an S3 client used by this store."""

    def put_object(self, **kwargs: Any) -> Mapping[str, Any]:
        ...

    def head_object(self, **kwargs: Any) -> Mapping[str, Any]:
        ...

    def get_object(self, **kwargs: Any) -> Mapping[str, Any]:
        ...


S3ClientFactory = Callable[
    ["S3RawPayloadStoreConfig", Optional[str], Optional[str]], _S3ObjectClient
]


@dataclass(frozen=True)
class S3RawPayloadStoreConfig:
    """Non-secret configuration for a private AWS S3 or Cloudflare R2 store.

    The future Render worker must receive a scoped S3 access-key pair for
    either backend. The adapter intentionally never falls back to ambient SDK
    credentials, which avoids a developer-machine credential chain becoming a
    surprise deployment dependency. Credentials deliberately do not live on
    this dataclass so its representation remains safe to inspect or log.
    """

    backend: str
    bucket: str
    prefix: str
    region: str
    endpoint_url: str | None = None
    max_payload_bytes: int = _DEFAULT_MAX_PAYLOAD_BYTES

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or self.backend not in _BACKENDS:
            raise RawPayloadStoreConfigurationError(
                "raw payload store backend must be aws_s3 or cloudflare_r2"
            )
        _validate_bucket_name(self.bucket)
        _validate_prefix(self.prefix)
        _validate_max_payload_bytes(self.max_payload_bytes)

        if self.backend == "aws_s3":
            if not isinstance(self.region, str) or not _AWS_REGION_RE.fullmatch(self.region):
                raise RawPayloadStoreConfigurationError("AWS object-store region is invalid")
            if self.endpoint_url is not None:
                _validate_aws_endpoint(self.endpoint_url, region=self.region)
        else:
            if self.region != "auto":
                raise RawPayloadStoreConfigurationError("Cloudflare R2 object-store region must be auto")
            if self.endpoint_url is None:
                raise RawPayloadStoreConfigurationError("Cloudflare R2 object-store endpoint is required")
            _validate_r2_endpoint(self.endpoint_url)

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> "S3RawPayloadStoreConfig":
        """Read only non-secret S3/R2 configuration from a future worker env."""

        values = os.environ if environ is None else environ
        backend = _required_config_value(values, "SAM_RAW_EVIDENCE_STORE_BACKEND")
        bucket, prefix = _parse_evidence_store_uri(
            _required_config_value(values, "SAM_RAW_EVIDENCE_STORE_URI")
        )
        endpoint_url = _optional_config_value(values, "SAM_RAW_EVIDENCE_S3_ENDPOINT_URL")
        region = _optional_config_value(values, "SAM_RAW_EVIDENCE_S3_REGION")
        if region is None:
            region = "auto" if backend == "cloudflare_r2" else ""
        max_payload_bytes = _parse_positive_int(
            _optional_config_value(values, "SAM_RAW_EVIDENCE_MAX_BYTES"),
            default=_DEFAULT_MAX_PAYLOAD_BYTES,
            label="SAM_RAW_EVIDENCE_MAX_BYTES",
        )
        return cls(
            backend=backend,
            bucket=bucket,
            prefix=prefix,
            region=region,
            endpoint_url=endpoint_url,
            max_payload_bytes=max_payload_bytes,
        )


class S3CompatibleRawPayloadStore:
    """Write byte-exact evidence to a private, conditional-create S3 key.

    The write uses ``If-None-Match: *`` so a SHA-256 addressed object cannot
    overwrite prior evidence.  A successful write (or an idempotent existing
    object) is followed by ``HeadObject`` verification of intrinsic object
    metadata and a bounded ``GetObject`` SHA-256 verification of retained
    bytes. Receipt provenance stays in PostgreSQL rather than the shared
    content-addressed object. This is an application-level verification
    boundary; provider-administered WORM/retention controls are separate. This
    store never lists a bucket, deletes an object, changes an ACL, generates a
    URL, or returns raw provider bytes.
    """

    def __init__(self, config: S3RawPayloadStoreConfig, *, client: _S3ObjectClient) -> None:
        if not isinstance(config, S3RawPayloadStoreConfig):
            raise RawPayloadStoreConfigurationError("raw payload store configuration is invalid")
        if not _is_s3_client(client):
            raise RawPayloadStoreConfigurationError("S3-compatible object-store client is invalid")
        self._config = config
        self._client = client

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        client_factory: S3ClientFactory | None = None,
    ) -> "S3CompatibleRawPayloadStore":
        """Build a private store for a future worker without exposing secrets.

        This method is intentionally not called by the web process or the
        current inert worker.  It exists so that a reviewed future worker can
        construct the concrete store only after the bucket and scoped service
        credentials have been provisioned.
        """

        values = os.environ if environ is None else environ
        config = S3RawPayloadStoreConfig.from_environment(values)
        access_key_id, secret_access_key = _read_credentials(values, backend=config.backend)
        factory = client_factory or _create_boto3_client
        client: _S3ObjectClient | None = None
        client_creation_failed = False
        try:
            client = factory(config, access_key_id, secret_access_key)
        except Exception:
            # Raise outside the exception handler so even programmatic
            # inspection of __context__ cannot expose a secret from an SDK.
            client_creation_failed = True
        if client_creation_failed or client is None:
            raise RawPayloadStoreConfigurationError("S3-compatible client could not be configured")
        return cls(config, client=client)

    @property
    def config(self) -> S3RawPayloadStoreConfig:
        """Expose non-secret configuration for a worker's local diagnostics."""

        return self._config

    def store(
        self,
        payload: bytes,
        *,
        metadata: RawPayloadMetadata,
        stored_at: datetime | None = None,
    ) -> StoredRawPayload:
        """Store an immutable payload or prove an identical object already exists.

        Any provider/SDK failure is converted to a credential-safe error.  A
        caller must treat that error as a failed ingest; normalized database
        facts may never be written after an unverified raw-object write.
        """

        validate_raw_payload_metadata(metadata)
        _validate_payload_and_time(payload, metadata=metadata, stored_at=stored_at)
        if len(payload) > self._config.max_payload_bytes:
            raise RawPayloadStoreViolation("payload exceeds the configured private-store byte limit")
        effective_stored_at = stored_at or metadata.received_at
        digest = hashlib.sha256(payload).hexdigest()
        key = self._object_key(digest)
        expected_metadata = _expected_object_metadata(payload_sha256=digest, byte_count=len(payload))

        write_parameters = {
            "Bucket": self._config.bucket,
            "Key": key,
            "Body": payload,
            "ContentLength": len(payload),
            # The raw object is byte evidence, not a typed provider response.
            # Keeping a constant object type allows identical bytes from two
            # distinct receipts to share one content-addressed object.
            "ContentType": _OBJECT_CONTENT_TYPE,
            "ContentMD5": _content_md5(payload),
            "CacheControl": "private, no-store",
            "Metadata": expected_metadata,
            "IfNoneMatch": "*",
        }
        # Cloudflare R2 explicitly does not implement S3's server-side
        # encryption header.  Its platform handles encryption at rest, while
        # AWS S3 receives an explicit SSE-S3 request and verification below.
        if self._config.backend == "aws_s3":
            write_parameters["ServerSideEncryption"] = "AES256"

        wrote_or_found = False
        write_failed = False
        for attempt in range(_MAX_CONDITIONAL_WRITE_ATTEMPTS):
            try:
                self._client.put_object(**write_parameters)
                wrote_or_found = True
                break
            except Exception as error:
                status = _conditional_write_status(error)
                if status == "already_exists":
                    wrote_or_found = True
                    break
                if status == "retry" and attempt + 1 < _MAX_CONDITIONAL_WRITE_ATTEMPTS:
                    continue
                write_failed = True
                break

        if write_failed or not wrote_or_found:  # Defensive: the loop must either return or fail.
            raise RawPayloadStoreUnavailable("private object-store write failed")

        self._verify_object(
            key=key,
            expected_metadata=expected_metadata,
            payload_sha256=digest,
            byte_count=len(payload),
        )
        payload_uri = f"s3://{self._config.bucket}/{key}"
        validate_private_payload_uri(payload_uri, payload_sha256=digest)
        return StoredRawPayload(
            payload_uri=payload_uri,
            payload_sha256=digest,
            byte_count=len(payload),
            metadata=metadata,
            stored_at=effective_stored_at,
        )

    def __repr__(self) -> str:
        """A deliberately credential-free debugging representation."""

        return (
            "S3CompatibleRawPayloadStore("
            f"backend={self._config.backend!r}, bucket={self._config.bucket!r}, "
            f"prefix={self._config.prefix!r})"
        )

    def _object_key(self, payload_sha256: str) -> str:
        return f"{self._config.prefix}/sha256/{payload_sha256}"

    def _verify_object(
        self,
        *,
        key: str,
        expected_metadata: Mapping[str, str],
        payload_sha256: str,
        byte_count: int,
    ) -> None:
        response: Mapping[str, Any] | None = None
        verification_request_failed = False
        try:
            response = self._client.head_object(Bucket=self._config.bucket, Key=key)
        except Exception:
            # As above, defer the safe error until no SDK exception is active.
            verification_request_failed = True

        if verification_request_failed or not isinstance(response, Mapping):
            if verification_request_failed:
                raise RawPayloadStoreUnavailable("private object-store verification failed")
            raise RawPayloadStoreViolation("private object-store verification failed")
        content_length = response.get("ContentLength")
        if isinstance(content_length, bool) or not isinstance(content_length, int) or content_length != byte_count:
            raise RawPayloadStoreViolation("private object-store verification failed")
        if response.get("ContentType") != _OBJECT_CONTENT_TYPE:
            raise RawPayloadStoreViolation("private object-store verification failed")
        if self._config.backend == "aws_s3" and response.get("ServerSideEncryption") != "AES256":
            raise RawPayloadStoreViolation("private object-store verification failed")
        remote_metadata = response.get("Metadata")
        if not isinstance(remote_metadata, Mapping):
            raise RawPayloadStoreViolation("private object-store verification failed")
        normalized_remote_metadata = {
            key.lower(): value for key, value in remote_metadata.items() if isinstance(key, str) and isinstance(value, str)
        }
        if normalized_remote_metadata != dict(expected_metadata):
            raise RawPayloadStoreViolation("private object-store verification failed")
        if normalized_remote_metadata.get("sam-sha256") != payload_sha256:
            raise RawPayloadStoreViolation("private object-store verification failed")
        self._verify_object_bytes(
            key=key,
            payload_sha256=payload_sha256,
            byte_count=byte_count,
        )

    def _verify_object_bytes(self, *, key: str, payload_sha256: str, byte_count: int) -> None:
        """Stream the retained object so a spoofed same-length object is rejected."""

        response: Mapping[str, Any] | None = None
        verification_request_failed = False
        try:
            response = self._client.get_object(Bucket=self._config.bucket, Key=key)
        except Exception:
            # Defer the safe error until no SDK exception is active. A GetObject
            # failure may otherwise retain a signed request URL in __context__.
            verification_request_failed = True
        if verification_request_failed or not isinstance(response, Mapping):
            if verification_request_failed:
                raise RawPayloadStoreUnavailable("private object-store verification failed")
            raise RawPayloadStoreViolation("private object-store verification failed")
        body = response.get("Body")
        retained_digest = _stream_sha256(body, expected_byte_count=byte_count)
        if retained_digest != payload_sha256:
            raise RawPayloadStoreViolation("private object-store verification failed")


def _validate_bucket_name(bucket: object) -> None:
    if not isinstance(bucket, str) or not _BUCKET_RE.fullmatch(bucket):
        raise RawPayloadStoreConfigurationError("object-store bucket name is invalid")
    if ".." in bucket or ".-" in bucket or "-." in bucket:
        raise RawPayloadStoreConfigurationError("object-store bucket name is invalid")
    if _IPV4_LIKE_BUCKET_RE.fullmatch(bucket) or bucket.startswith("xn--"):
        raise RawPayloadStoreConfigurationError("object-store bucket name is invalid")


def _validate_prefix(prefix: object) -> None:
    if not isinstance(prefix, str) or not prefix or len(prefix) > _MAX_PREFIX_LENGTH:
        raise RawPayloadStoreConfigurationError("object-store prefix is invalid")
    segments = prefix.split("/")
    if any(
        segment in {"", ".", "..", "sha256"} or not _PREFIX_SEGMENT_RE.fullmatch(segment)
        for segment in segments
    ):
        raise RawPayloadStoreConfigurationError("object-store prefix is invalid")


def _parse_evidence_store_uri(value: object) -> tuple[str, str]:
    """Parse the worker's canonical, credential-free evidence-store identity."""

    if not isinstance(value, str) or not value or len(value) > 1024:
        raise RawPayloadStoreConfigurationError("SAM_RAW_EVIDENCE_STORE_URI is invalid")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise RawPayloadStoreConfigurationError("SAM_RAW_EVIDENCE_STORE_URI is invalid")
    if "?" in value or "#" in value or "%3f" in value.lower() or "%23" in value.lower():
        raise RawPayloadStoreConfigurationError("SAM_RAW_EVIDENCE_STORE_URI is invalid")
    parsed = None
    port = None
    uri_parse_failed = False
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        uri_parse_failed = True
    if uri_parse_failed or parsed is None:
        raise RawPayloadStoreConfigurationError("SAM_RAW_EVIDENCE_STORE_URI is invalid")
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith("/")
    ):
        raise RawPayloadStoreConfigurationError("SAM_RAW_EVIDENCE_STORE_URI is invalid")
    _validate_bucket_name(parsed.netloc)
    prefix = parsed.path[1:]
    _validate_prefix(prefix)
    return parsed.netloc, prefix


def _validate_max_payload_bytes(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= _MAX_PAYLOAD_BYTES_HARD_LIMIT:
        raise RawPayloadStoreConfigurationError("raw payload byte limit is invalid")


def _validate_aws_endpoint(value: object, *, region: str) -> None:
    endpoint = _parse_private_https_endpoint(value)
    host = endpoint.hostname
    if host is None:
        raise RawPayloadStoreConfigurationError("AWS object-store endpoint is invalid")
    allowed_hosts = {
        "s3.amazonaws.com",
        f"s3.{region}.amazonaws.com",
        f"s3-{region}.amazonaws.com",
        f"s3.dualstack.{region}.amazonaws.com",
        f"s3-fips.{region}.amazonaws.com",
        f"s3-fips.{region}.api.aws",
    }
    if host.lower() not in allowed_hosts:
        raise RawPayloadStoreConfigurationError("AWS object-store endpoint is invalid")


def _validate_r2_endpoint(value: object) -> None:
    endpoint = _parse_private_https_endpoint(value)
    if endpoint.hostname is None or not _R2_ENDPOINT_RE.fullmatch(endpoint.hostname.lower()):
        raise RawPayloadStoreConfigurationError("Cloudflare R2 object-store endpoint is invalid")


def _parse_private_https_endpoint(value: object):
    if not isinstance(value, str) or not value or len(value) > 512 or "%" in value:
        raise RawPayloadStoreConfigurationError("object-store endpoint is invalid")
    parsed = None
    port = None
    endpoint_parse_failed = False
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        endpoint_parse_failed = True
    if endpoint_parse_failed or parsed is None:
        raise RawPayloadStoreConfigurationError("object-store endpoint is invalid")
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
        or not parsed.hostname
    ):
        raise RawPayloadStoreConfigurationError("object-store endpoint is invalid")
    return parsed


def _required_config_value(values: Mapping[str, str], name: str) -> str:
    value = _optional_config_value(values, name)
    if value is None:
        raise RawPayloadStoreConfigurationError(f"{name} is required")
    return value


def _optional_config_value(values: Mapping[str, str], name: str) -> str | None:
    value = values.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value or value != value.strip():
        raise RawPayloadStoreConfigurationError(f"{name} is invalid")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise RawPayloadStoreConfigurationError(f"{name} is invalid")
    return value


def _parse_positive_int(value: str | None, *, default: int, label: str) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        raise RawPayloadStoreConfigurationError(f"{label} is invalid") from None
    _validate_max_payload_bytes(parsed)
    return parsed


def _read_credentials(values: Mapping[str, str], *, backend: str) -> tuple[str | None, str | None]:
    access_key_id = _optional_secret_value(values, "SAM_RAW_EVIDENCE_S3_ACCESS_KEY_ID")
    secret_access_key = _optional_secret_value(values, "SAM_RAW_EVIDENCE_S3_SECRET_ACCESS_KEY")
    if (access_key_id is None) != (secret_access_key is None):
        raise RawPayloadStoreConfigurationError("object-store access-key pair is incomplete")
    if access_key_id is None:
        if backend == "cloudflare_r2":
            raise RawPayloadStoreConfigurationError("Cloudflare R2 object-store credentials are required")
        raise RawPayloadStoreConfigurationError("AWS object-store credentials are required")
    return access_key_id, secret_access_key


def _optional_secret_value(values: Mapping[str, str], name: str) -> str | None:
    value = values.get(name)
    if value is None or value == "":
        return None
    if not isinstance(value, str) or len(value) > 1024:
        raise RawPayloadStoreConfigurationError(f"{name} is invalid")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise RawPayloadStoreConfigurationError(f"{name} is invalid")
    return value


def _create_boto3_client(
    config: S3RawPayloadStoreConfig,
    access_key_id: str | None,
    secret_access_key: str | None,
) -> _S3ObjectClient:
    """Create a tightly configured SDK client without retaining a secret here."""

    if access_key_id is None or secret_access_key is None:
        raise RawPayloadStoreConfigurationError("object-store credentials are required")

    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        raise RawPayloadStoreConfigurationError("S3-compatible client library is unavailable") from None

    client_parameters: dict[str, Any] = {
        "service_name": "s3",
        "region_name": config.region,
        # Path style prevents a bucket name from becoming an uncontrolled host
        # component and is compatible with Cloudflare R2's account endpoint.
        "config": Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            connect_timeout=5,
            read_timeout=20,
            retries={"mode": "standard", "max_attempts": 3},
        ),
    }
    if config.endpoint_url is not None:
        client_parameters["endpoint_url"] = config.endpoint_url
    client_parameters["aws_access_key_id"] = access_key_id
    client_parameters["aws_secret_access_key"] = secret_access_key
    return boto3.client(**client_parameters)


def _is_s3_client(value: object) -> bool:
    return (
        callable(getattr(value, "put_object", None))
        and callable(getattr(value, "head_object", None))
        and callable(getattr(value, "get_object", None))
    )


def _validate_payload_and_time(
    payload: object, *, metadata: RawPayloadMetadata, stored_at: datetime | None
) -> None:
    if not isinstance(payload, bytes) or not payload:
        raise RawPayloadStoreViolation("payload must be non-empty bytes")
    if stored_at is not None:
        if not _is_aware(stored_at):
            raise RawPayloadStoreViolation("stored_at must be timezone-aware")
        if stored_at < metadata.received_at:
            raise RawPayloadStoreViolation("stored_at cannot precede received_at")


def _is_aware(value: object) -> bool:
    return isinstance(value, datetime) and value.tzinfo is not None and value.utcoffset() is not None


def _content_md5(payload: bytes) -> str:
    # Content-MD5 is the S3 transport-integrity protocol field; SHA-256
    # remains the evidence identity and content-addressed key.
    digest = hashlib.md5(payload, usedforsecurity=False).digest()  # nosec B324
    return base64.b64encode(digest).decode("ascii")


def _expected_object_metadata(*, payload_sha256: str, byte_count: int) -> dict[str, str]:
    """Return only metadata intrinsic to the immutable byte object.

    Receipt-specific fields (provider receipt ID, timestamps, license version,
    and claimed provider content type) intentionally stay in PostgreSQL's
    immutable receipt/provenance ledger. Including them here would make a
    second lawful receipt for identical bytes fail after ``If-None-Match``.
    """

    return {
        "sam-evidence-format": _EVIDENCE_FORMAT,
        "sam-sha256": payload_sha256,
        "sam-byte-count": str(byte_count),
    }


def _stream_sha256(body: object, *, expected_byte_count: int) -> str:
    """Read a bounded S3 body and return its SHA-256 without exposing bytes."""

    read = getattr(body, "read", None)
    if not callable(read):
        raise RawPayloadStoreViolation("private object-store verification failed")

    digest = hashlib.sha256()
    byte_count = 0
    read_failed = False
    length_exceeded = False
    try:
        while True:
            # Once the expected body length is reached, read exactly one more
            # byte to detect a forged object that lies about ContentLength.
            request_size = min(_VERIFY_READ_CHUNK_BYTES, expected_byte_count - byte_count + 1)
            chunk = read(request_size)
            if not isinstance(chunk, bytes):
                read_failed = True
                break
            if not chunk:
                break
            byte_count += len(chunk)
            if byte_count > expected_byte_count:
                length_exceeded = True
                break
            digest.update(chunk)
    except Exception:
        read_failed = True
    finally:
        close = getattr(body, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                # Closing a response never changes evidence validity and must
                # not retain a transport exception containing request details.
                pass

    if read_failed:
        raise RawPayloadStoreUnavailable("private object-store verification failed")
    if length_exceeded or byte_count != expected_byte_count:
        raise RawPayloadStoreViolation("private object-store verification failed")
    return digest.hexdigest()


def _conditional_write_status(error: BaseException) -> str:
    """Classify only S3 conditional-write outcomes without formatting ``error``."""

    response = getattr(error, "response", None)
    if not isinstance(response, Mapping):
        return "failed"
    response_error = response.get("Error")
    code = response_error.get("Code") if isinstance(response_error, Mapping) else None
    response_metadata = response.get("ResponseMetadata")
    status_code = response_metadata.get("HTTPStatusCode") if isinstance(response_metadata, Mapping) else None
    if code in {"PreconditionFailed", "412"} or status_code == 412:
        return "already_exists"
    if code in {"ConditionalRequestConflict", "409"} or status_code == 409:
        return "retry"
    return "failed"
