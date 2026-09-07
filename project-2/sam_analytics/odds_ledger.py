"""Transactional, evidence-first persistence for licensed odds and results.

This module is deliberately *not* a scheduler or a provider client.  A caller
first obtains a complete provider response, prepares it, and hands it to this
ledger.  The ledger then enforces the important ordering:

1. validate the approved provider contract and every admitted observation;
2. put the unmodified response in private, content-addressed storage;
3. atomically write the receipt, raw provenance, event identities, snapshots,
   provenance links, and a safe operational signal to PostgreSQL.

Nothing in this module returns a provider credential, raw response body, or a
private object URL.  It also intentionally has no network polling function; a
separately admitted private worker must be explicitly enabled only after this
persistence boundary and its object-storage configuration are verified.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol

from sam_analytics.data_contracts import (
    RawDataProvenance,
    validate_raw_data_provenance,
)
from sam_analytics.ingestion import NormalizedQuote, RawOddsQuote, normalize_quotes
from sam_analytics.provider_contracts import (
    ProviderContractRegistry,
    ProviderUse,
    validate_private_payload_metadata_for_use,
)
from sam_analytics.raw_payload_store import (
    RawPayloadMetadata,
    RawPayloadStore,
    StoredRawPayload,
    validate_private_payload_uri,
)

if TYPE_CHECKING:
    from sam_analytics.providers.the_odds_api import CompletedScore


_SENSITIVE_SCOPE_NAMES = frozenset(
    {
        "key",
    }
)
_SENSITIVE_SCOPE_KEY_RE = re.compile(
    r"(?:^|[-_])(?:api[-_]?key|token|secret|password|authorization|auth)(?:$|[-_])",
    re.IGNORECASE,
)
_MAX_PROVIDER_CLOCK_SKEW = timedelta(minutes=5)


class OddsLedgerError(RuntimeError):
    """Base error intentionally safe to log from an asynchronous worker."""


class OddsLedgerValidationError(OddsLedgerError, ValueError):
    """The caller supplied evidence that cannot safely become a ledger fact."""


class OddsLedgerUnavailable(OddsLedgerError):
    """The private database transaction could not be completed safely."""


@dataclass(frozen=True)
class PreparedOddsPayload:
    """One complete, immutable provider response ready for persistence.

    ``request_scope`` contains only non-secret fields such as sport, region,
    and market.  It is stored as a hash in PostgreSQL; any field name that
    resembles a credential is rejected even before hashing.
    """

    provider: str
    source_type: str
    raw_payload: bytes
    captured_at: datetime
    received_at: datetime
    schema_version: str
    license_scope: str
    license_version: str
    quotes: tuple[RawOddsQuote, ...]
    request_scope: tuple[tuple[str, str], ...] = ()
    provider_response_status: int = 200
    requests_remaining: int | None = None
    requests_used: int | None = None
    request_cost: int | None = None
    content_type: str = "application/json"
    source_available_at: datetime | None = None

    @property
    def request_fingerprint_sha256(self) -> str:
        return _payload_request_fingerprint(self)

    @property
    def receipt_sha256(self) -> str:
        """Stable identity for this exact provider receipt, without its body."""

        return _payload_receipt_sha256(self)


@dataclass(frozen=True)
class PreparedResultsPayload:
    """One complete scores response ready for immutable result persistence."""

    provider: str
    source_type: str
    raw_payload: bytes
    captured_at: datetime
    received_at: datetime
    schema_version: str
    license_scope: str
    license_version: str
    scores: tuple[CompletedScore, ...]
    request_scope: tuple[tuple[str, str], ...] = ()
    provider_response_status: int = 200
    requests_remaining: int | None = None
    requests_used: int | None = None
    request_cost: int | None = None
    content_type: str = "application/json"
    source_available_at: datetime | None = None

    @property
    def request_fingerprint_sha256(self) -> str:
        return _payload_request_fingerprint(self)

    @property
    def receipt_sha256(self) -> str:
        """Stable identity for this exact provider receipt, without its body."""

        return _payload_receipt_sha256(self)


@dataclass(frozen=True)
class LedgerWriteResult:
    """Credential-safe, aggregate outcome of one attempted evidence write."""

    status: str
    receipt_sha256: str
    provenance_sha256: str
    events_created: int
    snapshots_created: int
    snapshots_replayed: int
    provenance_links_created: int
    incidents_created: int


@dataclass(frozen=True)
class ResultsLedgerWriteResult:
    """Credential-safe outcome of one attempted completed-results write."""

    status: str
    receipt_sha256: str
    provenance_sha256: str
    events_created: int
    results_created: int
    results_replayed: int
    provenance_links_created: int
    incidents_created: int


@dataclass(frozen=True)
class _EventIdentity:
    provider: str
    provider_event_id: str
    sport: str
    league: str
    starts_at: datetime
    home_team: str
    away_team: str


class _Cursor(Protocol):
    def execute(self, query: str, params: Sequence[Any] | None = None) -> Any:
        ...

    def fetchone(self) -> Sequence[Any] | None:
        ...

    def __enter__(self) -> _Cursor:
        ...

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        ...


class _Connection(Protocol):
    def cursor(self) -> _Cursor:
        ...

    def transaction(self) -> Any:
        ...

    def close(self) -> None:
        ...


ConnectionFactory = Callable[[str], _Connection]


class OddsLedger:
    """Persist only prevalidated licensed odds under one PostgreSQL transaction."""

    def __init__(
        self,
        database_url: str,
        *,
        raw_payload_store: RawPayloadStore,
        provider_contracts: ProviderContractRegistry,
        connection_factory: ConnectionFactory | None = None,
    ):
        if not isinstance(database_url, str) or not database_url.strip():
            raise OddsLedgerValidationError("a database connection must be configured for the odds ledger")
        self._database_url = database_url
        self._raw_payload_store = raw_payload_store
        self._provider_contracts = provider_contracts
        self._connection_factory = connection_factory or _connect_postgres

    def persist(
        self, payload: PreparedOddsPayload, *, now: datetime | None = None
    ) -> LedgerWriteResult:
        """Write a provider receipt and its normalized observations safely.

        A raw object is deliberately stored *before* the database transaction.
        A transaction failure can leave a harmless content-addressed orphan,
        but can never create a database fact that lacks retained raw evidence.
        """

        validation_time = now or datetime.now(UTC)
        _validate_prepared_payload(payload, now=validation_time)
        provider_use = self._authorize(payload)
        try:
            normalized_quotes = tuple(
                normalize_quotes(payload.quotes, now=_payload_source_available_at(payload))
            )
        except ValueError:
            raise OddsLedgerValidationError("provider quotes failed pregame validation") from None
        event_identities = _event_identities(normalized_quotes)
        stored_payload, provenance = self._retain_raw_evidence(
            payload,
            provider_use=provider_use,
            validation_time=validation_time,
        )

        return self._write_transaction(
            payload=payload,
            stored_payload=stored_payload,
            provenance=provenance,
            normalized_quotes=normalized_quotes,
            event_identities=event_identities,
        )

    def persist_results(
        self, payload: PreparedResultsPayload, *, now: datetime | None = None
    ) -> ResultsLedgerWriteResult:
        """Write completed scores only after retaining their raw provider response."""

        validation_time = now or datetime.now(UTC)
        _validate_prepared_results_payload(payload, now=validation_time)
        provider_use = self._authorize(payload)
        event_identities = _result_event_identities(payload.scores)
        stored_payload, provenance = self._retain_raw_evidence(
            payload,
            provider_use=provider_use,
            validation_time=validation_time,
        )

        return self._write_results_transaction(
            payload=payload,
            stored_payload=stored_payload,
            provenance=provenance,
            event_identities=event_identities,
        )

    def _authorize(self, payload: PreparedOddsPayload | PreparedResultsPayload) -> ProviderUse:
        try:
            return self._provider_contracts.authorize_ingestion(
                provider=payload.provider,
                license_scope=payload.license_scope,
                license_version=payload.license_version,
                source_type=payload.source_type,
            )
        except ValueError:
            raise OddsLedgerValidationError("provider contract does not authorize this ingestion") from None

    def _retain_raw_evidence(
        self,
        payload: PreparedOddsPayload | PreparedResultsPayload,
        *,
        provider_use: ProviderUse,
        validation_time: datetime,
    ) -> tuple[StoredRawPayload, RawDataProvenance]:
        metadata = RawPayloadMetadata(
            provider=payload.provider,
            provider_record_id=f"receipt:{payload.receipt_sha256}",
            source_type=payload.source_type,
            captured_at=payload.captured_at,
            received_at=payload.received_at,
            schema_version=payload.schema_version,
            license_scope=provider_use.license_scope,
            license_version=provider_use.license_version,
            content_type=payload.content_type,
        )
        try:
            validate_private_payload_metadata_for_use(metadata, provider_use)
        except ValueError:
            raise OddsLedgerValidationError(
                "raw payload metadata is not authorized for this provider"
            ) from None
        stored_payload = self._store_raw_payload(payload.raw_payload, metadata)
        provenance = RawDataProvenance(
            provider=payload.provider,
            provider_record_id=metadata.provider_record_id,
            source_type=payload.source_type,
            payload_sha256=stored_payload.payload_sha256,
            payload_uri=stored_payload.payload_uri,
            captured_at=payload.captured_at,
            received_at=payload.received_at,
            schema_version=payload.schema_version,
            license_scope=provider_use.license_scope,
        )
        try:
            validate_raw_data_provenance(provenance, now=validation_time).require_valid(
                "raw provider evidence"
            )
        except ValueError:
            raise OddsLedgerValidationError("raw provider evidence failed validation") from None
        return stored_payload, provenance

    def _store_raw_payload(
        self, raw_payload: bytes, metadata: RawPayloadMetadata
    ) -> StoredRawPayload:
        expected_digest = hashlib.sha256(raw_payload).hexdigest()
        try:
            stored = self._raw_payload_store.store(
                raw_payload,
                metadata=metadata,
                stored_at=metadata.received_at,
            )
            if stored.payload_sha256 != expected_digest:
                raise OddsLedgerValidationError("raw payload store returned an unexpected digest")
            if stored.byte_count != len(raw_payload):
                raise OddsLedgerValidationError("raw payload store returned an unexpected byte count")
            if stored.metadata != metadata:
                raise OddsLedgerValidationError("raw payload store returned mismatched metadata")
            validate_private_payload_uri(stored.payload_uri, payload_sha256=expected_digest)
            return stored
        except OddsLedgerValidationError:
            raise
        except Exception:
            # The store implementation must not leak a signed URL or credential
            # in a worker-visible exception chain/message.
            raise OddsLedgerUnavailable("private raw payload storage failed") from None

    def _write_transaction(
        self,
        *,
        payload: PreparedOddsPayload,
        stored_payload: StoredRawPayload,
        provenance: RawDataProvenance,
        normalized_quotes: tuple[NormalizedQuote, ...],
        event_identities: tuple[_EventIdentity, ...],
    ) -> LedgerWriteResult:
        connection: _Connection | None = None
        try:
            connection = self._connection_factory(self._database_url)
            with connection.transaction():
                with connection.cursor() as cursor:
                    receipt_id = _insert_or_select_receipt(cursor, payload, stored_payload)
                    provenance_id = _insert_or_select_provenance(cursor, provenance, receipt_id)
                    event_ids, events_created, conflicts = _ensure_event_identities(cursor, event_identities)
                    if conflicts:
                        for conflict in conflicts:
                            _insert_event_identity_incident(cursor, conflict, provenance.provider)
                        _insert_provider_signal(
                            cursor,
                            payload=payload,
                            provenance=provenance,
                            status="blocked_event_identity",
                            source="odds_ledger",
                            counters={
                                "snapshots_created": 0,
                                "snapshots_replayed": 0,
                                "incidents_created": len(conflicts),
                            },
                        )
                        return LedgerWriteResult(
                            status="blocked_event_identity",
                            receipt_sha256=payload.receipt_sha256,
                            provenance_sha256=provenance.digest,
                            events_created=events_created,
                            snapshots_created=0,
                            snapshots_replayed=0,
                            provenance_links_created=0,
                            incidents_created=len(conflicts),
                        )

                    created, replayed, links_created = _insert_odds_snapshots(
                        cursor,
                        normalized_quotes=normalized_quotes,
                        event_ids=event_ids,
                        provenance_id=provenance_id,
                        payload_sha256=stored_payload.payload_sha256,
                        received_at=payload.received_at,
                        source_available_at=_payload_source_available_at(payload),
                    )
                    status = "accepted" if normalized_quotes else "accepted_empty"
                    _insert_provider_signal(
                        cursor,
                        payload=payload,
                        provenance=provenance,
                        status=status,
                        source="odds_ledger",
                        counters={
                            "snapshots_created": created,
                            "snapshots_replayed": replayed,
                            "incidents_created": 0,
                        },
                    )
                    return LedgerWriteResult(
                        status=status,
                        receipt_sha256=payload.receipt_sha256,
                        provenance_sha256=provenance.digest,
                        events_created=events_created,
                        snapshots_created=created,
                        snapshots_replayed=replayed,
                        provenance_links_created=links_created,
                        incidents_created=0,
                    )
        except OddsLedgerError:
            raise
        except Exception:
            raise OddsLedgerUnavailable("odds evidence ledger database transaction failed") from None
        finally:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    raise OddsLedgerUnavailable("odds evidence ledger connection cleanup failed") from None

    def _write_results_transaction(
        self,
        *,
        payload: PreparedResultsPayload,
        stored_payload: StoredRawPayload,
        provenance: RawDataProvenance,
        event_identities: tuple[_EventIdentity, ...],
    ) -> ResultsLedgerWriteResult:
        connection: _Connection | None = None
        try:
            connection = self._connection_factory(self._database_url)
            with connection.transaction():
                with connection.cursor() as cursor:
                    receipt_id = _insert_or_select_receipt(cursor, payload, stored_payload)
                    provenance_id = _insert_or_select_provenance(cursor, provenance, receipt_id)
                    event_ids, events_created, conflicts = _ensure_event_identities(
                        cursor,
                        event_identities,
                        allow_schedule_drift=True,
                        require_existing=any(
                            _result_event_key(score) != (score.provider, score.event_id)
                            for score in payload.scores
                        ),
                    )
                    if conflicts:
                        for conflict in conflicts:
                            _insert_event_identity_incident(cursor, conflict, provenance.provider)
                        _insert_provider_signal(
                            cursor,
                            payload=payload,
                            provenance=provenance,
                            status="blocked_event_identity",
                            source="results_ledger",
                            counters={
                                "results_created": 0,
                                "results_replayed": 0,
                                "incidents_created": len(conflicts),
                            },
                        )
                        return ResultsLedgerWriteResult(
                            status="blocked_event_identity",
                            receipt_sha256=payload.receipt_sha256,
                            provenance_sha256=provenance.digest,
                            events_created=events_created,
                            results_created=0,
                            results_replayed=0,
                            provenance_links_created=0,
                            incidents_created=len(conflicts),
                        )

                    created, replayed, links_created = _insert_event_results(
                        cursor,
                        scores=payload.scores,
                        event_ids=event_ids,
                        provenance_id=provenance_id,
                        payload_sha256=stored_payload.payload_sha256,
                        received_at=payload.received_at,
                        source_available_at=_payload_source_available_at(payload),
                    )
                    status = "accepted" if payload.scores else "accepted_empty"
                    _insert_provider_signal(
                        cursor,
                        payload=payload,
                        provenance=provenance,
                        status=status,
                        source="results_ledger",
                        counters={
                            "results_created": created,
                            "results_replayed": replayed,
                            "incidents_created": 0,
                        },
                    )
                    return ResultsLedgerWriteResult(
                        status=status,
                        receipt_sha256=payload.receipt_sha256,
                        provenance_sha256=provenance.digest,
                        events_created=events_created,
                        results_created=created,
                        results_replayed=replayed,
                        provenance_links_created=links_created,
                        incidents_created=0,
                    )
        except OddsLedgerError:
            raise
        except Exception:
            raise OddsLedgerUnavailable("result evidence ledger database transaction failed") from None
        finally:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    raise OddsLedgerUnavailable(
                        "result evidence ledger connection cleanup failed"
                    ) from None


def prepare_the_odds_api_payload(
    fetch: Any,
    *,
    license_scope: str,
    license_version: str,
    source_type: str = "odds",
    schema_version: str = "v4",
) -> PreparedOddsPayload:
    """Convert one already-fetched Odds API response into private evidence.

    This helper intentionally accepts the response object rather than a client
    or API key.  It performs no HTTP request.  A private worker can call it only
    after it has a reviewed provider contract and a real private object store.
    """

    raw_payload = getattr(fetch, "raw_payload", None)
    received_at = getattr(fetch, "received_at", None)
    quotes = getattr(fetch, "quotes", None)
    scope = getattr(fetch, "request_scope", None)
    if not isinstance(raw_payload, bytes) or not raw_payload:
        raise OddsLedgerValidationError("provider fetch does not contain raw response bytes")
    if not _aware(received_at):
        raise OddsLedgerValidationError("provider fetch does not contain a local receipt time")
    if not isinstance(quotes, list) or not all(isinstance(quote, RawOddsQuote) for quote in quotes):
        raise OddsLedgerValidationError("provider fetch does not contain a valid pregame quote collection")
    if scope is None:
        raise OddsLedgerValidationError("provider fetch does not contain a sanitized request scope")
    sport_key = getattr(scope, "sport_key", None)
    regions = getattr(scope, "regions", None)
    markets = getattr(scope, "markets", None)
    bookmakers = getattr(scope, "bookmakers", ())
    requested_snapshot_at = getattr(scope, "snapshot_at", None)
    fetch_source_available_at = getattr(fetch, "source_available_at", None)
    source_available_at = fetch_source_available_at or received_at
    if (
        not _nonempty_text(sport_key)
        or not isinstance(regions, tuple)
        or not isinstance(markets, tuple)
        or not isinstance(bookmakers, tuple)
    ):
        raise OddsLedgerValidationError("provider fetch request scope is incomplete")
    request_scope = [
        ("sport_key", sport_key),
        ("regions", ",".join(regions)),
        ("markets", ",".join(markets)),
    ]
    if bookmakers:
        request_scope.append(("bookmakers", ",".join(bookmakers)))
    if requested_snapshot_at is not None:
        if fetch_source_available_at is None:
            raise OddsLedgerValidationError(
                "historical provider fetch does not contain source availability"
            )
        if not _aware(requested_snapshot_at) or not _aware(source_available_at):
            raise OddsLedgerValidationError("historical provider timestamps are invalid")
        if source_available_at > requested_snapshot_at:
            raise OddsLedgerValidationError(
                "historical provider availability is after the requested snapshot"
            )
        request_scope.append(("snapshot_at", _utc(requested_snapshot_at)))
    # A valid empty response is still provider evidence.  With no provider
    # quote timestamp available, bind it to the provider snapshot time (or the
    # local receipt for live odds) and persist ``accepted_empty`` explicitly.
    captured_at = max((quote.captured_at for quote in quotes), default=source_available_at)
    return PreparedOddsPayload(
        provider="the_odds_api",
        source_type=source_type,
        raw_payload=raw_payload,
        captured_at=captured_at,
        received_at=received_at,
        schema_version=schema_version,
        license_scope=license_scope,
        license_version=license_version,
        quotes=tuple(quotes),
        request_scope=tuple(request_scope),
        provider_response_status=200,
        requests_remaining=getattr(fetch, "requests_remaining", None),
        requests_used=getattr(fetch, "requests_used", None),
        request_cost=getattr(fetch, "request_cost", None),
        source_available_at=source_available_at,
    )


def prepare_the_odds_api_results_payload(
    fetch: Any,
    *,
    license_scope: str,
    license_version: str,
    source_type: str = "result",
    schema_version: str = "v4",
) -> PreparedResultsPayload:
    """Convert one already-fetched Odds API scores response into private evidence."""

    from sam_analytics.providers.the_odds_api import CompletedScore

    raw_payload = getattr(fetch, "raw_payload", None)
    received_at = getattr(fetch, "received_at", None)
    scores = getattr(fetch, "scores", None)
    scope = getattr(fetch, "request_scope", None)
    if not isinstance(raw_payload, bytes) or not raw_payload:
        raise OddsLedgerValidationError("provider fetch does not contain raw response bytes")
    if not _aware(received_at):
        raise OddsLedgerValidationError("provider fetch does not contain a local receipt time")
    if not isinstance(scores, tuple) or not all(
        isinstance(score, CompletedScore) for score in scores
    ):
        raise OddsLedgerValidationError("provider fetch does not contain valid completed scores")
    _validated_completed_scores(scores, provider="the_odds_api")
    if scope is None:
        raise OddsLedgerValidationError("provider fetch does not contain a sanitized request scope")
    sport_key = getattr(scope, "sport_key", None)
    days_from = getattr(scope, "days_from", None)
    if (
        not _nonempty_text(sport_key)
        or isinstance(days_from, bool)
        or not isinstance(days_from, int)
        or not 1 <= days_from <= 3
    ):
        raise OddsLedgerValidationError("provider fetch request scope is incomplete")
    if any(score.sport != sport_key for score in scores):
        raise OddsLedgerValidationError("provider scores do not match the requested sport")

    captured_at = max((score.last_update for score in scores), default=received_at)
    return PreparedResultsPayload(
        provider="the_odds_api",
        source_type=source_type,
        raw_payload=raw_payload,
        captured_at=captured_at,
        received_at=received_at,
        schema_version=schema_version,
        license_scope=license_scope,
        license_version=license_version,
        scores=scores,
        request_scope=(("sport_key", sport_key), ("days_from", str(days_from))),
        provider_response_status=200,
        requests_remaining=getattr(fetch, "requests_remaining", None),
        requests_used=getattr(fetch, "requests_used", None),
        request_cost=getattr(fetch, "request_cost", None),
        source_available_at=received_at,
    )


def _connect_postgres(database_url: str) -> _Connection:
    try:
        import psycopg

        return psycopg.connect(
            database_url,
            application_name="sam-analytics-odds-ledger",
            connect_timeout=5,
            options="-c statement_timeout=10000",
        )
    except Exception:
        raise OddsLedgerUnavailable("odds evidence ledger database is unavailable") from None


def _validate_prepared_payload(payload: PreparedOddsPayload, *, now: datetime) -> None:
    if not isinstance(payload, PreparedOddsPayload):
        raise OddsLedgerValidationError("prepared odds payload is required")
    _validate_payload_envelope(payload, now=now, kind="odds")
    if not isinstance(payload.quotes, tuple) or not all(
        isinstance(quote, RawOddsQuote) for quote in payload.quotes
    ):
        raise OddsLedgerValidationError(
            "a provider payload must contain a valid pregame quote collection"
        )
    _validated_request_scope(payload.request_scope)


def _validate_payload_envelope(
    payload: PreparedOddsPayload | PreparedResultsPayload,
    *,
    now: datetime,
    kind: str,
) -> None:
    for field in ("provider", "source_type", "schema_version", "license_scope", "license_version", "content_type"):
        if not _nonempty_text(getattr(payload, field)):
            raise OddsLedgerValidationError(f"prepared {kind} payload has a required empty field")
    if not isinstance(payload.raw_payload, bytes) or not payload.raw_payload:
        raise OddsLedgerValidationError("provider response must be non-empty bytes")
    source_available_at = _payload_source_available_at(payload)
    if (
        not _aware(payload.captured_at)
        or not _aware(payload.received_at)
        or not _aware(source_available_at)
        or not _aware(now)
    ):
        raise OddsLedgerValidationError("provider and validation timestamps must be timezone-aware")
    if payload.captured_at > source_available_at + _MAX_PROVIDER_CLOCK_SKEW:
        raise OddsLedgerValidationError("provider payload capture time is after source availability")
    if source_available_at > payload.received_at + _MAX_PROVIDER_CLOCK_SKEW:
        raise OddsLedgerValidationError("source availability is after local receipt")
    if payload.received_at > now:
        raise OddsLedgerValidationError("provider payload receipt time cannot be in the future")
    if not isinstance(payload.provider_response_status, int) or not 100 <= payload.provider_response_status <= 599:
        raise OddsLedgerValidationError("provider response status must be an HTTP status code")
    for value in (payload.requests_remaining, payload.requests_used, payload.request_cost):
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
            raise OddsLedgerValidationError("provider quota values must be non-negative integers or None")


def _validate_prepared_results_payload(
    payload: PreparedResultsPayload, *, now: datetime
) -> None:
    if not isinstance(payload, PreparedResultsPayload):
        raise OddsLedgerValidationError("prepared results payload is required")
    _validate_payload_envelope(payload, now=now, kind="results")
    scores = _validated_completed_scores(payload.scores, provider=payload.provider)
    for score in scores:
        source_available_at = _score_source_available_at(score, payload)
        if not _aware(source_available_at):
            raise OddsLedgerValidationError("provider result availability must be timezone-aware")
        if score.last_update > source_available_at + _MAX_PROVIDER_CLOCK_SKEW:
            raise OddsLedgerValidationError("provider result update is after source availability")
        if source_available_at > payload.received_at + _MAX_PROVIDER_CLOCK_SKEW:
            raise OddsLedgerValidationError("provider result availability is after local receipt")
    _validated_request_scope(payload.request_scope)


def _validated_completed_scores(
    scores: object, *, provider: str
) -> tuple[CompletedScore, ...]:
    from sam_analytics.providers.the_odds_api import CompletedScore

    if not isinstance(scores, tuple) or not all(
        isinstance(score, CompletedScore) for score in scores
    ):
        raise OddsLedgerValidationError("a provider payload must contain valid completed scores")
    provider_events: set[tuple[str, str]] = set()
    matched_events: set[tuple[str, str]] = set()
    for score in scores:
        for field in ("provider", "event_id", "sport", "league", "home_team", "away_team"):
            if not _nonempty_text(getattr(score, field)):
                raise OddsLedgerValidationError("completed score is missing event metadata")
        if score.provider != provider:
            raise OddsLedgerValidationError("completed score provider does not match its payload")
        matched_provider = score.matched_event_provider
        matched_event_id = score.matched_provider_event_id
        if (matched_provider is None) != (matched_event_id is None):
            raise OddsLedgerValidationError("completed score event mapping must be complete")
        if matched_provider is not None and (
            not _nonempty_text(matched_provider) or not _nonempty_text(matched_event_id)
        ):
            raise OddsLedgerValidationError("completed score event mapping is invalid")
        if score.source_available_at is not None and not _aware(score.source_available_at):
            raise OddsLedgerValidationError("completed score source availability must be timezone-aware")
        if (
            matched_provider is not None
            and matched_provider != score.provider
            and score.source_available_at != score.last_update
        ):
            raise OddsLedgerValidationError(
                "cross-provider result availability must match its immutable update time"
            )
        if score.home_team == score.away_team:
            raise OddsLedgerValidationError("completed score teams must be distinct")
        if not _aware(score.commence_time) or not _aware(score.last_update):
            raise OddsLedgerValidationError("completed score timestamps must be timezone-aware")
        if score.last_update < score.commence_time:
            raise OddsLedgerValidationError("completed score update cannot predate event start")
        for value in (score.home_score, score.away_score):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise OddsLedgerValidationError("completed scores must be non-negative integers")
        event_key = (score.provider, score.event_id)
        if event_key in provider_events:
            raise OddsLedgerValidationError("provider payload contains duplicate completed events")
        provider_events.add(event_key)
        matched_event_key = _result_event_key(score)
        if matched_event_key in matched_events:
            raise OddsLedgerValidationError("provider payload maps multiple results to one event")
        matched_events.add(matched_event_key)
    return scores


def _validated_request_scope(scope: tuple[tuple[str, str], ...]) -> tuple[tuple[str, str], ...]:
    if not isinstance(scope, tuple):
        raise OddsLedgerValidationError("request scope must be an immutable tuple")
    validated: list[tuple[str, str]] = []
    keys: set[str] = set()
    for item in scope:
        if not isinstance(item, tuple) or len(item) != 2:
            raise OddsLedgerValidationError("request scope entries must be text pairs")
        key, value = item
        if not _nonempty_text(key) or not _nonempty_text(value):
            raise OddsLedgerValidationError("request scope entries must be non-empty text")
        normalized_key = key.strip().lower()
        if (
            normalized_key in keys
            or normalized_key in _SENSITIVE_SCOPE_NAMES
            or _SENSITIVE_SCOPE_KEY_RE.search(normalized_key)
        ):
            raise OddsLedgerValidationError("request scope cannot contain credentials or duplicate fields")
        keys.add(normalized_key)
        validated.append((normalized_key, value.strip()))
    return tuple(sorted(validated))


def _event_identities(quotes: tuple[NormalizedQuote, ...]) -> tuple[_EventIdentity, ...]:
    identities: dict[tuple[str, str], _EventIdentity] = {}
    for quote in quotes:
        raw = quote.raw
        for field in ("league", "home_team", "away_team", "bookmaker"):
            if not _nonempty_text(getattr(raw, field, None)):
                raise OddsLedgerValidationError("odds quote is missing immutable event or bookmaker metadata")
        identity = _EventIdentity(
            provider=raw.provider,
            provider_event_id=raw.event_id,
            sport=raw.sport,
            league=raw.league,
            starts_at=raw.starts_at,
            home_team=raw.home_team,
            away_team=raw.away_team,
        )
        key = (identity.provider, identity.provider_event_id)
        prior = identities.get(key)
        if prior is not None and prior != identity:
            raise OddsLedgerValidationError("provider payload contains conflicting identities for one event")
        identities[key] = identity
    return tuple(identities.values())


def _result_event_identities(scores: tuple[CompletedScore, ...]) -> tuple[_EventIdentity, ...]:
    return tuple(
        _EventIdentity(
            provider=_result_event_key(score)[0],
            provider_event_id=_result_event_key(score)[1],
            sport=score.sport,
            league=score.league,
            starts_at=score.commence_time,
            home_team=score.home_team,
            away_team=score.away_team,
        )
        for score in scores
    )


def _insert_or_select_receipt(
    cursor: _Cursor,
    payload: PreparedOddsPayload | PreparedResultsPayload,
    stored_payload: StoredRawPayload,
) -> Any:
    cursor.execute(
        """
        INSERT INTO provider_payload_receipt (
            provider, source_type, request_fingerprint_sha256, payload_sha256,
            payload_uri, captured_at, received_at, source_available_at,
            provider_response_status,
            payload_bytes, provider_quota_remaining, provider_quota_used,
            provider_quota_last, schema_version, license_scope, license_version,
            receipt_sha256
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s
        ) ON CONFLICT (receipt_sha256) DO NOTHING
        RETURNING id
        """,
        (
            payload.provider,
            payload.source_type,
            payload.request_fingerprint_sha256,
            stored_payload.payload_sha256,
            stored_payload.payload_uri,
            payload.captured_at,
            payload.received_at,
            _payload_source_available_at(payload),
            payload.provider_response_status,
            stored_payload.byte_count,
            payload.requests_remaining,
            payload.requests_used,
            payload.request_cost,
            payload.schema_version,
            payload.license_scope,
            payload.license_version,
            payload.receipt_sha256,
        ),
    )
    row = cursor.fetchone()
    if row is not None:
        return row[0]
    cursor.execute("SELECT id FROM provider_payload_receipt WHERE receipt_sha256 = %s", (payload.receipt_sha256,))
    existing = cursor.fetchone()
    if existing is None:
        raise OddsLedgerUnavailable("provider payload receipt could not be read after insert")
    return existing[0]


def _insert_or_select_provenance(
    cursor: _Cursor, provenance: RawDataProvenance, receipt_id: Any
) -> Any:
    cursor.execute(
        """
        INSERT INTO raw_data_provenance (
            provider, provider_record_id, source_type, payload_sha256, payload_uri,
            captured_at, received_at, schema_version, license_scope,
            provenance_sha256, provider_payload_receipt_id
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (provenance_sha256) DO NOTHING
        RETURNING id
        """,
        (
            provenance.provider,
            provenance.provider_record_id,
            provenance.source_type,
            provenance.payload_sha256,
            provenance.payload_uri,
            provenance.captured_at,
            provenance.received_at,
            provenance.schema_version,
            provenance.license_scope,
            provenance.digest,
            receipt_id,
        ),
    )
    row = cursor.fetchone()
    if row is not None:
        return row[0]
    cursor.execute("SELECT id FROM raw_data_provenance WHERE provenance_sha256 = %s", (provenance.digest,))
    existing = cursor.fetchone()
    if existing is None:
        raise OddsLedgerUnavailable("raw provenance could not be read after insert")
    return existing[0]


def _ensure_event_identities(
    cursor: _Cursor,
    identities: tuple[_EventIdentity, ...],
    *,
    allow_schedule_drift: bool = False,
    require_existing: bool = False,
) -> tuple[dict[tuple[str, str], Any], int, tuple[_EventIdentity, ...]]:
    event_ids: dict[tuple[str, str], Any] = {}
    conflicts: list[_EventIdentity] = []
    created = 0
    for identity in identities:
        if not require_existing:
            cursor.execute(
                """
                INSERT INTO sports_event (
                    provider, provider_event_id, sport, league, starts_at, home_team, away_team
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (provider, provider_event_id) DO NOTHING
                RETURNING id
                """,
                (
                    identity.provider,
                    identity.provider_event_id,
                    identity.sport,
                    identity.league,
                    identity.starts_at,
                    identity.home_team,
                    identity.away_team,
                ),
            )
            inserted = cursor.fetchone()
            if inserted is not None:
                event_ids[(identity.provider, identity.provider_event_id)] = inserted[0]
                created += 1
                continue
        cursor.execute(
            """
            SELECT id, sport, league, starts_at, home_team, away_team
            FROM sports_event
            WHERE provider = %s AND provider_event_id = %s
            FOR KEY SHARE
            """,
            (identity.provider, identity.provider_event_id),
        )
        existing = cursor.fetchone()
        if existing is None:
            if require_existing:
                raise OddsLedgerValidationError(
                    "completed score references an unavailable matched event"
                )
            raise OddsLedgerUnavailable("event identity could not be read after insert")
        event_ids[(identity.provider, identity.provider_event_id)] = existing[0]
        if allow_schedule_drift:
            if require_existing:
                candidate_fields = (
                    identity.sport,
                    identity.starts_at,
                    identity.home_team,
                    identity.away_team,
                )
                existing_fields = (existing[1], existing[3], existing[4], existing[5])
            else:
                candidate_fields = (identity.sport, identity.home_team, identity.away_team)
                existing_fields = (existing[1], existing[4], existing[5])
        else:
            candidate_fields = (
                identity.sport,
                identity.league,
                identity.starts_at,
                identity.home_team,
                identity.away_team,
            )
            existing_fields = tuple(existing[1:])
        if existing_fields != candidate_fields:
            conflicts.append(identity)
    return event_ids, created, tuple(conflicts)


def _insert_event_identity_incident(
    cursor: _Cursor, identity: _EventIdentity, provider: str
) -> None:
    details = {
        "reason": "immutable_event_identity_conflict",
        "provider_event_id": identity.provider_event_id,
        "candidate": {
            "sport": identity.sport,
            "league": identity.league,
            "starts_at": _utc(identity.starts_at),
            "home_team": identity.home_team,
            "away_team": identity.away_team,
        },
    }
    cursor.execute(
        """
        INSERT INTO data_quality_incident (severity, category, provider, details)
        VALUES (%s, %s, %s, %s::jsonb)
        """,
        ("critical", "event_identity_conflict", provider, json.dumps(details, sort_keys=True)),
    )


def _insert_odds_snapshots(
    cursor: _Cursor,
    *,
    normalized_quotes: tuple[NormalizedQuote, ...],
    event_ids: Mapping[tuple[str, str], Any],
    provenance_id: Any,
    payload_sha256: str,
    received_at: datetime,
    source_available_at: datetime,
) -> tuple[int, int, int]:
    created = 0
    replayed = 0
    provenance_links_created = 0
    for quote in normalized_quotes:
        raw = quote.raw
        event_id = event_ids[(raw.provider, raw.event_id)]
        cursor.execute(
            """
            INSERT INTO odds_snapshot (
                event_id, provider, provider_quote_id, bookmaker, market, selection,
                line, american_odds, decimal_odds, captured_at, received_at,
                source_available_at, source_payload_sha256, idempotency_key,
                primary_provenance_id
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            ) ON CONFLICT (idempotency_key) DO NOTHING
            RETURNING id
            """,
            (
                event_id,
                raw.provider,
                raw.provider_quote_id,
                raw.bookmaker,
                raw.market,
                raw.selection,
                raw.line,
                raw.american_odds,
                quote.decimal_odds,
                raw.captured_at,
                received_at,
                source_available_at,
                payload_sha256,
                quote.idempotency_key,
                provenance_id,
            ),
        )
        snapshot = cursor.fetchone()
        if snapshot is None:
            replayed += 1
            cursor.execute("SELECT id FROM odds_snapshot WHERE idempotency_key = %s", (quote.idempotency_key,))
            snapshot = cursor.fetchone()
            if snapshot is None:
                raise OddsLedgerUnavailable("odds snapshot could not be read after insert")
        else:
            created += 1
        cursor.execute(
            """
            INSERT INTO odds_snapshot_provenance (odds_snapshot_id, provenance_id)
            VALUES (%s, %s)
            ON CONFLICT DO NOTHING
            RETURNING odds_snapshot_id
            """,
            (snapshot[0], provenance_id),
        )
        if cursor.fetchone() is not None:
            provenance_links_created += 1
    return created, replayed, provenance_links_created


def _insert_event_results(
    cursor: _Cursor,
    *,
    scores: tuple[CompletedScore, ...],
    event_ids: Mapping[tuple[str, str], Any],
    provenance_id: Any,
    payload_sha256: str,
    received_at: datetime,
    source_available_at: datetime,
) -> tuple[int, int, int]:
    created = 0
    replayed = 0
    provenance_links_created = 0
    for score in scores:
        event_id = event_ids[_result_event_key(score)]
        provider_result_id = _provider_result_id(score)
        score_available_at = score.source_available_at or source_available_at
        cursor.execute(
            """
            INSERT INTO event_result (
                event_id, provider, provider_result_id, settled_at, home_score,
                away_score, source_payload_sha256, received_at, source_available_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (provider, provider_result_id) DO NOTHING
            RETURNING id
            """,
            (
                event_id,
                score.provider,
                provider_result_id,
                score.last_update,
                score.home_score,
                score.away_score,
                payload_sha256,
                received_at,
                score_available_at,
            ),
        )
        result = cursor.fetchone()
        if result is None:
            replayed += 1
            cursor.execute(
                "SELECT id FROM event_result WHERE provider = %s AND provider_result_id = %s",
                (score.provider, provider_result_id),
            )
            result = cursor.fetchone()
            if result is None:
                raise OddsLedgerUnavailable("event result could not be read after insert")
        else:
            created += 1
        cursor.execute(
            """
            INSERT INTO event_result_provenance (event_result_id, provenance_id)
            VALUES (%s, %s)
            ON CONFLICT DO NOTHING
            RETURNING event_result_id
            """,
            (result[0], provenance_id),
        )
        if cursor.fetchone() is not None:
            provenance_links_created += 1
    return created, replayed, provenance_links_created


def _provider_result_id(score: CompletedScore) -> str:
    identity = {
        "provider_event_id": score.event_id,
        "last_update": _utc(score.last_update),
        "home_score": score.home_score,
        "away_score": score.away_score,
    }
    # Preserve the established ID for ordinary same-provider results.  The
    # extra identity fields are needed only for historical/cross-provider
    # evidence; hashing explicit nulls here would replay every existing live
    # result as a new immutable version after this release.
    if (
        score.matched_event_provider is not None
        or score.matched_provider_event_id is not None
        or score.source_available_at is not None
    ):
        identity.update(
            {
                "matched_event_provider": score.matched_event_provider,
                "matched_provider_event_id": score.matched_provider_event_id,
                "source_available_at": (
                    _utc(score.source_available_at)
                    if score.source_available_at is not None
                    else None
                ),
            }
        )
    return _canonical_sha256(identity)


def _insert_provider_signal(
    cursor: _Cursor,
    *,
    payload: PreparedOddsPayload | PreparedResultsPayload,
    provenance: RawDataProvenance,
    status: str,
    source: str,
    counters: Mapping[str, int],
) -> None:
    safe_signal = {
        "status": status,
        "source_type": payload.source_type,
        "schema_version": payload.schema_version,
        "license_scope": payload.license_scope,
        "license_version": payload.license_version,
        "requests_remaining": payload.requests_remaining,
        "requests_used": payload.requests_used,
        "request_cost": payload.request_cost,
    }
    safe_signal.update(counters)
    cursor.execute(
        """
        INSERT INTO operational_signal (
            signal_type, observed_at, received_at, source, provenance_sha256, payload
        ) VALUES (%s, %s, %s, %s, %s, %s::jsonb)
        """,
        (
            "provider",
            payload.captured_at,
            payload.received_at,
            source,
            provenance.digest,
            json.dumps(safe_signal, sort_keys=True),
        ),
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _payload_request_fingerprint(
    payload: PreparedOddsPayload | PreparedResultsPayload,
) -> str:
    return _canonical_sha256(dict(_validated_request_scope(payload.request_scope)))


def _payload_receipt_sha256(payload: PreparedOddsPayload | PreparedResultsPayload) -> str:
    return _canonical_sha256(
        {
            "provider": payload.provider,
            "source_type": payload.source_type,
            "request_fingerprint_sha256": payload.request_fingerprint_sha256,
            "payload_sha256": hashlib.sha256(payload.raw_payload).hexdigest(),
            "captured_at": _utc(payload.captured_at),
            "received_at": _utc(payload.received_at),
            "source_available_at": _utc(_payload_source_available_at(payload)),
            "provider_response_status": payload.provider_response_status,
            "payload_bytes": len(payload.raw_payload),
            "requests_remaining": payload.requests_remaining,
            "requests_used": payload.requests_used,
            "request_cost": payload.request_cost,
            "schema_version": payload.schema_version,
            "license_scope": payload.license_scope,
            "license_version": payload.license_version,
        }
    )


def _payload_source_available_at(
    payload: PreparedOddsPayload | PreparedResultsPayload,
) -> datetime:
    return payload.source_available_at or payload.received_at


def _score_source_available_at(score: CompletedScore, payload: PreparedResultsPayload) -> datetime:
    return score.source_available_at or _payload_source_available_at(payload)


def _result_event_key(score: CompletedScore) -> tuple[str, str]:
    return (
        score.matched_event_provider or score.provider,
        score.matched_provider_event_id or score.event_id,
    )


def _aware(value: object) -> bool:
    return isinstance(value, datetime) and value.tzinfo is not None and value.utcoffset() is not None


def _utc(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _nonempty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())
