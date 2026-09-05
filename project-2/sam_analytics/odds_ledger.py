"""Transactional, evidence-first persistence for licensed pregame odds.

This module is deliberately *not* a scheduler or a provider client.  A caller
first obtains a complete provider response, builds :class:`PreparedOddsPayload`,
and hands it to this ledger.  The ledger then enforces the important ordering:

1. validate the approved provider contract and every pregame quote;
2. put the unmodified response in private, content-addressed storage;
3. atomically write the receipt, raw provenance, event identities, snapshots,
   provenance links, and a safe operational signal to PostgreSQL.

Nothing in this module returns a provider credential, raw response body, or a
private object URL.  It also intentionally has no network polling function;
the future worker must be explicitly enabled only after this persistence
boundary and its object-storage configuration are verified.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

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

    @property
    def request_fingerprint_sha256(self) -> str:
        return _canonical_sha256(dict(_validated_request_scope(self.request_scope)))

    @property
    def receipt_sha256(self) -> str:
        """Stable identity for this exact provider receipt, without its body."""

        return _canonical_sha256(
            {
                "provider": self.provider,
                "source_type": self.source_type,
                "request_fingerprint_sha256": self.request_fingerprint_sha256,
                "payload_sha256": hashlib.sha256(self.raw_payload).hexdigest(),
                "captured_at": _utc(self.captured_at),
                "received_at": _utc(self.received_at),
                "provider_response_status": self.provider_response_status,
                "payload_bytes": len(self.raw_payload),
                "requests_remaining": self.requests_remaining,
                "requests_used": self.requests_used,
                "request_cost": self.request_cost,
                "schema_version": self.schema_version,
                "license_scope": self.license_scope,
                "license_version": self.license_version,
            }
        )


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

    def __enter__(self) -> "_Cursor":
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

        validation_time = now or datetime.now(timezone.utc)
        _validate_prepared_payload(payload, now=validation_time)
        provider_use = self._authorize(payload)
        try:
            normalized_quotes = tuple(normalize_quotes(payload.quotes, now=validation_time))
        except ValueError:
            raise OddsLedgerValidationError("provider quotes failed pregame validation") from None
        event_identities = _event_identities(normalized_quotes)

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
            raise OddsLedgerValidationError("raw payload metadata is not authorized for this provider") from None
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

        return self._write_transaction(
            payload=payload,
            provider_use=provider_use,
            stored_payload=stored_payload,
            provenance=provenance,
            normalized_quotes=normalized_quotes,
            event_identities=event_identities,
        )

    def _authorize(self, payload: PreparedOddsPayload) -> ProviderUse:
        try:
            return self._provider_contracts.authorize_ingestion(
                provider=payload.provider,
                license_scope=payload.license_scope,
                license_version=payload.license_version,
                source_type=payload.source_type,
            )
        except ValueError:
            raise OddsLedgerValidationError("provider contract does not authorize this ingestion") from None

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
        provider_use: ProviderUse,
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
                            snapshots_created=0,
                            snapshots_replayed=0,
                            incidents_created=len(conflicts),
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
                    )
                    _insert_provider_signal(
                        cursor,
                        payload=payload,
                        provenance=provenance,
                        status="accepted",
                        snapshots_created=created,
                        snapshots_replayed=replayed,
                        incidents_created=0,
                    )
                    return LedgerWriteResult(
                        status="accepted",
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
    or API key.  It performs no HTTP request.  A future worker can call it only
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
    if not isinstance(quotes, list) or not quotes or not all(isinstance(quote, RawOddsQuote) for quote in quotes):
        raise OddsLedgerValidationError("provider fetch does not contain pregame quotes")
    if scope is None:
        raise OddsLedgerValidationError("provider fetch does not contain a sanitized request scope")
    sport_key = getattr(scope, "sport_key", None)
    regions = getattr(scope, "regions", None)
    markets = getattr(scope, "markets", None)
    bookmakers = getattr(scope, "bookmakers", ())
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
    captured_at = max(quote.captured_at for quote in quotes)
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
    for field in ("provider", "source_type", "schema_version", "license_scope", "license_version", "content_type"):
        if not _nonempty_text(getattr(payload, field)):
            raise OddsLedgerValidationError("prepared odds payload has a required empty field")
    if not isinstance(payload.raw_payload, bytes) or not payload.raw_payload:
        raise OddsLedgerValidationError("provider response must be non-empty bytes")
    if not _aware(payload.captured_at) or not _aware(payload.received_at) or not _aware(now):
        raise OddsLedgerValidationError("provider and validation timestamps must be timezone-aware")
    if payload.captured_at > payload.received_at + _MAX_PROVIDER_CLOCK_SKEW:
        raise OddsLedgerValidationError("provider payload capture time is after local receipt")
    if payload.received_at > now:
        raise OddsLedgerValidationError("provider payload receipt time cannot be in the future")
    if not isinstance(payload.provider_response_status, int) or not 100 <= payload.provider_response_status <= 599:
        raise OddsLedgerValidationError("provider response status must be an HTTP status code")
    for value in (payload.requests_remaining, payload.requests_used, payload.request_cost):
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
            raise OddsLedgerValidationError("provider quota values must be non-negative integers or None")
    if not isinstance(payload.quotes, tuple) or not payload.quotes:
        raise OddsLedgerValidationError("a provider payload must contain at least one pregame quote")
    _validated_request_scope(payload.request_scope)


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


def _insert_or_select_receipt(
    cursor: _Cursor, payload: PreparedOddsPayload, stored_payload: StoredRawPayload
) -> Any:
    cursor.execute(
        """
        INSERT INTO provider_payload_receipt (
            provider, source_type, request_fingerprint_sha256, payload_sha256,
            payload_uri, captured_at, received_at, provider_response_status,
            payload_bytes, provider_quota_remaining, provider_quota_used,
            provider_quota_last, schema_version, license_scope, license_version,
            receipt_sha256
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
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
    cursor: _Cursor, identities: tuple[_EventIdentity, ...]
) -> tuple[dict[tuple[str, str], Any], int, tuple[_EventIdentity, ...]]:
    event_ids: dict[tuple[str, str], Any] = {}
    conflicts: list[_EventIdentity] = []
    created = 0
    for identity in identities:
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
            raise OddsLedgerUnavailable("event identity could not be read after insert")
        event_ids[(identity.provider, identity.provider_event_id)] = existing[0]
        candidate_fields = (
            identity.sport,
            identity.league,
            identity.starts_at,
            identity.home_team,
            identity.away_team,
        )
        if tuple(existing[1:]) != candidate_fields:
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
                source_payload_sha256, idempotency_key, primary_provenance_id
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
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


def _insert_provider_signal(
    cursor: _Cursor,
    *,
    payload: PreparedOddsPayload,
    provenance: RawDataProvenance,
    status: str,
    snapshots_created: int,
    snapshots_replayed: int,
    incidents_created: int,
) -> None:
    safe_signal = {
        "status": status,
        "source_type": payload.source_type,
        "schema_version": payload.schema_version,
        "license_scope": payload.license_scope,
        "license_version": payload.license_version,
        "snapshots_created": snapshots_created,
        "snapshots_replayed": snapshots_replayed,
        "incidents_created": incidents_created,
        "requests_remaining": payload.requests_remaining,
        "requests_used": payload.requests_used,
        "request_cost": payload.request_cost,
    }
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
            "odds_ledger",
            provenance.digest,
            json.dumps(safe_signal, sort_keys=True),
        ),
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _aware(value: object) -> bool:
    return isinstance(value, datetime) and value.tzinfo is not None and value.utcoffset() is not None


def _utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _nonempty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())
