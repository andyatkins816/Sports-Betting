"""Provider-neutral odds-ingestion contracts and quality checks."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Protocol

from .odds import american_to_decimal


@dataclass(frozen=True)
class RawOddsQuote:
    """One provider price with enough context to reconstruct its market.

    The four trailing metadata fields deliberately remain optional for
    backwards compatibility with existing provider-neutral importers.  A
    production ledger can require them before persistence; first-party
    adapters should always populate them.
    """

    provider: str
    provider_quote_id: str
    event_id: str
    sport: str
    market: str
    selection: str
    american_odds: float
    line: float | None
    captured_at: datetime
    starts_at: datetime
    bookmaker: str | None = None
    league: str | None = None
    home_team: str | None = None
    away_team: str | None = None


@dataclass(frozen=True)
class NormalizedQuote:
    raw: RawOddsQuote
    decimal_odds: float
    idempotency_key: str


class OddsProvider(Protocol):
    """An adapter must preserve provider IDs and provider timestamps."""

    def fetch_quotes(self, sport: str) -> Iterable[RawOddsQuote]:
        ...


def normalize_quotes(quotes: Iterable[RawOddsQuote], *, now: datetime | None = None) -> List[NormalizedQuote]:
    now = now or datetime.now(timezone.utc)
    if not _is_aware_datetime(now):
        raise ValueError("now must be timezone-aware")
    normalized: List[NormalizedQuote] = []
    for quote in quotes:
        _validate_quote(quote, now)
        fingerprint = _canonical_quote_fingerprint(quote)
        normalized.append(
            NormalizedQuote(
                raw=quote,
                decimal_odds=american_to_decimal(quote.american_odds),
                idempotency_key=hashlib.sha256(fingerprint).hexdigest(),
            )
        )
    return normalized


def _validate_quote(quote: RawOddsQuote, now: datetime) -> None:
    if not isinstance(quote, RawOddsQuote):
        raise ValueError("quotes must use the RawOddsQuote contract")
    if not all(
        _nonempty_text(value)
        for value in (
            quote.provider,
            quote.provider_quote_id,
            quote.event_id,
            quote.sport,
            quote.market,
            quote.selection,
        )
    ):
        raise ValueError("provider, quote, event, sport, market, and selection identifiers are required")
    for field in ("bookmaker", "league", "home_team", "away_team"):
        value = getattr(quote, field)
        if value is not None and not _nonempty_text(value):
            raise ValueError(f"{field} must be a non-empty string or None")
    if not _is_aware_datetime(quote.captured_at) or not _is_aware_datetime(quote.starts_at):
        raise ValueError("provider timestamps must be timezone-aware")
    _require_finite_number(quote.american_odds, "American odds")
    if float(quote.american_odds) == 0.0:
        raise ValueError("American odds must be non-zero")
    # The immutable ledger's existing `odds_snapshot.american_odds` column is
    # an INTEGER. Reject a fractional value before raw evidence is retained or
    # a database transaction starts, rather than turning a schema mismatch
    # into a late, ambiguous ingestion failure.
    if not float(quote.american_odds).is_integer():
        raise ValueError("American odds must be whole-number values")
    if quote.line is not None:
        _require_finite_number(quote.line, "market line")
    if quote.captured_at > now:
        raise ValueError("provider quote timestamp is in the future")
    if quote.starts_at <= quote.captured_at:
        raise ValueError("cannot ingest an in-play or completed event through pregame pipeline")
    if quote.starts_at <= now:
        raise ValueError("cannot ingest an in-play or completed event through pregame pipeline")


def _canonical_quote_fingerprint(quote: RawOddsQuote) -> bytes:
    """Return a canonical, price-sensitive identity for an immutable quote.

    Provider feeds sometimes amend a price or line while retaining a source
    update timestamp.  Including both values ensures those corrections are
    retained instead of being silently treated as a duplicate.  Canonical
    JSON avoids delimiter collisions and normalizes equivalent UTC offsets.
    """

    value = {
        "american_odds": float(quote.american_odds),
        "away_team": quote.away_team.strip() if quote.away_team is not None else None,
        "bookmaker": quote.bookmaker.strip() if quote.bookmaker is not None else None,
        "captured_at": _utc_timestamp(quote.captured_at),
        "event_id": quote.event_id.strip(),
        "home_team": quote.home_team.strip() if quote.home_team is not None else None,
        "league": quote.league.strip() if quote.league is not None else None,
        "line": float(quote.line) if quote.line is not None else None,
        "market": quote.market.strip(),
        "provider": quote.provider.strip(),
        "provider_quote_id": quote.provider_quote_id.strip(),
        "selection": quote.selection.strip(),
        "sport": quote.sport.strip(),
        "starts_at": _utc_timestamp(quote.starts_at),
    }
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _is_aware_datetime(value: object) -> bool:
    return isinstance(value, datetime) and value.tzinfo is not None and value.utcoffset() is not None


def _utc_timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _nonempty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _require_finite_number(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
