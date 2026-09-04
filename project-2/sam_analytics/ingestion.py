"""Provider-neutral odds-ingestion contracts and quality checks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Protocol

from .odds import american_to_decimal


@dataclass(frozen=True)
class RawOddsQuote:
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
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    normalized: List[NormalizedQuote] = []
    for quote in quotes:
        _validate_quote(quote, now)
        fingerprint = "|".join(
            (quote.provider, quote.provider_quote_id, quote.event_id, quote.market, quote.selection, quote.captured_at.isoformat())
        )
        normalized.append(
            NormalizedQuote(
                raw=quote,
                decimal_odds=american_to_decimal(quote.american_odds),
                idempotency_key=hashlib.sha256(fingerprint.encode("utf-8")).hexdigest(),
            )
        )
    return normalized


def _validate_quote(quote: RawOddsQuote, now: datetime) -> None:
    if not all((quote.provider, quote.provider_quote_id, quote.event_id, quote.sport, quote.market, quote.selection)):
        raise ValueError("provider, quote, event, sport, market, and selection identifiers are required")
    if quote.captured_at.tzinfo is None or quote.starts_at.tzinfo is None:
        raise ValueError("provider timestamps must be timezone-aware")
    if quote.captured_at > now:
        raise ValueError("provider quote timestamp is in the future")
    if quote.starts_at <= quote.captured_at:
        raise ValueError("cannot ingest an in-play or completed event through pregame pipeline")
    if quote.starts_at <= now:
        raise ValueError("cannot ingest an in-play or completed event through pregame pipeline")
