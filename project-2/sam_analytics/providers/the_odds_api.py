"""Adapter for The Odds API v4 featured pregame markets.

The provider includes live events in its odds response. This adapter filters
them because the current SAM pregame pipeline must never mix live and pregame
prices. It also surfaces quota headers so a scheduler can alert before a hard
limit rather than silently degrading the feed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from sam_analytics.ingestion import RawOddsQuote


class TheOddsApiError(RuntimeError):
    """A provider failure safe to log without exposing an API key or URL."""


class _NoRedirect(HTTPRedirectHandler):
    """Reject redirects so a query-string provider key cannot leave the pinned host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


@dataclass(frozen=True)
class OddsApiFetch:
    """A parsed provider response plus immutable receipt evidence.

    ``raw_payload`` is the exact byte sequence returned by the provider.  It
    is intentionally not a request URL and therefore never contains the API
    key.  A future durable evidence store must retain these bytes before it
    writes parsed quotes to the analytics ledger.
    """

    quotes: list[RawOddsQuote]
    requests_remaining: int | None
    requests_used: int | None
    request_cost: int | None
    skipped_live_events: int
    raw_payload: bytes = b""
    received_at: datetime | None = None
    request_scope: OddsApiRequestScope | None = None


@dataclass(frozen=True)
class OddsApiRequestScope:
    """Safe-to-persist description of a request, excluding credentials."""

    sport_key: str
    regions: tuple[str, ...]
    markets: tuple[str, ...]
    bookmakers: tuple[str, ...] = ()


@dataclass(frozen=True)
class _OddsApiResponse:
    payload: Any
    raw_payload: bytes
    headers: Mapping[str, str]
    received_at: datetime


class TheOddsApiClient:
    base_url = "https://api.the-odds-api.com/v4"
    expected_host = "api.the-odds-api.com"
    supported_featured_markets = frozenset({"h2h", "spreads", "totals"})

    def __init__(self, api_key: str, *, timeout_seconds: float = 10.0):
        if not api_key or not api_key.strip():
            raise ValueError("The Odds API key is required")
        _validate_provider_base_url(self.base_url, self.expected_host)
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds

    def fetch_pregame_odds(
        self,
        sport_key: str,
        *,
        regions: str = "us",
        markets: Iterable[str] = ("h2h", "spreads", "totals"),
        bookmakers: str | None = None,
        now: datetime | None = None,
    ) -> OddsApiFetch:
        requested_markets = tuple(markets)
        if not sport_key or not regions:
            raise ValueError("sport_key and regions are required")
        if not _SPORT_KEY.fullmatch(sport_key):
            raise ValueError("sport_key must contain only lowercase letters, digits, and underscores")
        if not requested_markets or not all(isinstance(market, str) for market in requested_markets):
            raise ValueError("at least one market is required")
        if not set(requested_markets) <= self.supported_featured_markets:
            raise ValueError("only featured markets h2h, spreads, and totals are supported in this endpoint")
        requested_regions = _parse_scope_values(regions, field="regions")
        requested_bookmakers = (
            _parse_scope_values(bookmakers, field="bookmakers") if bookmakers is not None else ()
        )
        now = now or datetime.now(timezone.utc)
        if not _is_aware_datetime(now):
            raise ValueError("now must be timezone-aware")
        request_scope = OddsApiRequestScope(
            sport_key=sport_key,
            regions=requested_regions,
            markets=requested_markets,
            bookmakers=requested_bookmakers,
        )
        params = {
            "apiKey": self._api_key,
            "regions": ",".join(requested_regions),
            "markets": ",".join(requested_markets),
            "dateFormat": "iso",
            "oddsFormat": "american",
        }
        if requested_bookmakers:
            params["bookmakers"] = ",".join(requested_bookmakers)
        response = self._request(f"/sports/{sport_key}/odds", params)
        quotes, skipped = self.parse_response(response.payload, sport_key=sport_key, now=now)
        return OddsApiFetch(
            quotes=quotes,
            requests_remaining=_header_int(response.headers, "x-requests-remaining"),
            requests_used=_header_int(response.headers, "x-requests-used"),
            request_cost=_header_int(response.headers, "x-requests-last"),
            skipped_live_events=skipped,
            raw_payload=response.raw_payload,
            received_at=response.received_at,
            request_scope=request_scope,
        )

    @staticmethod
    def parse_response(
        payload: Any, *, sport_key: str, now: datetime
    ) -> tuple[list[RawOddsQuote], int]:
        """Convert the documented v4 response shape without losing book IDs."""
        if not isinstance(payload, list):
            raise TheOddsApiError("provider returned an unexpected odds payload")
        if not _is_aware_datetime(now):
            raise ValueError("now must be timezone-aware")
        quotes: list[RawOddsQuote] = []
        skipped_live_events = 0
        for event in payload:
            if not isinstance(event, Mapping):
                raise TheOddsApiError("provider event is not an object")
            event_id = _required_text(event, "id")
            commence_time = _parse_provider_time(event.get("commence_time"))
            if commence_time <= now:
                skipped_live_events += 1
                continue
            provider_sport = _required_text(event, "sport_key")
            if provider_sport != sport_key:
                raise TheOddsApiError("provider returned a sport that did not match the request")
            league = _optional_text(event, "sport_title") or provider_sport
            home_team = _required_text(event, "home_team")
            away_team = _required_text(event, "away_team")
            for bookmaker in _required_list(event, "bookmakers"):
                book_key = _required_text(bookmaker, "key")
                book_updated = _parse_provider_time(bookmaker.get("last_update"))
                for market in _required_list(bookmaker, "markets"):
                    market_key = _required_text(market, "key")
                    market_updated = _parse_provider_time(market.get("last_update"), default=book_updated)
                    for outcome in _required_list(market, "outcomes"):
                        selection = _required_text(outcome, "name")
                        price = outcome.get("price")
                        if (
                            isinstance(price, bool)
                            or not isinstance(price, (int, float))
                            or not math.isfinite(price)
                            or price == 0
                        ):
                            raise TheOddsApiError("provider returned invalid American odds")
                        line = outcome.get("point")
                        if line is not None and (
                            isinstance(line, bool)
                            or not isinstance(line, (int, float))
                            or not math.isfinite(line)
                        ):
                            raise TheOddsApiError("provider returned an invalid market line")
                        quote_id = _quote_id(
                            event_id,
                            book_key,
                            market_key,
                            selection,
                            market_updated,
                            float(price),
                            float(line) if line is not None else None,
                        )
                        quotes.append(
                            RawOddsQuote(
                                provider="the_odds_api",
                                provider_quote_id=quote_id,
                                event_id=event_id,
                                sport=provider_sport,
                                market=market_key,
                                selection=selection,
                                american_odds=float(price),
                                line=float(line) if line is not None else None,
                                captured_at=market_updated,
                                starts_at=commence_time,
                                bookmaker=book_key,
                                league=league,
                                home_team=home_team,
                                away_team=away_team,
                            )
                        )
        return quotes, skipped_live_events

    def _request(self, path: str, params: Mapping[str, str]) -> _OddsApiResponse:
        # Do not log request URLs: The provider requires the API key in the query string.
        url = f"{self.base_url}{path}?{urlencode(params)}"
        _validate_provider_request_url(url, self.expected_host)
        request = Request(url, headers={"Accept": "application/json"})
        opener = build_opener(_NoRedirect())
        try:
            with opener.open(request, timeout=self._timeout_seconds) as response:
                if response.status != 200:
                    raise TheOddsApiError(f"The Odds API returned HTTP {response.status}")
                raw_payload = response.read()
                if not isinstance(raw_payload, bytes):
                    raise TheOddsApiError("The Odds API returned a non-binary response body")
                received_at = _utc_now()
                if not _is_aware_datetime(received_at):
                    raise TheOddsApiError("local receipt clock is missing timezone")
                received_at = received_at.astimezone(timezone.utc)
                headers = {key.lower(): value for key, value in response.headers.items()}
            try:
                return _OddsApiResponse(
                    payload=json.loads(raw_payload),
                    raw_payload=raw_payload,
                    headers=headers,
                    received_at=received_at,
                )
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise TheOddsApiError("The Odds API returned invalid JSON") from None
        except HTTPError as error:
            # HTTPError retains the full request URL, including the provider's
            # query-string key. Never expose it through an exception chain.
            raise TheOddsApiError(f"The Odds API returned HTTP {error.code}") from None
        except (URLError, OSError):
            # A network exception can also retain the requested URL.
            raise TheOddsApiError("The Odds API request failed") from None


_SPORT_KEY = re.compile(r"^[a-z0-9_]{1,100}$")


def _validate_provider_base_url(value: object, expected_host: str) -> None:
    if not isinstance(value, str):
        raise TheOddsApiError("The Odds API endpoint is invalid")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname != expected_host
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith("/v4")
    ):
        raise TheOddsApiError("The Odds API endpoint is not the approved HTTPS host")


def _validate_provider_request_url(value: str, expected_host: str) -> None:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname != expected_host
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in (None, 443)
        or parsed.fragment
    ):
        raise TheOddsApiError("The Odds API request is not directed to the approved HTTPS host")


def _required_text(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TheOddsApiError(f"provider omitted required field {key}")
    return value


def _optional_text(data: Mapping[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise TheOddsApiError(f"provider returned invalid field {key}")
    return value


def _required_list(data: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    value = data.get(key)
    if not isinstance(value, list):
        raise TheOddsApiError(f"provider omitted required list {key}")
    if not all(isinstance(item, Mapping) for item in value):
        raise TheOddsApiError(f"provider list {key} contains a non-object")
    return value


def _parse_provider_time(value: Any, *, default: datetime | None = None) -> datetime:
    if value is None and default is not None:
        return default
    if not isinstance(value, str):
        raise TheOddsApiError("provider omitted timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise TheOddsApiError("provider timestamp is invalid") from None
    if parsed.tzinfo is None:
        raise TheOddsApiError("provider timestamp is missing timezone")
    return parsed.astimezone(timezone.utc)


def _quote_id(
    event_id: str,
    book: str,
    market: str,
    selection: str,
    updated: datetime,
    american_odds: float,
    line: float | None,
) -> str:
    """Create a source identity that cannot erase a same-time correction."""

    fingerprint = {
        "american_odds": float(american_odds),
        "book": book,
        "event_id": event_id,
        "line": float(line) if line is not None else None,
        "market": market,
        "selection": selection,
        "updated": updated.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z"),
    }
    encoded = json.dumps(fingerprint, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _header_int(headers: Mapping[str, str], key: str) -> int | None:
    value = headers.get(key)
    try:
        parsed = int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return parsed if parsed is None or parsed >= 0 else None


def _parse_scope_values(value: object, *, field: str) -> tuple[str, ...]:
    """Normalize a comma-separated request scope without retaining a URL."""

    if not isinstance(value, str):
        raise ValueError(f"{field} must be a comma-separated string")
    parts = tuple(part.strip() for part in value.split(","))
    if not parts or not all(_SCOPE_VALUE.fullmatch(part) for part in parts):
        raise ValueError(f"{field} must contain lowercase provider identifiers")
    return parts


def _is_aware_datetime(value: object) -> bool:
    return isinstance(value, datetime) and value.tzinfo is not None and value.utcoffset() is not None


def _utc_now() -> datetime:
    """Keep the receipt clock injectable in tests without accepting provider time."""

    return datetime.now(timezone.utc)


_SCOPE_VALUE = re.compile(r"^[a-z0-9_]{1,100}$")
