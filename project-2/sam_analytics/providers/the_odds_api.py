"""Adapter for The Odds API v4 featured pregame markets.

The provider includes live events in its odds response. This adapter filters
them because the current SAM pregame pipeline must never mix live and pregame
prices. It also surfaces quota headers so a scheduler can alert before a hard
limit rather than silently degrading the feed.
"""

from __future__ import annotations

import hashlib
import json
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
    quotes: list[RawOddsQuote]
    requests_remaining: int | None
    requests_used: int | None
    request_cost: int | None
    skipped_live_events: int


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
        if not requested_markets or not set(requested_markets) <= self.supported_featured_markets:
            raise ValueError("only featured markets h2h, spreads, and totals are supported in this endpoint")
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        params = {
            "apiKey": self._api_key,
            "regions": regions,
            "markets": ",".join(requested_markets),
            "dateFormat": "iso",
            "oddsFormat": "american",
        }
        if bookmakers:
            params["bookmakers"] = bookmakers
        payload, headers = self._request(f"/sports/{sport_key}/odds", params)
        quotes, skipped = self.parse_response(payload, sport_key=sport_key, now=now)
        return OddsApiFetch(
            quotes=quotes,
            requests_remaining=_header_int(headers, "x-requests-remaining"),
            requests_used=_header_int(headers, "x-requests-used"),
            request_cost=_header_int(headers, "x-requests-last"),
            skipped_live_events=skipped,
        )

    @staticmethod
    def parse_response(
        payload: Any, *, sport_key: str, now: datetime
    ) -> tuple[list[RawOddsQuote], int]:
        """Convert the documented v4 response shape without losing book IDs."""
        if not isinstance(payload, list):
            raise TheOddsApiError("provider returned an unexpected odds payload")
        if now.tzinfo is None:
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
            for bookmaker in _required_list(event, "bookmakers"):
                book_key = _required_text(bookmaker, "key")
                book_updated = _parse_provider_time(bookmaker.get("last_update"))
                for market in _required_list(bookmaker, "markets"):
                    market_key = _required_text(market, "key")
                    market_updated = _parse_provider_time(market.get("last_update"), default=book_updated)
                    for outcome in _required_list(market, "outcomes"):
                        selection = _required_text(outcome, "name")
                        price = outcome.get("price")
                        if isinstance(price, bool) or not isinstance(price, (int, float)) or price == 0:
                            raise TheOddsApiError("provider returned invalid American odds")
                        line = outcome.get("point")
                        if line is not None and (isinstance(line, bool) or not isinstance(line, (int, float))):
                            raise TheOddsApiError("provider returned an invalid market line")
                        quote_id = _quote_id(event_id, book_key, market_key, selection, market_updated)
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
                            )
                        )
        return quotes, skipped_live_events

    def _request(self, path: str, params: Mapping[str, str]) -> tuple[Any, Mapping[str, str]]:
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
                headers = {key.lower(): value for key, value in response.headers.items()}
            try:
                return json.loads(raw_payload), headers
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise TheOddsApiError("The Odds API returned invalid JSON") from error
        except HTTPError as error:
            raise TheOddsApiError(f"The Odds API returned HTTP {error.code}") from error
        except (URLError, OSError) as error:
            raise TheOddsApiError("The Odds API request failed") from error


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
    except ValueError as error:
        raise TheOddsApiError("provider timestamp is invalid") from error
    if parsed.tzinfo is None:
        raise TheOddsApiError("provider timestamp is missing timezone")
    return parsed.astimezone(timezone.utc)


def _quote_id(event_id: str, book: str, market: str, selection: str, updated: datetime) -> str:
    fingerprint = "|".join((event_id, book, market, selection, updated.isoformat()))
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()


def _header_int(headers: Mapping[str, str], key: str) -> int | None:
    value = headers.get(key)
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None
