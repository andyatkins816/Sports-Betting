"""Adapter for The Odds API v4 featured pregame markets and completed scores.

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
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from sam_analytics.ingestion import RawOddsQuote

_DEFAULT_MAX_RESPONSE_BYTES = 10 * 1024 * 1024
_MAX_RESPONSE_BYTES_HARD_LIMIT = 100 * 1024 * 1024
_RESPONSE_READ_CHUNK_BYTES = 64 * 1024
_H2H_PROVIDER_ADDED_MARKETS = frozenset({"h2h_lay"})


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
    source_available_at: datetime | None = None


@dataclass(frozen=True)
class OddsApiRequestScope:
    """Safe-to-persist description of a request, excluding credentials."""

    sport_key: str
    regions: tuple[str, ...]
    markets: tuple[str, ...]
    bookmakers: tuple[str, ...] = ()
    snapshot_at: datetime | None = None


@dataclass(frozen=True)
class CompletedScore:
    """One completed provider event, preserving its exact event identity."""

    provider: str
    event_id: str
    sport: str
    league: str
    commence_time: datetime
    last_update: datetime
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    source_available_at: datetime | None = None
    matched_event_provider: str | None = None
    matched_provider_event_id: str | None = None


@dataclass(frozen=True)
class ScoresApiRequestScope:
    """Safe score-request identity that contains no credential or URL."""

    sport_key: str
    days_from: int


@dataclass(frozen=True)
class ScoresApiFetch:
    """Completed scores plus the exact response bytes needed for provenance."""

    scores: tuple[CompletedScore, ...]
    requests_remaining: int | None
    requests_used: int | None
    request_cost: int | None
    skipped_incomplete_events: int
    raw_payload: bytes
    received_at: datetime
    request_scope: ScoresApiRequestScope


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

    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float = 10.0,
        max_response_bytes: int = _DEFAULT_MAX_RESPONSE_BYTES,
    ):
        if not api_key or not api_key.strip():
            raise ValueError("The Odds API key is required")
        if isinstance(max_response_bytes, bool) or not isinstance(max_response_bytes, int):
            raise ValueError("max_response_bytes must be a positive integer")
        if max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be a positive integer")
        if max_response_bytes > _MAX_RESPONSE_BYTES_HARD_LIMIT:
            raise ValueError("max_response_bytes exceeds the allowed maximum")
        _validate_provider_base_url(self.base_url, self.expected_host)
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._max_response_bytes = max_response_bytes

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
        now = now or datetime.now(UTC)
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
        quotes, skipped = self.parse_response(
            response.payload,
            sport_key=sport_key,
            requested_markets=requested_markets,
            now=now,
        )
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

    def fetch_historical_odds(
        self,
        sport_key: str,
        *,
        snapshot_at: datetime,
        regions: str = "us",
        markets: Iterable[str] = ("h2h", "spreads", "totals"),
        bookmakers: str | None = None,
    ) -> OddsApiFetch:
        """Fetch one historical pregame snapshot without retaining credentials."""
        requested_markets = tuple(markets)
        if not sport_key or not regions:
            raise ValueError("sport_key and regions are required")
        if not _SPORT_KEY.fullmatch(sport_key):
            raise ValueError("sport_key must contain only lowercase letters, digits, and underscores")
        if not requested_markets or not all(isinstance(market, str) for market in requested_markets):
            raise ValueError("at least one market is required")
        if not set(requested_markets) <= self.supported_featured_markets:
            raise ValueError("only featured markets h2h, spreads, and totals are supported in this endpoint")
        if not _is_aware_datetime(snapshot_at) or snapshot_at.utcoffset() != UTC.utcoffset(snapshot_at):
            raise ValueError("snapshot_at must be a timezone-aware UTC datetime")
        requested_regions = _parse_scope_values(regions, field="regions")
        requested_bookmakers = (
            _parse_scope_values(bookmakers, field="bookmakers") if bookmakers is not None else ()
        )
        snapshot_at = snapshot_at.astimezone(UTC)
        request_scope = OddsApiRequestScope(
            sport_key=sport_key,
            regions=requested_regions,
            markets=requested_markets,
            bookmakers=requested_bookmakers,
            snapshot_at=snapshot_at,
        )
        params = {
            "apiKey": self._api_key,
            "regions": ",".join(requested_regions),
            "markets": ",".join(requested_markets),
            "date": snapshot_at.isoformat().replace("+00:00", "Z"),
            "dateFormat": "iso",
            "oddsFormat": "american",
        }
        if requested_bookmakers:
            params["bookmakers"] = ",".join(requested_bookmakers)
        response = self._request(f"/historical/sports/{sport_key}/odds", params)
        if not isinstance(response.payload, Mapping):
            raise TheOddsApiError("provider returned an unexpected historical odds payload")
        returned_at = _parse_provider_time(response.payload.get("timestamp"))
        previous_at = _parse_provider_time(response.payload.get("previous_timestamp"))
        next_at = _parse_provider_time(response.payload.get("next_timestamp"))
        if returned_at > snapshot_at:
            raise TheOddsApiError("provider returned a historical snapshot after the requested time")
        if previous_at >= returned_at or next_at <= snapshot_at:
            raise TheOddsApiError("provider returned invalid historical snapshot navigation")
        data = _required_list(response.payload, "data")
        quotes, skipped = self.parse_response(
            data,
            sport_key=sport_key,
            requested_markets=requested_markets,
            now=returned_at,
        )
        return OddsApiFetch(
            quotes=quotes,
            requests_remaining=_header_int(response.headers, "x-requests-remaining"),
            requests_used=_header_int(response.headers, "x-requests-used"),
            request_cost=_header_int(response.headers, "x-requests-last"),
            skipped_live_events=skipped,
            raw_payload=response.raw_payload,
            received_at=response.received_at,
            request_scope=request_scope,
            source_available_at=returned_at,
        )

    def fetch_scores(self, sport_key: str, *, days_from: int = 3) -> ScoresApiFetch:
        """Fetch completed scores for one sport without retaining request credentials."""
        if not isinstance(sport_key, str) or not _SPORT_KEY.fullmatch(sport_key):
            raise ValueError("sport_key must contain only lowercase letters, digits, and underscores")
        if isinstance(days_from, bool) or not isinstance(days_from, int) or not 1 <= days_from <= 3:
            raise ValueError("days_from must be an integer from 1 to 3")
        request_scope = ScoresApiRequestScope(sport_key=sport_key, days_from=days_from)
        response = self._request(
            f"/sports/{sport_key}/scores",
            {
                "apiKey": self._api_key,
                "daysFrom": str(days_from),
                "dateFormat": "iso",
            },
        )
        scores, skipped = self.parse_scores_response(response.payload, sport_key=sport_key)
        return ScoresApiFetch(
            scores=scores,
            requests_remaining=_header_int(response.headers, "x-requests-remaining"),
            requests_used=_header_int(response.headers, "x-requests-used"),
            request_cost=_header_int(response.headers, "x-requests-last"),
            skipped_incomplete_events=skipped,
            raw_payload=response.raw_payload,
            received_at=response.received_at,
            request_scope=request_scope,
        )

    @staticmethod
    def parse_scores_response(
        payload: Any,
        *,
        sport_key: str,
    ) -> tuple[tuple[CompletedScore, ...], int]:
        """Parse completed matching-sport events and count ignored incomplete rows."""
        if not isinstance(payload, list):
            raise TheOddsApiError("provider returned an unexpected scores payload")
        if not isinstance(sport_key, str) or not _SPORT_KEY.fullmatch(sport_key):
            raise ValueError("sport_key must contain only lowercase letters, digits, and underscores")

        completed_scores: list[CompletedScore] = []
        skipped_incomplete_events = 0
        for event in payload:
            if not isinstance(event, Mapping):
                raise TheOddsApiError("provider score event is not an object")
            event_id = _required_text(event, "id")
            provider_sport = _required_text(event, "sport_key")
            if provider_sport != sport_key:
                raise TheOddsApiError("provider returned a score sport that did not match the request")
            completed = event.get("completed")
            if not isinstance(completed, bool):
                raise TheOddsApiError("provider returned invalid completed status")
            if not completed:
                skipped_incomplete_events += 1
                continue

            home_team = _required_text(event, "home_team")
            away_team = _required_text(event, "away_team")
            if home_team == away_team:
                raise TheOddsApiError("provider returned identical home and away teams")
            scores = _required_list(event, "scores")
            if len(scores) != 2:
                raise TheOddsApiError("completed provider event must contain exactly two scores")
            score_by_team: dict[str, int] = {}
            for score in scores:
                team = _required_text(score, "name")
                if team in score_by_team:
                    raise TheOddsApiError("completed provider event contains duplicate team scores")
                score_by_team[team] = _nonnegative_integer_score(score.get("score"))
            if set(score_by_team) != {home_team, away_team}:
                raise TheOddsApiError("completed provider scores did not match the event teams")

            completed_scores.append(
                CompletedScore(
                    provider="the_odds_api",
                    event_id=event_id,
                    sport=provider_sport,
                    league=_optional_text(event, "sport_title") or provider_sport,
                    commence_time=_parse_provider_time(event.get("commence_time")),
                    last_update=_parse_provider_time(event.get("last_update")),
                    home_team=home_team,
                    away_team=away_team,
                    home_score=score_by_team[home_team],
                    away_score=score_by_team[away_team],
                )
            )
        return tuple(completed_scores), skipped_incomplete_events

    @staticmethod
    def parse_response(
        payload: Any,
        *,
        sport_key: str,
        requested_markets: Iterable[str],
        now: datetime,
    ) -> tuple[list[RawOddsQuote], int]:
        """Convert only the requested v4 response scope without losing book IDs."""
        if not isinstance(payload, list):
            raise TheOddsApiError("provider returned an unexpected odds payload")
        if not _is_aware_datetime(now):
            raise ValueError("now must be timezone-aware")
        requested_market_keys = tuple(requested_markets)
        if (
            not requested_market_keys
            or not all(isinstance(market, str) for market in requested_market_keys)
            or not set(requested_market_keys) <= TheOddsApiClient.supported_featured_markets
        ):
            raise ValueError("requested markets must be supported featured markets")
        allowed_response_markets = set(requested_market_keys)
        if "h2h" in allowed_response_markets:
            # The provider can include this exchange companion market without
            # it appearing in the request's `markets` parameter.
            allowed_response_markets.update(_H2H_PROVIDER_ADDED_MARKETS)

        quotes: list[RawOddsQuote] = []
        skipped_live_events = 0
        for event in payload:
            if not isinstance(event, Mapping):
                raise TheOddsApiError("provider event is not an object")
            event_id = _required_text(event, "id")
            provider_sport = _required_text(event, "sport_key")
            if provider_sport != sport_key:
                raise TheOddsApiError("provider returned a sport that did not match the request")

            # Validate every returned market key before filtering live events
            # or parsing outcomes.  An out-of-scope empty market must not be
            # misclassified as an admitted empty response.
            bookmaker_markets: list[tuple[Mapping[str, Any], list[Mapping[str, Any]]]] = []
            for bookmaker in _required_list(event, "bookmakers"):
                markets = _required_list(bookmaker, "markets")
                for market in markets:
                    if _required_text(market, "key") not in allowed_response_markets:
                        raise TheOddsApiError(
                            "provider returned a market that did not match the request"
                        )
                bookmaker_markets.append((bookmaker, markets))

            commence_time = _parse_provider_time(event.get("commence_time"))
            if commence_time <= now:
                skipped_live_events += 1
                continue
            league = _optional_text(event, "sport_title") or provider_sport
            home_team = _required_text(event, "home_team")
            away_team = _required_text(event, "away_team")
            for bookmaker, markets in bookmaker_markets:
                book_key = _required_text(bookmaker, "key")
                book_updated = _parse_provider_time(bookmaker.get("last_update"))
                for market in markets:
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
                headers = {key.lower(): value for key, value in response.headers.items()}
                raw_payload = _read_bounded_response(
                    response,
                    headers=headers,
                    max_response_bytes=self._max_response_bytes,
                )
                received_at = _utc_now()
                if not _is_aware_datetime(received_at):
                    raise TheOddsApiError("local receipt clock is missing timezone")
                received_at = received_at.astimezone(UTC)
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


def _read_bounded_response(
    response: Any,
    *,
    headers: Mapping[str, str],
    max_response_bytes: int,
) -> bytes:
    """Read a provider body without trusting its optional Content-Length header."""

    declared_size = _header_int(headers, "content-length")
    if declared_size is not None and declared_size > max_response_bytes:
        raise TheOddsApiError("The Odds API response body exceeds the allowed size")

    try:
        read = getattr(response, "read", None)
    except Exception:
        raise TheOddsApiError("The Odds API response body could not be read") from None
    if not callable(read):
        raise TheOddsApiError("The Odds API returned a non-binary response body")

    chunks: list[bytes] = []
    received = 0
    while True:
        # Request at most one extra byte after the current total.  This makes
        # an omitted, malformed, or underestimated Content-Length harmless.
        read_size = min(_RESPONSE_READ_CHUNK_BYTES, max_response_bytes - received + 1)
        try:
            chunk = read(read_size)
        except Exception:
            raise TheOddsApiError("The Odds API response body could not be read") from None
        if not isinstance(chunk, bytes):
            raise TheOddsApiError("The Odds API returned a non-binary response body")
        if not chunk:
            if declared_size is not None and received != declared_size:
                raise TheOddsApiError("The Odds API response body length did not match Content-Length")
            return b"".join(chunks)
        if len(chunk) > max_response_bytes - received:
            raise TheOddsApiError("The Odds API response body exceeds the allowed size")
        chunks.append(chunk)
        received += len(chunk)


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
    return parsed.astimezone(UTC)


def _nonnegative_integer_score(value: object) -> int:
    if isinstance(value, bool):
        raise TheOddsApiError("provider returned invalid team score")
    if isinstance(value, int):
        score = value
    elif isinstance(value, str) and _SCORE.fullmatch(value):
        try:
            score = int(value)
        except ValueError:
            raise TheOddsApiError("provider returned invalid team score") from None
    else:
        raise TheOddsApiError("provider returned invalid team score")
    if score < 0:
        raise TheOddsApiError("provider returned invalid team score")
    return score


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
        "updated": updated.astimezone(UTC)
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

    return datetime.now(UTC)


_SCOPE_VALUE = re.compile(r"^[a-z0-9_]{1,100}$")
_SCORE = re.compile(r"^[0-9]+$")
