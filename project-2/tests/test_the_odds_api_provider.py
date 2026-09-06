import copy
import json
import unittest
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

from sam_analytics.providers.the_odds_api import (
    CompletedScore,
    TheOddsApiClient,
    TheOddsApiError,
)


class TheOddsApiProviderTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 1, 1, 15, tzinfo=UTC)
        self.payload = [
            {
                "id": "provider-event-1",
                "sport_key": "basketball_nba",
                "sport_title": "NBA",
                "commence_time": (self.now + timedelta(hours=2)).isoformat(),
                "home_team": "Home",
                "away_team": "Away",
                "bookmakers": [
                    {
                        "key": "example_book",
                        "last_update": (self.now - timedelta(seconds=10)).isoformat(),
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": [
                                    {"name": "Home", "price": -110},
                                    {"name": "Away", "price": -110},
                                ],
                            }
                        ],
                    }
                ],
            }
        ]
        self.scores_payload = [
            {
                "id": "provider-event-1",
                "sport_key": "basketball_nba",
                "sport_title": "NBA",
                "commence_time": (self.now - timedelta(hours=3)).isoformat(),
                "completed": True,
                "home_team": "Home",
                "away_team": "Away",
                "scores": [
                    {"name": "Away", "score": 98},
                    {"name": "Home", "score": "104"},
                ],
                "last_update": (self.now - timedelta(minutes=5)).isoformat(),
            }
        ]

    def test_scores_parser_preserves_completed_result_and_ignores_incomplete_event(self):
        incomplete = copy.deepcopy(self.scores_payload[0])
        incomplete.update(
            {
                "id": "provider-event-2",
                "completed": False,
                "scores": None,
                "last_update": None,
            }
        )

        scores, skipped = TheOddsApiClient.parse_scores_response(
            [*self.scores_payload, incomplete], sport_key="basketball_nba"
        )

        self.assertEqual(skipped, 1)
        self.assertIsInstance(scores, tuple)
        self.assertEqual(
            scores,
            (
                CompletedScore(
                    provider="the_odds_api",
                    event_id="provider-event-1",
                    sport="basketball_nba",
                    league="NBA",
                    commence_time=self.now - timedelta(hours=3),
                    last_update=self.now - timedelta(minutes=5),
                    home_team="Home",
                    away_team="Away",
                    home_score=104,
                    away_score=98,
                ),
            ),
        )
        with self.assertRaises(FrozenInstanceError):
            scores[0].home_score = 105

    def test_scores_parser_rejects_malformed_completed_results(self):
        invalid_payloads = []

        missing_score = copy.deepcopy(self.scores_payload)
        missing_score[0]["scores"] = missing_score[0]["scores"][:1]
        invalid_payloads.append(("exactly two", missing_score))

        wrong_team = copy.deepcopy(self.scores_payload)
        wrong_team[0]["scores"][0]["name"] = "Other"
        invalid_payloads.append(("did not match", wrong_team))

        negative_score = copy.deepcopy(self.scores_payload)
        negative_score[0]["scores"][0]["score"] = "-1"
        invalid_payloads.append(("invalid team score", negative_score))

        fractional_score = copy.deepcopy(self.scores_payload)
        fractional_score[0]["scores"][0]["score"] = 98.0
        invalid_payloads.append(("invalid team score", fractional_score))

        naive_commence_time = copy.deepcopy(self.scores_payload)
        naive_commence_time[0]["commence_time"] = "2026-01-01T12:00:00"
        invalid_payloads.append(("missing timezone", naive_commence_time))

        naive_last_update = copy.deepcopy(self.scores_payload)
        naive_last_update[0]["last_update"] = "2026-01-01T14:55:00"
        invalid_payloads.append(("missing timezone", naive_last_update))

        invalid_completed = copy.deepcopy(self.scores_payload)
        invalid_completed[0]["completed"] = "true"
        invalid_payloads.append(("completed status", invalid_completed))

        wrong_sport = copy.deepcopy(self.scores_payload)
        wrong_sport[0]["sport_key"] = "baseball_mlb"
        invalid_payloads.append(("sport.*did not match", wrong_sport))

        for error, payload in invalid_payloads:
            with self.subTest(error=error), self.assertRaisesRegex(TheOddsApiError, error):
                TheOddsApiClient.parse_scores_response(payload, sport_key="basketball_nba")

    def test_fetch_scores_preserves_exact_receipt_quota_and_sanitized_scope(self):
        raw_payload = b" " + json.dumps(self.scores_payload, separators=(",", ":")).encode() + b"\n"

        class FakeResponse:
            status = 200
            headers = {
                "x-requests-remaining": "93",
                "x-requests-used": "7",
                "x-requests-last": "2",
            }

            def __init__(self):
                self._was_read = False

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, _size):
                if self._was_read:
                    return b""
                self._was_read = True
                return raw_payload

        class FakeOpener:
            def open(self, request, timeout):
                self.request = request
                self.timeout = timeout
                return FakeResponse()

        receipt_time = self.now + timedelta(seconds=1)
        opener = FakeOpener()
        with (
            patch("sam_analytics.providers.the_odds_api.build_opener", return_value=opener),
            patch("sam_analytics.providers.the_odds_api._utc_now", return_value=receipt_time),
        ):
            fetched = TheOddsApiClient("never-expose-this-key").fetch_scores(
                "basketball_nba", days_from=2
            )

        request_url = urlsplit(opener.request.full_url)
        query = parse_qs(request_url.query)
        self.assertEqual(request_url.path, "/v4/sports/basketball_nba/scores")
        self.assertEqual(query["daysFrom"], ["2"])
        self.assertEqual(query["dateFormat"], ["iso"])
        self.assertEqual(fetched.raw_payload, raw_payload)
        self.assertEqual(fetched.received_at, receipt_time)
        self.assertEqual(fetched.requests_remaining, 93)
        self.assertEqual(fetched.requests_used, 7)
        self.assertEqual(fetched.request_cost, 2)
        self.assertEqual(fetched.skipped_incomplete_events, 0)
        self.assertEqual(fetched.request_scope.sport_key, "basketball_nba")
        self.assertEqual(fetched.request_scope.days_from, 2)
        self.assertEqual(len(fetched.scores), 1)
        self.assertNotIn("never-expose-this-key", repr(fetched))
        with self.assertRaises(FrozenInstanceError):
            fetched.request_cost = 3

    def test_fetch_scores_rejects_invalid_days_and_sport_before_request(self):
        client = TheOddsApiClient("test-key")
        for days_from in (True, "3", 0, 4):
            with self.subTest(days_from=days_from), self.assertRaisesRegex(ValueError, "days_from"):
                client.fetch_scores("basketball_nba", days_from=days_from)
        with self.assertRaisesRegex(ValueError, "sport_key"):
            client.fetch_scores("basketball_nba/../scores")

    def test_parser_preserves_event_market_selection_and_provider_time(self):
        quotes, skipped = TheOddsApiClient.parse_response(
            self.payload,
            sport_key="basketball_nba",
            requested_markets=("h2h",),
            now=self.now,
        )
        self.assertEqual(skipped, 0)
        self.assertEqual(len(quotes), 2)
        self.assertEqual(quotes[0].event_id, "provider-event-1")
        self.assertEqual(quotes[0].market, "h2h")
        self.assertEqual(quotes[0].american_odds, -110.0)
        self.assertEqual(len(quotes[0].provider_quote_id), 64)
        self.assertEqual(quotes[0].bookmaker, "example_book")
        self.assertEqual(quotes[0].league, "NBA")
        self.assertEqual(quotes[0].home_team, "Home")
        self.assertEqual(quotes[0].away_team, "Away")

    def test_quote_identity_changes_when_provider_corrects_price_or_line(self):
        original, _ = TheOddsApiClient.parse_response(
            self.payload,
            sport_key="basketball_nba",
            requested_markets=("h2h",),
            now=self.now,
        )
        price_correction = _payload_with_outcome(self.payload, price=-105)
        line_correction = _payload_with_outcome(self.payload, point=-1.5)
        corrected_price, _ = TheOddsApiClient.parse_response(
            price_correction,
            sport_key="basketball_nba",
            requested_markets=("h2h",),
            now=self.now,
        )
        corrected_line, _ = TheOddsApiClient.parse_response(
            line_correction,
            sport_key="basketball_nba",
            requested_markets=("h2h",),
            now=self.now,
        )
        self.assertNotEqual(original[0].provider_quote_id, corrected_price[0].provider_quote_id)
        self.assertNotEqual(original[0].provider_quote_id, corrected_line[0].provider_quote_id)

    def test_client_preserves_exact_response_bytes_receipt_and_sanitized_scope(self):
        raw_payload = (
            b' [ {"id":"provider-event-1","sport_key":"basketball_nba","sport_title":"NBA",'
            b'"commence_time":"2026-01-01T17:00:00+00:00","home_team":"Home",'
            b'"away_team":"Away","bookmakers":[]} ] '
        )
        class FakeResponse:
            status = 200
            headers = {
                "x-requests-remaining": "99",
                "x-requests-used": "1",
                "x-requests-last": "1",
            }

            def __init__(self):
                self._was_read = False

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, _size):
                if self._was_read:
                    return b""
                self._was_read = True
                return raw_payload

        class FakeOpener:
            def open(self, request, timeout):
                self.request = request
                self.timeout = timeout
                return FakeResponse()

        receipt = self.now - timedelta(seconds=1)
        fake_opener = FakeOpener()
        with (
            patch("sam_analytics.providers.the_odds_api.build_opener", return_value=fake_opener),
            patch("sam_analytics.providers.the_odds_api._utc_now", return_value=receipt),
        ):
            fetched = TheOddsApiClient("never-expose-this-key").fetch_pregame_odds(
                "basketball_nba", regions="us,eu", bookmakers="example_book", now=self.now
            )
        self.assertEqual(fetched.raw_payload, raw_payload)
        self.assertEqual(fetched.received_at, receipt)
        self.assertEqual(fetched.requests_remaining, 99)
        self.assertEqual(fetched.request_scope.sport_key, "basketball_nba")
        self.assertEqual(fetched.request_scope.regions, ("us", "eu"))
        self.assertEqual(fetched.request_scope.markets, ("h2h", "spreads", "totals"))
        self.assertEqual(fetched.request_scope.bookmakers, ("example_book",))
        self.assertNotIn("never-expose-this-key", repr(fetched))

    def test_client_rejects_declared_response_larger_than_limit_without_reading(self):
        class FakeResponse:
            status = 200
            headers = {"Content-Length": "5"}

            def __init__(self):
                self.read_called = False

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, _size):
                self.read_called = True
                raise AssertionError("the oversized response must not be read")

        class FakeOpener:
            def __init__(self, response):
                self.response = response

            def open(self, request, timeout):
                return self.response

        secret = "never-expose-this-key"
        fake_response = FakeResponse()
        with patch(
            "sam_analytics.providers.the_odds_api.build_opener",
            return_value=FakeOpener(fake_response),
        ):
            with self.assertRaisesRegex(TheOddsApiError, "response body exceeds") as raised:
                TheOddsApiClient(secret, max_response_bytes=4).fetch_pregame_odds(
                    "basketball_nba", now=self.now
                )
        self.assertFalse(fake_response.read_called)
        self.assertNotIn(secret, str(raised.exception))

    def test_client_bounds_streams_when_content_length_is_missing_or_untrustworthy(self):
        class FakeResponse:
            status = 200

            def __init__(self, headers):
                self.headers = headers
                self.read_sizes = []
                self._chunks = iter((b"1234", b"5"))

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, size):
                self.read_sizes.append(size)
                return next(self._chunks, b"")

        class FakeOpener:
            def __init__(self, response):
                self.response = response

            def open(self, request, timeout):
                return self.response

        secret = "never-expose-this-key"
        for headers in ({}, {"Content-Length": "1"}, {"Content-Length": "unknown"}):
            with self.subTest(headers=headers):
                fake_response = FakeResponse(headers)
                with patch(
                    "sam_analytics.providers.the_odds_api.build_opener",
                    return_value=FakeOpener(fake_response),
                ):
                    with self.assertRaisesRegex(TheOddsApiError, "response body exceeds") as raised:
                        TheOddsApiClient(secret, max_response_bytes=4).fetch_pregame_odds(
                            "basketball_nba", now=self.now
                        )
                self.assertEqual(fake_response.read_sizes, [5, 1])
                self.assertNotIn(secret, str(raised.exception))

    def test_client_enforces_a_hard_response_ceiling_and_accepts_exactly_limited_streams(self):
        for invalid_limit in (True, "4", 0, -1, 101 * 1024 * 1024):
            with self.subTest(invalid_limit=invalid_limit), self.assertRaises(ValueError):
                TheOddsApiClient("test-key", max_response_bytes=invalid_limit)

        class FakeResponse:
            status = 200

            def __init__(self):
                self.headers = {"Content-Length": "4"}
                self.read_sizes = []
                self._chunks = iter((b"[ ", b" ]"))

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, size):
                self.read_sizes.append(size)
                return next(self._chunks, b"")

        class FakeOpener:
            def __init__(self, response):
                self.response = response

            def open(self, request, timeout):
                return self.response

        response = FakeResponse()
        with patch(
            "sam_analytics.providers.the_odds_api.build_opener",
            return_value=FakeOpener(response),
        ):
            fetched = TheOddsApiClient("test-key", max_response_bytes=4).fetch_pregame_odds(
                "basketball_nba", now=self.now
            )
        self.assertEqual(fetched.raw_payload, b"[  ]")
        self.assertEqual(fetched.quotes, [])
        self.assertEqual(response.read_sizes, [5, 3, 1])

    def test_client_rejects_content_length_mismatches_without_retaining_payload(self):
        class FakeResponse:
            status = 200

            def __init__(self, declared_size):
                self.headers = {"Content-Length": declared_size}
                self._was_read = False

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, _size):
                if self._was_read:
                    return b""
                self._was_read = True
                return b"[]"

        class FakeOpener:
            def __init__(self, response):
                self.response = response

            def open(self, request, timeout):
                return self.response

        for declared_size in ("1", "3"):
            with self.subTest(declared_size=declared_size):
                with patch(
                    "sam_analytics.providers.the_odds_api.build_opener",
                    return_value=FakeOpener(FakeResponse(declared_size)),
                ):
                    with self.assertRaisesRegex(TheOddsApiError, "length did not match"):
                        TheOddsApiClient("test-key", max_response_bytes=4).fetch_pregame_odds(
                            "basketball_nba", now=self.now
                        )

    def test_client_rejects_malformed_scope_and_non_finite_provider_prices(self):
        client = TheOddsApiClient("test-key")
        with self.assertRaisesRegex(ValueError, "regions"):
            client.fetch_pregame_odds("basketball_nba", regions="us,https://elsewhere.invalid")
        invalid_payload = _payload_with_outcome(self.payload, price=float("nan"))
        with self.assertRaisesRegex(TheOddsApiError, "invalid American odds"):
            TheOddsApiClient.parse_response(
                invalid_payload,
                sport_key="basketball_nba",
                requested_markets=("h2h",),
                now=self.now,
            )

    def test_parser_filters_live_events_instead_of_relabeling_them_pregame(self):
        self.payload[0]["commence_time"] = (self.now - timedelta(seconds=1)).isoformat()
        quotes, skipped = TheOddsApiClient.parse_response(
            self.payload,
            sport_key="basketball_nba",
            requested_markets=("h2h",),
            now=self.now,
        )
        self.assertEqual(quotes, [])
        self.assertEqual(skipped, 1)

    def test_parser_rejects_sport_mismatch(self):
        with self.assertRaises(TheOddsApiError):
            TheOddsApiClient.parse_response(
                self.payload,
                sport_key="baseball_mlb",
                requested_markets=("h2h",),
                now=self.now,
            )

    def test_parser_validates_scope_before_live_filter_or_empty_outcomes(self):
        live_wrong_sport = copy.deepcopy(self.payload)
        live_wrong_sport[0]["commence_time"] = (
            self.now - timedelta(seconds=1)
        ).isoformat()
        with self.assertRaisesRegex(TheOddsApiError, "sport.*did not match"):
            TheOddsApiClient.parse_response(
                live_wrong_sport,
                sport_key="baseball_mlb",
                requested_markets=("h2h",),
                now=self.now,
            )

        empty_wrong_market = copy.deepcopy(self.payload)
        empty_wrong_market[0]["bookmakers"][0]["markets"][0]["key"] = "spreads"
        empty_wrong_market[0]["bookmakers"][0]["markets"][0]["outcomes"] = []
        with self.assertRaisesRegex(TheOddsApiError, "market.*did not match"):
            TheOddsApiClient.parse_response(
                empty_wrong_market,
                sport_key="basketball_nba",
                requested_markets=("h2h",),
                now=self.now,
            )

    def test_parser_admits_documented_h2h_lay_as_a_provider_added_companion(self):
        provider_added = copy.deepcopy(self.payload)
        provider_added[0]["bookmakers"][0]["markets"][0]["key"] = "h2h_lay"

        quotes, skipped = TheOddsApiClient.parse_response(
            provider_added,
            sport_key="basketball_nba",
            requested_markets=("h2h",),
            now=self.now,
        )

        self.assertEqual(skipped, 0)
        self.assertEqual({quote.market for quote in quotes}, {"h2h_lay"})

    def test_client_rejects_unpinned_or_non_https_provider_endpoint(self):
        class UnsafeClient(TheOddsApiClient):
            base_url = "http://not-the-odds-api.example/v4"

        with self.assertRaises(TheOddsApiError):
            UnsafeClient("test-key")

    def test_fetch_rejects_sport_key_that_could_change_the_request_path(self):
        client = TheOddsApiClient("test-key")
        with self.assertRaises(ValueError):
            client.fetch_pregame_odds("basketball_nba/../other")


def _payload_with_outcome(payload, *, price=None, point=None):
    """Return an isolated payload with the first outcome selectively amended."""

    amended = copy.deepcopy(payload)
    outcome = amended[0]["bookmakers"][0]["markets"][0]["outcomes"][0]
    if price is not None:
        outcome["price"] = price
    if point is not None:
        outcome["point"] = point
    return amended
