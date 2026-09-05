import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from sam_analytics.providers.the_odds_api import TheOddsApiClient, TheOddsApiError


class TheOddsApiProviderTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 1, 1, 15, tzinfo=timezone.utc)
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

    def test_parser_preserves_event_market_selection_and_provider_time(self):
        quotes, skipped = TheOddsApiClient.parse_response(
            self.payload, sport_key="basketball_nba", now=self.now
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
            self.payload, sport_key="basketball_nba", now=self.now
        )
        price_correction = _payload_with_outcome(self.payload, price=-105)
        line_correction = _payload_with_outcome(self.payload, point=-1.5)
        corrected_price, _ = TheOddsApiClient.parse_response(
            price_correction, sport_key="basketball_nba", now=self.now
        )
        corrected_line, _ = TheOddsApiClient.parse_response(
            line_correction, sport_key="basketball_nba", now=self.now
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

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
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

    def test_client_rejects_malformed_scope_and_non_finite_provider_prices(self):
        client = TheOddsApiClient("test-key")
        with self.assertRaisesRegex(ValueError, "regions"):
            client.fetch_pregame_odds("basketball_nba", regions="us,https://elsewhere.invalid")
        invalid_payload = _payload_with_outcome(self.payload, price=float("nan"))
        with self.assertRaisesRegex(TheOddsApiError, "invalid American odds"):
            TheOddsApiClient.parse_response(invalid_payload, sport_key="basketball_nba", now=self.now)

    def test_parser_filters_live_events_instead_of_relabeling_them_pregame(self):
        self.payload[0]["commence_time"] = (self.now - timedelta(seconds=1)).isoformat()
        quotes, skipped = TheOddsApiClient.parse_response(
            self.payload, sport_key="basketball_nba", now=self.now
        )
        self.assertEqual(quotes, [])
        self.assertEqual(skipped, 1)

    def test_parser_rejects_sport_mismatch(self):
        with self.assertRaises(TheOddsApiError):
            TheOddsApiClient.parse_response(self.payload, sport_key="baseball_mlb", now=self.now)

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

    import copy

    amended = copy.deepcopy(payload)
    outcome = amended[0]["bookmakers"][0]["markets"][0]["outcomes"][0]
    if price is not None:
        outcome["price"] = price
    if point is not None:
        outcome["point"] = point
    return amended
