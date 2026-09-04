import unittest
from datetime import datetime, timedelta, timezone

from sam_analytics.providers.the_odds_api import TheOddsApiClient, TheOddsApiError


class TheOddsApiProviderTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 1, 1, 15, tzinfo=timezone.utc)
        self.payload = [
            {
                "id": "provider-event-1",
                "sport_key": "basketball_nba",
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
