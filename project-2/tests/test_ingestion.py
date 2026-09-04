import unittest
from datetime import datetime, timedelta, timezone

from sam_analytics.ingestion import RawOddsQuote, normalize_quotes


class IngestionTests(unittest.TestCase):
    def test_normalization_preserves_provider_identity_and_is_idempotent(self):
        now = datetime(2026, 1, 1, 15, tzinfo=timezone.utc)
        quote = RawOddsQuote(
            provider="licensed-provider",
            provider_quote_id="q-1",
            event_id="e-1",
            sport="basketball",
            market="h2h",
            selection="home",
            american_odds=-110,
            line=None,
            captured_at=now - timedelta(seconds=1),
            starts_at=now + timedelta(hours=1),
        )
        normalized = normalize_quotes([quote], now=now)
        self.assertEqual(len(normalized), 1)
        self.assertAlmostEqual(normalized[0].decimal_odds, 1.9090909)
        self.assertEqual(len(normalized[0].idempotency_key), 64)

    def test_completed_or_live_events_are_not_sent_to_pregame_pipeline(self):
        now = datetime(2026, 1, 1, 15, tzinfo=timezone.utc)
        quote = RawOddsQuote(
            provider="licensed-provider",
            provider_quote_id="q-1",
            event_id="e-1",
            sport="basketball",
            market="h2h",
            selection="home",
            american_odds=100,
            line=None,
            captured_at=now,
            starts_at=now,
        )
        with self.assertRaisesRegex(ValueError, "pregame"):
            normalize_quotes([quote], now=now)
