import unittest
from dataclasses import replace
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

    def test_price_or_line_correction_at_same_provider_time_is_not_a_duplicate(self):
        now = datetime(2026, 1, 1, 15, tzinfo=timezone.utc)
        quote = RawOddsQuote(
            provider="licensed-provider",
            provider_quote_id="q-1",
            event_id="e-1",
            sport="basketball",
            market="spreads",
            selection="home",
            american_odds=-110,
            line=-1.5,
            captured_at=now - timedelta(seconds=1),
            starts_at=now + timedelta(hours=1),
        )
        normalized = normalize_quotes(
            [quote, replace(quote, american_odds=-105), replace(quote, line=-2.0)], now=now
        )
        self.assertEqual(len({record.idempotency_key for record in normalized}), 3)

    def test_equivalent_utc_offsets_have_the_same_canonical_identity(self):
        now = datetime(2026, 1, 1, 15, tzinfo=timezone.utc)
        captured_utc = now - timedelta(minutes=1)
        starts_utc = now + timedelta(hours=1)
        quote = RawOddsQuote(
            provider="licensed-provider",
            provider_quote_id="q-1",
            event_id="e-1",
            sport="basketball",
            market="h2h",
            selection="home",
            american_odds=-110,
            line=None,
            captured_at=captured_utc,
            starts_at=starts_utc,
        )
        offset = timezone(timedelta(hours=-5))
        offset_quote = replace(
            quote,
            captured_at=captured_utc.astimezone(offset),
            starts_at=starts_utc.astimezone(offset),
        )
        normalized = normalize_quotes([quote, offset_quote], now=now)
        self.assertEqual(normalized[0].idempotency_key, normalized[1].idempotency_key)

    def test_rejects_non_finite_or_boolean_numeric_values(self):
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
        for invalid in (float("nan"), float("inf"), True, 0, -110.5):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "odds"):
                normalize_quotes([replace(quote, american_odds=invalid)], now=now)
        for invalid_line in (float("nan"), float("-inf"), False):
            with self.subTest(invalid_line=invalid_line), self.assertRaisesRegex(ValueError, "line"):
                normalize_quotes([replace(quote, line=invalid_line)], now=now)

    def test_metadata_fields_are_optional_but_must_be_meaningful_when_present(self):
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
            bookmaker="example_book",
            league="NBA",
            home_team="Home",
            away_team="Away",
        )
        self.assertEqual(normalize_quotes([quote], now=now)[0].raw.bookmaker, "example_book")
        with self.assertRaisesRegex(ValueError, "bookmaker"):
            normalize_quotes([replace(quote, bookmaker="  ")], now=now)

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
