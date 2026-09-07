"""PostgreSQL integration test for the immutable provider-receipt path.

It is intentionally skipped for a normal local test run.  GitHub Actions
supplies a disposable PostgreSQL URL after applying all checked-in migrations,
which proves that the Python transaction and database integrity triggers agree.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import unittest
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from sam_analytics.ingestion import RawOddsQuote
from sam_analytics.modeling import load_h2h_market_training_rows
from sam_analytics.odds_ledger import OddsLedger, PreparedOddsPayload, PreparedResultsPayload
from sam_analytics.provider_contracts import ApprovedProviderContract, ProviderContractRegistry
from sam_analytics.providers.the_odds_api import CompletedScore
from sam_analytics.raw_payload_store import InMemoryRawPayloadStore

_DATABASE_URL = os.getenv("DATABASE_URL")
_PSYCOPG_AVAILABLE = importlib.util.find_spec("psycopg") is not None


@unittest.skipUnless(_DATABASE_URL and _PSYCOPG_AVAILABLE, "requires disposable PostgreSQL")
class OddsLedgerPostgresTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime.now(UTC)
        self.event_id = f"ci-event-{uuid4().hex}"
        self.contracts = ProviderContractRegistry(
            [
                ApprovedProviderContract(
                    provider="the_odds_api",
                    license_scope="internal_analytics_only",
                    license_version="terms-2026-08-31",
                    permitted_source_types=frozenset({"odds", "result"}),
                ),
                ApprovedProviderContract(
                    provider="retrosheet",
                    license_scope="commercial_use_with_attribution",
                    license_version="notice-2026-09-06",
                    permitted_source_types=frozenset({"result"}),
                ),
            ]
        )
        self.store = InMemoryRawPayloadStore(namespace="sam-ci-private-evidence")
        self.ledger = OddsLedger(
            _DATABASE_URL,
            raw_payload_store=self.store,
            provider_contracts=self.contracts,
        )

    def _quote(self, *, american_odds=-110.0):
        return RawOddsQuote(
            provider="the_odds_api",
            provider_quote_id=f"quote-{self.event_id}",
            event_id=self.event_id,
            sport="basketball_nba",
            market="h2h",
            selection="Home",
            american_odds=american_odds,
            line=None,
            captured_at=self.now - timedelta(seconds=10),
            starts_at=self.now + timedelta(hours=2),
            bookmaker="example_book",
            league="NBA",
            home_team="Home",
            away_team="Away",
        )

    def _payload(self, raw_payload: bytes, *, received_at: datetime, american_odds=-110.0):
        quote = self._quote(american_odds=american_odds)
        return PreparedOddsPayload(
            provider="the_odds_api",
            source_type="odds",
            raw_payload=raw_payload,
            captured_at=quote.captured_at,
            received_at=received_at,
            schema_version="v4",
            license_scope="internal_analytics_only",
            license_version="terms-2026-08-31",
            quotes=(quote,),
            request_scope=(("sport_key", "basketball_nba"), ("regions", "us"), ("markets", "h2h")),
            requests_remaining=499,
            requests_used=1,
            request_cost=1,
        )

    def _results_payload(
        self,
        raw_payload: bytes,
        *,
        received_at: datetime,
        last_update: datetime,
        home_score: int,
    ) -> PreparedResultsPayload:
        return PreparedResultsPayload(
            provider="the_odds_api",
            source_type="result",
            raw_payload=raw_payload,
            captured_at=last_update,
            received_at=received_at,
            schema_version="v4",
            license_scope="internal_analytics_only",
            license_version="terms-2026-08-31",
            scores=(
                CompletedScore(
                    provider="the_odds_api",
                    event_id=self.event_id,
                    sport="basketball_nba",
                    league="NBA",
                    commence_time=self.now - timedelta(hours=3),
                    last_update=last_update,
                    home_team="Home",
                    away_team="Away",
                    home_score=home_score,
                    away_score=99,
                ),
            ),
            request_scope=(("sport_key", "basketball_nba"), ("days_from", "3")),
            requests_remaining=498,
            requests_used=2,
            request_cost=2,
        )

    def test_receipts_provenance_and_snapshot_links_are_written_together(self):
        first = self.ledger.persist(self._payload(b'{"response":1}', received_at=self.now), now=self.now)
        later = self.now + timedelta(seconds=1)
        second = self.ledger.persist(
            self._payload(b'{"response":1,"replayed":true}', received_at=later), now=later
        )

        self.assertEqual(first.status, "accepted")
        self.assertEqual(first.snapshots_created, 1)
        self.assertEqual(second.snapshots_replayed, 1)
        self.assertEqual(second.provenance_links_created, 1)

        import psycopg

        with psycopg.connect(_DATABASE_URL) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        count(DISTINCT receipt.id),
                        count(DISTINCT provenance.id),
                        count(DISTINCT snapshot.id),
                        count(DISTINCT snapshot_link.provenance_id),
                        bool_and(snapshot.primary_provenance_id IS NOT NULL),
                        min(snapshot.captured_at),
                        min(snapshot.received_at)
                    FROM sports_event event
                    JOIN odds_snapshot snapshot ON snapshot.event_id = event.id
                    JOIN odds_snapshot_provenance snapshot_link
                      ON snapshot_link.odds_snapshot_id = snapshot.id
                    JOIN raw_data_provenance provenance
                      ON provenance.id = snapshot_link.provenance_id
                    JOIN provider_payload_receipt receipt
                      ON receipt.id = provenance.provider_payload_receipt_id
                    WHERE event.provider = %s AND event.provider_event_id = %s
                    """,
                    ("the_odds_api", self.event_id),
                )
                (
                    receipt_count,
                    provenance_count,
                    snapshot_count,
                    linked_provenance_count,
                    primary_provenance_present,
                    snapshot_captured_at,
                    snapshot_received_at,
                ) = cursor.fetchone()

        self.assertEqual(receipt_count, 2)
        self.assertEqual(provenance_count, 2)
        self.assertEqual(snapshot_count, 1)
        self.assertEqual(linked_provenance_count, 2)
        self.assertTrue(primary_provenance_present)
        self.assertEqual(snapshot_captured_at, self.now - timedelta(seconds=10))
        self.assertEqual(snapshot_received_at, self.now)

    def test_same_provider_quote_time_with_a_corrected_price_is_a_new_snapshot(self):
        first = self.ledger.persist(self._payload(b'{"response":1}', received_at=self.now), now=self.now)
        later = self.now + timedelta(seconds=1)
        correction = self.ledger.persist(
            self._payload(
                b'{"response":2,"price":-105}',
                received_at=later,
                american_odds=-105.0,
            ),
            now=later,
        )

        self.assertEqual(first.snapshots_created, 1)
        self.assertEqual(correction.snapshots_created, 1)

        import psycopg

        with psycopg.connect(_DATABASE_URL) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT count(*)
                    FROM sports_event event
                    JOIN odds_snapshot snapshot ON snapshot.event_id = event.id
                    WHERE event.provider = %s AND event.provider_event_id = %s
                    """,
                    ("the_odds_api", self.event_id),
                )
                (snapshot_count,) = cursor.fetchone()
        self.assertEqual(snapshot_count, 2)

    def test_result_correction_appends_a_new_version_with_provenance(self):
        first_update = self.now - timedelta(seconds=10)
        first = self.ledger.persist_results(
            self._results_payload(
                b'{"score":101}',
                received_at=self.now,
                last_update=first_update,
                home_score=101,
            ),
            now=self.now,
        )
        later = datetime.now(UTC)
        correction = self.ledger.persist_results(
            self._results_payload(
                b'{"score":102}',
                received_at=later,
                last_update=later,
                home_score=102,
            ),
            now=later,
        )

        self.assertEqual(first.results_created, 1)
        self.assertEqual(correction.results_created, 1)

        import psycopg

        with psycopg.connect(_DATABASE_URL) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT count(DISTINCT result.id),
                           count(DISTINCT result_link.provenance_id),
                           array_agg(result.home_score ORDER BY result.settled_at)
                    FROM sports_event event
                    JOIN event_result result ON result.event_id = event.id
                    JOIN event_result_provenance result_link
                      ON result_link.event_result_id = result.id
                    WHERE event.provider = %s AND event.provider_event_id = %s
                    """,
                    ("the_odds_api", self.event_id),
                )
                result_count, provenance_count, home_scores = cursor.fetchone()

        self.assertEqual(result_count, 2)
        self.assertEqual(provenance_count, 2)
        self.assertEqual(home_scores, [101, 102])

    def test_persisted_odds_and_result_become_a_model_training_row(self):
        starts_at = self.now - timedelta(hours=3)
        captured_at = starts_at - timedelta(hours=1)
        quotes = []
        prices = {
            "book-a": {"Home": -100.0, "Away": -100.0},
            "book-b": {"Home": 120.0, "Away": -125.0},
        }
        for bookmaker, selections in prices.items():
            for selection, american_odds in selections.items():
                quotes.append(
                    RawOddsQuote(
                        provider="the_odds_api",
                        provider_quote_id=f"{self.event_id}-{bookmaker}-{selection}",
                        event_id=self.event_id,
                        sport="basketball_nba",
                        market="h2h",
                        selection=selection,
                        american_odds=american_odds,
                        line=None,
                        captured_at=captured_at,
                        starts_at=starts_at,
                        bookmaker=bookmaker,
                        league="NBA",
                        home_team="Home",
                        away_team="Away",
                    )
                )
        odds_payload = PreparedOddsPayload(
            provider="the_odds_api",
            source_type="odds",
            raw_payload=b'{"historical":"odds"}',
            captured_at=captured_at,
            received_at=self.now,
            schema_version="v4",
            license_scope="internal_analytics_only",
            license_version="terms-2026-08-31",
            quotes=tuple(quotes),
            request_scope=(("sport_key", "basketball_nba"), ("regions", "us"), ("markets", "h2h")),
            requests_remaining=499,
            requests_used=1,
            request_cost=1,
            source_available_at=captured_at,
        )
        odds_write = self.ledger.persist(odds_payload, now=self.now)
        result_available_at = self.now - timedelta(seconds=10)
        result_write = self.ledger.persist_results(
            PreparedResultsPayload(
                provider="retrosheet",
                source_type="result",
                raw_payload=b'{"historical":"result"}',
                captured_at=result_available_at,
                received_at=self.now,
                schema_version="gl2025-v1",
                license_scope="commercial_use_with_attribution",
                license_version="notice-2026-09-06",
                scores=(
                    CompletedScore(
                        provider="retrosheet",
                        event_id=f"retrosheet-{self.event_id}",
                        sport="basketball_nba",
                        league="NBA",
                        commence_time=starts_at,
                        last_update=result_available_at,
                        home_team="Home",
                        away_team="Away",
                        home_score=101,
                        away_score=99,
                        source_available_at=result_available_at,
                        matched_event_provider="the_odds_api",
                        matched_provider_event_id=self.event_id,
                    ),
                ),
                request_scope=(("season", "2025"),),
                source_available_at=result_available_at,
            ),
            now=self.now,
        )

        rows = load_h2h_market_training_rows(
            _DATABASE_URL,
            sport="basketball_nba",
            training_cutoff=self.now + timedelta(seconds=1),
        )
        row = next(item for item in rows if item.event_id and item.label_source_snapshot_id)

        self.assertEqual(odds_write.snapshots_created, 4)
        self.assertEqual(result_write.results_created, 1)
        self.assertEqual(row.outcome, 1)
        self.assertAlmostEqual(row.features["market_probability"], 0.475)
        self.assertEqual(len(row.source_snapshot_ids), 4)

    def test_empty_provider_response_retains_receipt_and_provenance_without_snapshots(self):
        import psycopg

        payload = PreparedOddsPayload(
            provider="the_odds_api",
            source_type="odds",
            raw_payload=b"[]",
            captured_at=self.now,
            received_at=self.now,
            schema_version="v4",
            license_scope="internal_analytics_only",
            license_version="terms-2026-08-31",
            quotes=(),
            request_scope=(
                ("sport_key", "basketball_nba"),
                ("regions", "us"),
                ("markets", "h2h"),
            ),
            requests_remaining=500,
            requests_used=0,
            request_cost=0,
        )

        result = self.ledger.persist(payload, now=self.now)

        self.assertEqual(result.status, "accepted_empty")
        self.assertEqual(result.events_created, 0)
        self.assertEqual(result.snapshots_created, 0)
        with psycopg.connect(_DATABASE_URL) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT count(DISTINCT receipt.id),
                           count(DISTINCT provenance.id),
                           count(snapshot.id),
                           signal.payload->>'status'
                    FROM provider_payload_receipt receipt
                    JOIN raw_data_provenance provenance
                      ON provenance.provider_payload_receipt_id = receipt.id
                    LEFT JOIN odds_snapshot snapshot
                      ON snapshot.primary_provenance_id = provenance.id
                    JOIN operational_signal signal
                      ON signal.provenance_sha256 = provenance.provenance_sha256
                    WHERE receipt.receipt_sha256 = %s
                    GROUP BY signal.payload->>'status'
                    """,
                    (payload.receipt_sha256,),
                )
                row = cursor.fetchone()

        self.assertEqual(row, (1, 1, 0, "accepted_empty"))

    def test_database_rejects_forged_provenance_bad_availability_and_event_mutation(self):
        payload = self._payload(b'{"response":1}', received_at=self.now)
        self.ledger.persist(payload, now=self.now)

        import psycopg

        with psycopg.connect(_DATABASE_URL, autocommit=True) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT receipt.id, receipt.payload_sha256, receipt.payload_uri,
                           receipt.captured_at, receipt.received_at, receipt.schema_version,
                           receipt.license_scope, receipt.receipt_sha256, event.id,
                           snapshot.id
                    FROM provider_payload_receipt receipt
                    JOIN raw_data_provenance provenance
                      ON provenance.provider_payload_receipt_id = receipt.id
                    JOIN odds_snapshot snapshot
                      ON snapshot.primary_provenance_id = provenance.id
                    JOIN sports_event event ON event.id = snapshot.event_id
                    WHERE receipt.receipt_sha256 = %s
                    """,
                    (payload.receipt_sha256,),
                )
                (
                    receipt_id,
                    payload_sha256,
                    payload_uri,
                    captured_at,
                    received_at,
                    schema_version,
                    license_scope,
                    receipt_sha256,
                    event_id,
                    snapshot_id,
                ) = cursor.fetchone()

                with self.assertRaises(psycopg.Error):
                    cursor.execute(
                        """
                        INSERT INTO raw_data_provenance (
                            provider, provider_record_id, source_type, payload_sha256,
                            payload_uri, captured_at, received_at, schema_version,
                            license_scope, provenance_sha256, provider_payload_receipt_id
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            "the_odds_api",
                            f"receipt:{receipt_sha256}",
                            "forged_source",
                            payload_sha256,
                            payload_uri,
                            captured_at,
                            received_at,
                            schema_version,
                            license_scope,
                            hashlib.sha256(f"forged:{self.event_id}".encode()).hexdigest(),
                            receipt_id,
                        ),
                    )

                with self.assertRaises(psycopg.Error):
                    cursor.execute(
                        """
                        INSERT INTO odds_snapshot (
                            event_id, provider, provider_quote_id, bookmaker, market, selection,
                            line, american_odds, decimal_odds, captured_at, received_at,
                            source_available_at, source_payload_sha256, idempotency_key,
                            primary_provenance_id
                        ) SELECT
                            event_id, provider, provider_quote_id, bookmaker, market, selection,
                            line, american_odds, decimal_odds, captured_at, received_at + interval '1 second',
                            source_available_at, source_payload_sha256, %s, primary_provenance_id
                        FROM odds_snapshot WHERE id = %s
                        """,
                        (hashlib.sha256(f"bad-time:{self.event_id}".encode()).hexdigest(), snapshot_id),
                    )

                with self.assertRaises(psycopg.Error):
                    cursor.execute(
                        "UPDATE sports_event SET home_team = %s WHERE id = %s",
                        ("Forged Change", event_id),
                    )


if __name__ == "__main__":
    unittest.main()
