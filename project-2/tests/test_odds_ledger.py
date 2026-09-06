"""Tests for the evidence-first transactional odds ledger.

These deliberately use a small in-memory SQL recorder instead of a provider or
network connection.  PostgreSQL migrations are exercised separately in CI;
these tests prove the important behavioral ordering and immutable-link logic
without ever needing an odds-provider credential.
"""

from __future__ import annotations

import copy
import json
import unittest
from contextlib import AbstractContextManager
from datetime import UTC, datetime, timedelta

from sam_analytics.ingestion import RawOddsQuote
from sam_analytics.odds_ledger import (
    OddsLedger,
    OddsLedgerUnavailable,
    OddsLedgerValidationError,
    PreparedOddsPayload,
    PreparedResultsPayload,
    prepare_the_odds_api_payload,
    prepare_the_odds_api_results_payload,
)
from sam_analytics.provider_contracts import (
    ApprovedProviderContract,
    ProviderContractRegistry,
)
from sam_analytics.providers.the_odds_api import (
    CompletedScore,
    OddsApiFetch,
    OddsApiRequestScope,
    ScoresApiFetch,
    ScoresApiRequestScope,
)
from sam_analytics.raw_payload_store import InMemoryRawPayloadStore, RawPayloadStoreViolation


class _LedgerTransaction(AbstractContextManager[None]):
    def __init__(self, connection: _LedgerConnection):
        self.connection = connection
        self._state = None

    def __enter__(self):
        self._state = copy.deepcopy(self.connection.state)
        self.connection.transaction_count += 1
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            self.connection.state = self._state
        return False


class _LedgerCursor:
    def __init__(self, connection: _LedgerConnection):
        self.connection = connection
        self._row = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def execute(self, query, params=None):
        normalized = " ".join(query.split())
        self.connection.executions.append((normalized, params))
        self._row = None
        state = self.connection.state

        if normalized.startswith("INSERT INTO provider_payload_receipt"):
            receipt_sha = params[-1]
            existing = state["receipts"].get(receipt_sha)
            if existing is None:
                receipt_id = f"receipt-{len(state['receipts']) + 1}"
                state["receipts"][receipt_sha] = (receipt_id, params)
                self._row = (receipt_id,)
            return
        if normalized.startswith("SELECT id FROM provider_payload_receipt"):
            receipt = state["receipts"].get(params[0])
            self._row = (receipt[0],) if receipt is not None else None
            return
        if normalized.startswith("INSERT INTO raw_data_provenance"):
            provenance_sha = params[-2]
            existing = state["provenance"].get(provenance_sha)
            if existing is None:
                provenance_id = f"provenance-{len(state['provenance']) + 1}"
                state["provenance"][provenance_sha] = (provenance_id, params)
                self._row = (provenance_id,)
            return
        if normalized.startswith("SELECT id FROM raw_data_provenance"):
            provenance = state["provenance"].get(params[0])
            self._row = (provenance[0],) if provenance is not None else None
            return
        if normalized.startswith("INSERT INTO sports_event"):
            key = (params[0], params[1])
            if key not in state["events"]:
                event_id = f"event-{len(state['events']) + 1}"
                state["events"][key] = (event_id, *params[2:])
                self._row = (event_id,)
            return
        if normalized.startswith("SELECT id, sport, league, starts_at, home_team, away_team FROM sports_event"):
            event = state["events"].get((params[0], params[1]))
            self._row = event
            return
        if normalized.startswith("INSERT INTO data_quality_incident"):
            state["incidents"].append(params)
            return
        if normalized.startswith("INSERT INTO event_result_provenance"):
            link = (params[0], params[1])
            if link not in state["result_provenance"]:
                state["result_provenance"].add(link)
                self._row = (params[0],)
            return
        if normalized.startswith("INSERT INTO event_result"):
            if self.connection.fail_on_result:
                raise RuntimeError("database failure containing an opaque connection value")
            key = (params[1], params[2])
            if key not in state["results"]:
                result_id = f"result-{len(state['results']) + 1}"
                state["results"][key] = (result_id, params)
                self._row = (result_id,)
            return
        if normalized.startswith("SELECT id FROM event_result"):
            result = state["results"].get((params[0], params[1]))
            self._row = (result[0],) if result is not None else None
            return
        if normalized.startswith("INSERT INTO odds_snapshot_provenance"):
            link = (params[0], params[1])
            if link not in state["snapshot_provenance"]:
                state["snapshot_provenance"].add(link)
                self._row = (params[0],)
            return
        if normalized.startswith("INSERT INTO odds_snapshot"):
            if self.connection.fail_on_snapshot:
                raise RuntimeError("database failure containing an opaque connection value")
            idempotency_key = params[-2]
            if idempotency_key not in state["snapshots"]:
                snapshot_id = f"snapshot-{len(state['snapshots']) + 1}"
                state["snapshots"][idempotency_key] = (snapshot_id, params)
                self._row = (snapshot_id,)
            return
        if normalized.startswith("SELECT id FROM odds_snapshot"):
            snapshot = state["snapshots"].get(params[0])
            self._row = (snapshot[0],) if snapshot is not None else None
            return
        if normalized.startswith("INSERT INTO operational_signal"):
            state["signals"].append(params)
            return
        raise AssertionError(f"unexpected SQL in ledger test: {normalized}")

    def fetchone(self):
        return self._row


class _LedgerConnection:
    def __init__(self, *, fail_on_snapshot: bool = False, fail_on_result: bool = False):
        self.fail_on_snapshot = fail_on_snapshot
        self.fail_on_result = fail_on_result
        self.executions = []
        self.transaction_count = 0
        self.closed = False
        self.state = {
            "receipts": {},
            "provenance": {},
            "events": {},
            "snapshots": {},
            "snapshot_provenance": set(),
            "results": {},
            "result_provenance": set(),
            "signals": [],
            "incidents": [],
        }

    def cursor(self):
        return _LedgerCursor(self)

    def transaction(self):
        return _LedgerTransaction(self)

    def close(self):
        self.closed = True


class _FailingPayloadStore:
    def store(self, *args, **kwargs):
        raise RawPayloadStoreViolation("storage configuration is unavailable")


class OddsLedgerTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 9, 4, 18, tzinfo=UTC)
        self.registry = ProviderContractRegistry(
            [
                ApprovedProviderContract(
                    provider="the_odds_api",
                    license_scope="internal_analytics_only",
                    license_version="terms-2026-08-31",
                    permitted_source_types=frozenset({"odds", "result"}),
                )
            ]
        )
        self.store = InMemoryRawPayloadStore()
        self.connection = _LedgerConnection()

    def _ledger(self, *, connection=None, store=None):
        selected_connection = connection or self.connection
        return OddsLedger(
            "postgresql://sam:opaque@db.example/sam",
            raw_payload_store=store or self.store,
            provider_contracts=self.registry,
            connection_factory=lambda _: selected_connection,
        )

    def _quote(self, *, price=-110.0, home_team="Home", quote_id="quote-1"):
        return RawOddsQuote(
            provider="the_odds_api",
            provider_quote_id=quote_id,
            event_id="event-1",
            sport="basketball_nba",
            market="h2h",
            selection="Home",
            american_odds=price,
            line=None,
            captured_at=self.now - timedelta(seconds=10),
            starts_at=self.now + timedelta(hours=2),
            bookmaker="example_book",
            league="NBA",
            home_team=home_team,
            away_team="Away",
        )

    def _payload(self, *, quote=None, raw_payload=b'{"response":1}', received_at=None):
        quote = quote or self._quote()
        received_at = received_at or self.now
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
            request_scope=(
                ("sport_key", "basketball_nba"),
                ("regions", "us"),
                ("markets", "h2h"),
            ),
            requests_remaining=499,
            requests_used=1,
            request_cost=1,
        )

    def _score(
        self,
        *,
        home_score=101,
        away_score=99,
        last_update=None,
        commence_time=None,
        league="NBA",
        home_team="Home",
    ):
        return CompletedScore(
            provider="the_odds_api",
            event_id="event-1",
            sport="basketball_nba",
            league=league,
            commence_time=commence_time or self.now - timedelta(hours=3),
            last_update=last_update or self.now - timedelta(seconds=10),
            home_team=home_team,
            away_team="Away",
            home_score=home_score,
            away_score=away_score,
        )

    def _results_payload(
        self,
        *,
        score=None,
        raw_payload=b'{"scores":1}',
        received_at=None,
    ):
        score = score or self._score()
        return PreparedResultsPayload(
            provider="the_odds_api",
            source_type="result",
            raw_payload=raw_payload,
            captured_at=score.last_update,
            received_at=received_at or self.now,
            schema_version="v4",
            license_scope="internal_analytics_only",
            license_version="terms-2026-08-31",
            scores=(score,),
            request_scope=(("sport_key", "basketball_nba"), ("days_from", "3")),
            requests_remaining=498,
            requests_used=2,
            request_cost=1,
        )

    def test_exact_replay_is_idempotent_and_stays_linked_to_private_evidence(self):
        payload = self._payload()
        first = self._ledger().persist(payload, now=self.now)
        replay = self._ledger().persist(payload, now=self.now)

        self.assertEqual(first.status, "accepted")
        self.assertEqual(first.events_created, 1)
        self.assertEqual(first.snapshots_created, 1)
        self.assertEqual(first.provenance_links_created, 1)
        self.assertEqual(replay.snapshots_created, 0)
        self.assertEqual(replay.snapshots_replayed, 1)
        self.assertEqual(replay.provenance_links_created, 0)
        self.assertEqual(len(self.connection.state["snapshots"]), 1)
        self.assertEqual(len(self.connection.state["snapshot_provenance"]), 1)
        self.assertEqual(self.store.stored_count, 1)
        self.assertTrue(self.connection.closed)
        self.assertNotIn(payload.raw_payload.decode("utf-8"), repr(first))

    def test_same_quote_from_a_different_raw_payload_retains_additional_provenance(self):
        quote = self._quote()
        self._ledger().persist(self._payload(quote=quote, raw_payload=b'{"response":1}'), now=self.now)
        observed_again = self._ledger().persist(
            self._payload(
                quote=quote,
                raw_payload=b'{"response":1,"unrelated_provider_field":"changed"}',
                received_at=self.now + timedelta(seconds=1),
            ),
            now=self.now + timedelta(seconds=1),
        )

        self.assertEqual(observed_again.snapshots_created, 0)
        self.assertEqual(observed_again.snapshots_replayed, 1)
        self.assertEqual(observed_again.provenance_links_created, 1)
        self.assertEqual(len(self.connection.state["snapshots"]), 1)
        self.assertEqual(len(self.connection.state["snapshot_provenance"]), 2)
        self.assertEqual(len(self.connection.state["provenance"]), 2)

    def test_same_timestamp_price_correction_creates_a_distinct_immutable_snapshot(self):
        first = self._ledger().persist(self._payload(), now=self.now)
        corrected = self._ledger().persist(
            self._payload(
                quote=self._quote(price=-105.0),
                raw_payload=b'{"response":2,"price":-105}',
                received_at=self.now + timedelta(seconds=1),
            ),
            now=self.now + timedelta(seconds=1),
        )

        self.assertEqual(first.snapshots_created, 1)
        self.assertEqual(corrected.snapshots_created, 1)
        self.assertEqual(len(self.connection.state["snapshots"]), 2)

    def test_completed_result_replay_is_idempotent_and_linked_to_private_evidence(self):
        payload = self._results_payload()

        first = self._ledger().persist_results(payload, now=self.now)
        replay = self._ledger().persist_results(payload, now=self.now)

        self.assertEqual(first.status, "accepted")
        self.assertEqual(first.events_created, 1)
        self.assertEqual(first.results_created, 1)
        self.assertEqual(first.provenance_links_created, 1)
        self.assertEqual(replay.results_created, 0)
        self.assertEqual(replay.results_replayed, 1)
        self.assertEqual(replay.provenance_links_created, 0)
        self.assertEqual(len(self.connection.state["results"]), 1)
        self.assertEqual(len(self.connection.state["result_provenance"]), 1)
        self.assertEqual(self.store.stored_count, 1)

    def test_provider_result_correction_appends_a_new_immutable_version(self):
        self._ledger().persist_results(self._results_payload(), now=self.now)
        corrected_score = self._score(
            home_score=102,
            last_update=self.now,
        )

        corrected = self._ledger().persist_results(
            self._results_payload(
                score=corrected_score,
                raw_payload=b'{"scores":2,"home":102}',
                received_at=self.now + timedelta(seconds=1),
            ),
            now=self.now + timedelta(seconds=1),
        )

        self.assertEqual(corrected.events_created, 0)
        self.assertEqual(corrected.results_created, 1)
        self.assertEqual(corrected.results_replayed, 0)
        self.assertEqual(len(self.connection.state["results"]), 2)

    def test_result_tolerates_existing_event_start_and_league_drift(self):
        self._ledger().persist(self._payload(), now=self.now)
        drifted_score = self._score(
            league="National Basketball Association",
            commence_time=self.now - timedelta(hours=4),
        )

        result = self._ledger().persist_results(
            self._results_payload(score=drifted_score),
            now=self.now,
        )

        self.assertEqual(result.status, "accepted")
        self.assertEqual(result.events_created, 0)
        self.assertEqual(result.results_created, 1)
        self.assertEqual(result.incidents_created, 0)

    def test_result_identity_conflict_records_incident_without_result_fact(self):
        self._ledger().persist(self._payload(), now=self.now)
        conflicting_score = self._score(home_team="Different Home")

        result = self._ledger().persist_results(
            self._results_payload(score=conflicting_score),
            now=self.now,
        )

        self.assertEqual(result.status, "blocked_event_identity")
        self.assertEqual(result.results_created, 0)
        self.assertEqual(result.incidents_created, 1)
        self.assertEqual(self.connection.state["results"], {})
        signal = json.loads(self.connection.state["signals"][-1][-1])
        self.assertEqual(signal["status"], "blocked_event_identity")

    def test_completed_result_update_cannot_predate_event_start(self):
        impossible_score = self._score(
            commence_time=self.now - timedelta(hours=1),
            last_update=self.now - timedelta(hours=2),
        )

        with self.assertRaisesRegex(
            OddsLedgerValidationError,
            "cannot predate event start",
        ):
            self._ledger().persist_results(
                self._results_payload(score=impossible_score),
                now=self.now,
            )

        self.assertEqual(self.store.stored_count, 0)
        self.assertFalse(self.connection.executions)

    def test_event_identity_conflict_records_incident_without_appending_quote_facts(self):
        self._ledger().persist(self._payload(), now=self.now)
        conflicted = self._ledger().persist(
            self._payload(
                quote=self._quote(home_team="Different Home"),
                raw_payload=b'{"response":"conflicting-event"}',
                received_at=self.now + timedelta(seconds=1),
            ),
            now=self.now + timedelta(seconds=1),
        )

        self.assertEqual(conflicted.status, "blocked_event_identity")
        self.assertEqual(conflicted.incidents_created, 1)
        self.assertEqual(len(self.connection.state["snapshots"]), 1)
        self.assertEqual(len(self.connection.state["incidents"]), 1)
        signal = json.loads(self.connection.state["signals"][-1][-1])
        self.assertEqual(signal["status"], "blocked_event_identity")

    def test_invalid_pregame_input_and_unapproved_contract_never_store_or_write_evidence(self):
        live = self._quote()
        live = RawOddsQuote(**{**live.__dict__, "starts_at": self.now})
        with self.assertRaises(OddsLedgerValidationError):
            self._ledger().persist(self._payload(quote=live), now=self.now)
        self.assertEqual(self.store.stored_count, 0)
        self.assertFalse(self.connection.executions)

    def test_request_scope_rejects_credential_like_names_but_allows_sport_key(self):
        for unsafe_name in ("x-api-key", "api-key", "client_secret", "access_token"):
            with self.subTest(unsafe_name=unsafe_name):
                payload = self._payload()
                payload = PreparedOddsPayload(
                    **{**payload.__dict__, "request_scope": ((unsafe_name, "must-not-hash"),)}
                )
                with self.assertRaises(OddsLedgerValidationError):
                    self._ledger().persist(payload, now=self.now)
        self.assertEqual(self.store.stored_count, 0)
        self.assertFalse(self.connection.executions)

        unapproved = self._payload()
        unapproved = PreparedOddsPayload(
            **{**unapproved.__dict__, "license_version": "terms-unreviewed"}
        )
        with self.assertRaises(OddsLedgerValidationError):
            self._ledger().persist(unapproved, now=self.now)
        self.assertEqual(self.store.stored_count, 0)
        self.assertFalse(self.connection.executions)

    def test_raw_storage_failure_writes_no_database_facts(self):
        with self.assertRaises(OddsLedgerUnavailable):
            self._ledger(store=_FailingPayloadStore()).persist(self._payload(), now=self.now)
        self.assertFalse(self.connection.executions)

    def test_prepares_a_private_ledger_payload_from_a_completed_provider_fetch(self):
        quote = self._quote()
        fetched = OddsApiFetch(
            quotes=[quote],
            requests_remaining=499,
            requests_used=1,
            request_cost=1,
            skipped_live_events=2,
            raw_payload=b'{"provider":"response"}',
            received_at=self.now,
            request_scope=OddsApiRequestScope(
                sport_key="basketball_nba",
                regions=("us",),
                markets=("h2h",),
                bookmakers=("example_book",),
            ),
        )

        prepared = prepare_the_odds_api_payload(
            fetched,
            license_scope="internal_analytics_only",
            license_version="terms-2026-08-31",
        )

        self.assertEqual(prepared.provider, "the_odds_api")
        self.assertEqual(prepared.raw_payload, fetched.raw_payload)
        self.assertEqual(prepared.received_at, self.now)
        self.assertIn(("sport_key", "basketball_nba"), prepared.request_scope)
        self.assertNotIn("apiKey", repr(prepared.request_scope))
        self.assertEqual(prepared.requests_remaining, 499)

    def test_prepares_a_private_results_payload_from_a_scores_fetch(self):
        score = self._score()
        fetched = ScoresApiFetch(
            scores=(score,),
            requests_remaining=498,
            requests_used=2,
            request_cost=1,
            skipped_incomplete_events=3,
            raw_payload=b'{"provider":"scores"}',
            received_at=self.now,
            request_scope=ScoresApiRequestScope(
                sport_key="basketball_nba",
                days_from=3,
            ),
        )

        prepared = prepare_the_odds_api_results_payload(
            fetched,
            license_scope="internal_analytics_only",
            license_version="terms-2026-08-31",
        )

        self.assertEqual(prepared.provider, "the_odds_api")
        self.assertEqual(prepared.source_type, "result")
        self.assertEqual(prepared.raw_payload, fetched.raw_payload)
        self.assertEqual(prepared.captured_at, score.last_update)
        self.assertEqual(prepared.scores, (score,))
        self.assertEqual(
            prepared.request_scope,
            (("sport_key", "basketball_nba"), ("days_from", "3")),
        )
        self.assertNotIn("apiKey", repr(prepared.request_scope))

    def test_empty_provider_response_is_retained_as_explicit_evidence(self):
        fetched = OddsApiFetch(
            quotes=[],
            requests_remaining=500,
            requests_used=0,
            request_cost=0,
            skipped_live_events=0,
            raw_payload=b"[]",
            received_at=self.now,
            request_scope=OddsApiRequestScope(
                sport_key="basketball_nba",
                regions=("us",),
                markets=("h2h",),
            ),
        )

        prepared = prepare_the_odds_api_payload(
            fetched,
            license_scope="internal_analytics_only",
            license_version="terms-2026-08-31",
        )
        result = self._ledger().persist(prepared, now=self.now)

        self.assertEqual(prepared.captured_at, self.now)
        self.assertEqual(result.status, "accepted_empty")
        self.assertEqual(result.events_created, 0)
        self.assertEqual(result.snapshots_created, 0)
        self.assertEqual(len(self.connection.state["receipts"]), 1)
        self.assertEqual(len(self.connection.state["provenance"]), 1)
        self.assertEqual(self.connection.state["events"], {})
        self.assertEqual(self.connection.state["snapshots"], {})
        signal = json.loads(self.connection.state["signals"][-1][-1])
        self.assertEqual(signal["status"], "accepted_empty")
        self.assertEqual(self.store.stored_count, 1)

    def test_database_failure_never_commits_a_success_signal(self):
        failing_connection = _LedgerConnection(fail_on_snapshot=True)
        with self.assertRaises(OddsLedgerUnavailable) as context:
            self._ledger(connection=failing_connection).persist(self._payload(), now=self.now)

        self.assertNotIn("postgresql://", str(context.exception))
        self.assertEqual(failing_connection.state["signals"], [])
        self.assertEqual(failing_connection.state["snapshots"], {})
        self.assertEqual(failing_connection.state["receipts"], {})
        self.assertEqual(self.store.stored_count, 1)
        self.assertTrue(failing_connection.closed)

    def test_result_database_failure_leaves_only_content_addressed_raw_evidence(self):
        failing_connection = _LedgerConnection(fail_on_result=True)

        with self.assertRaises(OddsLedgerUnavailable):
            self._ledger(connection=failing_connection).persist_results(
                self._results_payload(),
                now=self.now,
            )

        self.assertEqual(failing_connection.state["signals"], [])
        self.assertEqual(failing_connection.state["results"], {})
        self.assertEqual(failing_connection.state["receipts"], {})
        self.assertEqual(self.store.stored_count, 1)
        self.assertTrue(failing_connection.closed)


if __name__ == "__main__":
    unittest.main()
