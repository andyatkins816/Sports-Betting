import csv
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from contextlib import redirect_stdout
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from sam_analytics.historical_backfill import (
    MAX_CREDITS_PER_SNAPSHOT,
    RESULT_AVAILABILITY_CONVENTION,
    ExistingMlbEvent,
    HistoricalBackfillError,
    _historical_request_fingerprint,
    _read_archive_bytes,
    historical_odds_schedule,
    import_retrosheet_results,
    load_persisted_historical_snapshots,
    main,
    plan_historical_odds_backfill,
    plan_retrosheet_results,
    run_historical_odds_backfill,
)
from sam_analytics.odds_ledger import LedgerWriteResult, ResultsLedgerWriteResult
from sam_analytics.providers.the_odds_api import OddsApiFetch, OddsApiRequestScope
from sam_analytics.readiness import DatabaseReadiness


class HistoricalBackfillTests(unittest.TestCase):
    def test_network_free_default_plan_is_exact(self):
        schedule = historical_odds_schedule()
        plan = plan_historical_odds_backfill()

        self.assertEqual(len(schedule), 186)
        self.assertEqual(schedule[0], datetime(2025, 3, 27, 16, tzinfo=UTC))
        self.assertEqual(schedule[-1], datetime(2025, 9, 28, 16, tzinfo=UTC))
        self.assertTrue(all(moment.hour == 16 and moment.tzinfo is UTC for moment in schedule))
        self.assertEqual(plan.max_calls, 186)
        self.assertEqual(plan.max_credits, 1_860)

        output = io.StringIO()
        with patch(
            "sam_analytics.historical_backfill._runtime_components",
            side_effect=AssertionError("dry run constructed live dependencies"),
        ), redirect_stdout(output):
            self.assertEqual(main([]), 0)
        rendered = json.loads(output.getvalue())
        self.assertEqual(rendered["snapshots"], 186)
        self.assertEqual(rendered["max_credits"], 1_860)
        self.assertFalse(rendered["execution"])

    def test_run_requires_exact_outstanding_confirmation_before_fetch(self):
        schedule = historical_odds_schedule()
        client = _RecordingHistoricalClient()
        ledger = _RecordingLedger(client.actions)
        progress = []

        def resume_reader(_database_url):
            return schedule[:-2]

        with self.assertRaisesRegex(
            HistoricalBackfillError,
            "--max-calls 2 and --max-credits 20",
        ):
            run_historical_odds_backfill(
                client,
                ledger,
                database_url="postgresql://unused",
                license_scope="internal_analytics_only",
                license_version="terms-2026-08-31",
                max_calls=186,
                max_credits=1_860,
                resume_reader=resume_reader,
            )
        self.assertEqual(client.actions, [])

        report = run_historical_odds_backfill(
            client,
            ledger,
            database_url="postgresql://unused",
            license_scope="internal_analytics_only",
            license_version="terms-2026-08-31",
            max_calls=2,
            max_credits=2 * MAX_CREDITS_PER_SNAPSHOT,
            resume_reader=resume_reader,
            progress_writer=progress.append,
        )

        self.assertEqual(
            client.actions,
            [
                ("fetch", schedule[-2]),
                ("persist", schedule[-2]),
                ("fetch", schedule[-1]),
                ("persist", schedule[-1]),
            ],
        )
        self.assertTrue(report.complete)
        self.assertEqual(report.already_persisted, 184)
        self.assertEqual(report.calls_completed, 2)
        self.assertEqual(report.credits_used, 20)
        self.assertEqual(
            [json.loads(message)["completed"] for message in progress],
            [1, 2],
        )

    def test_stale_schema_preflight_prevents_paid_odds_fetch(self):
        client = _RecordingHistoricalClient()
        ledger = _RecordingLedger(client.actions)

        with (
            patch(
                "sam_analytics.historical_backfill._runtime_components",
                return_value=(
                    client,
                    ledger,
                    "postgresql://unused",
                    "internal_analytics_only",
                    "terms-2026-08-31",
                ),
            ),
            patch(
                "sam_analytics.historical_backfill.probe_postgres",
                return_value=DatabaseReadiness(reachable=True, migrations_current=False),
            ),
            self.assertRaisesRegex(HistoricalBackfillError, "current migrations"),
        ):
            main(["--execute", "--max-calls", "186", "--max-credits", "1860"])

        self.assertEqual(client.actions, [])

    def test_stale_schema_preflight_prevents_results_persistence(self):
        client = _RecordingHistoricalClient()
        ledger = _RecordingResultsLedger()

        with (
            patch(
                "sam_analytics.historical_backfill._runtime_components",
                return_value=(
                    client,
                    ledger,
                    "postgresql://unused",
                    "internal_analytics_only",
                    "terms-2026-08-31",
                ),
            ),
            patch(
                "sam_analytics.historical_backfill.probe_postgres",
                return_value=DatabaseReadiness(reachable=True, migrations_current=False),
            ),
            patch(
                "sam_analytics.historical_backfill.import_retrosheet_results"
            ) as import_results,
            self.assertRaisesRegex(HistoricalBackfillError, "current migrations"),
        ):
            main(
                [
                    "--execute",
                    "--retrosheet-zip",
                    "/tmp/gl2025.zip",
                    "--retrosheet-sha256",
                    "a" * 64,
                ]
            )

        import_results.assert_not_called()
        self.assertIsNone(ledger.payload)

    def test_resume_accepts_only_receipt_provenance_and_accepted_signal_query(self):
        snapshot = historical_odds_schedule()[7]
        digest = _historical_request_fingerprint(snapshot)
        cursor = _ResumeCursor([(digest,)])
        connection = _FakeConnection(cursor)

        with patch(
            "sam_analytics.historical_backfill._connect_read_only",
            return_value=connection,
        ):
            persisted = load_persisted_historical_snapshots("postgresql://unused")

        self.assertEqual(persisted, frozenset({snapshot}))
        query = " ".join(cursor.query.split())
        self.assertIn("JOIN raw_data_provenance", query)
        self.assertIn("JOIN operational_signal", query)
        self.assertIn("'accepted', 'accepted_empty'", query)
        self.assertTrue(connection.closed)

    def test_run_rejects_changed_cost_and_stops_before_insufficient_quota(self):
        schedule = historical_odds_schedule()
        persisted = schedule[:-2]

        wrong_cost = _RecordingHistoricalClient(request_cost=9)
        wrong_cost_ledger = _RecordingLedger(wrong_cost.actions)
        with self.assertRaisesRegex(HistoricalBackfillError, "invalid credit cost"):
            run_historical_odds_backfill(
                wrong_cost,
                wrong_cost_ledger,
                database_url="postgresql://unused",
                license_scope="internal_analytics_only",
                license_version="terms-2026-08-31",
                max_calls=2,
                max_credits=20,
                resume_reader=lambda _database_url: persisted,
            )
        self.assertEqual(wrong_cost.actions, [("fetch", schedule[-2])])

        missing_headers = _RecordingHistoricalClient(requests_remaining=None)
        missing_headers_ledger = _RecordingLedger(missing_headers.actions)
        with self.assertRaisesRegex(HistoricalBackfillError, "invalid quota headers"):
            run_historical_odds_backfill(
                missing_headers,
                missing_headers_ledger,
                database_url="postgresql://unused",
                license_scope="internal_analytics_only",
                license_version="terms-2026-08-31",
                max_calls=2,
                max_credits=20,
                resume_reader=lambda _database_url: persisted,
            )
        self.assertEqual(missing_headers.actions, [("fetch", schedule[-2])])

        low_quota = _RecordingHistoricalClient(requests_remaining=9)
        low_quota_ledger = _RecordingLedger(low_quota.actions)
        with self.assertRaisesRegex(HistoricalBackfillError, "cannot cover the next"):
            run_historical_odds_backfill(
                low_quota,
                low_quota_ledger,
                database_url="postgresql://unused",
                license_scope="internal_analytics_only",
                license_version="terms-2026-08-31",
                max_calls=2,
                max_credits=20,
                resume_reader=lambda _database_url: persisted,
            )
        self.assertEqual(
            low_quota.actions,
            [("fetch", schedule[-2]), ("persist", schedule[-2])],
        )

    def test_retrosheet_import_matches_ordered_games_and_fails_closed(self):
        rows = [
            _game_row("20250327", "0", "CHN", "ARI", "10", "6"),
            _game_row("20250401", "1", "BOS", "NYA", "2", "3"),
            _game_row("20250401", "2", "BOS", "NYA", "7", "1"),
            _game_row("20250402", "0", "MIA", "ATL", "4", "5"),
            _game_row("20250403", "0", "CIN", "PIT", "4", "2"),
            _game_row("20250405", "0", "ANA", "HOU", "5", "3"),
            _game_row(
                "20250404",
                "0",
                "SEA",
                "TEX",
                "3",
                "1",
                completion_info="20250405,TEX01,7,0,42",
            ),
        ]
        archive = _archive(rows)
        digest = hashlib.sha256(archive).hexdigest()
        events = (
            # A 10:10pm Pacific start is 05:10 UTC next day but belongs to 03/27.
            ExistingMlbEvent(
                "single",
                datetime(2025, 3, 28, 5, 10, tzinfo=UTC),
                "Arizona Diamondbacks",
                "Chicago Cubs",
            ),
            ExistingMlbEvent(
                "dh-early",
                datetime(2025, 4, 1, 17, tzinfo=UTC),
                "New York Yankees",
                "Boston Red Sox",
            ),
            ExistingMlbEvent(
                "dh-late",
                datetime(2025, 4, 1, 23, tzinfo=UTC),
                "New York Yankees",
                "Boston Red Sox",
            ),
            # Two targets for one source game are ambiguous and must both be ignored.
            ExistingMlbEvent(
                "ambiguous-1",
                datetime(2025, 4, 3, 17, tzinfo=UTC),
                "Pittsburgh Pirates",
                "Cincinnati Reds",
            ),
            ExistingMlbEvent(
                "angels",
                datetime(2025, 4, 5, 20, tzinfo=UTC),
                "Houston Astros",
                "Los Angeles Angels",
            ),
            ExistingMlbEvent(
                "ambiguous-2",
                datetime(2025, 4, 3, 23, tzinfo=UTC),
                "Pittsburgh Pirates",
                "Cincinnati Reds",
            ),
        )
        now = datetime(2025, 9, 30, tzinfo=UTC)
        ledger = _RecordingResultsLedger()
        training_calls = []

        def training_loader(database_url, *, sport, training_cutoff):
            training_calls.append((database_url, sport, training_cutoff))
            return (object(), object())

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gl2025.zip"
            path.write_bytes(archive)
            dry_run = plan_retrosheet_results(path, expected_sha256=digest)
            report = import_retrosheet_results(
                path,
                expected_sha256=digest,
                database_url="postgresql://unused",
                ledger=ledger,
                existing_events=events,
                received_at=now,
                training_row_loader=training_loader,
            )

        self.assertEqual(dry_run.source_games, 7)
        self.assertEqual(dry_run.eligible_games, 6)
        self.assertEqual(report.matched_games, 4)
        self.assertEqual(report.unmatched_games, 1)
        self.assertEqual(report.ambiguous_games, 1)
        self.assertEqual(report.completion_info_games_excluded, 1)
        self.assertEqual(report.results_created, 4)
        self.assertEqual(report.eligible_training_rows, 2)
        self.assertEqual(report.availability_convention, RESULT_AVAILABILITY_CONVENTION)
        self.assertEqual(training_calls, [("postgresql://unused", "baseball_mlb", now)])

        payload = ledger.payload
        self.assertEqual(payload.provider, "retrosheet")
        self.assertEqual(payload.source_type, "result")
        self.assertEqual(payload.raw_payload, archive)
        self.assertIn(RESULT_AVAILABILITY_CONVENTION, payload.schema_version)
        self.assertIn(
            ("availability_convention", RESULT_AVAILABILITY_CONVENTION),
            payload.request_scope,
        )
        by_target = {score.matched_provider_event_id: score for score in payload.scores}
        self.assertEqual(by_target["dh-early"].away_score, 2)
        self.assertEqual(by_target["dh-late"].away_score, 7)
        self.assertEqual(by_target["angels"].away_team, "Los Angeles Angels")
        self.assertEqual(
            by_target["single"].source_available_at,
            by_target["single"].commence_time + timedelta(hours=12),
        )
        self.assertTrue(
            all(
                score.provider == "retrosheet"
                and score.matched_event_provider == "the_odds_api"
                and score.last_update == score.source_available_at
                for score in payload.scores
            )
        )

    def test_retrosheet_import_rejects_any_target_event_creation(self):
        archive = _archive([_game_row("20250327", "0", "CHN", "ARI", "10", "6")])
        event = ExistingMlbEvent(
            "single",
            datetime(2025, 3, 27, 23, tzinfo=UTC),
            "Arizona Diamondbacks",
            "Chicago Cubs",
        )
        ledger = _RecordingResultsLedger(events_created=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gl2025.zip"
            path.write_bytes(archive)
            with self.assertRaisesRegex(HistoricalBackfillError, "create target events"):
                import_retrosheet_results(
                    path,
                    expected_sha256=hashlib.sha256(archive).hexdigest(),
                    database_url="postgresql://unused",
                    ledger=ledger,
                    existing_events=(event,),
                    received_at=datetime(2025, 9, 30, tzinfo=UTC),
                    training_row_loader=lambda *_args, **_kwargs: (),
                )

    def test_retrosheet_import_read_is_bounded_before_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oversized.zip"
            path.write_bytes(b"12345")
            with patch(
                "sam_analytics.historical_backfill._MAX_RETROSHEET_ARCHIVE_BYTES",
                4,
            ), self.assertRaisesRegex(HistoricalBackfillError, "exceeds the size limit"):
                _read_archive_bytes(path)


class _RecordingHistoricalClient:
    def __init__(self, *, request_cost=10, requests_remaining=1_000, requests_used=10):
        self.actions = []
        self.request_cost = request_cost
        self.requests_remaining = requests_remaining
        self.requests_used = requests_used

    def fetch_historical_odds(self, sport_key, *, snapshot_at, regions, markets):
        self.assert_scope = (sport_key, regions, markets)
        if self.assert_scope != ("baseball_mlb", "us", ("h2h",)):
            raise AssertionError("unsafe historical request scope")
        self.actions.append(("fetch", snapshot_at))
        return OddsApiFetch(
            quotes=[],
            requests_remaining=self.requests_remaining,
            requests_used=self.requests_used,
            request_cost=self.request_cost,
            skipped_live_events=0,
            raw_payload=("snapshot:" + snapshot_at.isoformat()).encode(),
            received_at=datetime(2026, 9, 6, tzinfo=UTC),
            request_scope=OddsApiRequestScope(
                sport_key=sport_key,
                regions=(regions,),
                markets=markets,
                snapshot_at=snapshot_at,
            ),
            source_available_at=snapshot_at,
        )


class _RecordingLedger:
    def __init__(self, actions):
        self.actions = actions

    def persist(self, payload):
        snapshot_at = dict(payload.request_scope)["snapshot_at"]
        parsed = datetime.fromisoformat(snapshot_at.replace("Z", "+00:00"))
        self.actions.append(("persist", parsed))
        return LedgerWriteResult(
            status="accepted_empty",
            receipt_sha256="a" * 64,
            provenance_sha256="b" * 64,
            events_created=0,
            snapshots_created=0,
            snapshots_replayed=0,
            provenance_links_created=0,
            incidents_created=0,
        )


class _RecordingResultsLedger:
    def __init__(self, *, events_created=0):
        self.events_created = events_created
        self.payload = None

    def persist_results(self, payload, *, now):
        self.payload = payload
        return ResultsLedgerWriteResult(
            status="accepted",
            receipt_sha256="a" * 64,
            provenance_sha256="b" * 64,
            events_created=self.events_created,
            results_created=len(payload.scores),
            results_replayed=0,
            provenance_links_created=len(payload.scores),
            incidents_created=0,
        )


class _ResumeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.query = ""

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def execute(self, query, params=None):
        if params is not None:
            raise AssertionError("resume query needs no dynamic values")
        self.query = query

    def fetchall(self):
        return self.rows


class _FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.closed = False

    def cursor(self):
        return self._cursor

    def close(self):
        self.closed = True


def _game_row(
    day,
    game_number,
    away,
    home,
    away_score,
    home_score,
    *,
    completion_info="",
):
    row = [""] * 161
    row[0] = day
    row[1] = game_number
    row[3] = away
    row[4] = "NL" if away in {"ARI", "ATL", "CHN", "CIN", "MIA", "PIT"} else "AL"
    row[6] = home
    row[7] = "NL" if home in {"ARI", "ATL", "CHN", "CIN", "MIA", "PIT"} else "AL"
    row[9] = away_score
    row[10] = home_score
    row[13] = completion_info
    row[160] = "Y"
    return row


def _archive(rows):
    text = io.StringIO(newline="")
    csv.writer(text, lineterminator="\n").writerows(rows)
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("gl2025.txt", text.getvalue().encode())
    return payload.getvalue()


if __name__ == "__main__":
    unittest.main()
