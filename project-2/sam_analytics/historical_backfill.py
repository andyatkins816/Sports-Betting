"""Bounded, resumable 2025 MLB evidence backfill.

Running this module without ``--execute`` is a network-free dry run.  Live
historical-odds execution is deliberately sequential and requires confirmations
that exactly match the outstanding request count and its worst-case credit cost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from collections.abc import Callable, Collection, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

from sam_analytics.odds_ledger import (
    OddsLedger,
    PreparedResultsPayload,
    prepare_the_odds_api_payload,
)
from sam_analytics.providers.retrosheet import RetrosheetGame, parse_retrosheet_2025
from sam_analytics.providers.the_odds_api import CompletedScore, TheOddsApiClient
from sam_analytics.readiness import probe_postgres

MLB_SPORT_KEY = "baseball_mlb"
HISTORICAL_SOURCE_TYPE = "historical_odds"
RESULT_SOURCE_TYPE = "result"
SNAPSHOT_FIRST_DATE = date(2025, 3, 27)
SNAPSHOT_LAST_DATE = date(2025, 9, 28)
SNAPSHOT_HOUR_UTC = 16
MAX_CREDITS_PER_SNAPSHOT = 10
RESULT_AVAILABILITY_CONVENTION = "derived-game-final-v1"
RESULT_AVAILABILITY_DELAY = timedelta(hours=12)
RETROSHEET_LICENSE_SCOPE = "commercial_use_with_attribution"
RETROSHEET_LICENSE_VERSION = "notice-2026-09-06"
RETROSHEET_SCHEMA_VERSION = f"retrosheet-gl2025-{RESULT_AVAILABILITY_CONVENTION}"

_MLB_OFFICIAL_DATE_OFFSET = timedelta(hours=12)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_RETROSHEET_ARCHIVE_BYTES = 25 * 1024 * 1024

# The Odds API's exact MLB display names, with the franchise's transitional
# 2025 name accepted explicitly.  No fuzzy or reversed-team matching is used.
_TEAM_CODE_BY_PROVIDER_NAME = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago White Sox": "CHA",
    "Chicago Cubs": "CHN",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KCA",
    "Los Angeles Angels": "ANA",
    "Los Angeles Dodgers": "LAN",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Yankees": "NYA",
    "New York Mets": "NYN",
    "Athletics": "ATH",
    "Oakland Athletics": "ATH",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDN",
    "Seattle Mariners": "SEA",
    "San Francisco Giants": "SFN",
    "St. Louis Cardinals": "SLN",
    "Tampa Bay Rays": "TBA",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WAS",
}


class HistoricalBackfillError(RuntimeError):
    """A backfill precondition or fail-closed integrity check failed."""


@dataclass(frozen=True)
class HistoricalOddsPlan:
    first_snapshot_at: datetime
    last_snapshot_at: datetime
    snapshots: int
    max_calls: int
    max_credits_per_call: int
    max_credits: int
    execution: bool = False


@dataclass(frozen=True)
class HistoricalOddsReport:
    scheduled_snapshots: int
    already_persisted: int
    outstanding_snapshots: int
    calls_completed: int
    snapshots_persisted: int
    credits_used: int
    max_calls_confirmed: int
    max_credits_confirmed: int
    complete: bool


@dataclass(frozen=True)
class ExistingMlbEvent:
    provider_event_id: str
    starts_at: datetime
    home_team: str
    away_team: str


@dataclass(frozen=True)
class RetrosheetPlan:
    source_games: int
    eligible_games: int
    completion_info_games_excluded: int
    availability_convention: str
    availability_delay_hours: int
    execution: bool = False


@dataclass(frozen=True)
class RetrosheetImportReport:
    source_games: int
    eligible_source_games: int
    matched_games: int
    unmatched_games: int
    ambiguous_games: int
    completion_info_games_excluded: int
    results_created: int
    results_replayed: int
    eligible_training_rows: int | None
    availability_convention: str
    availability_delay_hours: int


def historical_odds_schedule() -> tuple[datetime, ...]:
    """Return every requested 2025 MLB snapshot at exactly 16:00 UTC."""

    days = (SNAPSHOT_LAST_DATE - SNAPSHOT_FIRST_DATE).days + 1
    return tuple(
        datetime.combine(SNAPSHOT_FIRST_DATE + timedelta(days=offset), time(16), UTC)
        for offset in range(days)
    )


def plan_historical_odds_backfill() -> HistoricalOddsPlan:
    """Return the fresh-run ceiling without reading a database or provider."""

    schedule = historical_odds_schedule()
    maximum = len(schedule) * MAX_CREDITS_PER_SNAPSHOT
    return HistoricalOddsPlan(
        first_snapshot_at=schedule[0],
        last_snapshot_at=schedule[-1],
        snapshots=len(schedule),
        max_calls=len(schedule),
        max_credits_per_call=MAX_CREDITS_PER_SNAPSHOT,
        max_credits=maximum,
    )


def load_persisted_historical_snapshots(database_url: str) -> frozenset[datetime]:
    """Load only request fingerprints backed by complete accepted evidence."""

    connection = _connect_read_only(database_url, application_name="sam-historical-resume")
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT DISTINCT receipt.request_fingerprint_sha256
                FROM provider_payload_receipt AS receipt
                JOIN raw_data_provenance AS provenance
                  ON provenance.provider_payload_receipt_id = receipt.id
                 AND provenance.provider = receipt.provider
                 AND provenance.source_type = receipt.source_type
                 AND provenance.payload_sha256 = receipt.payload_sha256
                 AND provenance.received_at = receipt.received_at
                JOIN operational_signal AS signal
                  ON signal.provenance_sha256 = provenance.provenance_sha256
                 AND signal.signal_type = 'provider'
                 AND signal.source = 'odds_ledger'
                 AND signal.observed_at = receipt.captured_at
                 AND signal.received_at = receipt.received_at
                WHERE receipt.provider = 'the_odds_api'
                  AND receipt.source_type = 'historical_odds'
                  AND signal.payload->>'source_type' = receipt.source_type
                  AND signal.payload->>'status' IN ('accepted', 'accepted_empty')
                """
            )
            rows = tuple(cursor.fetchall())
    except Exception:
        raise HistoricalBackfillError("historical resume evidence could not be read") from None
    finally:
        _close_connection(connection)

    schedule_by_digest = {
        _historical_request_fingerprint(snapshot_at): snapshot_at
        for snapshot_at in historical_odds_schedule()
    }
    found: set[datetime] = set()
    for row in rows:
        if not isinstance(row, Sequence) or len(row) != 1:
            raise HistoricalBackfillError("historical resume evidence is malformed")
        digest = str(row[0]).strip()
        if not _SHA256_RE.fullmatch(digest):
            raise HistoricalBackfillError("historical resume evidence is malformed")
        snapshot_at = schedule_by_digest.get(digest)
        if snapshot_at is not None:
            found.add(snapshot_at)
    return frozenset(found)


def run_historical_odds_backfill(
    client: Any,
    ledger: Any,
    *,
    database_url: str,
    license_scope: str,
    license_version: str,
    max_calls: int,
    max_credits: int,
    resume_reader: Callable[[str], Collection[datetime]] = load_persisted_historical_snapshots,
    progress_writer: Callable[[str], None] | None = None,
) -> HistoricalOddsReport:
    """Fetch and immediately persist each outstanding snapshot, without retries."""

    schedule = historical_odds_schedule()
    persisted = _validated_persisted_schedule(resume_reader(database_url), schedule=schedule)
    outstanding = tuple(snapshot for snapshot in schedule if snapshot not in persisted)
    expected_calls = len(outstanding)
    expected_credits = expected_calls * MAX_CREDITS_PER_SNAPSHOT
    _require_exact_execution_confirmation(
        max_calls=max_calls,
        max_credits=max_credits,
        expected_calls=expected_calls,
        expected_credits=expected_credits,
    )
    if progress_writer is not None and not callable(progress_writer):
        raise TypeError("progress_writer must be callable or None")

    calls_completed = 0
    credits_used = 0
    persisted_count = 0
    for snapshot_at in outstanding:
        if calls_completed >= max_calls or credits_used + MAX_CREDITS_PER_SNAPSHOT > max_credits:
            raise HistoricalBackfillError("historical request ceiling would be exceeded")
        fetched = client.fetch_historical_odds(
            MLB_SPORT_KEY,
            snapshot_at=snapshot_at,
            regions="us",
            markets=("h2h",),
        )
        request_cost = getattr(fetched, "request_cost", None)
        quota_values = (
            getattr(fetched, "requests_remaining", None),
            getattr(fetched, "requests_used", None),
        )
        if request_cost != MAX_CREDITS_PER_SNAPSHOT or isinstance(request_cost, bool):
            raise HistoricalBackfillError("historical request returned an invalid credit cost")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in quota_values
        ):
            raise HistoricalBackfillError("historical request returned invalid quota headers")
        payload = prepare_the_odds_api_payload(
            fetched,
            license_scope=license_scope,
            license_version=license_version,
            source_type=HISTORICAL_SOURCE_TYPE,
        )
        result = ledger.persist(payload)
        if getattr(result, "status", None) not in {"accepted", "accepted_empty"}:
            raise HistoricalBackfillError("historical snapshot was not accepted by the ledger")
        calls_completed += 1
        credits_used += request_cost
        persisted_count += 1
        if progress_writer is not None:
            progress_writer(
                json.dumps(
                    {
                        "completed": calls_completed,
                        "credits_used": credits_used,
                        "snapshot_at": snapshot_at.isoformat().replace("+00:00", "Z"),
                        "total": expected_calls,
                        "type": "historical_odds_progress",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        if calls_completed < expected_calls and quota_values[0] < MAX_CREDITS_PER_SNAPSHOT:
            raise HistoricalBackfillError("provider quota cannot cover the next confirmed call")

    return HistoricalOddsReport(
        scheduled_snapshots=len(schedule),
        already_persisted=len(persisted),
        outstanding_snapshots=expected_calls,
        calls_completed=calls_completed,
        snapshots_persisted=persisted_count,
        credits_used=credits_used,
        max_calls_confirmed=max_calls,
        max_credits_confirmed=max_credits,
        complete=len(persisted) + persisted_count == len(schedule),
    )


def plan_retrosheet_results(
    archive_path: str | os.PathLike[str], *, expected_sha256: str
) -> RetrosheetPlan:
    """Validate a pinned local archive and report import eligibility only."""

    games = parse_retrosheet_2025(archive_path, expected_sha256=expected_sha256)
    excluded = sum(game.completion_info is not None for game in games)
    return RetrosheetPlan(
        source_games=len(games),
        eligible_games=len(games) - excluded,
        completion_info_games_excluded=excluded,
        availability_convention=RESULT_AVAILABILITY_CONVENTION,
        availability_delay_hours=int(RESULT_AVAILABILITY_DELAY.total_seconds() // 3600),
    )


def load_existing_mlb_events(database_url: str) -> tuple[ExistingMlbEvent, ...]:
    """Read only existing The Odds API MLB identities in the target date window."""

    # A 12:00Z-to-12:00Z slate window covers the bounded US regular season,
    # including 10:10pm Pacific starts that occur after midnight Eastern/UTC.
    first = datetime.combine(SNAPSHOT_FIRST_DATE, time.min, UTC) + _MLB_OFFICIAL_DATE_OFFSET
    last = (
        datetime.combine(SNAPSHOT_LAST_DATE + timedelta(days=1), time.min, UTC)
        + _MLB_OFFICIAL_DATE_OFFSET
    )
    connection = _connect_read_only(database_url, application_name="sam-results-match")
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT provider_event_id, starts_at, home_team, away_team
                FROM sports_event
                WHERE provider = 'the_odds_api'
                  AND sport = 'baseball_mlb'
                  AND starts_at >= %s
                  AND starts_at < %s
                ORDER BY starts_at, provider_event_id
                """,
                (first, last),
            )
            rows = tuple(cursor.fetchall())
    except Exception:
        raise HistoricalBackfillError("existing MLB event identities could not be read") from None
    finally:
        _close_connection(connection)
    return _validated_existing_events(ExistingMlbEvent(*row) for row in rows)


def import_retrosheet_results(
    archive_path: str | os.PathLike[str],
    *,
    expected_sha256: str,
    database_url: str,
    ledger: Any,
    existing_events: Sequence[ExistingMlbEvent] | None = None,
    received_at: datetime | None = None,
    training_row_loader: Callable[..., Sequence[Any]] | None = None,
) -> RetrosheetImportReport:
    """Match a pinned local game log to existing events and persist results."""

    archive_bytes = _read_archive_bytes(archive_path)
    games = parse_retrosheet_2025(archive_bytes, expected_sha256=expected_sha256)
    digest = hashlib.sha256(archive_bytes).hexdigest()
    targets = (
        load_existing_mlb_events(database_url)
        if existing_events is None
        else _validated_existing_events(existing_events)
    )
    received_at = received_at or datetime.now(UTC)
    if not _aware(received_at):
        raise ValueError("received_at must be timezone-aware")
    received_at = received_at.astimezone(UTC)

    excluded = tuple(game for game in games if game.completion_info is not None)
    eligible = tuple(game for game in games if game.completion_info is None)
    scores, unmatched, ambiguous = _match_retrosheet_games(eligible, targets)
    created = replayed = 0
    if scores:
        latest_availability = max(score.source_available_at for score in scores)
        if latest_availability is None or latest_availability > received_at:
            raise HistoricalBackfillError("derived result availability is after local receipt")
        payload = PreparedResultsPayload(
            provider="retrosheet",
            source_type=RESULT_SOURCE_TYPE,
            raw_payload=archive_bytes,
            captured_at=latest_availability,
            received_at=received_at,
            schema_version=RETROSHEET_SCHEMA_VERSION,
            license_scope=RETROSHEET_LICENSE_SCOPE,
            license_version=RETROSHEET_LICENSE_VERSION,
            scores=scores,
            request_scope=(
                ("archive_sha256", digest),
                ("availability_convention", RESULT_AVAILABILITY_CONVENTION),
                ("matched_event_provider", "the_odds_api"),
                ("season", "2025"),
            ),
            content_type="application/zip",
            source_available_at=latest_availability,
        )
        result = ledger.persist_results(payload, now=received_at)
        if getattr(result, "status", None) != "accepted":
            raise HistoricalBackfillError("Retrosheet results were not accepted by the ledger")
        if getattr(result, "events_created", None) != 0:
            raise HistoricalBackfillError("Retrosheet import attempted to create target events")
        created = _nonnegative_result_count(result, "results_created")
        replayed = _nonnegative_result_count(result, "results_replayed")

    eligible_training_rows = _count_training_rows(
        database_url,
        training_cutoff=received_at,
        loader=training_row_loader,
    )
    return RetrosheetImportReport(
        source_games=len(games),
        eligible_source_games=len(eligible),
        matched_games=len(scores),
        unmatched_games=unmatched,
        ambiguous_games=ambiguous,
        completion_info_games_excluded=len(excluded),
        results_created=created,
        results_replayed=replayed,
        eligible_training_rows=eligible_training_rows,
        availability_convention=RESULT_AVAILABILITY_CONVENTION,
        availability_delay_hours=int(RESULT_AVAILABILITY_DELAY.total_seconds() // 3600),
    )


def _match_retrosheet_games(
    games: Sequence[RetrosheetGame], events: Sequence[ExistingMlbEvent]
) -> tuple[tuple[CompletedScore, ...], int, int]:
    source_groups: dict[tuple[date, str, str], list[RetrosheetGame]] = defaultdict(list)
    for game in games:
        source_groups[(game.game_date, game.away_team_code, game.home_team_code)].append(game)

    target_groups: dict[tuple[date, str, str], list[ExistingMlbEvent]] = defaultdict(list)
    for event in events:
        away_code = _TEAM_CODE_BY_PROVIDER_NAME.get(event.away_team)
        home_code = _TEAM_CODE_BY_PROVIDER_NAME.get(event.home_team)
        if away_code is None or home_code is None:
            continue
        event_date = (event.starts_at.astimezone(UTC) - _MLB_OFFICIAL_DATE_OFFSET).date()
        target_groups[(event_date, away_code, home_code)].append(event)

    matched: list[CompletedScore] = []
    unmatched = ambiguous = 0
    for key in sorted(source_groups):
        source = sorted(source_groups[key], key=lambda game: game.game_number)
        targets = sorted(
            target_groups.get(key, ()),
            key=lambda event: (event.starts_at, event.provider_event_id),
        )
        if not targets:
            unmatched += len(source)
            continue
        if len(source) != len(targets) or len({event.starts_at for event in targets}) != len(targets):
            ambiguous += len(source)
            continue
        for game, target in zip(source, targets, strict=True):
            available_at = target.starts_at + RESULT_AVAILABILITY_DELAY
            matched.append(
                CompletedScore(
                    provider="retrosheet",
                    event_id=(
                        f"{game.game_date:%Y%m%d}-{game.away_team_code}-"
                        f"{game.home_team_code}-{game.game_number}"
                    ),
                    sport=MLB_SPORT_KEY,
                    league="MLB",
                    commence_time=target.starts_at,
                    last_update=available_at,
                    home_team=target.home_team,
                    away_team=target.away_team,
                    home_score=game.home_score,
                    away_score=game.away_score,
                    source_available_at=available_at,
                    matched_event_provider="the_odds_api",
                    matched_provider_event_id=target.provider_event_id,
                )
            )
    return tuple(matched), unmatched, ambiguous


def _validated_existing_events(
    events: Collection[ExistingMlbEvent],
) -> tuple[ExistingMlbEvent, ...]:
    if isinstance(events, (str, bytes)):
        raise ValueError("existing_events must be a collection of event identities")
    validated = tuple(events)
    seen: set[str] = set()
    for event in validated:
        if not isinstance(event, ExistingMlbEvent):
            raise ValueError("existing_events contains an invalid event identity")
        if (
            not event.provider_event_id.strip()
            or not event.home_team.strip()
            or not event.away_team.strip()
            or event.home_team == event.away_team
            or not _aware(event.starts_at)
            or event.provider_event_id in seen
        ):
            raise ValueError("existing_events contains an invalid event identity")
        seen.add(event.provider_event_id)
    return tuple(
        sorted(validated, key=lambda event: (event.starts_at, event.provider_event_id))
    )


def _count_training_rows(
    database_url: str,
    *,
    training_cutoff: datetime,
    loader: Callable[..., Sequence[Any]] | None,
) -> int:
    if loader is None:
        from sam_analytics.modeling import load_h2h_market_training_rows

        loader = load_h2h_market_training_rows
    rows = loader(database_url, sport=MLB_SPORT_KEY, training_cutoff=training_cutoff)
    return len(rows)


def _historical_request_fingerprint(snapshot_at: datetime) -> str:
    scope = {
        "markets": "h2h",
        "regions": "us",
        "snapshot_at": snapshot_at.astimezone(UTC)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z"),
        "sport_key": MLB_SPORT_KEY,
    }
    encoded = json.dumps(scope, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validated_persisted_schedule(
    values: Collection[datetime], *, schedule: Sequence[datetime]
) -> frozenset[datetime]:
    if isinstance(values, (str, bytes)):
        raise HistoricalBackfillError("historical resume reader returned malformed evidence")
    expected = frozenset(schedule)
    try:
        persisted = frozenset(values)
    except TypeError:
        raise HistoricalBackfillError("historical resume reader returned malformed evidence") from None
    if any(not _aware(value) for value in persisted):
        raise HistoricalBackfillError("historical resume reader returned malformed evidence")
    normalized = frozenset(value.astimezone(UTC) for value in persisted)
    if not normalized <= expected:
        raise HistoricalBackfillError("historical resume reader returned an unexpected snapshot")
    return normalized


def _require_exact_execution_confirmation(
    *, max_calls: int, max_credits: int, expected_calls: int, expected_credits: int
) -> None:
    if (
        isinstance(max_calls, bool)
        or not isinstance(max_calls, int)
        or isinstance(max_credits, bool)
        or not isinstance(max_credits, int)
        or max_calls != expected_calls
        or max_credits != expected_credits
    ):
        raise HistoricalBackfillError(
            f"execution requires --max-calls {expected_calls} "
            f"and --max-credits {expected_credits}"
        )


def _connect_read_only(database_url: str, *, application_name: str) -> Any:
    if not isinstance(database_url, str) or not database_url.strip():
        raise HistoricalBackfillError("DATABASE_URL is required")
    try:
        import psycopg

        return psycopg.connect(
            database_url,
            application_name=application_name,
            connect_timeout=5,
            options="-c statement_timeout=30000 -c default_transaction_read_only=on",
        )
    except Exception:
        raise HistoricalBackfillError("historical backfill database is unavailable") from None


def _close_connection(connection: Any) -> None:
    try:
        connection.close()
    except Exception:
        raise HistoricalBackfillError("historical database cleanup failed") from None


def _read_archive_bytes(archive_path: str | os.PathLike[str]) -> bytes:
    if not isinstance(archive_path, (str, os.PathLike)):
        raise TypeError("archive_path must be a local filesystem path")
    try:
        path = Path(archive_path)
        if path.stat().st_size > _MAX_RETROSHEET_ARCHIVE_BYTES:
            raise HistoricalBackfillError("Retrosheet archive exceeds the size limit")
        with path.open("rb") as handle:
            payload = handle.read(_MAX_RETROSHEET_ARCHIVE_BYTES + 1)
    except OSError:
        raise HistoricalBackfillError("Retrosheet archive could not be read") from None
    if len(payload) > _MAX_RETROSHEET_ARCHIVE_BYTES:
        raise HistoricalBackfillError("Retrosheet archive exceeds the size limit")
    return payload


def _nonnegative_result_count(result: Any, name: str) -> int:
    value = getattr(result, name, None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HistoricalBackfillError("Retrosheet ledger returned malformed counters")
    return value


def _aware(value: object) -> bool:
    return isinstance(value, datetime) and value.tzinfo is not None and value.utcoffset() is not None


def _runtime_components() -> tuple[Any, OddsLedger, str, str, str]:
    from sam_analytics.provider_contracts import (
        ApprovedProviderContract,
        ProviderContractRegistry,
    )
    from sam_analytics.provider_shadow_settings import ProviderShadowSettings
    from sam_analytics.s3_payload_store import S3CompatibleRawPayloadStore

    settings = ProviderShadowSettings.from_environment(os.environ)
    if settings.sport_key != MLB_SPORT_KEY or settings.regions != ("us",) or settings.markets != (
        "h2h",
    ):
        raise HistoricalBackfillError("worker must be admitted for MLB h2h/us only")
    database_url = os.environ.get("DATABASE_URL", "")
    contracts = ProviderContractRegistry(
        [
            ApprovedProviderContract(
                provider="the_odds_api",
                license_scope=settings.license_scope,
                license_version=settings.license_version,
                permitted_source_types=frozenset({HISTORICAL_SOURCE_TYPE}),
            ),
            ApprovedProviderContract(
                provider="retrosheet",
                license_scope=RETROSHEET_LICENSE_SCOPE,
                license_version=RETROSHEET_LICENSE_VERSION,
                permitted_source_types=frozenset({RESULT_SOURCE_TYPE}),
            ),
        ]
    )
    ledger = OddsLedger(
        database_url,
        raw_payload_store=S3CompatibleRawPayloadStore.from_environment(os.environ),
        provider_contracts=contracts,
    )
    client = TheOddsApiClient(
        os.environ.get("ODDS_PROVIDER_API_KEY", ""),
        max_response_bytes=settings.raw_evidence_max_bytes,
    )
    return client, ledger, database_url, settings.license_scope, settings.license_version


def _require_current_schema(database_url: str) -> None:
    readiness = probe_postgres(database_url)
    if readiness.reachable is not True or readiness.migrations_current is not True:
        raise HistoricalBackfillError(
            "historical backfill requires a reachable database with current migrations"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bounded 2025 MLB evidence backfill")
    parser.add_argument("--execute", action="store_true", help="perform writes and provider calls")
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--max-credits", type=int)
    parser.add_argument("--retrosheet-zip")
    parser.add_argument("--retrosheet-sha256")
    args = parser.parse_args(argv)

    if not args.execute:
        if args.max_calls is not None or args.max_credits is not None:
            parser.error("credit confirmations are accepted only with --execute")
        report: object
        if args.retrosheet_zip is not None or args.retrosheet_sha256 is not None:
            if not args.retrosheet_zip or not args.retrosheet_sha256:
                parser.error("Retrosheet dry-run requires both archive and SHA-256")
            report = plan_retrosheet_results(
                args.retrosheet_zip, expected_sha256=args.retrosheet_sha256
            )
        else:
            report = plan_historical_odds_backfill()
        print(_report_json(report))
        return 0

    importing_results = args.retrosheet_zip is not None or args.retrosheet_sha256 is not None
    if importing_results:
        if not args.retrosheet_zip or not args.retrosheet_sha256:
            parser.error("Retrosheet execution requires both archive and SHA-256")
        if args.max_calls is not None or args.max_credits is not None:
            parser.error("credit confirmations do not apply to a local results import")
    elif args.max_calls is None or args.max_credits is None:
        parser.error("odds execution requires --max-calls and --max-credits")

    client, ledger, database_url, license_scope, license_version = _runtime_components()
    _require_current_schema(database_url)
    if importing_results:
        report = import_retrosheet_results(
            args.retrosheet_zip,
            expected_sha256=args.retrosheet_sha256,
            database_url=database_url,
            ledger=ledger,
        )
    else:
        report = run_historical_odds_backfill(
            client,
            ledger,
            database_url=database_url,
            license_scope=license_scope,
            license_version=license_version,
            max_calls=args.max_calls,
            max_credits=args.max_credits,
            progress_writer=lambda message: print(message, flush=True),
        )
    print(_report_json(report))
    return 0


def _report_json(report: object) -> str:
    return json.dumps(
        asdict(report),
        sort_keys=True,
        separators=(",", ":"),
        default=lambda value: value.isoformat().replace("+00:00", "Z"),
    )


if __name__ == "__main__":  # pragma: no cover - exercised through the module command.
    raise SystemExit(main())
