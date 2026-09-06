"""Auditable, time-aware probability-model evaluation.

This module deliberately does *not* create sports features, scrape data, or
turn a model score into a betting recommendation.  Callers must provide
timestamped, source-backed observations that were available at the decision
time.  The module then evaluates probability models chronologically, using
out-of-fold predictions and forward-only calibration.

The scikit-learn candidates are loaded lazily.  That keeps ingestion and
governance tooling usable in a minimal runtime and makes a missing modelling
dependency fail closed instead of quietly substituting a made-up model.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import groupby
from numbers import Real
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .calibration import IsotonicCalibrator
from .metrics import brier_score, calibration_bins, log_loss
from .odds import implied_probability, market_consensus_two_way


class ModelDataError(ValueError):
    """Raised when a row is incomplete, non-reproducible, or time-leaking."""


class OptionalModelDependencyError(RuntimeError):
    """Raised when an explicitly requested modelling library is unavailable."""


class ModelTrainingError(RuntimeError):
    """Raised when a candidate cannot be trained safely on the supplied fold."""


@dataclass(frozen=True)
class NumericFeature:
    """A single, numeric feature in an explicit point-in-time contract."""

    name: str
    minimum: Optional[float] = None
    maximum: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ModelDataError("feature names must be non-empty")
        for label, value in (("minimum", self.minimum), ("maximum", self.maximum)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, Real)):
                raise ModelDataError("feature %s must be numeric" % label)
            if value is not None and not math.isfinite(float(value)):
                raise ModelDataError("feature %s must be finite" % label)
        if self.minimum is not None and self.maximum is not None and float(self.minimum) > float(self.maximum):
            raise ModelDataError("feature minimum cannot exceed maximum")


@dataclass(frozen=True)
class FeatureSchema:
    """Versioned, strict input contract for one model family.

    Features are intentionally numeric and complete.  This package does not
    silently impute missing values because invented values obscure source-data
    faults and can create false confidence in an evaluation.
    """

    schema_id: str
    features: Tuple[NumericFeature, ...]
    reject_unknown_features: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.schema_id, str) or not self.schema_id.strip():
            raise ModelDataError("schema_id must be non-empty")
        if not self.features:
            raise ModelDataError("a feature schema needs at least one feature")
        if any(not isinstance(feature, NumericFeature) for feature in self.features):
            raise ModelDataError("a feature schema may contain only NumericFeature definitions")
        names = [feature.name for feature in self.features]
        if len(names) != len(set(names)):
            raise ModelDataError("feature names must be unique")
        # The target must never be passed through the model feature contract.
        prohibited = {"outcome", "label", "target"}
        found = prohibited.intersection(names)
        if found:
            raise ModelDataError("target fields are not valid model features: " + ", ".join(sorted(found)))

    @property
    def feature_names(self) -> Tuple[str, ...]:
        return tuple(feature.name for feature in self.features)

    def vector(self, values: Mapping[str, Any]) -> Tuple[float, ...]:
        """Validate a record and return values in stable schema order."""

        if not isinstance(values, Mapping):
            raise ModelDataError("feature values must be a mapping")
        known = set(self.feature_names)
        supplied = set(values.keys())
        missing = known - supplied
        if missing:
            raise ModelDataError("missing required features: " + ", ".join(sorted(missing)))
        if self.reject_unknown_features:
            unexpected = supplied - known
            if unexpected:
                raise ModelDataError("unexpected features: " + ", ".join(sorted(str(item) for item in unexpected)))

        vector: List[float] = []
        for feature in self.features:
            value = values[feature.name]
            # bool is a Real in Python but is rarely an intentional probability
            # feature.  Require an explicit 0.0/1.0 upstream instead.
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ModelDataError("feature %s must be a finite numeric value" % feature.name)
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ModelDataError("feature %s must be finite" % feature.name)
            if feature.minimum is not None and numeric < feature.minimum:
                raise ModelDataError("feature %s is below its declared minimum" % feature.name)
            if feature.maximum is not None and numeric > feature.maximum:
                raise ModelDataError("feature %s is above its declared maximum" % feature.name)
            vector.append(numeric)
        return tuple(vector)


@dataclass(frozen=True)
class PredictionInput:
    """Source-backed model inputs known no later than ``decision_at``."""

    event_id: str
    event_starts_at: datetime
    decision_at: datetime
    features_available_at: datetime
    source_snapshot_ids: Tuple[str, ...]
    features: Mapping[str, Any]


@dataclass(frozen=True)
class OutcomeTrainingRow:
    """A settled historical prediction opportunity.

    ``label_available_at`` is distinct from the event decision time.  It lets
    the rolling splitter exclude a game's result until that result would have
    been known by the next historical training cut-off.
    """

    event_id: str
    event_starts_at: datetime
    decision_at: datetime
    features_available_at: datetime
    label_available_at: datetime
    source_snapshot_ids: Tuple[str, ...]
    label_source_snapshot_id: str
    features: Mapping[str, Any]
    outcome: int


# One fixed decision horizon keeps every row comparable and prevents a later
# quote from quietly changing the meaning of the feature. Changing either the
# horizon or the calculation requires a new schema id.
PREGAME_H2H_DECISION_LEAD = timedelta(minutes=30)
PREGAME_H2H_FEATURE_SCHEMA = FeatureSchema(
    schema_id="pregame-h2h-market-consensus-30m-v1",
    features=(NumericFeature("market_probability", minimum=0.0, maximum=1.0),),
)


@dataclass(frozen=True)
class AdaptedTrainingDataset:
    """Model-ready rows derived from the immutable evidence-layer contracts."""

    schema: FeatureSchema
    feature_contract_digest: str
    rows: Tuple[OutcomeTrainingRow, ...]
    row_digests: Tuple[str, ...]


def _is_aware(value: datetime) -> bool:
    return isinstance(value, datetime) and value.tzinfo is not None and value.utcoffset() is not None


def feature_schema_fingerprint(schema: FeatureSchema) -> str:
    """Return the canonical SHA-256 identity of an ordered feature schema."""

    if not isinstance(schema, FeatureSchema):
        raise ModelDataError("feature schema fingerprinting requires a FeatureSchema")
    payload = {
        "schema_id": schema.schema_id,
        "features": [
            {
                "name": feature.name,
                "minimum": feature.minimum,
                "maximum": feature.maximum,
            }
            for feature in schema.features
        ],
        "reject_unknown_features": schema.reject_unknown_features,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_source_snapshots(source_snapshot_ids: Sequence[str]) -> None:
    if isinstance(source_snapshot_ids, str):
        raise ModelDataError("source snapshot ids must be a sequence, not one string")
    if not source_snapshot_ids:
        raise ModelDataError("at least one immutable source snapshot id is required")
    if any(not isinstance(item, str) or not item.strip() for item in source_snapshot_ids):
        raise ModelDataError("source snapshot ids must be non-empty strings")
    if len(source_snapshot_ids) != len(set(source_snapshot_ids)):
        raise ModelDataError("source snapshot ids must not contain duplicates")
    if tuple(source_snapshot_ids) != tuple(sorted(source_snapshot_ids)):
        raise ModelDataError("source snapshot ids must be sorted for deterministic lineage")


def validate_prediction_input(request: PredictionInput, schema: FeatureSchema) -> Tuple[float, ...]:
    """Fail closed unless an inference request is reproducible and point-in-time safe."""

    if not isinstance(request.event_id, str) or not request.event_id.strip():
        raise ModelDataError("event_id must be non-empty")
    if not all(
        _is_aware(value)
        for value in (request.event_starts_at, request.decision_at, request.features_available_at)
    ):
        raise ModelDataError("event, decision, and feature timestamps must be timezone-aware")
    if request.decision_at >= request.event_starts_at:
        raise ModelDataError("prediction decisions must be recorded before the scheduled event start")
    if request.features_available_at > request.decision_at:
        raise ModelDataError("features became available after the requested decision time")
    _validate_source_snapshots(request.source_snapshot_ids)
    return schema.vector(request.features)


def validate_training_rows(rows: Iterable[OutcomeTrainingRow], schema: FeatureSchema) -> Tuple[OutcomeTrainingRow, ...]:
    """Validate rows without sorting, imputing, or deriving any observations."""

    materialized = tuple(rows)
    if not materialized:
        raise ModelDataError("at least one settled training row is required")
    event_ids = set()
    for row in materialized:
        if not isinstance(row, OutcomeTrainingRow):
            raise ModelDataError("training rows must be OutcomeTrainingRow instances")
        if not isinstance(row.event_id, str) or not row.event_id.strip():
            raise ModelDataError("event_id must be non-empty")
        if row.event_id in event_ids:
            raise ModelDataError("each event_id may appear only once in a binary outcome data set")
        event_ids.add(row.event_id)
        if any(
            not _is_aware(value)
            for value in (
                row.event_starts_at,
                row.decision_at,
                row.features_available_at,
                row.label_available_at,
            )
        ):
            raise ModelDataError("event, decision, feature, and label timestamps must be timezone-aware")
        if row.decision_at >= row.event_starts_at:
            raise ModelDataError("training decisions must be recorded before the scheduled event start")
        if row.features_available_at > row.decision_at:
            raise ModelDataError("features became available after the recorded decision time")
        if row.label_available_at < row.decision_at:
            raise ModelDataError("a label cannot be available before its decision time")
        if row.label_available_at < row.event_starts_at:
            raise ModelDataError("a label cannot be available before the scheduled event start")
        if row.outcome not in (0, 1, False, True):
            raise ModelDataError("outcomes must be binary")
        _validate_source_snapshots(row.source_snapshot_ids)
        if not isinstance(row.label_source_snapshot_id, str) or not row.label_source_snapshot_id.strip():
            raise ModelDataError("a non-empty immutable label source snapshot id is required")
        if row.label_source_snapshot_id in row.source_snapshot_ids:
            raise ModelDataError("the label source snapshot cannot also be an input feature source")
        schema.vector(row.features)
    return materialized


def build_h2h_market_training_rows(
    records: Iterable[Mapping[str, Any]],
    *,
    sport: str,
    training_cutoff: datetime,
) -> Tuple[OutcomeTrainingRow, ...]:
    """Build point-in-time home-win rows from persisted odds and results.

    The latest locally available, coherent home/away quote batch is selected
    independently for each bookmaker at a fixed 30-minute pregame horizon.
    At least two complete books are required. Provider corrections are handled
    by selecting the latest result received by ``training_cutoff``; ties and
    incomplete markets are excluded instead of inventing binary labels.
    """

    if not isinstance(sport, str) or not sport.strip():
        raise ModelDataError("sport must be non-empty")
    if not _is_aware(training_cutoff):
        raise ModelDataError("training_cutoff must be timezone-aware")

    events: Dict[str, Dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ModelDataError("training evidence rows must be mappings")
        if str(record.get("sport", "")).strip() != sport:
            continue

        event_id = str(record.get("event_id", "")).strip()
        home_team = str(record.get("home_team", "")).strip()
        away_team = str(record.get("away_team", "")).strip()
        starts_at = record.get("starts_at")
        if (
            not event_id
            or not home_team
            or not away_team
            or home_team == away_team
            or not _is_aware(starts_at)
        ):
            raise ModelDataError("persisted event evidence is incomplete")
        if starts_at >= training_cutoff:
            continue
        decision_at = starts_at - PREGAME_H2H_DECISION_LEAD

        event = events.setdefault(
            event_id,
            {
                "starts_at": starts_at,
                "decision_at": decision_at,
                "home_team": home_team,
                "away_team": away_team,
                "results": {},
                "quote_batches": {},
            },
        )
        if (
            event["starts_at"] != starts_at
            or event["home_team"] != home_team
            or event["away_team"] != away_team
        ):
            raise ModelDataError("persisted event identity changed across evidence rows")

        result_id = str(record.get("result_id", "")).strip()
        result_received_at = record.get("result_received_at")
        settled_at = record.get("settled_at")
        home_score = record.get("home_score")
        away_score = record.get("away_score")
        if (
            result_id
            and _is_aware(result_received_at)
            and _is_aware(settled_at)
            and result_received_at <= training_cutoff
            and settled_at <= training_cutoff
        ):
            if any(isinstance(score, bool) or not isinstance(score, int) or score < 0 for score in (home_score, away_score)):
                raise ModelDataError("persisted result scores must be non-negative integers")
            event["results"][result_id] = {
                "id": result_id,
                "received_at": result_received_at,
                "settled_at": settled_at,
                "home_score": home_score,
                "away_score": away_score,
            }

        if str(record.get("market", "")).strip() != "h2h":
            continue
        snapshot_id = str(record.get("odds_snapshot_id", "")).strip()
        provenance_id = str(record.get("primary_provenance_id", "")).strip()
        bookmaker = str(record.get("bookmaker", "")).strip()
        selection = str(record.get("selection", "")).strip()
        captured_at = record.get("captured_at")
        received_at = record.get("quote_received_at")
        if not snapshot_id or not provenance_id or not bookmaker:
            continue
        if selection not in {home_team, away_team}:
            continue
        if not _is_aware(captured_at) or not _is_aware(received_at):
            raise ModelDataError("persisted quote timestamps must be timezone-aware")
        if captured_at > decision_at or received_at > decision_at:
            continue
        try:
            decimal_odds = float(record["decimal_odds"])
            implied_probability(decimal_odds)
        except (KeyError, TypeError, ValueError):
            raise ModelDataError("persisted decimal odds are invalid") from None

        batch_key = (bookmaker, provenance_id)
        batch = event["quote_batches"].setdefault(batch_key, {})
        candidate = {
            "snapshot_id": snapshot_id,
            "captured_at": captured_at,
            "received_at": received_at,
            "decimal_odds": decimal_odds,
        }
        previous = batch.get(selection)
        if previous is None or (
            received_at,
            captured_at,
            snapshot_id,
        ) > (
            previous["received_at"],
            previous["captured_at"],
            previous["snapshot_id"],
        ):
            batch[selection] = candidate

    training_rows: List[OutcomeTrainingRow] = []
    for event_id, event in events.items():
        if not event["results"]:
            continue
        result = max(
            event["results"].values(),
            key=lambda item: (item["settled_at"], item["received_at"], item["id"]),
        )
        if result["received_at"] < event["starts_at"]:
            raise ModelDataError("persisted result became available before its event started")
        if result["home_score"] == result["away_score"]:
            continue

        complete_by_book: Dict[str, Tuple[Dict[str, Any], Dict[str, Any], str]] = {}
        for (bookmaker, provenance_id), selections in event["quote_batches"].items():
            home_quote = selections.get(event["home_team"])
            away_quote = selections.get(event["away_team"])
            if home_quote is None or away_quote is None:
                continue
            candidate = (home_quote, away_quote, provenance_id)
            previous = complete_by_book.get(bookmaker)
            if previous is None or _quote_pair_order(candidate) > _quote_pair_order(previous):
                complete_by_book[bookmaker] = candidate
        if len(complete_by_book) < 2:
            continue

        quote_pairs = list(complete_by_book.values())
        home_probability, _ = market_consensus_two_way(
            (home["decimal_odds"], away["decimal_odds"])
            for home, away, _provenance_id in quote_pairs
        )
        source_snapshot_ids = tuple(
            sorted(
                quote["snapshot_id"]
                for home, away, _provenance_id in quote_pairs
                for quote in (home, away)
            )
        )
        features_available_at = max(
            quote["received_at"]
            for home, away, _provenance_id in quote_pairs
            for quote in (home, away)
        )
        training_rows.append(
            OutcomeTrainingRow(
                event_id=event_id,
                event_starts_at=event["starts_at"],
                decision_at=event["decision_at"],
                features_available_at=features_available_at,
                label_available_at=result["received_at"],
                source_snapshot_ids=source_snapshot_ids,
                label_source_snapshot_id=result["id"],
                features={"market_probability": home_probability},
                outcome=int(result["home_score"] > result["away_score"]),
            )
        )

    ordered = tuple(sorted(training_rows, key=lambda row: (row.decision_at, row.event_id)))
    return validate_training_rows(ordered, PREGAME_H2H_FEATURE_SCHEMA) if ordered else ()


def build_h2h_market_prediction_inputs(
    records: Iterable[Mapping[str, Any]],
    *,
    sport: str,
    now: datetime,
    released_at: datetime,
) -> Tuple[PredictionInput, ...]:
    """Build reproducible live inputs at the same fixed horizon used in training."""

    if not isinstance(sport, str) or not sport.strip():
        raise ModelDataError("sport must be non-empty")
    if not _is_aware(now):
        raise ModelDataError("now must be timezone-aware")
    if not _is_aware(released_at):
        raise ModelDataError("released_at must be timezone-aware")
    if released_at > now:
        raise ModelDataError("released_at cannot be in the future")

    events: Dict[str, Dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ModelDataError("prediction evidence rows must be mappings")
        if str(record.get("sport", "")).strip() != sport:
            continue

        event_id = str(record.get("event_id", "")).strip()
        home_team = str(record.get("home_team", "")).strip()
        away_team = str(record.get("away_team", "")).strip()
        starts_at = record.get("starts_at")
        if (
            not event_id
            or not home_team
            or not away_team
            or home_team == away_team
            or not _is_aware(starts_at)
        ):
            raise ModelDataError("persisted event evidence is incomplete")
        decision_at = starts_at - PREGAME_H2H_DECISION_LEAD
        if not released_at <= decision_at <= now < starts_at:
            continue

        event = events.setdefault(
            event_id,
            {
                "starts_at": starts_at,
                "decision_at": decision_at,
                "home_team": home_team,
                "away_team": away_team,
                "quote_batches": {},
            },
        )
        if (
            event["starts_at"] != starts_at
            or event["home_team"] != home_team
            or event["away_team"] != away_team
        ):
            raise ModelDataError("persisted event identity changed across evidence rows")

        if str(record.get("market", "")).strip() != "h2h":
            continue
        snapshot_id = str(record.get("odds_snapshot_id", "")).strip()
        provenance_id = str(record.get("primary_provenance_id", "")).strip()
        bookmaker = str(record.get("bookmaker", "")).strip()
        selection = str(record.get("selection", "")).strip()
        captured_at = record.get("captured_at")
        received_at = record.get("quote_received_at")
        if not snapshot_id or not provenance_id or not bookmaker:
            continue
        if selection not in {home_team, away_team}:
            continue
        if not _is_aware(captured_at) or not _is_aware(received_at):
            raise ModelDataError("persisted quote timestamps must be timezone-aware")
        if captured_at > decision_at or received_at > decision_at:
            continue
        try:
            decimal_odds = float(record["decimal_odds"])
            implied_probability(decimal_odds)
        except (KeyError, TypeError, ValueError):
            raise ModelDataError("persisted decimal odds are invalid") from None

        batch_key = (bookmaker, provenance_id)
        batch = event["quote_batches"].setdefault(batch_key, {})
        candidate = {
            "snapshot_id": snapshot_id,
            "captured_at": captured_at,
            "received_at": received_at,
            "decimal_odds": decimal_odds,
        }
        previous = batch.get(selection)
        if previous is None or (
            received_at,
            captured_at,
            snapshot_id,
        ) > (
            previous["received_at"],
            previous["captured_at"],
            previous["snapshot_id"],
        ):
            batch[selection] = candidate

    inputs: List[PredictionInput] = []
    for event_id, event in events.items():
        complete_by_book: Dict[str, Tuple[Dict[str, Any], Dict[str, Any], str]] = {}
        for (bookmaker, provenance_id), selections in event["quote_batches"].items():
            home_quote = selections.get(event["home_team"])
            away_quote = selections.get(event["away_team"])
            if home_quote is None or away_quote is None:
                continue
            candidate = (home_quote, away_quote, provenance_id)
            previous = complete_by_book.get(bookmaker)
            if previous is None or _quote_pair_order(candidate) > _quote_pair_order(previous):
                complete_by_book[bookmaker] = candidate
        if len(complete_by_book) < 2:
            continue

        quote_pairs = list(complete_by_book.values())
        home_probability, _ = market_consensus_two_way(
            (home["decimal_odds"], away["decimal_odds"])
            for home, away, _provenance_id in quote_pairs
        )
        request = PredictionInput(
            event_id=event_id,
            event_starts_at=event["starts_at"],
            decision_at=event["decision_at"],
            features_available_at=max(
                quote["received_at"]
                for home, away, _provenance_id in quote_pairs
                for quote in (home, away)
            ),
            source_snapshot_ids=tuple(
                sorted(
                    quote["snapshot_id"]
                    for home, away, _provenance_id in quote_pairs
                    for quote in (home, away)
                )
            ),
            features={"market_probability": home_probability},
        )
        validate_prediction_input(request, PREGAME_H2H_FEATURE_SCHEMA)
        inputs.append(request)
    return tuple(sorted(inputs, key=lambda request: (request.decision_at, request.event_id)))


def load_h2h_market_prediction_inputs(
    database_url: str,
    *,
    sport: str,
    model_id: str,
    now: datetime,
    released_at: datetime,
    limit: int = 100,
) -> Tuple[PredictionInput, ...]:
    """Load unscored upcoming events with receipt-backed pregame market evidence."""

    if not isinstance(database_url, str) or not database_url.strip():
        raise ModelDataError("database_url must be configured")
    if not isinstance(sport, str) or not sport.strip():
        raise ModelDataError("sport must be non-empty")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ModelDataError("model_id must be non-empty")
    if not _is_aware(now):
        raise ModelDataError("now must be timezone-aware")
    if not _is_aware(released_at):
        raise ModelDataError("released_at must be timezone-aware")
    if released_at > now:
        raise ModelDataError("released_at cannot be in the future")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
        raise ModelDataError("limit must be an integer from 1 to 1000")

    try:
        import psycopg
        from psycopg.rows import dict_row

        connection = psycopg.connect(
            database_url,
            application_name="sam-analytics-model-inference",
            connect_timeout=5,
            options="-c statement_timeout=10000 -c default_transaction_read_only=on",
            row_factory=dict_row,
        )
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    WITH candidate_events AS (
                        SELECT event.id, event.sport, event.starts_at,
                               event.home_team, event.away_team
                        FROM sports_event AS event
                        WHERE event.sport = %(sport)s
                          AND event.starts_at > %(now)s
                          AND event.starts_at - %(decision_lead)s <= %(now)s
                          AND event.starts_at - %(decision_lead)s >= %(released_at)s
                          AND NOT EXISTS (
                              SELECT 1
                              FROM prediction
                              WHERE prediction.event_id = event.id
                                AND prediction.model_id = %(model_id)s
                                AND prediction.as_of = event.starts_at - %(decision_lead)s
                          )
                        ORDER BY event.starts_at, event.id
                        LIMIT %(limit)s
                    ),
                    eligible_quotes AS (
                        SELECT event.id AS event_id, event.sport, event.starts_at,
                               event.home_team, event.away_team,
                               quote.id AS odds_snapshot_id, quote.bookmaker,
                               quote.primary_provenance_id, quote.market, quote.selection,
                               quote.decimal_odds, quote.captured_at,
                               quote.received_at AS quote_received_at
                        FROM candidate_events AS event
                        JOIN odds_snapshot AS quote ON quote.event_id = event.id
                        JOIN raw_data_provenance AS provenance
                          ON provenance.id = quote.primary_provenance_id
                        JOIN provider_payload_receipt AS receipt
                          ON receipt.id = provenance.provider_payload_receipt_id
                        WHERE quote.market = 'h2h'
                          AND quote.line IS NULL
                          AND quote.bookmaker IS NOT NULL
                          AND quote.selection IN (event.home_team, event.away_team)
                          AND quote.captured_at <= event.starts_at - %(decision_lead)s
                          AND quote.received_at <= event.starts_at - %(decision_lead)s
                          AND provenance.source_type = 'odds'
                          AND provenance.payload_sha256 = quote.source_payload_sha256
                          AND provenance.received_at = quote.received_at
                          AND receipt.provider = quote.provider
                          AND receipt.source_type = 'odds'
                          AND receipt.payload_sha256 = quote.source_payload_sha256
                          AND receipt.received_at = quote.received_at
                    ),
                    latest_batch_selections AS (
                        SELECT DISTINCT ON (
                                   event_id, bookmaker, primary_provenance_id, selection
                               )
                               *
                        FROM eligible_quotes
                        ORDER BY event_id, bookmaker, primary_provenance_id, selection,
                                 quote_received_at DESC, captured_at DESC,
                                 odds_snapshot_id DESC
                    ),
                    complete_batches AS (
                        SELECT event_id, bookmaker, primary_provenance_id,
                               max(quote_received_at) AS pair_received_at,
                               max(captured_at) AS pair_captured_at
                        FROM latest_batch_selections
                        GROUP BY event_id, bookmaker, primary_provenance_id
                        HAVING count(*) = 2
                    ),
                    selected_batches AS (
                        SELECT DISTINCT ON (event_id, bookmaker)
                               event_id, bookmaker, primary_provenance_id
                        FROM complete_batches
                        ORDER BY event_id, bookmaker, pair_received_at DESC,
                                 pair_captured_at DESC, primary_provenance_id DESC
                    )
                    SELECT quote.event_id, quote.sport, quote.starts_at,
                           quote.home_team, quote.away_team,
                           quote.odds_snapshot_id, quote.bookmaker,
                           quote.primary_provenance_id, quote.market, quote.selection,
                           quote.decimal_odds, quote.captured_at,
                           quote.quote_received_at
                    FROM latest_batch_selections AS quote
                    JOIN selected_batches AS selected
                      ON selected.event_id = quote.event_id
                     AND selected.bookmaker = quote.bookmaker
                     AND selected.primary_provenance_id = quote.primary_provenance_id
                    ORDER BY quote.starts_at, quote.event_id, quote.bookmaker,
                             quote.quote_received_at, quote.captured_at,
                             quote.odds_snapshot_id
                    """,
                    {
                        "sport": sport,
                        "model_id": model_id,
                        "now": now,
                        "released_at": released_at,
                        "decision_lead": PREGAME_H2H_DECISION_LEAD,
                        "limit": limit,
                    },
                )
                records = tuple(cursor.fetchall())
        finally:
            connection.close()
    except Exception as error:
        if isinstance(error, ModelDataError):
            raise
        raise ModelTrainingError("persisted prediction evidence could not be loaded") from None
    return build_h2h_market_prediction_inputs(
        records,
        sport=sport,
        now=now,
        released_at=released_at,
    )


def load_h2h_market_training_rows(
    database_url: str,
    *,
    sport: str,
    training_cutoff: datetime,
) -> Tuple[OutcomeTrainingRow, ...]:
    """Load the immutable PostgreSQL evidence needed by the model evaluator."""

    if not isinstance(database_url, str) or not database_url.strip():
        raise ModelDataError("database_url must be configured")
    if not isinstance(sport, str) or not sport.strip():
        raise ModelDataError("sport must be non-empty")
    if not _is_aware(training_cutoff):
        raise ModelDataError("training_cutoff must be timezone-aware")

    try:
        import psycopg
        from psycopg.rows import dict_row

        connection = psycopg.connect(
            database_url,
            application_name="sam-analytics-model-training",
            connect_timeout=5,
            options="-c statement_timeout=30000 -c default_transaction_read_only=on",
            row_factory=dict_row,
        )
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    WITH latest_results AS (
                        SELECT DISTINCT ON (result.event_id)
                               result.id, result.event_id, result.settled_at,
                               result.received_at, result.home_score, result.away_score
                        FROM event_result AS result
                        JOIN sports_event AS result_event ON result_event.id = result.event_id
                        WHERE result.provider = result_event.provider
                          AND result_event.sport = %(sport)s
                          AND result.received_at <= %(training_cutoff)s
                          AND result.settled_at <= %(training_cutoff)s
                          AND result.settled_at >= result_event.starts_at
                          AND EXISTS (
                              SELECT 1
                              FROM event_result_provenance AS result_link
                              JOIN raw_data_provenance AS result_provenance
                                ON result_provenance.id = result_link.provenance_id
                              JOIN provider_payload_receipt AS result_receipt
                                ON result_receipt.id = result_provenance.provider_payload_receipt_id
                              WHERE result_link.event_result_id = result.id
                                AND result_provenance.provider = result.provider
                                AND result_provenance.source_type = 'result'
                                AND result_provenance.payload_sha256 = result.source_payload_sha256
                                AND result_provenance.received_at = result.received_at
                                AND result_receipt.provider = result.provider
                                AND result_receipt.source_type = 'result'
                                AND result_receipt.payload_sha256 = result.source_payload_sha256
                                AND result_receipt.received_at = result.received_at
                          )
                        ORDER BY result.event_id, result.settled_at DESC,
                                 result.received_at DESC, result.id DESC
                    ),
                    eligible_quotes AS (
                        SELECT event.id AS event_id, event.sport, event.starts_at,
                               event.home_team, event.away_team,
                               quote.id AS odds_snapshot_id, quote.bookmaker,
                               quote.primary_provenance_id, quote.market, quote.selection,
                               quote.decimal_odds, quote.captured_at,
                               quote.received_at AS quote_received_at,
                               result.id AS result_id, result.settled_at,
                               result.received_at AS result_received_at,
                               result.home_score, result.away_score
                        FROM sports_event AS event
                        JOIN latest_results AS result ON result.event_id = event.id
                        JOIN odds_snapshot AS quote ON quote.event_id = event.id
                        WHERE event.sport = %(sport)s
                          AND event.starts_at < %(training_cutoff)s
                          AND quote.market = 'h2h'
                          AND quote.line IS NULL
                          AND quote.bookmaker IS NOT NULL
                          AND quote.primary_provenance_id IS NOT NULL
                          AND quote.selection IN (event.home_team, event.away_team)
                          AND quote.captured_at <= event.starts_at - %(decision_lead)s
                          AND quote.received_at <= event.starts_at - %(decision_lead)s
                    ),
                    latest_batch_selections AS (
                        SELECT DISTINCT ON (
                                   event_id, bookmaker, primary_provenance_id, selection
                               )
                               *
                        FROM eligible_quotes
                        ORDER BY event_id, bookmaker, primary_provenance_id, selection,
                                 quote_received_at DESC, captured_at DESC,
                                 odds_snapshot_id DESC
                    ),
                    complete_batches AS (
                        SELECT event_id, bookmaker, primary_provenance_id,
                               max(quote_received_at) AS pair_received_at,
                               max(captured_at) AS pair_captured_at
                        FROM latest_batch_selections
                        GROUP BY event_id, bookmaker, primary_provenance_id
                        HAVING count(*) = 2
                    ),
                    selected_batches AS (
                        SELECT DISTINCT ON (event_id, bookmaker)
                               event_id, bookmaker, primary_provenance_id
                        FROM complete_batches
                        ORDER BY event_id, bookmaker, pair_received_at DESC,
                                 pair_captured_at DESC, primary_provenance_id DESC
                    )
                    SELECT quote.event_id, quote.sport, quote.starts_at,
                           quote.home_team, quote.away_team,
                           quote.odds_snapshot_id, quote.bookmaker,
                           quote.primary_provenance_id, quote.market, quote.selection,
                           quote.decimal_odds, quote.captured_at,
                           quote.quote_received_at, quote.result_id,
                           quote.settled_at, quote.result_received_at,
                           quote.home_score, quote.away_score
                    FROM latest_batch_selections AS quote
                    JOIN selected_batches AS selected
                      ON selected.event_id = quote.event_id
                     AND selected.bookmaker = quote.bookmaker
                     AND selected.primary_provenance_id = quote.primary_provenance_id
                    ORDER BY quote.starts_at, quote.event_id, quote.bookmaker,
                             quote.quote_received_at, quote.captured_at,
                             quote.odds_snapshot_id
                    """,
                    {
                        "sport": sport,
                        "training_cutoff": training_cutoff,
                        "decision_lead": PREGAME_H2H_DECISION_LEAD,
                    },
                )
                records = tuple(cursor.fetchall())
        finally:
            connection.close()
    except Exception as error:
        if isinstance(error, ModelDataError):
            raise
        raise ModelTrainingError("persisted training evidence could not be loaded") from None
    return build_h2h_market_training_rows(
        records,
        sport=sport,
        training_cutoff=training_cutoff,
    )


def _quote_pair_order(
    pair: Tuple[Dict[str, Any], Dict[str, Any], str],
) -> Tuple[datetime, datetime, str]:
    home, away, provenance_id = pair
    return (
        max(home["received_at"], away["received_at"]),
        max(home["captured_at"], away["captured_at"]),
        provenance_id,
    )


def _data_contract_components() -> Dict[str, Any]:
    """Load the evidence-layer contract lazily to keep this module standalone."""

    try:
        from .data_contracts import (
            DataContractViolation,
            FeatureContract,
            LabeledTrainingExample,
            validate_feature_contract,
            validate_labeled_training_example,
        )
    except ImportError as error:  # pragma: no cover - the bundled package includes this module.
        raise OptionalModelDependencyError(
            "the point-in-time data-contract module is required to adapt evidence-layer training rows"
        ) from error
    return {
        "DataContractViolation": DataContractViolation,
        "FeatureContract": FeatureContract,
        "LabeledTrainingExample": LabeledTrainingExample,
        "validate_feature_contract": validate_feature_contract,
        "validate_labeled_training_example": validate_labeled_training_example,
    }


def schema_from_feature_contract(contract: Any) -> FeatureSchema:
    """Build a strict model schema from an immutable evidence-layer contract.

    A feature allowed to be missing at the evidence layer still cannot enter a
    model by default.  It needs a separate, explicit, versioned transform
    before a model schema may include it; this prevents silent imputation.
    """

    components = _data_contract_components()
    if not isinstance(contract, components["FeatureContract"]):
        raise ModelDataError("contract must be a FeatureContract")
    report = components["validate_feature_contract"](contract)
    try:
        report.require_valid("feature contract")
    except components["DataContractViolation"] as error:
        raise ModelDataError(str(error)) from error
    permissive = [
        definition.name
        for definition in contract.features
        if definition.allow_missing or not definition.required
    ]
    if permissive:
        raise ModelDataError(
            "model schema refuses nullable/optional features without an explicit, "
            "contract-versioned missing-data transform: "
            + ", ".join(permissive)
        )
    return FeatureSchema(
        schema_id=contract.digest,
        features=tuple(
            NumericFeature(
                name=definition.name,
                minimum=definition.minimum,
                maximum=definition.maximum,
            )
            for definition in contract.features
        ),
    )


def adapt_labeled_training_examples(
    contract: Any, examples: Iterable[Any]
) -> AdaptedTrainingDataset:
    """Convert validated evidence-layer examples into strict model rows.

    This is intentionally a one-way adapter: all provenance stays represented
    as immutable raw-payload digests, and it never looks up newer data or fills
    a missing observation.  The data-contract manifest remains responsible for
    binding a training run to its declared training cut-off.
    """

    components = _data_contract_components()
    schema = schema_from_feature_contract(contract)
    materialized = tuple(examples)
    if not materialized:
        raise ModelDataError("at least one labeled example is required")

    rows: List[OutcomeTrainingRow] = []
    row_digests: List[str] = []
    for example in materialized:
        if not isinstance(example, components["LabeledTrainingExample"]):
            raise ModelDataError("examples must be LabeledTrainingExample instances")
        report = components["validate_labeled_training_example"](example)
        try:
            report.require_valid("labeled training example")
        except components["DataContractViolation"] as error:
            raise ModelDataError(str(error)) from error
        if example.vector.contract.digest != contract.digest:
            raise ModelDataError("every example must use exactly the requested feature contract")

        observations = tuple(example.vector.observations)
        if any(observation.value is None for observation in observations):
            raise ModelDataError(
                "model adapter refuses missing feature values without an explicit, "
                "contract-versioned transform"
            )
        feature_values = {observation.name: observation.value for observation in observations}
        feature_available_at = max(observation.available_at for observation in observations)
        source_digests = tuple(sorted({observation.source.digest for observation in observations}))
        rows.append(
            OutcomeTrainingRow(
                event_id=example.vector.event_id,
                event_starts_at=example.vector.event_starts_at,
                decision_at=example.vector.as_of,
                features_available_at=feature_available_at,
                label_available_at=example.result_source.received_at,
                source_snapshot_ids=source_digests,
                label_source_snapshot_id=example.result_source.digest,
                features=feature_values,
                outcome=int(example.target),
            )
        )
        row_digests.append(example.digest)
    strict_rows = validate_training_rows(rows, schema)
    return AdaptedTrainingDataset(
        schema=schema,
        feature_contract_digest=contract.digest,
        rows=strict_rows,
        row_digests=tuple(row_digests),
    )


def training_dataset_fingerprint(rows: Iterable[OutcomeTrainingRow], schema: FeatureSchema) -> str:
    """Return a stable audit fingerprint for the exact evaluated training data.

    The fingerprint contains metadata and declared feature values, not a secret
    provider key or raw provider response.  A registry can use it to prove that
    the data fitted at promotion time is the data evaluated in walk-forward
    validation.
    """

    validated_rows = validate_training_rows(rows, schema)
    payload = {
        "schema": {
            "schema_id": schema.schema_id,
            "features": [
                {"name": feature.name, "minimum": feature.minimum, "maximum": feature.maximum}
                for feature in schema.features
            ],
            "reject_unknown_features": schema.reject_unknown_features,
        },
        "rows": [
            {
                "event_id": row.event_id,
                "event_starts_at": row.event_starts_at.isoformat(),
                "decision_at": row.decision_at.isoformat(),
                "features_available_at": row.features_available_at.isoformat(),
                "label_available_at": row.label_available_at.isoformat(),
                "source_snapshot_ids": list(row.source_snapshot_ids),
                "label_source_snapshot_id": row.label_source_snapshot_id,
                "features": list(schema.vector(row.features)),
                "outcome": int(row.outcome),
            }
            for row in validated_rows
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_chronological(rows: Sequence[OutcomeTrainingRow]) -> None:
    for previous, current in zip(rows, rows[1:]):
        if previous.decision_at > current.decision_at:
            raise ModelDataError("training rows must be supplied in chronological decision order")


@dataclass(frozen=True)
class ChronologicalFold:
    """Indices for one walk-forward train/validation split."""

    number: int
    train_indices: Tuple[int, ...]
    validation_indices: Tuple[int, ...]
    training_cutoff_at: datetime
    validation_start_at: datetime


@dataclass(frozen=True)
class RollingTimeSplitter:
    """Expanding-window, chronological splitter with label-settlement checks.

    Validation windows never overlap.  The embargo provides a configurable
    row-level gap between the last candidate training event and the validation
    window; source availability and label availability remain the authoritative
    safeguards when event times are irregular.
    """

    # By default, score every eligible historical validation era. Restricting
    # this to an arbitrary four folds made the old promotion coverage gate
    # mathematically unreachable once a data set became moderately large.
    n_splits: Optional[int] = None
    min_train_rows: int = 250
    validation_rows: int = 100
    step_rows: Optional[int] = None
    embargo_rows: int = 0

    def __post_init__(self) -> None:
        if self.n_splits is not None and self.n_splits < 1:
            raise ModelDataError("n_splits must be at least one when supplied")
        if self.min_train_rows < 2:
            raise ModelDataError("min_train_rows must be at least two")
        if self.validation_rows < 1:
            raise ModelDataError("validation_rows must be at least one")
        if self.embargo_rows < 0:
            raise ModelDataError("embargo_rows cannot be negative")
        effective_step = self.step_rows if self.step_rows is not None else self.validation_rows
        if effective_step < self.validation_rows:
            raise ModelDataError("step_rows must prevent overlapping validation windows")

    @property
    def effective_step_rows(self) -> int:
        return self.step_rows if self.step_rows is not None else self.validation_rows

    def split(self, rows: Sequence[OutcomeTrainingRow]) -> Tuple[ChronologicalFold, ...]:
        _assert_chronological(rows)
        initial_start = self.min_train_rows + self.embargo_rows
        starts = list(
            range(
                initial_start,
                len(rows) - self.validation_rows + 1,
                self.effective_step_rows,
            )
        )
        if self.n_splits is not None and len(starts) < self.n_splits:
            raise ModelDataError(
                "not enough chronological rows for %d folds with the requested training/validation windows"
                % self.n_splits
            )

        folds: List[ChronologicalFold] = []
        # Favor the most recent validation eras.  This reflects the data regime
        # a production challenger would actually face and avoids cherry-picking
        # a favorable early period.
        selected_starts = starts if self.n_splits is None else starts[-self.n_splits :]
        for number, start in enumerate(selected_starts):
            validation_indices = tuple(range(start, start + self.validation_rows))
            validation_start_at = rows[validation_indices[0]].decision_at
            candidate_train_indices = range(0, start - self.embargo_rows)
            train_indices = tuple(
                index
                for index in candidate_train_indices
                if rows[index].decision_at < validation_start_at
                and rows[index].label_available_at <= validation_start_at
            )
            if len(train_indices) < self.min_train_rows:
                raise ModelDataError(
                    "fold %d has only %d settled rows before its validation window; "
                    "increase history or reduce the configured minimum"
                    % (number, len(train_indices))
                )
            folds.append(
                ChronologicalFold(
                    number=number,
                    train_indices=train_indices,
                    validation_indices=validation_indices,
                    training_cutoff_at=rows[train_indices[-1]].decision_at,
                    validation_start_at=validation_start_at,
                )
            )
        return tuple(folds)


@dataclass(frozen=True)
class ModelCandidate:
    """A deterministic sklearn candidate specification, not a trained model."""

    name: str
    family: str
    random_state: int = 20260904
    hyperparameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        allowed = {"logistic_regression", "hist_gradient_boosting", "neural_mlp"}
        if not isinstance(self.name, str) or not self.name.strip():
            raise ModelDataError("candidate name must be non-empty")
        if not isinstance(self.family, str) or self.family not in allowed:
            raise ModelDataError("unsupported candidate family: %s" % self.family)
        if isinstance(self.random_state, bool) or not isinstance(self.random_state, int):
            raise ModelDataError("random_state must be an integer")
        if not isinstance(self.hyperparameters, Mapping):
            raise ModelDataError("candidate hyperparameters must be a mapping")


def default_model_candidates(random_state: int = 20260904, *, include_neural: bool = True) -> Tuple[ModelCandidate, ...]:
    """Return the baseline and challenger families used by the evaluator.

    The neural candidate is a challenger, not an automatic upgrade.  It must
    pass the same time-aware validation and promotion gates as every other
    model.
    """

    candidates = [
        ModelCandidate("logistic_baseline", "logistic_regression", random_state),
        ModelCandidate("hist_gradient_boosting", "hist_gradient_boosting", random_state),
    ]
    if include_neural:
        candidates.append(ModelCandidate("neural_mlp", "neural_mlp", random_state))
    return tuple(candidates)


def _load_sklearn_components() -> Dict[str, Any]:
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as error:  # pragma: no cover - exercised via a mock below.
        raise OptionalModelDependencyError(
            "scikit-learn is required to train probability candidates. "
            "Install the project's modelling dependencies before invoking model training."
        ) from error
    return {
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
        "LogisticRegression": LogisticRegression,
        "MLPClassifier": MLPClassifier,
        "Pipeline": Pipeline,
        "StandardScaler": StandardScaler,
    }


def build_estimator(candidate: ModelCandidate) -> Any:
    """Create a new deterministic sklearn estimator for one fold.

    A fresh estimator is deliberately created for every fold, so preprocessing
    statistics and fitted state can never flow from a validation era into a
    training era.
    """

    sklearn = _load_sklearn_components()
    params = dict(candidate.hyperparameters)
    if candidate.family == "logistic_regression":
        defaults = {
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs",
            "random_state": candidate.random_state,
            "n_jobs": 1,
        }
        defaults.update(params)
        estimator = sklearn["LogisticRegression"](**defaults)
        return sklearn["Pipeline"](
            [("scale", sklearn["StandardScaler"]()), ("model", estimator)]
        )
    if candidate.family == "hist_gradient_boosting":
        defaults = {
            "learning_rate": 0.05,
            "max_iter": 250,
            "max_leaf_nodes": 15,
            "l2_regularization": 1.0,
            "random_state": candidate.random_state,
        }
        defaults.update(params)
        return sklearn["HistGradientBoostingClassifier"](**defaults)
    if candidate.family == "neural_mlp":
        defaults = {
            "hidden_layer_sizes": (64, 32),
            "activation": "relu",
            "solver": "adam",
            "alpha": 0.01,
            "batch_size": "auto",
            "early_stopping": True,
            "validation_fraction": 0.15,
            "max_iter": 500,
            "random_state": candidate.random_state,
        }
        defaults.update(params)
        estimator = sklearn["MLPClassifier"](**defaults)
        return sklearn["Pipeline"](
            [("scale", sklearn["StandardScaler"]()), ("model", estimator)]
        )
    # ModelCandidate validates family, but fail closed if an object was
    # constructed by an unusual deserialization path.
    raise ModelDataError("unsupported candidate family: %s" % candidate.family)


def _positive_probabilities(estimator: Any, features: Sequence[Sequence[float]]) -> List[float]:
    try:
        classes = list(estimator.classes_)
        positive_index = classes.index(1)
    except (AttributeError, ValueError) as error:
        raise ModelTrainingError("trained candidate does not expose a binary positive class") from error
    values = estimator.predict_proba(features)
    probabilities = [float(row[positive_index]) for row in values]
    if len(probabilities) != len(features) or any(not 0.0 <= value <= 1.0 for value in probabilities):
        raise ModelTrainingError("candidate emitted invalid probability values")
    return probabilities


@dataclass(frozen=True)
class OOFProbability:
    """One raw out-of-fold prediction with the metadata needed for calibration."""

    row_index: int
    fold_number: int
    event_id: str
    decision_at: datetime
    label_available_at: datetime
    raw_probability: float
    outcome: int

    def __post_init__(self) -> None:
        if self.row_index < 0 or self.fold_number < 0:
            raise ModelDataError("OOF indices and fold numbers must be non-negative")
        if not self.event_id or not self.event_id.strip():
            raise ModelDataError("OOF event_id must be non-empty")
        if not _is_aware(self.decision_at) or not _is_aware(self.label_available_at):
            raise ModelDataError("OOF decision and label timestamps must be timezone-aware")
        if self.label_available_at < self.decision_at:
            raise ModelDataError("OOF labels cannot be available before their decision time")
        if not isinstance(self.raw_probability, Real) or not 0.0 <= float(self.raw_probability) <= 1.0:
            raise ModelDataError("OOF probabilities must be in [0, 1]")
        if self.outcome not in (0, 1, False, True):
            raise ModelDataError("OOF outcomes must be binary")


@dataclass(frozen=True)
class CrossFittedCalibration:
    """Forward-only calibrated OOF values plus a final inference calibrator."""

    probabilities: Tuple[float, ...]
    calibrated_rows: int
    uncalibrated_rows: int
    final_calibrator: Optional[IsotonicCalibrator]


def cross_fit_isotonic_calibration(
    records: Sequence[OOFProbability], *, min_calibration_rows: int = 100
) -> CrossFittedCalibration:
    """Calibrate each validation fold with *earlier settled* OOF predictions.

    A calibrator is never fit on the predictions it transforms.  Early folds
    pass through uncalibrated when there is insufficient earlier coverage,
    which is intentional: pretending those labels existed would bias reported
    calibration quality.
    """

    if min_calibration_rows < 2:
        raise ModelDataError("min_calibration_rows must be at least two")
    if not records:
        raise ModelDataError("at least one OOF prediction is required for calibration")
    ordered = sorted(records, key=lambda item: (item.fold_number, item.decision_at, item.row_index))
    if list(records) != ordered:
        raise ModelDataError("OOF predictions must be supplied in chronological fold order")
    if len({record.row_index for record in records}) != len(records):
        raise ModelDataError("OOF predictions must not contain duplicate validation rows")

    calibrated: List[float] = []
    historical: List[OOFProbability] = []
    calibrated_rows = 0
    for fold_number, group_iterator in groupby(records, key=lambda item: item.fold_number):
        group = list(group_iterator)
        if fold_number < 0:
            raise ModelDataError("fold numbers must be non-negative")
        decision_cutoff = group[0].decision_at
        if any(record.decision_at < decision_cutoff for record in group):
            raise ModelDataError("OOF fold records must be chronological")
        eligible_history = [
            record for record in historical if record.label_available_at <= decision_cutoff
        ]
        if len(eligible_history) >= min_calibration_rows:
            calibrator = IsotonicCalibrator().fit(
                [record.raw_probability for record in eligible_history],
                [record.outcome for record in eligible_history],
            )
            calibrated.extend(calibrator.predict(record.raw_probability for record in group))
            calibrated_rows += len(group)
        else:
            calibrated.extend(record.raw_probability for record in group)
        historical.extend(group)

    final_calibrator: Optional[IsotonicCalibrator] = None
    if len(records) >= min_calibration_rows:
        final_calibrator = IsotonicCalibrator().fit(
            [record.raw_probability for record in records], [record.outcome for record in records]
        )
    return CrossFittedCalibration(
        probabilities=tuple(calibrated),
        calibrated_rows=calibrated_rows,
        uncalibrated_rows=len(records) - calibrated_rows,
        final_calibrator=final_calibrator,
    )


@dataclass(frozen=True)
class ProbabilityMetrics:
    """Proper probability scores only; no profit or ROI is used for promotion."""

    sample_size: int
    brier: float
    logloss: float
    expected_calibration_error: float

    def __post_init__(self) -> None:
        if self.sample_size < 1:
            raise ModelDataError("probability metrics require at least one observation")
        for name in ("brier", "logloss", "expected_calibration_error"):
            value = getattr(self, name)
            if not isinstance(value, Real) or not math.isfinite(float(value)) or value < 0.0:
                raise ModelDataError("%s must be a non-negative finite metric" % name)

    @classmethod
    def from_probabilities(
        cls, probabilities: Sequence[float], outcomes: Sequence[int], *, calibration_bin_count: int = 10
    ) -> "ProbabilityMetrics":
        if calibration_bin_count < 2:
            raise ModelDataError("calibration_bin_count must be at least two")
        bins = calibration_bins(probabilities, outcomes, bins=calibration_bin_count)
        total = len(outcomes)
        ece = sum(
            bucket.count * abs(bucket.mean_prediction - bucket.observed_rate) for bucket in bins
        ) / total
        return cls(
            sample_size=total,
            brier=brier_score(probabilities, outcomes),
            logloss=log_loss(probabilities, outcomes),
            expected_calibration_error=ece,
        )


@dataclass(frozen=True)
class CandidateScore:
    """Metrics for a single candidate on a known chronological OOF sample."""

    candidate_name: str
    metrics: ProbabilityMetrics
    evaluated_rows: int
    total_rows: int
    fold_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_name, str) or not self.candidate_name:
            raise ModelDataError("candidate_name must be non-empty")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (self.total_rows, self.evaluated_rows, self.fold_count)
        ):
            raise ModelDataError("candidate score row and fold counts must be integers")
        if self.total_rows < 1 or self.evaluated_rows < 1 or self.fold_count < 1:
            raise ModelDataError("candidate score requires positive row and fold counts")
        if self.evaluated_rows > self.total_rows:
            raise ModelDataError("evaluated rows cannot exceed total rows")
        if self.metrics.sample_size != self.evaluated_rows:
            raise ModelDataError("metric sample size must equal evaluated rows")

    @property
    def coverage(self) -> float:
        return self.evaluated_rows / self.total_rows


@dataclass(frozen=True)
class CandidateEvaluation:
    """Raw and forward-calibrated OOF evaluation for one candidate."""

    candidate: ModelCandidate
    schema_id: str
    data_fingerprint: str
    score: CandidateScore
    raw_metrics: ProbabilityMetrics
    oof_predictions: Tuple[OOFProbability, ...]
    calibration: CrossFittedCalibration


@dataclass(frozen=True)
class PromotionPolicy:
    """Conservative approval gates for a model registry promotion.

    Thresholds intentionally evaluate probability accuracy and calibration,
    rather than historical wager profit.  Set sport/market-specific limits in
    deployment configuration after a documented validation study.
    """

    minimum_evaluated_rows: int = 500
    minimum_coverage: float = 0.60
    maximum_brier: Optional[float] = 0.25
    maximum_logloss: Optional[float] = 0.70
    maximum_expected_calibration_error: Optional[float] = 0.05
    minimum_brier_improvement: float = 0.0005
    minimum_logloss_improvement: float = 0.001
    minimum_calibration_improvement: float = 0.001

    def __post_init__(self) -> None:
        if isinstance(self.minimum_evaluated_rows, bool) or not isinstance(self.minimum_evaluated_rows, int):
            raise ModelDataError("minimum_evaluated_rows must be an integer")
        if self.minimum_evaluated_rows < 2:
            raise ModelDataError("minimum_evaluated_rows must be at least two")
        if (
            isinstance(self.minimum_coverage, bool)
            or not isinstance(self.minimum_coverage, Real)
            or not math.isfinite(float(self.minimum_coverage))
            or not 0.0 < self.minimum_coverage <= 1.0
        ):
            raise ModelDataError("minimum_coverage must be in (0, 1]")
        for name in (
            "maximum_brier",
            "maximum_logloss",
            "maximum_expected_calibration_error",
        ):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or value < 0.0
            ):
                raise ModelDataError("%s must be a non-negative finite value" % name)
        for name in (
            "minimum_brier_improvement",
            "minimum_logloss_improvement",
            "minimum_calibration_improvement",
        ):
            value = getattr(self, name)
            if not isinstance(value, Real) or not math.isfinite(float(value)) or value < 0.0:
                raise ModelDataError("%s must be a non-negative finite value" % name)


@dataclass(frozen=True)
class PromotionDecision:
    approved: bool
    reasons: Tuple[str, ...]


def evaluate_promotion(
    challenger: CandidateScore,
    policy: PromotionPolicy,
    *,
    incumbent: Optional[CandidateScore] = None,
) -> PromotionDecision:
    """Apply coverage, quality, calibration, and incumbent-comparison gates."""

    reasons: List[str] = []
    metrics = challenger.metrics
    if challenger.evaluated_rows < policy.minimum_evaluated_rows:
        reasons.append(
            "insufficient out-of-fold sample: %d < %d"
            % (challenger.evaluated_rows, policy.minimum_evaluated_rows)
        )
    if challenger.coverage < policy.minimum_coverage:
        reasons.append(
            "insufficient chronological coverage: %.3f < %.3f"
            % (challenger.coverage, policy.minimum_coverage)
        )
    absolute_gates = (
        ("brier", metrics.brier, policy.maximum_brier),
        ("log loss", metrics.logloss, policy.maximum_logloss),
        (
            "expected calibration error",
            metrics.expected_calibration_error,
            policy.maximum_expected_calibration_error,
        ),
    )
    for label, value, maximum in absolute_gates:
        if maximum is not None and value > maximum:
            reasons.append("%s gate failed: %.6f > %.6f" % (label, value, maximum))

    if incumbent is not None:
        if incumbent.evaluated_rows != challenger.evaluated_rows:
            reasons.append("incumbent and challenger must use the same OOF sample size")
        comparisons = (
            ("brier", metrics.brier, incumbent.metrics.brier, policy.minimum_brier_improvement),
            ("log loss", metrics.logloss, incumbent.metrics.logloss, policy.minimum_logloss_improvement),
            (
                "expected calibration error",
                metrics.expected_calibration_error,
                incumbent.metrics.expected_calibration_error,
                policy.minimum_calibration_improvement,
            ),
        )
        for label, challenger_value, incumbent_value, improvement in comparisons:
            if challenger_value > incumbent_value - improvement:
                reasons.append(
                    "%s did not improve enough versus incumbent: %.6f vs %.6f (required %.6f)"
                    % (label, challenger_value, incumbent_value, improvement)
                )
    return PromotionDecision(approved=not reasons, reasons=tuple(reasons))


def evaluate_candidate_promotion(
    challenger: CandidateEvaluation,
    policy: PromotionPolicy,
    *,
    incumbent: Optional[CandidateEvaluation] = None,
) -> PromotionDecision:
    """Promote only comparable, calibrated candidate evaluations.

    ``evaluate_promotion`` is intentionally usable for stored scorecards.  New
    registry code should prefer this function because it additionally binds the
    challenger and incumbent to the same feature schema, exact data fingerprint,
    and chronological validation rows.
    """

    decision = evaluate_promotion(
        challenger.score, policy, incumbent=incumbent.score if incumbent is not None else None
    )
    reasons = list(decision.reasons)
    if challenger.calibration.final_calibrator is None:
        reasons.append("no final isotonic calibrator was produced from the OOF sample")
    if incumbent is not None:
        if challenger.schema_id != incumbent.schema_id:
            reasons.append("incumbent and challenger use different feature schemas")
        if challenger.data_fingerprint != incumbent.data_fingerprint:
            reasons.append("incumbent and challenger use different evaluated data fingerprints")
        challenger_rows = tuple(
            (record.row_index, record.fold_number) for record in challenger.oof_predictions
        )
        incumbent_rows = tuple(
            (record.row_index, record.fold_number) for record in incumbent.oof_predictions
        )
        if challenger_rows != incumbent_rows:
            reasons.append("incumbent and challenger use different chronological OOF rows")
    return PromotionDecision(approved=not reasons, reasons=tuple(reasons))


def select_best_candidate(evaluations: Sequence[CandidateEvaluation]) -> CandidateEvaluation:
    """Rank only comparable OOF candidates, favoring proper probability scores."""

    if not evaluations:
        raise ModelDataError("at least one candidate evaluation is required")
    first = evaluations[0].score
    first_evaluation = evaluations[0]
    first_oof_rows = tuple(
        (record.row_index, record.fold_number) for record in first_evaluation.oof_predictions
    )
    if any(
        evaluation.score.evaluated_rows != first.evaluated_rows
        or evaluation.score.total_rows != first.total_rows
        or evaluation.score.fold_count != first.fold_count
        or evaluation.schema_id != first_evaluation.schema_id
        or evaluation.data_fingerprint != first_evaluation.data_fingerprint
        or tuple((record.row_index, record.fold_number) for record in evaluation.oof_predictions)
        != first_oof_rows
        for evaluation in evaluations[1:]
    ):
        raise ModelDataError("candidates must be evaluated on matching OOF coverage before selection")
    return min(
        evaluations,
        key=lambda evaluation: (
            evaluation.score.metrics.brier,
            evaluation.score.metrics.logloss,
            evaluation.score.metrics.expected_calibration_error,
            evaluation.candidate.name,
        ),
    )


class ProbabilityModelEvaluator:
    """Run leakage-resistant walk-forward model evaluation on supplied rows."""

    def __init__(
        self,
        schema: FeatureSchema,
        splitter: RollingTimeSplitter,
        *,
        min_calibration_rows: int = 100,
        calibration_bin_count: int = 10,
    ) -> None:
        if min_calibration_rows < 2:
            raise ModelDataError("min_calibration_rows must be at least two")
        if calibration_bin_count < 2:
            raise ModelDataError("calibration_bin_count must be at least two")
        self.schema = schema
        self.splitter = splitter
        self.min_calibration_rows = min_calibration_rows
        self.calibration_bin_count = calibration_bin_count

    def evaluate(self, candidate: ModelCandidate, rows: Iterable[OutcomeTrainingRow]) -> CandidateEvaluation:
        validated_rows = validate_training_rows(rows, self.schema)
        data_fingerprint = training_dataset_fingerprint(validated_rows, self.schema)
        folds = self.splitter.split(validated_rows)
        matrix = [list(self.schema.vector(row.features)) for row in validated_rows]
        outcomes = [int(row.outcome) for row in validated_rows]
        predictions: List[OOFProbability] = []

        for fold in folds:
            train_outcomes = [outcomes[index] for index in fold.train_indices]
            if len(set(train_outcomes)) != 2:
                raise ModelTrainingError(
                    "%s cannot train fold %d because its settled history has one outcome class"
                    % (candidate.name, fold.number)
                )
            estimator = build_estimator(candidate)
            train_matrix = [matrix[index] for index in fold.train_indices]
            validation_matrix = [matrix[index] for index in fold.validation_indices]
            try:
                estimator.fit(train_matrix, train_outcomes)
                probabilities = _positive_probabilities(estimator, validation_matrix)
            except OptionalModelDependencyError:
                raise
            except Exception as error:
                raise ModelTrainingError(
                    "%s failed to train or score fold %d; do not replace this failure with synthetic values"
                    % (candidate.name, fold.number)
                ) from error
            for index, probability in zip(fold.validation_indices, probabilities):
                row = validated_rows[index]
                predictions.append(
                    OOFProbability(
                        row_index=index,
                        fold_number=fold.number,
                        event_id=row.event_id,
                        decision_at=row.decision_at,
                        label_available_at=row.label_available_at,
                        raw_probability=probability,
                        outcome=int(row.outcome),
                    )
                )

        if not predictions:
            raise ModelTrainingError("evaluation produced no out-of-fold predictions")
        calibration = cross_fit_isotonic_calibration(
            predictions, min_calibration_rows=self.min_calibration_rows
        )
        raw_probabilities = [record.raw_probability for record in predictions]
        oof_outcomes = [record.outcome for record in predictions]
        raw_metrics = ProbabilityMetrics.from_probabilities(
            raw_probabilities, oof_outcomes, calibration_bin_count=self.calibration_bin_count
        )
        calibrated_metrics = ProbabilityMetrics.from_probabilities(
            calibration.probabilities, oof_outcomes, calibration_bin_count=self.calibration_bin_count
        )
        score = CandidateScore(
            candidate_name=candidate.name,
            metrics=calibrated_metrics,
            evaluated_rows=len(predictions),
            total_rows=len(validated_rows),
            fold_count=len(folds),
        )
        return CandidateEvaluation(
            candidate=candidate,
            schema_id=self.schema.schema_id,
            data_fingerprint=data_fingerprint,
            score=score,
            raw_metrics=raw_metrics,
            oof_predictions=tuple(predictions),
            calibration=calibration,
        )

    def evaluate_many(
        self, candidates: Iterable[ModelCandidate], rows: Iterable[OutcomeTrainingRow]
    ) -> Tuple[CandidateEvaluation, ...]:
        """Evaluate every requested candidate on exactly the same supplied rows."""

        validated_rows = validate_training_rows(rows, self.schema)
        candidates = tuple(candidates)
        if not candidates:
            raise ModelDataError("at least one model candidate is required")
        return tuple(self.evaluate(candidate, validated_rows) for candidate in candidates)


@dataclass
class FittedProbabilityModel:
    """A final, approved estimator with its schema and OOF calibrator attached."""

    candidate: ModelCandidate
    schema: FeatureSchema
    estimator: Any
    calibrator: Optional[IsotonicCalibrator]
    trained_at: datetime
    released_at: datetime
    training_rows: int

    def predict_probability(self, request: PredictionInput) -> float:
        vector = validate_prediction_input(request, self.schema)
        if not _is_aware(self.trained_at) or not _is_aware(self.released_at):
            raise ModelDataError("fitted model timestamps must be timezone-aware")
        if self.released_at < self.trained_at:
            raise ModelDataError("a model cannot be released before it was trained")
        if request.decision_at < self.released_at:
            raise ModelDataError("model cannot score a decision from before its release time")
        raw_probability = _positive_probabilities(self.estimator, [list(vector)])[0]
        return self.calibrator.predict_one(raw_probability) if self.calibrator and self.calibrator.fitted else raw_probability


def fit_approved_model(
    evaluation: CandidateEvaluation,
    rows: Iterable[OutcomeTrainingRow],
    schema: FeatureSchema,
    *,
    trained_at: datetime,
    promotion_policy: PromotionPolicy,
    incumbent: Optional[CandidateEvaluation] = None,
    released_at: Optional[datetime] = None,
) -> FittedProbabilityModel:
    """Fit only after re-evaluating a policy-bound promotion and a settled cut-off.

    The database registry must still persist the candidate configuration,
    evaluation, source snapshots, artifact digest, and an independently
    authorized approval before serving. This function deliberately computes the
    promotion itself rather than accepting a caller-constructed approval flag.
    """

    promotion = evaluate_candidate_promotion(evaluation, promotion_policy, incumbent=incumbent)
    if not promotion.approved:
        raise ModelTrainingError("refusing to fit an unapproved model: " + "; ".join(promotion.reasons))
    if not _is_aware(trained_at):
        raise ModelDataError("trained_at must be timezone-aware")
    effective_at = released_at or trained_at
    if not _is_aware(effective_at):
        raise ModelDataError("released_at must be timezone-aware")
    if effective_at < trained_at:
        raise ModelDataError("released_at cannot precede trained_at")
    validated_rows = validate_training_rows(rows, schema)
    if evaluation.schema_id != schema.schema_id:
        raise ModelDataError("evaluation schema does not match the final-fit schema")
    if evaluation.data_fingerprint != training_dataset_fingerprint(validated_rows, schema):
        raise ModelDataError("final-fit rows differ from the rows used for walk-forward evaluation")
    if evaluation.calibration.final_calibrator is None:
        raise ModelTrainingError("refusing to fit a model without an OOF calibration artifact")
    unsettled = [
        row.event_id
        for row in validated_rows
        if row.decision_at >= trained_at or row.label_available_at > trained_at
    ]
    if unsettled:
        raise ModelDataError(
            "refusing final fit with rows unavailable at trained_at (first event: %s)" % unsettled[0]
        )
    outcomes = [int(row.outcome) for row in validated_rows]
    if len(set(outcomes)) != 2:
        raise ModelTrainingError("final model training requires both outcome classes")
    estimator = build_estimator(evaluation.candidate)
    try:
        estimator.fit([list(schema.vector(row.features)) for row in validated_rows], outcomes)
    except OptionalModelDependencyError:
        raise
    except Exception as error:
        raise ModelTrainingError(
            "%s failed during final fit; no substitute model was created" % evaluation.candidate.name
        ) from error
    return FittedProbabilityModel(
        candidate=evaluation.candidate,
        schema=schema,
        estimator=estimator,
        calibrator=evaluation.calibration.final_calibrator,
        trained_at=trained_at,
        released_at=effective_at,
        training_rows=len(validated_rows),
    )
