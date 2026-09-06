"""Public, read-safe API routes.

No route here trains a model, writes fabricated outcomes, or submits a wager.
Administrative ingestion, approval, and model promotion belong on private
worker interfaces protected by the deployment identity provider.
"""

from __future__ import annotations

import hmac
import math
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from flask import Blueprint, current_app, jsonify, request

from sam_analytics.odds import (
    american_to_decimal,
    expected_roi,
    implied_probability,
    market_consensus_two_way,
)
from sam_analytics.readiness import DependencyReadiness, check_dependencies
from sam_analytics.risk import BankrollPolicy, ExposureState, size_moneyline
from sam_analytics.service_contract import OperationalSignals, build_integration_status

bp = Blueprint("api", __name__, url_prefix="/api")


@bp.after_request
def protect_versioned_api_responses(response):
    """Do not let a browser, proxy, or shared UI cache authenticated analytics."""
    if request.path.startswith("/api/v1/"):
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
    return response


@bp.get("/healthz")
def healthz():
    """Liveness probe; never reveals credentials or provider status."""
    return jsonify({"status": "ok", "service": "sam-analytics"})


@bp.get("/readyz")
def readyz():
    """Readiness probe that fails closed without exposing infrastructure facts."""

    settings = current_app.config["SAM_SETTINGS"]
    probe = current_app.config.get("SAM_DEPENDENCY_READINESS_PROBE", check_dependencies)
    try:
        result = probe(settings.database_url, settings.redis_url)
    except Exception:
        result = None
    is_ready = isinstance(result, DependencyReadiness) and result.ready
    return jsonify({"status": "ready" if is_ready else "not_ready"}), 200 if is_ready else 503


@bp.get("/v1/integration/status")
def integration_status():
    """Return the safe UI gateway contract without exposing operational secrets.

    Base44 (or another UI) should call this only from a server-side function
    using its secret store.  A missing health repository is deliberately
    represented as blocked rather than being treated as a healthy feed/model.
    """
    if not _status_authorized():
        return jsonify({"error": "unauthorized"}), 401
    settings = current_app.config["SAM_SETTINGS"]
    signals = current_app.config.get("SAM_OPERATIONAL_SIGNALS", OperationalSignals())
    if not isinstance(signals, OperationalSignals):
        current_app.logger.error("invalid SAM_OPERATIONAL_SIGNALS configuration")
        return jsonify({"error": "operational status is unavailable"}), 503
    return jsonify(
        build_integration_status(
            database_configured=bool(settings.database_url),
            redis_configured=bool(settings.redis_url),
            quote_max_age_seconds=settings.quote_max_age_seconds,
            signals=signals,
        )
    )


@bp.get("/v1/predictions")
def current_predictions():
    """Return current consensus, overlaid with eligible approved predictions."""
    if not _status_authorized():
        return jsonify({"error": "unauthorized"}), 401
    try:
        limit = _prediction_limit(request.args.get("limit", "20"))
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    settings = current_app.config["SAM_SETTINGS"]
    if not settings.database_url:
        return jsonify({"error": "predictions unavailable"}), 503
    now = datetime.now(UTC)
    reader = current_app.config.get("SAM_PREDICTION_READER", _read_current_h2h_quotes)
    try:
        rows = reader(
            database_url=settings.database_url,
            now=now,
            quote_max_age_seconds=settings.quote_max_age_seconds,
            limit=limit,
        )
        predictions = _build_market_consensus_predictions(
            rows,
            now=now,
            quote_max_age_seconds=settings.quote_max_age_seconds,
            limit=limit,
        )
    except Exception:
        # Database and row errors can contain connection details.  Keep both
        # logs and the response deliberately generic.
        current_app.logger.error("market consensus prediction lookup failed")
        return jsonify({"error": "predictions unavailable"}), 503

    _initialize_prediction_sources(predictions)
    trained_count = 0
    if predictions:
        trained_reader = current_app.config.get(
            "SAM_APPROVED_PREDICTION_READER",
            _read_current_approved_predictions,
        )
        try:
            trained_rows = trained_reader(
                database_url=settings.database_url,
                now=now,
                event_ids=tuple(prediction["event_id"] for prediction in predictions),
            )
            trained_count = _overlay_approved_predictions(
                predictions,
                trained_rows,
                now=now,
            )
        except Exception:
            # An unavailable or malformed trained-model record must never hide
            # the independently computed market baseline or leak database facts.
            current_app.logger.error("approved trained prediction lookup failed")

    method = "market_consensus_v1"
    classification = "market_consensus_baseline"
    trained_model = False
    notice = (
        "Descriptive consensus from current bookmaker prices; not a trained model, "
        "independent betting edge, or wager recommendation."
    )
    trained_summary: dict[str, int] = {}
    if trained_count:
        fallback_count = len(predictions) - trained_count
        method = "approved_trained_model_v1"
        classification = (
            "approved_trained_model"
            if fallback_count == 0
            else "approved_trained_model_with_market_consensus_fallback"
        )
        trained_model = True
        trained_summary = {
            "trained_model_count": trained_count,
            "market_consensus_fallback_count": fallback_count,
        }
        notice = (
            "Approved trained-model probabilities are shown where eligible alongside "
            "current bookmaker consensus and prices; remaining events use the descriptive "
            "market baseline. Analytical output only, not a wager recommendation or guarantee."
        )

    return jsonify(
        {
            "status": "ok",
            "method": method,
            "classification": classification,
            "trained_model": trained_model,
            "generated_at": _timestamp(now),
            "quote_max_age_seconds": settings.quote_max_age_seconds,
            "count": len(predictions),
            **trained_summary,
            "predictions": predictions,
            "notice": notice,
        }
    )


@bp.post("/v1/evaluate")
def evaluate_moneyline():
    """Evaluate one timestamped price against an approved model probability.

    The endpoint returns an analytic sizing control only. It records neither a
    bet nor a performance result, and requires an API key outside local dev.
    """
    if not _authorized():
        return jsonify({"error": "unauthorized"}), 401
    settings = current_app.config["SAM_SETTINGS"]
    if settings.environment not in {"development", "test"}:
        return jsonify(
            {
                "error": "private research endpoint disabled",
                "message": "Production serving must resolve immutable approved predictions, quotes, policy, and exposure server-side.",
            }
        ), 403
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "expected a JSON object"}), 400
    try:
        decimal_odds = _decimal_odds(payload)
        quote_at = _parse_timestamp(payload["quote_captured_at"])
        age_seconds = (datetime.now(UTC) - quote_at).total_seconds()
        policy = BankrollPolicy(
            bankroll=float(payload.get("bankroll", 1000.0)),
            fractional_kelly=float(payload.get("fractional_kelly", 0.25)),
            max_stake_fraction=float(payload.get("max_stake_fraction", 0.01)),
            max_event_fraction=float(payload.get("max_event_fraction", 0.02)),
            max_daily_fraction=float(payload.get("max_daily_fraction", 0.05)),
            min_expected_roi=float(payload.get("min_expected_roi", 0.015)),
        )
        event_id = str(payload["event_id"])
        state = ExposureState(
            event_exposure={event_id: float(payload.get("event_exposure", 0.0))},
            daily_exposure=float(payload.get("daily_exposure", 0.0)),
        )
        decision = size_moneyline(
            event_id=event_id,
            model_probability=float(payload["model_probability"]),
            decimal_odds=decimal_odds,
            policy=policy,
            exposure=state,
            quote_is_fresh=0 <= age_seconds <= settings.quote_max_age_seconds,
            # A client-supplied label is not an approval.  Until the database
            # registry is wired into this request path, use an explicit
            # server-side allow-list so callers cannot self-approve a model.
            model_is_approved=str(payload.get("model_version", "")) in settings.approved_model_versions,
        )
        return jsonify(
            {
                "event_id": event_id,
                "model_version": str(payload.get("model_version", "")),
                "model_is_approved": str(payload.get("model_version", "")) in settings.approved_model_versions,
                "decimal_odds": decimal_odds,
                "vig_included_implied_probability": implied_probability(decimal_odds),
                "expected_roi": expected_roi(float(payload["model_probability"]), decimal_odds),
                "quote_age_seconds": round(age_seconds, 3),
                "decision": decision.__dict__,
                "notice": "Analytical output only. It does not place a wager or establish profitability.",
            }
        )
    except (KeyError, TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400


@bp.route("/<path:_legacy>", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
def legacy_api_disabled(_legacy: str):
    """Prevent old demo/simulation endpoints from generating misleading data."""
    return jsonify(
        {
            "error": "legacy endpoint retired",
            "message": "Use /api/v1/evaluate with timestamped provider odds and an approved model version.",
        }
    ), 410


def _authorized() -> bool:
    settings = current_app.config["SAM_SETTINGS"]
    if not settings.api_key and settings.environment in {"development", "test"}:
        return True
    supplied = request.headers.get("X-API-Key", "")
    return bool(settings.api_key) and hmac.compare_digest(supplied, settings.api_key)


def _status_authorized() -> bool:
    """Authorize the read-safe server-side UI capability.

    The Base44 gateway receives a separate key which can read sanitized status
    and market output, but cannot invoke research evaluation. In local
    development and tests an intentionally absent key keeps the contract easy
    to exercise without copying credentials.
    """

    settings = current_app.config["SAM_SETTINGS"]
    if not settings.status_api_key and settings.environment in {"development", "test"}:
        return True
    supplied = request.headers.get("X-API-Key", "")
    return bool(settings.status_api_key) and hmac.compare_digest(supplied, settings.status_api_key)


def _decimal_odds(payload: dict[str, Any]) -> float:
    has_decimal = "decimal_odds" in payload
    has_american = "american_odds" in payload
    if has_decimal == has_american:
        raise ValueError("provide exactly one of decimal_odds or american_odds")
    return float(payload["decimal_odds"]) if has_decimal else american_to_decimal(float(payload["american_odds"]))


def _parse_timestamp(value: Any) -> datetime:
    timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("quote_captured_at must include an offset or Z")
    return timestamp.astimezone(UTC)


def _prediction_limit(value: object) -> int:
    try:
        limit = int(str(value))
    except (TypeError, ValueError):
        raise ValueError("limit must be an integer from 1 to 100") from None
    if str(limit) != str(value) or not 1 <= limit <= 100:
        raise ValueError("limit must be an integer from 1 to 100")
    return limit


def _read_current_h2h_quotes(
    *,
    database_url: str,
    now: datetime,
    quote_max_age_seconds: int,
    limit: int,
) -> list[Mapping[str, Any]]:
    """Read only the latest fresh home/away quote per event and bookmaker."""
    import psycopg
    from psycopg.rows import dict_row

    oldest = now - timedelta(seconds=quote_max_age_seconds)
    connection = psycopg.connect(
        database_url,
        application_name="sam-analytics-predictions",
        connect_timeout=2,
        options="-c statement_timeout=3000 -c default_transaction_read_only=on",
        row_factory=dict_row,
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                WITH candidate_events AS (
                    SELECT event.id, event.provider, event.provider_event_id,
                           event.sport, event.league, event.starts_at,
                           event.home_team, event.away_team
                    FROM sports_event AS event
                    WHERE event.starts_at > %(now)s
                      AND EXISTS (
                          SELECT 1
                          FROM odds_snapshot AS candidate_quote
                          WHERE candidate_quote.event_id = event.id
                            AND candidate_quote.market = 'h2h'
                            AND candidate_quote.bookmaker IS NOT NULL
                            AND candidate_quote.captured_at BETWEEN %(oldest)s AND %(now)s
                            AND candidate_quote.selection IN (event.home_team, event.away_team)
                      )
                    ORDER BY event.starts_at, event.id
                    LIMIT %(limit)s
                ),
                latest_quotes AS (
                    SELECT DISTINCT ON (event.id, quote.bookmaker, quote.selection)
                           event.id AS event_id, event.provider, event.provider_event_id,
                           event.sport, event.league, event.starts_at,
                           event.home_team, event.away_team,
                           quote.bookmaker, quote.selection, quote.decimal_odds,
                           quote.captured_at
                    FROM candidate_events AS event
                    JOIN odds_snapshot AS quote ON quote.event_id = event.id
                    WHERE quote.market = 'h2h'
                      AND quote.bookmaker IS NOT NULL
                      AND quote.captured_at BETWEEN %(oldest)s AND %(now)s
                      AND quote.selection IN (event.home_team, event.away_team)
                    ORDER BY event.id, quote.bookmaker, quote.selection,
                             quote.captured_at DESC, quote.received_at DESC, quote.id DESC
                )
                SELECT event_id, provider, provider_event_id, sport, league, starts_at,
                       home_team, away_team, bookmaker, selection, decimal_odds, captured_at
                FROM latest_quotes
                ORDER BY starts_at, event_id, bookmaker, selection
                """,
                {"now": now, "oldest": oldest, "limit": limit},
            )
            return list(cursor.fetchall())
    finally:
        connection.close()


def _read_current_approved_predictions(
    *,
    database_url: str,
    now: datetime,
    event_ids: tuple[str, ...],
) -> list[Mapping[str, Any]]:
    """Read one currently authorized persisted prediction per requested event."""
    import psycopg
    from psycopg.rows import dict_row

    if not event_ids:
        return []
    connection = psycopg.connect(
        database_url,
        application_name="sam-analytics-approved-predictions",
        connect_timeout=2,
        options="-c statement_timeout=3000 -c default_transaction_read_only=on",
        row_factory=dict_row,
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                WITH latest_decision AS (
                    SELECT DISTINCT ON (decision.model_id)
                           decision.model_id, decision.decision,
                           decision.decided_by, decision.decided_at,
                           decision.created_at, decision.id
                    FROM model_governance_decision AS decision
                    WHERE decision.decided_at <= %(now)s
                    ORDER BY decision.model_id, decision.decided_at DESC,
                             decision.created_at DESC, decision.id DESC
                ),
                eligible_predictions AS (
                    SELECT prediction.id::text AS prediction_id,
                           prediction.event_id::text AS event_id,
                           prediction.model_id::text AS model_id,
                           prediction.as_of, prediction.features_available_at,
                           prediction.created_at AS prediction_created_at,
                           prediction.home_win_probability,
                           prediction.feature_values_uri,
                           prediction.feature_values_sha256,
                           event.sport AS event_sport,
                           event.starts_at,
                           model.version,
                           model.sport AS model_sport,
                           model.target_definition,
                           model.approval_status AS registry_approval_status,
                           model.approved_by, model.approved_at,
                           model.created_at AS model_created_at,
                           model.artifact_format, model.artifact_sha256,
                           octet_length(model.artifact_bytes) > 0
                               AS artifact_bytes_present,
                           model.artifact_sha256 = encode(
                               public.digest(model.artifact_bytes, 'sha256'), 'hex'
                           ) AS artifact_digest_verified,
                           latest.decision AS governance_decision,
                           latest.decided_by,
                           latest.decided_at AS released_at
                    FROM prediction
                    JOIN sports_event AS event ON event.id = prediction.event_id
                    JOIN model_registry AS model ON model.id = prediction.model_id
                    JOIN latest_decision AS latest ON latest.model_id = model.id
                    WHERE prediction.event_id = ANY(%(event_ids)s::uuid[])
                      AND event.starts_at > %(now)s
                      AND event.sport = model.sport
                      AND model.target_definition = 'home_team_wins'
                      AND model.approval_status = 'approved'
                      AND model.approved_by = latest.decided_by
                      AND model.approved_at = latest.decided_at
                      AND latest.decision = 'approved'
                      AND latest.decided_at >= model.created_at
                      AND length(btrim(model.version)) > 0
                      AND model.artifact_format = 'sam-joblib-envelope-v1'
                      AND model.artifact_bytes IS NOT NULL
                      AND octet_length(model.artifact_bytes) > 0
                      AND model.artifact_sha256 ~ '^[0-9a-f]{64}$'
                      AND model.artifact_sha256 = encode(
                          public.digest(model.artifact_bytes, 'sha256'), 'hex'
                      )
                      AND prediction.as_of >= latest.decided_at
                      AND prediction.as_of <= %(now)s
                      AND prediction.as_of < event.starts_at
                      AND prediction.features_available_at <= prediction.as_of
                      AND prediction.created_at >= latest.decided_at
                      AND prediction.created_at BETWEEN prediction.as_of AND %(now)s
                      AND prediction.feature_values_uri LIKE
                          'data:application/vnd.sam.feature-vector+json;base64,%'
                      AND prediction.feature_values_sha256 ~ '^[0-9a-f]{64}$'
                      AND prediction.home_win_probability BETWEEN 0 AND 1
                )
                SELECT DISTINCT ON (event_id) *
                FROM eligible_predictions
                ORDER BY event_id, released_at DESC, as_of DESC,
                         model_created_at DESC, model_id DESC, prediction_id DESC
                """,
                {"now": now, "event_ids": list(event_ids)},
            )
            return list(cursor.fetchall())
    finally:
        connection.close()


def _initialize_prediction_sources(predictions: Iterable[dict[str, Any]]) -> None:
    """Make the per-event fallback explicit without changing consensus fields."""
    for prediction in predictions:
        prediction["trained_model"] = False
        prediction["prediction_source"] = "market_consensus_v1"
        prediction["model"] = None


def _overlay_approved_predictions(
    predictions: list[dict[str, Any]],
    rows: Iterable[Mapping[str, Any]],
    *,
    now: datetime,
) -> int:
    """Overlay only complete, internally consistent approved model records."""
    events = {prediction["event_id"]: prediction for prediction in predictions}
    candidates: dict[str, tuple[tuple[datetime, datetime, str, str], dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        event_id = _bounded_identifier(row.get("event_id"))
        if event_id is None or event_id not in events:
            continue
        candidate = _approved_prediction_candidate(row, events[event_id], now=now)
        if candidate is None:
            continue
        rank, model = candidate
        previous = candidates.get(event_id)
        if previous is None or rank > previous[0]:
            candidates[event_id] = (rank, model)

    for event_id, (_rank, model) in candidates.items():
        prediction = events[event_id]
        prediction["trained_model"] = True
        prediction["prediction_source"] = "approved_trained_model_v1"
        prediction["model"] = model

        home_probability = model["home_win_probability"]
        away_probability = model["away_win_probability"]
        prediction["home"]["trained_model_expected_roi"] = round(
            expected_roi(
                home_probability,
                prediction["home"]["best_available_price"]["decimal_odds"],
            ),
            6,
        )
        prediction["home"]["probability_edge_vs_consensus"] = round(
            home_probability - prediction["home"]["consensus_probability"],
            6,
        )
        prediction["away"]["trained_model_expected_roi"] = round(
            expected_roi(
                away_probability,
                prediction["away"]["best_available_price"]["decimal_odds"],
            ),
            6,
        )
        prediction["away"]["probability_edge_vs_consensus"] = round(
            away_probability - prediction["away"]["consensus_probability"],
            6,
        )
    return len(candidates)


def _approved_prediction_candidate(
    row: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    now: datetime,
) -> tuple[tuple[datetime, datetime, str, str], dict[str, Any]] | None:
    prediction_id = _bounded_identifier(row.get("prediction_id"))
    model_id = _bounded_identifier(row.get("model_id"))
    version = _model_version(row.get("version"))
    approved_by = _bounded_identifier(row.get("approved_by"))
    decided_by = _bounded_identifier(row.get("decided_by"))
    feature_values_uri = row.get("feature_values_uri")
    if (
        prediction_id is None
        or model_id is None
        or version is None
        or approved_by is None
        or approved_by != decided_by
        or row.get("registry_approval_status") != "approved"
        or row.get("governance_decision") != "approved"
        or row.get("target_definition") != "home_team_wins"
        or not isinstance(feature_values_uri, str)
        or not feature_values_uri.startswith(
            "data:application/vnd.sam.feature-vector+json;base64,"
        )
        or not _is_sha256(row.get("feature_values_sha256"))
        or row.get("artifact_format") != "sam-joblib-envelope-v1"
        or not _is_sha256(row.get("artifact_sha256"))
        or row.get("artifact_bytes_present") is not True
        or row.get("artifact_digest_verified") is not True
    ):
        return None

    baseline_sport = str(baseline.get("sport", "")).strip()
    event_sport = str(row.get("event_sport", "")).strip()
    model_sport = str(row.get("model_sport", "")).strip()
    if not baseline_sport or event_sport != baseline_sport or model_sport != event_sport:
        return None

    model_created_at = _utc_datetime(row.get("model_created_at"))
    approved_at = _utc_datetime(row.get("approved_at"))
    released_at = _utc_datetime(row.get("released_at"))
    as_of = _utc_datetime(row.get("as_of"))
    features_available_at = _utc_datetime(row.get("features_available_at"))
    prediction_created_at = _utc_datetime(row.get("prediction_created_at"))
    event_starts_at = _utc_datetime(row.get("starts_at"))
    baseline_starts_at = _iso_datetime(baseline.get("starts_at"))
    if (
        model_created_at is None
        or approved_at is None
        or released_at is None
        or approved_at != released_at
        or released_at < model_created_at
        or as_of is None
        or features_available_at is None
        or prediction_created_at is None
        or event_starts_at is None
        or baseline_starts_at is None
        or event_starts_at != baseline_starts_at
        or released_at > as_of
        or as_of > now
        or as_of >= event_starts_at
        or features_available_at > as_of
        or prediction_created_at < released_at
        or not as_of <= prediction_created_at <= now
    ):
        return None

    value = row.get("home_win_probability")
    if isinstance(value, bool):
        return None
    try:
        home_probability = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(home_probability) or not 0.0 <= home_probability <= 1.0:
        return None
    home_probability = round(home_probability, 6)
    away_probability = round(1.0 - home_probability, 6)
    model = {
        "prediction_id": prediction_id,
        "version": version,
        "as_of": _timestamp(as_of),
        "features_available_at": _timestamp(features_available_at),
        "home_win_probability": home_probability,
        "away_win_probability": away_probability,
    }
    return (released_at, as_of, version, prediction_id), model


def _bounded_identifier(value: object, *, maximum_length: int = 200) -> str | None:
    text = str(value).strip() if value is not None else ""
    if not text or len(text) > maximum_length or any(ord(character) < 32 for character in text):
        return None
    return text


def _model_version(value: object) -> str | None:
    version = _bounded_identifier(value, maximum_length=128)
    if version is None or any(
        not (character.isalnum() or character in "._:-") for character in version
    ):
        return None
    return version


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _iso_datetime(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return _utc_datetime(parsed)


def _build_market_consensus_predictions(
    rows: Iterable[Mapping[str, Any]],
    *,
    now: datetime,
    quote_max_age_seconds: int,
    limit: int,
) -> list[dict[str, Any]]:
    events: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        event_id = str(row.get("event_id", "")).strip()
        bookmaker = str(row.get("bookmaker", "")).strip()
        home_team = str(row.get("home_team", "")).strip()
        away_team = str(row.get("away_team", "")).strip()
        selection = str(row.get("selection", "")).strip()
        starts_at = _utc_datetime(row.get("starts_at"))
        captured_at = _utc_datetime(row.get("captured_at"))
        if (
            not event_id
            or not bookmaker
            or not home_team
            or not away_team
            or home_team == away_team
            or selection not in {home_team, away_team}
            or starts_at is None
            or starts_at <= now
            or captured_at is None
        ):
            continue
        quote_age = (now - captured_at).total_seconds()
        if not 0 <= quote_age <= quote_max_age_seconds:
            continue
        try:
            decimal_odds = float(row["decimal_odds"])
            implied_probability(decimal_odds)
        except (KeyError, TypeError, ValueError):
            continue

        event = events.setdefault(
            event_id,
            {
                "event_id": event_id,
                "provider": str(row.get("provider", "")),
                "provider_event_id": str(row.get("provider_event_id", "")),
                "sport": str(row.get("sport", "")),
                "league": str(row.get("league", "")),
                "starts_at": starts_at,
                "home_team": home_team,
                "away_team": away_team,
                "books": {},
            },
        )
        if event["home_team"] != home_team or event["away_team"] != away_team:
            continue
        book = event["books"].setdefault(bookmaker, {})
        previous = book.get(selection)
        if previous is None or captured_at > previous["captured_at"]:
            book[selection] = {
                "decimal_odds": decimal_odds,
                "captured_at": captured_at,
            }

    predictions: list[dict[str, Any]] = []
    ordered_events = sorted(events.values(), key=lambda event: (event["starts_at"], event["event_id"]))
    for event in ordered_events:
        complete_books: list[dict[str, Any]] = []
        for bookmaker, selections in event["books"].items():
            home = selections.get(event["home_team"])
            away = selections.get(event["away_team"])
            if home is None or away is None:
                continue
            complete_books.append(
                {
                    "bookmaker": bookmaker,
                    "home": home,
                    "away": away,
                }
            )
        if len(complete_books) < 2:
            continue

        home_probability, away_probability = market_consensus_two_way(
            (book["home"]["decimal_odds"], book["away"]["decimal_odds"])
            for book in complete_books
        )
        best_home = min(
            complete_books,
            key=lambda book: (-book["home"]["decimal_odds"], book["bookmaker"]),
        )
        best_away = min(
            complete_books,
            key=lambda book: (-book["away"]["decimal_odds"], book["bookmaker"]),
        )
        as_of = max(
            quote["captured_at"]
            for book in complete_books
            for quote in (book["home"], book["away"])
        )
        predictions.append(
            {
                "event_id": event["event_id"],
                "provider": event["provider"],
                "provider_event_id": event["provider_event_id"],
                "sport": event["sport"],
                "league": event["league"],
                "starts_at": _timestamp(event["starts_at"]),
                "as_of": _timestamp(as_of),
                "market": "h2h",
                "method": "market_consensus_v1",
                "book_count": len(complete_books),
                "home": _consensus_side(
                    selection=event["home_team"],
                    probability=home_probability,
                    best_book=best_home,
                    side="home",
                ),
                "away": _consensus_side(
                    selection=event["away_team"],
                    probability=away_probability,
                    best_book=best_away,
                    side="away",
                ),
            }
        )
        if len(predictions) == limit:
            break
    return predictions


def _consensus_side(
    *, selection: str, probability: float, best_book: Mapping[str, Any], side: str
) -> dict[str, Any]:
    quote = best_book[side]
    decimal_odds = quote["decimal_odds"]
    return {
        "selection": selection,
        "consensus_probability": round(probability, 6),
        "best_available_price": {
            "decimal_odds": round(decimal_odds, 6),
            "bookmaker": best_book["bookmaker"],
            "captured_at": _timestamp(quote["captured_at"]),
        },
        "best_price_implied_probability": round(implied_probability(decimal_odds), 6),
        "consensus_expected_roi": round(expected_roi(probability, decimal_odds), 6),
    }


def _utc_datetime(value: object) -> datetime | None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        return None
    return value.astimezone(UTC)


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
