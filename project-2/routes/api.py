"""Public, read-safe API routes.

No route here trains a model, writes fabricated outcomes, or submits a wager.
Administrative ingestion, approval, and model promotion belong on private
worker interfaces protected by the deployment identity provider.
"""

from __future__ import annotations

import hmac
from datetime import datetime, timezone
from typing import Any, Dict

from flask import Blueprint, current_app, jsonify, request

from sam_analytics.odds import american_to_decimal, expected_roi, implied_probability
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
        age_seconds = (datetime.now(timezone.utc) - quote_at).total_seconds()
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
    """Authorize only the sanitized status capability.

    The Base44 gateway receives a separate key which cannot invoke research
    evaluation. In local development and tests an intentionally absent key
    keeps the status contract easy to exercise without copying credentials.
    """

    settings = current_app.config["SAM_SETTINGS"]
    if not settings.status_api_key and settings.environment in {"development", "test"}:
        return True
    supplied = request.headers.get("X-API-Key", "")
    return bool(settings.status_api_key) and hmac.compare_digest(supplied, settings.status_api_key)


def _decimal_odds(payload: Dict[str, Any]) -> float:
    has_decimal = "decimal_odds" in payload
    has_american = "american_odds" in payload
    if has_decimal == has_american:
        raise ValueError("provide exactly one of decimal_odds or american_odds")
    return float(payload["decimal_odds"]) if has_decimal else american_to_decimal(float(payload["american_odds"]))


def _parse_timestamp(value: Any) -> datetime:
    timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("quote_captured_at must include an offset or Z")
    return timestamp.astimezone(timezone.utc)
