"""Chronological, price-aware backtest runner.

It does not create, shuffle, or fill missing historical records.  A record
with data unavailable at decision time fails closed so reported performance
cannot accidentally use hindsight.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from typing import Iterable, List

from .metrics import evaluation_report
from .risk import BankrollPolicy, ExposureState, size_moneyline


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ApprovedModelRelease:
    """A registry-derived release identity, not a caller-supplied version label."""

    version: str
    artifact_sha256: str
    effective_at: datetime


@dataclass(frozen=True)
class BacktestObservation:
    event_id: str
    event_starts_at: datetime
    decision_at: datetime
    quote_at: datetime
    features_available_at: datetime
    quote_snapshot_id: str
    prediction_id: str
    model_release: ApprovedModelRelease
    model_probability: float
    decimal_odds: float
    outcome: int


@dataclass(frozen=True)
class BacktestResult:
    evaluated: int
    rejected: int
    stake: float
    profit: float
    roi_on_staked: float
    metrics: dict


def run_backtest(
    observations: Iterable[BacktestObservation], policy: BankrollPolicy, *, max_quote_age_seconds: int = 300
) -> BacktestResult:
    rows = list(observations)
    if rows != sorted(rows, key=lambda row: row.decision_at):
        raise ValueError("observations must be passed in chronological decision order")
    probabilities: List[float] = []
    outcomes: List[int] = []
    profit = stake_total = 0.0
    rejected = 0
    exposure = ExposureState()
    exposure_day = None
    for row in rows:
        _assert_no_lookahead(row)
        decision_day = row.decision_at.date()
        if exposure_day is not None and decision_day != exposure_day:
            exposure = ExposureState(
                event_exposure=dict(exposure.event_exposure),
                daily_exposure=0.0,
            )
        exposure_day = decision_day
        fresh = (row.decision_at - row.quote_at).total_seconds() <= max_quote_age_seconds
        decision = size_moneyline(
            event_id=row.event_id,
            model_probability=row.model_probability,
            decimal_odds=row.decimal_odds,
            policy=policy,
            exposure=exposure,
            quote_is_fresh=fresh,
            model_is_approved=True,
        )
        probabilities.append(row.model_probability)
        outcomes.append(row.outcome)
        if decision.status == "rejected":
            rejected += 1
            continue
        stake_total += decision.stake
        profit += decision.stake * (row.decimal_odds - 1.0 if row.outcome else -1.0)
        event_exposure = dict(exposure.event_exposure)
        event_exposure[row.event_id] = event_exposure.get(row.event_id, 0.0) + decision.stake
        exposure = ExposureState(event_exposure=event_exposure, daily_exposure=exposure.daily_exposure + decision.stake)
    return BacktestResult(
        evaluated=len(rows),
        rejected=rejected,
        stake=round(stake_total, 2),
        profit=round(profit, 2),
        roi_on_staked=profit / stake_total if stake_total else 0.0,
        metrics=evaluation_report(probabilities, outcomes) if rows else {},
    )


def _assert_no_lookahead(row: BacktestObservation) -> None:
    if not isinstance(row.model_release, ApprovedModelRelease):
        raise ValueError("backtest rows require a registry-derived ApprovedModelRelease")
    if any(
        not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None
        for value in (
            row.event_starts_at,
            row.decision_at,
            row.quote_at,
            row.features_available_at,
            row.model_release.effective_at,
        )
    ):
        raise ValueError("all backtest timestamps must be timezone-aware")
    if not isinstance(row.event_id, str) or not row.event_id.strip():
        raise ValueError("event_id must be non-empty")
    if row.decision_at >= row.event_starts_at:
        raise ValueError("decision timestamp must precede the scheduled event start")
    if row.model_release.effective_at > row.decision_at:
        raise ValueError("model release became effective after the recorded decision time")
    if row.features_available_at > row.decision_at:
        raise ValueError("feature data arrived after the recorded decision time")
    if row.quote_at > row.decision_at:
        raise ValueError("quote timestamp is after the recorded decision time")
    for label, value in (
        ("quote_snapshot_id", row.quote_snapshot_id),
        ("prediction_id", row.prediction_id),
        ("model version", row.model_release.version),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be a non-empty immutable identifier")
    if not isinstance(row.model_release.artifact_sha256, str) or not _SHA256_RE.fullmatch(
        row.model_release.artifact_sha256
    ):
        raise ValueError("model release requires a lowercase SHA-256 artifact digest")
    if row.outcome not in (0, 1) or isinstance(row.outcome, bool):
        raise ValueError("backtest outcomes must be binary integers")
