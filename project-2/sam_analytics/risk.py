"""Portfolio guardrails for research decisions.

The policy is intentionally conservative: a positive expected value alone
does not override stale data, event concentration, or daily exposure limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Dict, List

from .odds import expected_roi, kelly_fraction


@dataclass(frozen=True)
class BankrollPolicy:
    bankroll: float
    fractional_kelly: float = 0.25
    max_stake_fraction: float = 0.01
    max_event_fraction: float = 0.02
    max_daily_fraction: float = 0.05
    min_expected_roi: float = 0.015

    def __post_init__(self) -> None:
        if not _finite_number(self.bankroll) or self.bankroll <= 0:
            raise ValueError("bankroll must be a positive finite number")
        for value in (
            self.fractional_kelly,
            self.max_stake_fraction,
            self.max_event_fraction,
            self.max_daily_fraction,
        ):
            if not _finite_number(value) or not 0 < value <= 1:
                raise ValueError("risk fractions must be finite values in (0, 1]")
        if not _finite_number(self.min_expected_roi) or self.min_expected_roi < 0:
            raise ValueError("min_expected_roi must be a non-negative finite number")


@dataclass(frozen=True)
class ExposureState:
    event_exposure: Dict[str, float] = field(default_factory=dict)
    daily_exposure: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.event_exposure, dict):
            raise ValueError("event_exposure must be a dictionary")
        if not _finite_number(self.daily_exposure) or self.daily_exposure < 0:
            raise ValueError("daily exposure must be a non-negative finite number")
        for event_id, value in self.event_exposure.items():
            if not isinstance(event_id, str) or not event_id.strip():
                raise ValueError("event exposure keys must be non-empty event IDs")
            if not _finite_number(value) or value < 0:
                raise ValueError("event exposure must be a non-negative finite number")


@dataclass(frozen=True)
class SizingDecision:
    status: str
    stake: float
    expected_roi: float
    full_kelly: float
    reasons: List[str]


def size_moneyline(
    *,
    event_id: str,
    model_probability: float,
    decimal_odds: float,
    policy: BankrollPolicy,
    exposure: ExposureState,
    quote_is_fresh: bool,
    model_is_approved: bool,
) -> SizingDecision:
    """Return a capped stake or an explicit rejection reason.

    This is a research sizing control.  It does not place a wager and assumes
    the caller has already determined that betting is lawful for the user.
    """
    roi = expected_roi(model_probability, decimal_odds)
    full_kelly = kelly_fraction(model_probability, decimal_odds)
    reasons: List[str] = []
    if not quote_is_fresh:
        reasons.append("quote is stale or missing a provider timestamp")
    if not model_is_approved:
        reasons.append("model version is not approved for live evaluation")
    if roi < policy.min_expected_roi:
        reasons.append("expected ROI is below the configured minimum")
    if exposure.daily_exposure >= policy.bankroll * policy.max_daily_fraction:
        reasons.append("daily exposure limit is exhausted")
    if reasons:
        return SizingDecision("rejected", 0.0, roi, full_kelly, reasons)

    raw_stake = policy.bankroll * full_kelly * policy.fractional_kelly
    event_remaining = max(
        0.0,
        policy.bankroll * policy.max_event_fraction - exposure.event_exposure.get(event_id, 0.0),
    )
    daily_remaining = max(
        0.0, policy.bankroll * policy.max_daily_fraction - exposure.daily_exposure
    )
    stake = min(
        raw_stake,
        policy.bankroll * policy.max_stake_fraction,
        event_remaining,
        daily_remaining,
    )
    if stake <= 0:
        return SizingDecision("rejected", 0.0, roi, full_kelly, ["portfolio exposure limit is exhausted"])
    return SizingDecision("accepted", round(stake, 2), roi, full_kelly, [])


def _finite_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(float(value))
