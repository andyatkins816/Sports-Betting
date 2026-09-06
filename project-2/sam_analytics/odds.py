"""Odds conversion and value math with explicit validation.

The functions in this module operate on *decimal* returns internally.  They
are deterministic and intentionally do not infer a line, a probability, or a
bookmaker from incomplete data.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from statistics import median


class OddsValidationError(ValueError):
    """Raised when an odds value cannot describe a valid wager."""


def american_to_decimal(american: float) -> float:
    """Convert American odds to decimal odds, including the returned stake."""
    american = float(american)
    if not math.isfinite(american) or american == 0:
        raise OddsValidationError("American odds must be finite and non-zero")
    return 1.0 + (american / 100.0 if american > 0 else 100.0 / abs(american))


def decimal_to_american(decimal: float) -> float:
    """Convert valid decimal odds to American odds."""
    decimal = float(decimal)
    if not math.isfinite(decimal) or decimal <= 1.0:
        raise OddsValidationError("Decimal odds must be finite and greater than 1")
    return (decimal - 1.0) * 100.0 if decimal >= 2.0 else -100.0 / (decimal - 1.0)


def implied_probability(decimal: float) -> float:
    """Return the bookmaker's vig-included implied probability."""
    decimal = float(decimal)
    if not math.isfinite(decimal) or decimal <= 1.0:
        raise OddsValidationError("Decimal odds must be finite and greater than 1")
    return 1.0 / decimal


def devig_two_way(home_decimal: float, away_decimal: float) -> tuple[float, float]:
    """Remove proportional margin from a two-outcome market.

    This is a simple normalization, not a claim that the result is a true
    probability.  Three-way markets, alternate lines, and correlated markets
    must use a market-specific method upstream.
    """
    home = implied_probability(home_decimal)
    away = implied_probability(away_decimal)
    overround = home + away
    if overround <= 0:
        raise OddsValidationError("Market probabilities must have positive mass")
    return home / overround, away / overround


def market_consensus_two_way(
    bookmaker_prices: Iterable[tuple[float, float]],
) -> tuple[float, float]:
    """Return a robust descriptive consensus from complete two-way books.

    Each pair is de-vigged independently before the median home and away
    probabilities are normalized.  This is a market baseline, not a trained
    model or evidence of an independent betting edge.
    """
    probabilities = [devig_two_way(home, away) for home, away in bookmaker_prices]
    if not probabilities:
        raise OddsValidationError("At least one complete two-way market is required")
    home = median(probability[0] for probability in probabilities)
    away = median(probability[1] for probability in probabilities)
    total = home + away
    if not math.isfinite(total) or total <= 0:
        raise OddsValidationError("Market consensus must have positive finite mass")
    return home / total, away / total


def expected_roi(model_probability: float, decimal_odds: float) -> float:
    """Expected profit per unit staked; 0.02 means 2 cents per dollar."""
    probability = _probability(model_probability)
    return probability * float(decimal_odds) - 1.0


def kelly_fraction(model_probability: float, decimal_odds: float) -> float:
    """Full Kelly fraction, floored at zero.

    A caller should apply a conservative fractional multiplier and portfolio
    limits; this value is never a standalone staking recommendation.
    """
    probability = _probability(model_probability)
    decimal_odds = float(decimal_odds)
    if not math.isfinite(decimal_odds) or decimal_odds <= 1.0:
        raise OddsValidationError("Decimal odds must be finite and greater than 1")
    fraction = (probability * decimal_odds - 1.0) / (decimal_odds - 1.0)
    return max(0.0, fraction)


def _probability(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise OddsValidationError("Probability must be finite and between zero and one")
    return value
