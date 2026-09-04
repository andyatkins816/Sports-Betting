"""SAM Analytics: auditable sports-market research primitives.

This package deliberately contains no automated bet placement.  A decision is
only valid when it can be reproduced from a timestamped model version and a
timestamped market quote.
"""

from .odds import american_to_decimal, implied_probability
from .risk import BankrollPolicy, ExposureState

__all__ = [
    "BankrollPolicy",
    "ExposureState",
    "american_to_decimal",
    "implied_probability",
]
