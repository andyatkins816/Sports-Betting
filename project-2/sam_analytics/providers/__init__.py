"""Licensed provider adapters.

Adapters normalize a provider response but do not hide its source identity or
turn unavailable data into sample data.
"""

from .the_odds_api import OddsApiFetch, TheOddsApiClient, TheOddsApiError

__all__ = ["OddsApiFetch", "TheOddsApiClient", "TheOddsApiError"]
