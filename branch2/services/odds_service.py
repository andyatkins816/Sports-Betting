import logging
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class OddsService:
    def __init__(self):
        logger.info("Odds Service initialized")

    def format_odds(self, odds_data: Dict) -> Optional[Dict]:
        """Format odds data from ESPN API response"""
        try:
            if not odds_data:
                return None

            spread = odds_data.get('details', '')
            over_under = odds_data.get('overUnder', 0.0)
            
            # Parse money line from provider
            money_line_home = None
            money_line_away = None
            
            try:
                money_line = odds_data.get('moneyLine', {})
                money_line_home = int(money_line.get('homeTeamOdds', {}).get('moneyLine', 0))
                money_line_away = int(money_line.get('awayTeamOdds', {}).get('moneyLine', 0))
            except (ValueError, TypeError, AttributeError) as e:
                logger.warning(f"Error parsing money line odds: {e}")

            return {
                'spread': spread,
                'over_under': over_under,
                'money_line': {
                    'home': money_line_home,
                    'away': money_line_away
                },
                'provider': odds_data.get('provider', {}).get('name', 'Unknown'),
                'updated': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error formatting odds data: {str(e)}", exc_info=True)
            return None

    def calculate_implied_probability(self, money_line: int) -> float:
        """Calculate implied probability from money line odds"""
        try:
            if money_line > 0:
                return 100 / (money_line + 100)
            else:
                return (-money_line) / (-money_line + 100)
        except (ValueError, ZeroDivisionError):
            return 0.0
