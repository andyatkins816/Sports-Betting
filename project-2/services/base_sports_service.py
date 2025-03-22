import logging
from datetime import datetime
from typing import Dict, List, Optional
import requests

logger = logging.getLogger(__name__)

class BaseSportsService:
    def __init__(self, sport_name: str):
        self.sport_name = sport_name
        # ESPN API configuration - free public API
        self.base_url = 'http://site.api.espn.com/apis/site/v2'
        self.headers = {
            'Accept': 'application/json'
        }
        logger.info(f"{sport_name} Data Service initialized using ESPN API")
        logger.debug(f"Using ESPN API endpoint: {self.base_url}")

    def check_api_status(self) -> bool:
        """Check if ESPN API is accessible and returning valid game data"""
        try:
            today = datetime.now()
            date_str = today.strftime('%Y%m%d')
            endpoint = f"sports/basketball/nba/scoreboard?dates={date_str}"
            response = self._make_request(endpoint)

            # Check if we have a valid response with events data
            if response and isinstance(response, dict) and 'events' in response:
                events = response.get('events', [])
                if len(events) > 0:
                    logger.info("ESPN API is working and returning game data")
                    return True
                else:
                    logger.info("ESPN API is working but no games found for today")
                    return True  # Still consider API working even if no games today

            logger.warning("ESPN API response invalid or missing events data")
            return False

        except Exception as e:
            logger.error(f"Error checking ESPN API status: {str(e)}")
            return False

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make ESPN API request with error handling"""
        try:
            url = f"{self.base_url}/{endpoint}"
            logger.info(f"Making ESPN API request for {self.sport_name} to: {url}")
            logger.debug(f"Request parameters: {params}")

            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            logger.debug(f"Response status code: {response.status_code}")

            if response.status_code != 200:
                logger.error(f"ESPN API request failed with status {response.status_code}")
                logger.error(f"Error response: {response.text}")
                return None

            data = response.json()
            logger.debug(f"ESPN API response structure: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
            return data

        except Exception as e:
            logger.error(f"Error making ESPN API request: {str(e)}")
            return None

    def _format_date(self, date: datetime) -> str:
        """Format date consistently across services"""
        return date.strftime('%Y-%m-%d')

    def _format_time(self, time: datetime) -> str:
        """Format time consistently across services - Central Time"""
        return time.strftime('%I:%M %p').lstrip('0')  # 12-hour format with AM/PM

    def _calculate_win_pct(self, wins: int, losses: int) -> float:
        """Calculate win percentage"""
        total_games = wins + losses
        return round(wins / total_games, 3) if total_games > 0 else 0.0