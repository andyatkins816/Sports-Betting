import trafilatura
from datetime import datetime, timedelta
import logging
import json

logger = logging.getLogger(__name__)

class NBAScheduleValidator:
    def __init__(self):
        self.schedule_url = "https://www.nba.com/schedule"
    
    def get_current_schedule(self, days=7):
        """Fetch the current NBA schedule from NBA.com"""
        try:
            downloaded = trafilatura.fetch_url(self.schedule_url)
            text = trafilatura.extract(downloaded)
            logger.info(f"Retrieved NBA schedule text: {text[:200]}...")  # Log first 200 chars
            return text
        except Exception as e:
            logger.error(f"Error fetching NBA schedule: {e}")
            return None
