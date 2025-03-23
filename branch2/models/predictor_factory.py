from typing import Dict, Optional
from models.predictor import NBAPredictor
from models.mlb_predictor import MLBPredictor
from models.nfl_predictor import NFLPredictor
import logging

logger = logging.getLogger(__name__)

class PredictorFactory:
    """Factory class to manage sport-specific predictors"""

    def __init__(self):
        self._predictors: Dict[str, object] = {}
        self._supported_sports = {
            'NBA': NBAPredictor,
            'MLB': MLBPredictor,
            'NFL': NFLPredictor
            # NHL predictor will be added here
        }

    def get_predictor(self, sport: str) -> Optional[object]:
        """Get or create a predictor for the specified sport"""
        sport = sport.upper()
        if sport not in self._supported_sports:
            logger.error(f"Unsupported sport: {sport}")
            return None

        if sport not in self._predictors:
            try:
                self._predictors[sport] = self._supported_sports[sport]()
                logger.info(f"Created new predictor for {sport}")
            except Exception as e:
                logger.error(f"Error creating predictor for {sport}: {e}")
                return None

        return self._predictors[sport]

    def get_supported_sports(self) -> list:
        """Get list of supported sports"""
        return list(self._supported_sports.keys())