from abc import ABC, abstractmethod
from typing import Dict, Any
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class BasePredictor(ABC):
    """Base class for all sport predictors"""
    
    def __init__(self, sport_name: str):
        self.sport_name = sport_name
        self.is_trained = False
        self.last_training = None
    
    @abstractmethod
    def predict(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction for a single game"""
        pass
    
    @abstractmethod
    def train(self, X, y) -> float:
        """Train the model with game data"""
        pass
    
    @abstractmethod
    def preprocess_features(self, data: Dict[str, Any]):
        """Convert raw game data into features"""
        pass
    
    def _validate_game_data(self, game_data: Dict[str, Any]) -> bool:
        """Validate required fields in game data"""
        required_fields = [
            'home_team_win_pct', 'away_team_win_pct',
            'home_team_points_avg', 'away_team_points_avg',
            'home_team_points_allowed_avg', 'away_team_points_allowed_avg'
        ]
        return all(field in game_data for field in required_fields)

    def _record_training(self):
        """Record successful training"""
        self.is_trained = True
        self.last_training = datetime.now()
