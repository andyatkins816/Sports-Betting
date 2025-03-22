import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class NBADataProcessor:
    def __init__(self):
        self.games_data = []
        
    def process_game_data(self, raw_data):
        """Process raw NBA game data into features"""
        try:
            df = pd.DataFrame(raw_data)
            
            # Calculate rolling averages
            df['points_scored'] = df.groupby('team_id')['points'].rolling(
                window=10, min_periods=1
            ).mean()
            
            df['points_allowed'] = df.groupby('team_id')['opponent_points'].rolling(
                window=10, min_periods=1
            ).mean()
            
            # Calculate win percentage
            df['wins'] = (df['points'] > df['opponent_points']).astype(int)
            df['win_pct'] = df.groupby('team_id')['wins'].rolling(
                window=20, min_periods=1
            ).mean()
            
            # Detect back-to-back games
            df['days_rest'] = df.groupby('team_id')['date'].diff().dt.days
            df['back_to_back'] = df['days_rest'] <= 1
            
            return df
        except Exception as e:
            logger.error(f"Error processing game data: {e}")
            raise

    def prepare_game_features(self, game):
        """Prepare features for a single game prediction"""
        try:
            return {
                'home_team_win_pct': game['home_team_win_pct'],
                'away_team_win_pct': game['away_team_win_pct'],
                'home_team_points_avg': game['home_team_points_scored'],
                'away_team_points_avg': game['away_team_points_scored'],
                'home_team_points_allowed_avg': game['home_team_points_allowed'],
                'away_team_points_allowed_avg': game['away_team_points_allowed'],
                'home_team_back_to_back': game['home_team_back_to_back'],
                'away_team_back_to_back': game['away_team_back_to_back']
            }
        except Exception as e:
            logger.error(f"Error preparing game features: {e}")
            raise
