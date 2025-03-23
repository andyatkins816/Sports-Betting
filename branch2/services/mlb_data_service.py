import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List
from services.base_sports_service import BaseSportsService

logger = logging.getLogger(__name__)

class MLBDataService(BaseSportsService):
    def __init__(self):
        super().__init__('MLB')
        self.teams = {
            'NYY': {'name': 'New York Yankees', 'wins': 92, 'losses': 70},
            'BOS': {'name': 'Boston Red Sox', 'wins': 89, 'losses': 73},
            'LAD': {'name': 'Los Angeles Dodgers', 'wins': 100, 'losses': 62},
            'SFG': {'name': 'San Francisco Giants', 'wins': 85, 'losses': 77},
            'HOU': {'name': 'Houston Astros', 'wins': 95, 'losses': 67},
            'TEX': {'name': 'Texas Rangers', 'wins': 90, 'losses': 72},
            'CHC': {'name': 'Chicago Cubs', 'wins': 83, 'losses': 79},
            'STL': {'name': 'St. Louis Cardinals', 'wins': 87, 'losses': 75},
            'ATL': {'name': 'Atlanta Braves', 'wins': 104, 'losses': 58},
            'PHI': {'name': 'Philadelphia Phillies', 'wins': 88, 'losses': 74}
        }

    def get_upcoming_games(self, days: int = 7) -> List[Dict]:
        """Get upcoming MLB games using sample data"""
        try:
            logger.info(f"Fetching upcoming MLB games for next {days} days")
            current_date = datetime.now()

            # Generate realistic game schedule
            matchups = [
                ('NYY', 'BOS'), ('LAD', 'SFG'), ('HOU', 'TEX'),
                ('CHC', 'STL'), ('ATL', 'PHI'), ('BOS', 'LAD'),
                ('SFG', 'HOU'), ('TEX', 'NYY'), ('STL', 'ATL'),
                ('PHI', 'CHC'), ('NYY', 'LAD'), ('BOS', 'SFG')
            ]

            upcoming_games = []
            for i, (home_code, away_code) in enumerate(matchups):
                game_date = current_date + timedelta(days=i//3)
                game_time = '19:05' if i % 2 == 0 else '13:05'

                home_team = self.teams[home_code]
                away_team = self.teams[away_code]

                game = {
                    'id': i + 1,
                    'date': self._format_date(game_date),
                    'time': game_time,
                    'home_team': {
                        'name': home_team['name'],
                        'win_pct': self._calculate_win_pct(
                            home_team['wins'],
                            home_team['losses']
                        ),
                        'runs_avg': 4.8,
                        'runs_against_avg': 3.9
                    },
                    'away_team': {
                        'name': away_team['name'],
                        'win_pct': self._calculate_win_pct(
                            away_team['wins'],
                            away_team['losses']
                        ),
                        'runs_avg': 4.5,
                        'runs_against_avg': 4.1
                    }
                }
                upcoming_games.append(game)

                if len(upcoming_games) >= days * 3:  # Max 3 games per day
                    break

            logger.info(f"Generated {len(upcoming_games)} sample upcoming games")
            return upcoming_games

        except Exception as e:
            logger.error(f"Error generating sample MLB games: {str(e)}", exc_info=True)
            return []

    def _format_date(self, dt: datetime) -> str:
        return dt.strftime('%Y-%m-%d')

    def _calculate_win_pct(self, wins: int, losses: int) -> float:
        """Calculate win percentage"""
        total_games = wins + losses
        return round(wins / total_games, 3) if total_games > 0 else 0.0