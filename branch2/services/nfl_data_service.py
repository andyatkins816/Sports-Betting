import os
import requests
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class NFLDataService:
    def __init__(self):
        self.api_key = os.environ.get('NFL_API_KEY')
        self.base_url = 'https://api.sportsdata.io/v3/nfl/scores/json'

    def _make_request(self, endpoint, params=None):
        """Make API request with error handling"""
        try:
            headers = {'Ocp-Apim-Subscription-Key': self.api_key}
            response = requests.get(f"{self.base_url}/{endpoint}", headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error making NFL API request: {e}")
            return None

    def get_upcoming_games(self):
        """Get upcoming NFL games for the next week"""
        try:
            # For development/testing, return sample data if no API key
            if not self.api_key:
                return self._get_sample_upcoming_games()

            # Get current week's schedule
            today = datetime.now()
            games = self._make_request('Scores/2024')  # Current season

            if not games:
                return self._get_sample_upcoming_games()

            # Filter for upcoming games
            upcoming = []
            for game in games:
                game_date = datetime.strptime(game['Day'], '%Y-%m-%d')
                if game_date >= today and game_date <= today + timedelta(days=7):
                    upcoming.append(self._format_game(game))

            return upcoming

        except Exception as e:
            logger.error(f"Error fetching upcoming NFL games: {e}")
            return self._get_sample_upcoming_games()

    def _format_game(self, game):
        """Format NFL game data for prediction system"""
        return {
            'id': game.get('GameID', 0),
            'date': game.get('Day'),
            'time': game.get('DateTime', '').split('T')[1].split('.')[0],
            'home_team': {
                'name': game.get('HomeTeam', 'Home Team'),
                'win_pct': self._calculate_win_pct(game.get('HomeTeamWins', 0), game.get('HomeTeamLosses', 0)),
                'points_avg': float(game.get('HomeTeamScore', 0)),
                'points_against_avg': float(game.get('AwayTeamScore', 0))
            },
            'away_team': {
                'name': game.get('AwayTeam', 'Away Team'),
                'win_pct': self._calculate_win_pct(game.get('AwayTeamWins', 0), game.get('AwayTeamLosses', 0)),
                'points_avg': float(game.get('AwayTeamScore', 0)),
                'points_against_avg': float(game.get('HomeTeamScore', 0))
            }
        }

    def _calculate_win_pct(self, wins, losses):
        """Calculate win percentage"""
        total_games = wins + losses
        return wins / total_games if total_games > 0 else 0.0

    def _get_sample_upcoming_games(self):
        """Return sample NFL game data for development"""
        today = datetime.now()
        sample_games = []
        
        teams = [
            ('Kansas City Chiefs', 'Las Vegas Raiders'),
            ('San Francisco 49ers', 'Los Angeles Rams'),
            ('Buffalo Bills', 'New York Jets'),
            ('Dallas Cowboys', 'Philadelphia Eagles')
        ]
        
        for i, (home, away) in enumerate(teams):
            game_date = today + timedelta(days=i)
            sample_games.append({
                'id': i + 1,
                'date': game_date.strftime('%Y-%m-%d'),
                'time': '13:00',
                'home_team': {
                    'name': home,
                    'win_pct': 0.65,
                    'points_avg': 27.5,
                    'points_against_avg': 20.3
                },
                'away_team': {
                    'name': away,
                    'win_pct': 0.55,
                    'points_avg': 24.8,
                    'points_against_avg': 22.1
                }
            })
            
        return sample_games
