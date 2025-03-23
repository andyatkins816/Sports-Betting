import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
from services.news_analyzer import NewsAnalyzer

logger = logging.getLogger(__name__)

class FeatureEngineeringService:
    def __init__(self):
        self.news_analyzer = NewsAnalyzer()
        self.weather_impact_threshold = 0.1
        logger.info("Feature Engineering Service initialized")

    def enhance_game_features(self, game_data: Dict, historical_data: List[Dict]) -> Dict:
        """Enhance game features with additional analytics"""
        try:
            enhanced_features = game_data.copy()
            
            # Add team momentum features
            home_momentum = self._calculate_team_momentum(
                historical_data, 
                enhanced_features['home_team']['name']
            )
            away_momentum = self._calculate_team_momentum(
                historical_data, 
                enhanced_features['away_team']['name']
            )
            
            # Add head-to-head features
            h2h_stats = self._analyze_head_to_head(
                historical_data,
                enhanced_features['home_team']['name'],
                enhanced_features['away_team']['name']
            )
            
            # Get injury impact
            home_injury_impact = self._calculate_injury_impact(
                enhanced_features['home_team']['name']
            )
            away_injury_impact = self._calculate_injury_impact(
                enhanced_features['away_team']['name']
            )
            
            # Get weather impact for outdoor sports
            weather_impact = self._get_weather_impact(game_data)
            
            # Add news sentiment analysis
            sentiment_data = self._analyze_news_sentiment(
                enhanced_features['home_team']['name'],
                enhanced_features['away_team']['name']
            )
            
            # Combine all enhanced features
            enhanced_features.update({
                'momentum_factors': {
                    'home_team_momentum': home_momentum,
                    'away_team_momentum': away_momentum,
                    'momentum_advantage': home_momentum - away_momentum
                },
                'head_to_head_stats': h2h_stats,
                'injury_impact': {
                    'home_team': home_injury_impact,
                    'away_team': away_injury_impact,
                    'relative_impact': home_injury_impact - away_injury_impact
                },
                'weather_impact': weather_impact,
                'sentiment_analysis': sentiment_data
            })
            
            return enhanced_features
            
        except Exception as e:
            logger.error(f"Error enhancing game features: {str(e)}", exc_info=True)
            return game_data

    def _calculate_team_momentum(self, historical_data: List[Dict], team_name: str) -> float:
        """Calculate team momentum based on recent performance"""
        try:
            # Get last 10 games
            recent_games = [
                game for game in historical_data 
                if team_name in [game['home_team']['name'], game['away_team']['name']]
            ][-10:]
            
            if not recent_games:
                return 0.0
                
            # Calculate weighted momentum (more recent games have higher weight)
            momentum = 0.0
            weights = np.linspace(0.5, 1.0, len(recent_games))
            
            for game, weight in zip(recent_games, weights):
                is_home = game['home_team']['name'] == team_name
                team_score = game['home_score'] if is_home else game['away_score']
                opp_score = game['away_score'] if is_home else game['home_score']
                
                # Win/loss momentum
                win_factor = 1 if team_score > opp_score else -1
                # Margin of victory/defeat impact
                margin_factor = (team_score - opp_score) / max(team_score, opp_score)
                
                momentum += weight * (win_factor * (1 + margin_factor))
            
            return momentum / len(recent_games)
            
        except Exception as e:
            logger.error(f"Error calculating team momentum: {str(e)}", exc_info=True)
            return 0.0

    def _analyze_head_to_head(
        self, 
        historical_data: List[Dict], 
        home_team: str, 
        away_team: str
    ) -> Dict:
        """Analyze head-to-head matchup history"""
        try:
            # Get historical matchups
            h2h_games = [
                game for game in historical_data
                if (game['home_team']['name'] == home_team and 
                    game['away_team']['name'] == away_team) or
                   (game['home_team']['name'] == away_team and 
                    game['away_team']['name'] == home_team)
            ]
            
            if not h2h_games:
                return {
                    'home_team_wins': 0,
                    'away_team_wins': 0,
                    'average_point_diff': 0,
                    'home_team_advantage': 0
                }
            
            home_wins = sum(1 for game in h2h_games 
                          if game['home_team']['name'] == home_team and 
                          game['home_score'] > game['away_score'])
            away_wins = sum(1 for game in h2h_games 
                          if game['away_team']['name'] == away_team and 
                          game['away_score'] > game['home_score'])
            
            point_diffs = []
            for game in h2h_games:
                if game['home_team']['name'] == home_team:
                    point_diffs.append(game['home_score'] - game['away_score'])
                else:
                    point_diffs.append(game['away_score'] - game['home_score'])
            
            return {
                'home_team_wins': home_wins,
                'away_team_wins': away_wins,
                'average_point_diff': np.mean(point_diffs) if point_diffs else 0,
                'home_team_advantage': home_wins / len(h2h_games) if h2h_games else 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing head-to-head: {str(e)}", exc_info=True)
            return {
                'home_team_wins': 0,
                'away_team_wins': 0,
                'average_point_diff': 0,
                'home_team_advantage': 0
            }

    def _calculate_injury_impact(self, team_name: str) -> float:
        """Calculate impact of team injuries on performance"""
        try:
            # TODO: Implement injury data retrieval from ESPN API
            # For now, return random impact for demonstration
            return np.random.uniform(-0.1, 0.1)
            
        except Exception as e:
            logger.error(f"Error calculating injury impact: {str(e)}", exc_info=True)
            return 0.0

    def _get_weather_impact(self, game_data: Dict) -> Optional[Dict]:
        """Get weather impact for outdoor sports"""
        try:
            # TODO: Implement weather API integration
            # For now, return None for indoor sports
            return None
            
        except Exception as e:
            logger.error(f"Error getting weather impact: {str(e)}", exc_info=True)
            return None

    def _analyze_news_sentiment(self, home_team: str, away_team: str) -> Dict:
        """Analyze recent news sentiment for both teams"""
        try:
            home_sentiment = self.news_analyzer.analyze_team_news(home_team)
            away_sentiment = self.news_analyzer.analyze_team_news(away_team)
            
            return {
                'home_team_sentiment': home_sentiment,
                'away_team_sentiment': away_sentiment,
                'sentiment_advantage': home_sentiment - away_sentiment
            }
            
        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {str(e)}", exc_info=True)
            return {
                'home_team_sentiment': 0,
                'away_team_sentiment': 0,
                'sentiment_advantage': 0
            }
