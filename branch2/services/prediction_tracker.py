from datetime import datetime, timedelta
import logging
from typing import Dict, List
from services.nba_data_service import NBADataService
import os
import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np

logger = logging.getLogger(__name__)

class PredictionTracker:
    def __init__(self):
        self.last_update = None
        self.nba_service = NBADataService()
        self.db_url = os.environ.get("DATABASE_URL")
        self.accuracy_window = 30  # Days to consider for accuracy trends

    def _get_db_connection(self):
        return psycopg2.connect(self.db_url)

    def record_prediction(self, game_id: int, prediction: Dict):
        """Record a new prediction"""
        try:
            with self._get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO predictions 
                        (game_id, timestamp, home_team, away_team, predicted_winner, 
                         predicted_spread, win_probability)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (game_id) DO UPDATE SET
                        timestamp = EXCLUDED.timestamp,
                        predicted_winner = EXCLUDED.predicted_winner,
                        predicted_spread = EXCLUDED.predicted_spread,
                        win_probability = EXCLUDED.win_probability
                    """, (
                        game_id,
                        datetime.now(),
                        prediction['home_team'],
                        prediction['away_team'],
                        prediction['predicted_winner'],
                        prediction['point_spread'],
                        prediction['win_probability']
                    ))
                    conn.commit()
                    logger.info(f"Successfully recorded prediction for game {game_id}")
        except Exception as e:
            logger.error(f"Error recording prediction: {e}", exc_info=True)

    def record_result(self, game_id: int, result: Dict):
        """Record actual game result and update model weights"""
        try:
            with self._get_db_connection() as conn:
                with conn.cursor() as cur:
                    # Check if result already exists
                    cur.execute("SELECT 1 FROM game_results WHERE game_id = %s", (game_id,))
                    if cur.fetchone() is None:
                        cur.execute("""
                            INSERT INTO game_results 
                            (game_id, timestamp, home_team_score, away_team_score, 
                             actual_winner, final_spread)
                            VALUES (%s, %s, %s, %s, %s, %s)
                        """, (
                            game_id,
                            datetime.now(),
                            result['home_score'],
                            result['away_score'],
                            result['winner'],
                            result['final_spread']
                        ))
                        conn.commit()
                        logger.info(f"Successfully recorded result for game {game_id}")

                        # Update model weights based on prediction accuracy
                        self._update_model_weights(game_id, result)
                    else:
                        logger.info(f"Result for game {game_id} already exists")
        except Exception as e:
            logger.error(f"Error recording result: {e}", exc_info=True)

    def _update_model_weights(self, game_id: int, actual_result: Dict):
        """Update model weights based on prediction accuracy"""
        try:
            with self._get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Get prediction for this game
                    cur.execute("""
                        SELECT * FROM predictions 
                        WHERE game_id = %s
                    """, (game_id,))
                    prediction = cur.fetchone()

                    if not prediction:
                        return

                    # Calculate prediction error
                    predicted_spread = float(prediction['predicted_spread'])
                    actual_spread = float(actual_result['final_spread'])
                    spread_error = abs(predicted_spread - actual_spread)

                    # Update accuracy metrics in database
                    cur.execute("""
                        INSERT INTO model_metrics 
                        (game_id, spread_error, winner_correct, timestamp)
                        VALUES (%s, %s, %s, %s)
                    """, (
                        game_id,
                        spread_error,
                        prediction['predicted_winner'] == actual_result['winner'],
                        datetime.now()
                    ))
                    conn.commit()

                    logger.info(f"Updated model metrics for game {game_id}")
        except Exception as e:
            logger.error(f"Error updating model weights: {e}", exc_info=True)

    def get_model_accuracy_trends(self) -> Dict:
        """Get model accuracy trends over time"""
        try:
            with self._get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Get accuracy metrics for last 30 days
                    start_date = datetime.now() - timedelta(days=self.accuracy_window)
                    cur.execute("""
                        SELECT 
                            DATE(timestamp) as date,
                            AVG(CASE WHEN winner_correct THEN 1 ELSE 0 END) as win_accuracy,
                            AVG(spread_error) as avg_spread_error,
                            COUNT(*) as total_predictions
                        FROM model_metrics
                        WHERE timestamp >= %s
                        GROUP BY DATE(timestamp)
                        HAVING COUNT(*) > 0
                        ORDER BY date DESC
                    """, (start_date,))

                    trends = cur.fetchall()
                    logger.info(f"Found {len(trends)} days with prediction metrics")

                    if not trends:
                        return {
                            'dates': [],
                            'win_accuracy': [],
                            'spread_accuracy': []
                        }

                    return {
                        'dates': [trend['date'].strftime('%Y-%m-%d') for trend in trends],
                        'win_accuracy': [float(trend['win_accuracy']) for trend in trends],
                        'spread_accuracy': [float(trend['avg_spread_error']) for trend in trends]
                    }

        except Exception as e:
            logger.error(f"Error getting model accuracy trends: {e}", exc_info=True)
            return {
                'dates': [],
                'win_accuracy': [],
                'spread_accuracy': []
            }

    def get_yesterday_accuracy(self) -> Dict:
        """Get prediction accuracy for yesterday's games"""
        try:
            yesterday = datetime.now() - timedelta(days=1)
            yesterday_start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
            yesterday_end = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)

            logger.info(f"Fetching predictions between {yesterday_start} and {yesterday_end}")

            with self._get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Get yesterday's predictions and results with metrics
                    cur.execute("""
                        SELECT 
                            p.game_id,
                            p.home_team,
                            p.away_team,
                            p.predicted_winner,
                            p.predicted_spread,
                            r.home_team_score,
                            r.away_team_score,
                            r.actual_winner,
                            r.final_spread,
                            m.spread_error,
                            m.winner_correct
                        FROM predictions p
                        JOIN game_results r ON p.game_id = r.game_id
                        JOIN model_metrics m ON p.game_id = m.game_id
                        WHERE DATE(p.timestamp) = DATE(CURRENT_DATE - INTERVAL '1 day')
                    """)

                    games = cur.fetchall()
                    logger.info(f"Found {len(games)} games with predictions from yesterday")

                    games_analyzed = []
                    correct_predictions = 0
                    spread_differences = []

                    for game in games:
                        if game['actual_winner']:  # Only analyze games with results
                            spread_diff = abs(float(game['final_spread']) - float(game['predicted_spread']))
                            prediction_correct = game['winner_correct']

                            if prediction_correct:
                                correct_predictions += 1

                            spread_differences.append(spread_diff)
                            games_analyzed.append({
                                'home_team': game['home_team'],
                                'away_team': game['away_team'],
                                'predicted_winner': game['predicted_winner'],
                                'predicted_spread': float(game['predicted_spread']),
                                'actual_score': f"{game['home_team_score']}-{game['away_team_score']}",
                                'prediction_correct': prediction_correct,
                                'spread_error': game['spread_error']
                            })

                    total_games = len(games_analyzed)
                    accuracy = correct_predictions / total_games if total_games > 0 else 0

                    # Get accuracy trends
                    trends = self.get_model_accuracy_trends()

                    response = {
                        'yesterday_games': games_analyzed,
                        'accuracy_stats': {
                            'correct': correct_predictions,
                            'incorrect': total_games - correct_predictions,
                            'accuracy': accuracy
                        },
                        'spread_accuracy': {
                            'ranges': ['0-5', '6-10', '11-15', '16+'],
                            'counts': [
                                len([s for s in spread_differences if 0 <= s <= 5]),
                                len([s for s in spread_differences if 6 <= s <= 10]),
                                len([s for s in spread_differences if 11 <= s <= 15]),
                                len([s for s in spread_differences if s > 15])
                            ]
                        },
                        'trends': trends,
                        'last_update': datetime.now().isoformat()
                    }

                    logger.info(f"Generated accuracy report: {response}")
                    return response

        except Exception as e:
            logger.error(f"Error generating accuracy report: {e}", exc_info=True)
            return {
                'yesterday_games': [],
                'accuracy_stats': {'correct': 0, 'incorrect': 0, 'accuracy': 0},
                'spread_accuracy': {'ranges': [], 'counts': []},
                'trends': {'dates': [], 'win_accuracy': [], 'spread_accuracy': []},
                'last_update': None
            }

    def _result_exists(self, game_id: int) -> bool:
        """Check if a result already exists for a game"""
        try:
            with self._get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1 FROM game_results WHERE game_id = %s", (game_id,))
                    return cur.fetchone() is not None
        except Exception as e:
            logger.error(f"Error checking result existence: {e}")
            return False