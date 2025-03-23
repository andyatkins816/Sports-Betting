import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from services.news_analyzer import NewsAnalyzer
from models.base_predictor import BasePredictor
import logging

logger = logging.getLogger(__name__)

class MLBPredictor(BasePredictor):
    def __init__(self):
        super().__init__('MLB')
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.news_analyzer = NewsAnalyzer()
        self._initialize_with_sample_data()
    
    def _calculate_run_line(self, game_data):
        """Calculate estimated run line based on team stats"""
        home_offense = game_data['home_team_runs_avg']
        home_defense = game_data['home_team_runs_allowed_avg']
        away_offense = game_data['away_team_runs_avg']
        away_defense = game_data['away_team_runs_allowed_avg']

        # Basic run line calculation
        home_expected = (home_offense + away_defense) / 2
        away_expected = (away_offense + home_defense) / 2

        # Add home field advantage (typically 0.1-0.2 runs in MLB)
        home_advantage = 0.15
        run_line = (home_expected - away_expected) + home_advantage

        # Adjust for factors like pitcher matchups
        if game_data.get('home_pitcher_era', 0) > 0:
            run_line -= (game_data['home_pitcher_era'] - 4.0) * 0.2
        if game_data.get('away_pitcher_era', 0) > 0:
            run_line += (game_data['away_pitcher_era'] - 4.0) * 0.2

        return {
            'line': round(run_line, 1),
            'factors': {
                'offensive_advantage': round(home_offense - away_offense, 2),
                'defensive_advantage': round(away_defense - home_defense, 2),
                'home_advantage': home_advantage,
                'pitching_impact': 0  # Will be calculated if pitcher data available
            }
        }

    def _initialize_with_sample_data(self):
        """Initialize model with sample MLB historical data"""
        try:
            np.random.seed(42)
            n_samples = 1000

            # Generate features
            X = np.random.rand(n_samples, 8)
            X[:, 0] = np.random.uniform(0.3, 0.7, n_samples)  # win %
            X[:, 1] = np.random.uniform(0.3, 0.7, n_samples)  # win %
            X[:, 2] = np.random.uniform(3.5, 6.0, n_samples)  # runs/game
            X[:, 3] = np.random.uniform(3.5, 6.0, n_samples)  # runs/game
            X[:, 4] = np.random.uniform(3.5, 6.0, n_samples)  # runs allowed
            X[:, 5] = np.random.uniform(3.5, 6.0, n_samples)  # runs allowed
            X[:, 6] = np.random.uniform(2.5, 5.0, n_samples)  # pitcher ERA
            X[:, 7] = np.random.uniform(2.5, 5.0, n_samples)  # pitcher ERA

            # Generate outcomes (home team wins)
            y = np.zeros(n_samples)
            for i in range(n_samples):
                home_strength = X[i, 0] + (6.0 - X[i, 6])/10
                away_strength = X[i, 1] + (6.0 - X[i, 7])/10
                y[i] = 1 if (home_strength + 0.05) > away_strength else 0

            self.train(X, y)
            logger.info("MLB model initialized with sample data")

        except Exception as e:
            logger.error(f"Error initializing MLB model: {e}")
            raise

    def predict(self, game_data):
        """Make prediction for a single MLB game"""
        if not self.is_trained:
            raise ValueError("Model needs to be trained first")

        try:
            # Get sentiment factors
            home_sentiment = self.news_analyzer.get_team_sentiment_factor(
                game_data.get('home_team_name', 'Unknown')
            )
            away_sentiment = self.news_analyzer.get_team_sentiment_factor(
                game_data.get('away_team_name', 'Unknown')
            )

            features = self.preprocess_features([game_data])
            prediction = self.model.predict_proba(features)[0]
            run_line = self._calculate_run_line(game_data)

            # Adjust probabilities based on sentiment
            home_prob = min(max(prediction[1] + home_sentiment - away_sentiment, 0.1), 0.9)
            away_prob = 1 - home_prob

            return {
                'home_team_win_probability': float(home_prob),
                'away_team_win_probability': float(away_prob),
                'run_line': float(run_line['line']),
                'analysis': {
                    'factors': {
                        **run_line['factors'],
                        'sentiment_impact': home_sentiment - away_sentiment
                    },
                    'confidence': 'high' if abs(run_line['line']) > 2 else 'medium' if abs(run_line['line']) > 1 else 'low'
                }
            }

        except Exception as e:
            logger.error(f"Error making MLB prediction: {e}")
            raise

    def preprocess_features(self, data):
        """Convert raw MLB game data into features"""
        features = []
        for game in data:
            feature_vector = [
                game['home_team_win_pct'],
                game['away_team_win_pct'],
                game['home_team_runs_avg'],
                game['away_team_runs_avg'],
                game['home_team_runs_allowed_avg'],
                game['away_team_runs_allowed_avg'],
                game.get('home_pitcher_era', 4.50),  # Default MLB average ERA
                game.get('away_pitcher_era', 4.50)
            ]
            features.append(feature_vector)
        return np.array(features)

    def train(self, X, y):
        """Train the MLB prediction model"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            self._record_training()
            logger.info(f"MLB model accuracy: {accuracy:.2f}")
            
            return accuracy

        except Exception as e:
            logger.error(f"Error training MLB model: {e}")
            raise
