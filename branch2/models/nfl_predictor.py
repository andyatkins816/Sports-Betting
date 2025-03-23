import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from services.news_analyzer import NewsAnalyzer
from models.base_predictor import BasePredictor
import logging

logger = logging.getLogger(__name__)

class NFLPredictor(BasePredictor):
    def __init__(self):
        super().__init__('NFL')
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.news_analyzer = NewsAnalyzer()
        self._initialize_with_sample_data()
    
    def _calculate_point_spread(self, game_data):
        """Calculate estimated point spread based on team stats"""
        home_offense = game_data['home_team_points_avg']
        home_defense = game_data['home_team_points_allowed_avg']
        away_offense = game_data['away_team_points_avg']
        away_defense = game_data['away_team_points_allowed_avg']

        # Basic point spread calculation
        home_expected = (home_offense + away_defense) / 2
        away_expected = (away_offense + home_defense) / 2

        # Add home field advantage (typically 2-3 points in NFL)
        home_advantage = 2.5
        spread = (home_expected - away_expected) + home_advantage

        return {
            'spread': round(spread, 1),
            'factors': {
                'offensive_advantage': round(home_offense - away_offense, 1),
                'defensive_advantage': round(away_defense - home_defense, 1),
                'home_advantage': home_advantage
            }
        }

    def _initialize_with_sample_data(self):
        """Initialize model with sample NFL historical data"""
        try:
            np.random.seed(42)
            n_samples = 1000

            # Generate features
            X = np.random.rand(n_samples, 6)  # 6 features for NFL
            X[:, 0] = np.random.uniform(0.3, 0.7, n_samples)  # win %
            X[:, 1] = np.random.uniform(0.3, 0.7, n_samples)  # win %
            X[:, 2] = np.random.uniform(17, 35, n_samples)    # points/game
            X[:, 3] = np.random.uniform(17, 35, n_samples)    # points/game
            X[:, 4] = np.random.uniform(17, 35, n_samples)    # points allowed
            X[:, 5] = np.random.uniform(17, 35, n_samples)    # points allowed

            # Generate outcomes (home team wins)
            y = np.zeros(n_samples)
            for i in range(n_samples):
                home_strength = X[i, 0] + (X[i, 2] - X[i, 4])/50
                away_strength = X[i, 1] + (X[i, 3] - X[i, 5])/50
                y[i] = 1 if (home_strength + 0.05) > away_strength else 0

            self.train(X, y)
            logger.info("NFL model initialized with sample data")

        except Exception as e:
            logger.error(f"Error initializing NFL model: {e}")
            raise

    def predict(self, game_data):
        """Make prediction for a single NFL game"""
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
            spread_analysis = self._calculate_point_spread(game_data)

            # Adjust probabilities based on sentiment
            home_prob = min(max(prediction[1] + home_sentiment - away_sentiment, 0.1), 0.9)
            away_prob = 1 - home_prob

            return {
                'home_team_win_probability': float(home_prob),
                'away_team_win_probability': float(away_prob),
                'point_spread': float(spread_analysis['spread']),
                'analysis': {
                    'factors': {
                        **spread_analysis['factors'],
                        'sentiment_impact': home_sentiment - away_sentiment
                    },
                    'confidence': 'high' if abs(spread_analysis['spread']) > 7 else 'medium' if abs(spread_analysis['spread']) > 3 else 'low'
                }
            }

        except Exception as e:
            logger.error(f"Error making NFL prediction: {e}")
            raise

    def preprocess_features(self, data):
        """Convert raw NFL game data into features"""
        features = []
        for game in data:
            feature_vector = [
                game['home_team_win_pct'],
                game['away_team_win_pct'],
                game['home_team_points_avg'],
                game['away_team_points_avg'],
                game['home_team_points_allowed_avg'],
                game['away_team_points_allowed_avg']
            ]
            features.append(feature_vector)
        return np.array(features)

    def train(self, X, y):
        """Train the NFL prediction model"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            self._record_training()
            logger.info(f"NFL model accuracy: {accuracy:.2f}")
            
            return accuracy

        except Exception as e:
            logger.error(f"Error training NFL model: {e}")
            raise
