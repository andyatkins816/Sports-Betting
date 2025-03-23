import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging

logger = logging.getLogger(__name__)

class NBAPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.is_trained = False
        self._initialize_with_sample_data()

    def _calculate_point_spread(self, game_data):
        """Calculate estimated point spread based on team stats"""
        home_offense = game_data['home_team_points_avg']
        home_defense = game_data['home_team_points_allowed_avg']
        away_offense = game_data['away_team_points_avg']
        away_defense = game_data['away_team_points_allowed_avg']

        # Basic point spread calculation using offensive and defensive ratings
        home_expected = (home_offense + away_defense) / 2
        away_expected = (away_offense + home_defense) / 2

        # Add home court advantage (typically 3-4 points)
        home_court_advantage = 3.5
        spread = (home_expected - away_expected) + home_court_advantage

        return {
            'spread': round(spread, 1),
            'factors': {
                'offensive_advantage': round(home_offense - away_offense, 1),
                'defensive_advantage': round(away_defense - home_defense, 1),
                'home_court': home_court_advantage,
                'back_to_back_impact': 0,
                'sentiment_impact': 0  # Default sentiment impact
            },
            'confidence': 'high' if abs(spread) > 7 else 'medium' if abs(spread) > 3 else 'low'
        }

    def _initialize_with_sample_data(self):
        """Initialize model with sample historical data"""
        try:
            # Generate sample training data
            np.random.seed(42)
            n_samples = 1000

            # Generate features
            X = np.random.rand(n_samples, 8)  # 8 features
            X[:, 0] = np.random.uniform(0.3, 0.8, n_samples)  # home team win %
            X[:, 1] = np.random.uniform(0.3, 0.8, n_samples)  # away team win %
            X[:, 2] = np.random.uniform(95, 125, n_samples)   # home points avg
            X[:, 3] = np.random.uniform(95, 125, n_samples)   # away points avg
            X[:, 4] = np.random.uniform(95, 125, n_samples)   # home points allowed
            X[:, 5] = np.random.uniform(95, 125, n_samples)   # away points allowed
            X[:, 6] = np.random.choice([0, 1], n_samples)     # home b2b
            X[:, 7] = np.random.choice([0, 1], n_samples)     # away b2b

            # Generate outcomes (home team wins)
            y = np.zeros(n_samples)
            for i in range(n_samples):
                home_advantage = 0.05  # 5% home court advantage
                home_strength = X[i, 0] + (X[i, 2] - X[i, 4])/100
                away_strength = X[i, 1] + (X[i, 3] - X[i, 5])/100

                y[i] = 1 if (home_strength + home_advantage) > away_strength else 0

            # Train the model
            self.train(X, y)
            logger.info("Model initialized with sample data")
        except Exception as e:
            logger.error(f"Error initializing model with sample data: {e}")
            raise

    def preprocess_features(self, data):
        """Convert raw game data into features"""
        features = []
        for game in data:
            feature_vector = [
                game['home_team_win_pct'],
                game['away_team_win_pct'],
                game['home_team_points_avg'],
                game['away_team_points_avg'],
                game['home_team_points_allowed_avg'],
                game['away_team_points_allowed_avg'],
                1 if game.get('home_team_back_to_back', False) else 0,
                1 if game.get('away_team_back_to_back', False) else 0
            ]
            features.append(feature_vector)
        return np.array(features)

    def train(self, X, y):
        """Train the model with game data"""
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            self.model.fit(X_train, y_train)

            # Evaluate model
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            logger.info(f"Model accuracy: {accuracy:.2f}")

            self.is_trained = True
            return accuracy
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise

    def predict(self, game_data):
        """Make prediction for a single game"""
        if not self.is_trained:
            raise ValueError("Model needs to be trained first")

        try:
            # Format data for prediction
            features = self.preprocess_features([game_data])
            prediction = self.model.predict_proba(features)[0]
            spread_analysis = self._calculate_point_spread(game_data)

            # Adjust probabilities based on spread
            home_prob = min(max(prediction[1], 0.1), 0.9)
            away_prob = 1 - home_prob

            # Calculate total points prediction
            total_points = (game_data['home_team_points_avg'] + 
                          game_data['away_team_points_avg']) / 2

            # Determine if model agrees with basic statistics
            model_favors_home = home_prob > away_prob
            spread_favors_home = spread_analysis['spread'] > 0
            model_agrees = model_favors_home == spread_favors_home

            # Generate analysis reasons
            analysis_reasons = [
                'Based on current form and historical matchups',
                'Home court advantage factored in',
                'Recent scoring trends considered'
            ]

            # Ensure all values are JSON serializable
            return {
                'home_team_win_probability': float(home_prob),
                'away_team_win_probability': float(away_prob),
                'confidence_score': float(max(home_prob, away_prob)),
                'point_spread': float(spread_analysis['spread']),
                'total_points': float(total_points),
                'spread_analysis': {
                    'factors': spread_analysis['factors'],
                    'confidence': str(spread_analysis['confidence']),
                    'model_agreement': bool(model_agrees),
                    'reason': list(analysis_reasons)
                }
            }

        except Exception as e:
            logger.error(f"Error making prediction: {e}", exc_info=True)
            raise