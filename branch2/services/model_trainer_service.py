import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from utils.constants import MODEL_PARAMS, FEATURE_NAMES

logger = logging.getLogger(__name__)

class ModelTrainerService:
    def __init__(self):
        self.rf_model = RandomForestClassifier(
            n_estimators=MODEL_PARAMS['N_ESTIMATORS'],
            max_depth=MODEL_PARAMS['MAX_DEPTH'],
            random_state=MODEL_PARAMS['RANDOM_STATE']
        )
        self.gb_model = GradientBoostingClassifier(
            n_estimators=MODEL_PARAMS['N_ESTIMATORS'],
            max_depth=MODEL_PARAMS['MAX_DEPTH'],
            random_state=MODEL_PARAMS['RANDOM_STATE']
        )
        self.nn_model = self._build_neural_network()
        self.feature_names = FEATURE_NAMES
        logger.info("Model Trainer Service initialized")

    def _build_neural_network(self) -> Sequential:
        """Build and compile neural network model"""
        model = Sequential([
            Dense(64, activation='relu', input_shape=(len(FEATURE_NAMES),)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
        return model

    def train_models(self, training_data: List[Dict]) -> Dict:
        """Train all models with the latest data"""
        try:
            # Prepare data
            X = np.array([[game[feature] for feature in self.feature_names] 
                         for game in training_data])
            y = np.array([game['home_team_won'] for game in training_data])

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=MODEL_PARAMS['RANDOM_STATE']
            )

            # Train models
            self.rf_model.fit(X_train, y_train)
            self.gb_model.fit(X_train, y_train)
            self.nn_model.fit(X_train, y_train,
                            epochs=50,
                            batch_size=32,
                            verbose=0,
                            validation_split=0.2)

            # Evaluate models
            rf_score = self.rf_model.score(X_test, y_test)
            gb_score = self.gb_model.score(X_test, y_test)
            nn_score = self.nn_model.evaluate(X_test, y_test, verbose=0)[1]

            logger.info(f"Models trained successfully. Scores - RF: {rf_score:.3f}, GB: {gb_score:.3f}, NN: {nn_score:.3f}")

            return {
                'random_forest_accuracy': rf_score,
                'gradient_boosting_accuracy': gb_score,
                'neural_network_accuracy': nn_score,
                'training_size': len(X_train),
                'test_size': len(X_test),
                'training_date': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error training models: {str(e)}", exc_info=True)
            return None

    def predict_game(self, game_features: Dict) -> Dict:
        """Make predictions using all models"""
        try:
            # Prepare features
            X = np.array([[game_features[feature] for feature in self.feature_names]])

            # Get predictions from all models
            rf_pred = self.rf_model.predict_proba(X)[0]
            gb_pred = self.gb_model.predict_proba(X)[0]
            nn_pred = self.nn_model.predict(X)[0]

            # Ensemble predictions with weights
            weights = [0.4, 0.4, 0.2]  # RF, GB, NN weights
            ensemble_pred = (
                weights[0] * rf_pred +
                weights[1] * gb_pred +
                weights[2] * nn_pred
            )

            return {
                'ensemble_probability': float(ensemble_pred[1]),
                'model_predictions': {
                    'random_forest': float(rf_pred[1]),
                    'gradient_boosting': float(gb_pred[1]),
                    'neural_network': float(nn_pred[0])
                },
                'prediction_confidence': self._calculate_confidence(rf_pred[1], gb_pred[1], nn_pred[0])
            }

        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}", exc_info=True)
            return None

    def _calculate_confidence(self, *predictions) -> float:
        """Calculate prediction confidence based on model agreement"""
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        # Higher confidence when models agree (low std dev)
        confidence = 1.0 - min(std_pred * 2, 0.5)  
        
        # Adjust confidence based on how close to 0.5 the prediction is
        certainty_factor = abs(mean_pred - 0.5) * 2
        final_confidence = confidence * certainty_factor
        
        return float(final_confidence)
