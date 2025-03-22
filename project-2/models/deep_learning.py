import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class NBADeepLearningPredictor:
    def __init__(self):
        self.lstm_model = None
        self.nn_model = None
        self.is_trained = False
        self._initialize_models()

    def _create_lstm_model(self, sequence_length=10, n_features=8):
        """Create LSTM model for sequence-based prediction"""
        model = Sequential([
            LSTM(64, activation='relu', input_shape=(sequence_length, n_features), return_sequences=True),
            Dropout(0.2),
            LSTM(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(2, activation='softmax')  # Binary classification (home/away win)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def _create_nn_model(self, n_features=8):
        """Create Neural Network for point spread prediction"""
        model = Sequential([
            Dense(64, activation='relu', input_shape=(n_features,)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')  # Point spread prediction
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model

    def _initialize_models(self):
        """Initialize both LSTM and NN models"""
        try:
            self.lstm_model = self._create_lstm_model()
            self.nn_model = self._create_nn_model()
            self._initialize_with_sample_data()
            logger.info("Deep learning models initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing deep learning models: {e}")
            raise

    def _initialize_with_sample_data(self):
        """Initialize models with sample data"""
        try:
            # Generate sample training data
            np.random.seed(42)
            n_samples = 1000
            sequence_length = 10

            # Generate features for both models
            X_seq = np.random.rand(n_samples, sequence_length, 8)  # Sequential data for LSTM
            X_single = np.random.rand(n_samples, 8)  # Single-game data for NN
            
            # Generate target variables
            y_classification = np.random.randint(0, 2, (n_samples, 2))  # One-hot encoded win/loss
            y_regression = np.random.normal(0, 10, (n_samples, 1))  # Point spreads

            # Train models
            self.lstm_model.fit(X_seq, y_classification, epochs=5, batch_size=32, verbose=0)
            self.nn_model.fit(X_single, y_regression, epochs=5, batch_size=32, verbose=0)
            
            self.is_trained = True
            logger.info("Models initialized with sample data")
        except Exception as e:
            logger.error(f"Error training with sample data: {e}")
            raise

    def predict_with_sequence(self, sequence_data):
        """Make prediction using LSTM model with historical sequence"""
        if not self.is_trained:
            raise ValueError("Models need to be trained first")
        
        try:
            # Ensure proper shape for LSTM input
            if len(sequence_data.shape) == 2:
                sequence_data = np.expand_dims(sequence_data, axis=0)
            
            win_probabilities = self.lstm_model.predict(sequence_data)[0]
            return {
                'home_win_probability': float(win_probabilities[0]),
                'away_win_probability': float(win_probabilities[1])
            }
        except Exception as e:
            logger.error(f"Error making LSTM prediction: {e}")
            raise

    def predict_point_spread(self, game_features):
        """Predict point spread using Neural Network"""
        if not self.is_trained:
            raise ValueError("Models need to be trained first")
        
        try:
            # Reshape features if needed
            features = np.array(game_features).reshape(1, -1)
            predicted_spread = self.nn_model.predict(features)[0][0]
            
            return float(predicted_spread)
        except Exception as e:
            logger.error(f"Error predicting point spread: {e}")
            raise

    def train_models(self, sequence_data, single_game_data, sequence_labels, spread_labels):
        """Train both LSTM and NN models with new data"""
        try:
            lstm_history = self.lstm_model.fit(
                sequence_data, sequence_labels,
                epochs=10, batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            nn_history = self.nn_model.fit(
                single_game_data, spread_labels,
                epochs=10, batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            self.is_trained = True
            
            return {
                'lstm_accuracy': float(lstm_history.history['accuracy'][-1]),
                'lstm_loss': float(lstm_history.history['loss'][-1]),
                'nn_mae': float(nn_history.history['mae'][-1]),
                'nn_loss': float(nn_history.history['loss'][-1])
            }
        except Exception as e:
            logger.error(f"Error training models: {e}")
            raise
