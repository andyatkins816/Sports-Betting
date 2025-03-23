# Model parameters
MODEL_PARAMS = {
    'N_ESTIMATORS': 100,
    'MAX_DEPTH': 10,
    'RANDOM_STATE': 42
}

# Feature names
FEATURE_NAMES = [
    'home_team_win_pct',
    'away_team_win_pct',
    'home_team_points_avg',
    'away_team_points_avg',
    'home_team_points_allowed_avg',
    'away_team_points_allowed_avg',
    'home_team_momentum',
    'away_team_momentum',
    'home_team_injury_impact',
    'away_team_injury_impact',
    'head_to_head_advantage',
    'weather_impact',
    'sentiment_advantage',
    'home_team_back_to_back',
    'away_team_back_to_back'
]

# API response keys
PREDICTION_KEYS = [
    'home_team_win_probability',
    'away_team_win_probability',
    'confidence_score',
    'model_agreement',
    'feature_importance'
]

# Model training parameters
TRAINING_PARAMS = {
    'TEST_SIZE': 0.2,
    'VALIDATION_SIZE': 0.2,
    'BATCH_SIZE': 32,
    'EPOCHS': 50,
    'LEARNING_RATE': 0.001
}

# Feature importance thresholds
FEATURE_IMPORTANCE_THRESHOLDS = {
    'HIGH': 0.15,
    'MEDIUM': 0.08,
    'LOW': 0.03
}

# Prediction confidence levels
CONFIDENCE_LEVELS = {
    'HIGH': 0.8,
    'MEDIUM': 0.6,
    'LOW': 0.4
}