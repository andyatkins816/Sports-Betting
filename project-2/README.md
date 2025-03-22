# NBA Prediction System

A cutting-edge multi-sport predictive analytics platform that combines advanced AI technologies with comprehensive betting tracking capabilities.

## Features

- Game outcome predictions with AI-powered analysis
- Real-time betting tracking and management
- Multiple bet types supported:
  - Point spread betting
  - Money line betting
  - Parlay betting
- Interactive dashboard with performance metrics
- AI-powered betting suggestions
- Historical performance tracking

## Tech Stack

- Python Flask backend
- Bootstrap for frontend styling
- Chart.js for data visualization
- SQLAlchemy for database management
- Machine learning integration for predictions

## Setup

1. Clone the repository
```bash
git clone https://github.com/yourusername/nba-prediction-system.git
cd nba-prediction-system
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```
DATABASE_URL=your_database_url
NBA_API_KEY=your_api_key
OPENAI_API_KEY=your_openai_key
```

4. Run the application
```bash
python main.py
```

The application will be available at `http://localhost:5000`

## Usage

1. **View Predictions**: Navigate to the Predictions page to see upcoming games and make predictions
2. **Track Bets**: After making a prediction, use the bet tracking feature to monitor your wagers
3. **Dashboard**: Check the dashboard for prediction accuracy and betting performance metrics

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
