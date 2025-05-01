from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from feature_engineering import add_elo_and_form_features, add_team_strength_proxy, add_head_to_head_features, add_betting_odds_features
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests (for React frontend)

# Load the pre-trained model (you only have 'football_predictor_model.pkl')
model = joblib.load('football_predictor_model.pkl')

# Load team data for dropdowns
EPL_TEAMS = [
    "Arsenal", "Chelsea", "Manchester United", "Liverpool", "Tottenham", "Manchester City",
    "Leicester City", "West Ham", "Aston Villa", "Everton", "Bournemouth", "Brentford",
    "Crystal Palace", "Fulham", "Leeds United", "Southampton", "Wolves", "Nottingham Forest"
]
LA_LIGA_TEAMS = [
    "Real Madrid", "Barcelona", "Atletico Madrid", "Sevilla", "Real Sociedad", "Villarreal",
    "Valencia", "Athletic Bilbao", "Getafe", "Granada", "Celta Vigo", "Levante", "Osasuna",
    "Espanyol", "Mallorca", "Alaves", "Elche"
]

# Endpoint to get teams based on league
@app.route('/teams', methods=['GET'])
def get_teams():
    league = request.args.get('league', default='EPL', type=str)  # Get league from query params
    if league == "La Liga":
        return jsonify({"teams": LA_LIGA_TEAMS})
    else:
        return jsonify({"teams": EPL_TEAMS})

# Endpoint to predict match outcome and scoreline
@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get JSON data sent from the frontend
    league = data['league']
    home_team = data['home_team']
    away_team = data['away_team']

    # Prepare the input data (you should map this properly as per your model)
    features = {
        'HomeTeam': home_team,
        'AwayTeam': away_team,
        # Add additional features here (e.g., team form, Elo ratings, etc.)
    }

    # Convert the data into a DataFrame for processing
    df = pd.DataFrame([features])

    # Add dummy values for missing columns if needed
    df['FTHG'] = 0  # Add default values for Home Team Goals if missing
    df['FTAG'] = 0  # Add default values for Away Team Goals if missing

    # Feature engineering (using the same functions you used to train your model)
    df = add_elo_and_form_features(df)
    df = add_team_strength_proxy(df)
    df = add_head_to_head_features(df)
    df = add_betting_odds_features(df)

    # Make prediction for match outcome
    X = df.drop(columns=['HomeTeam', 'AwayTeam'])  # Adjust as necessary
    outcome_prediction = model.predict(X)  # Predict outcome (0 = Home Win, 1 = Draw, 2 = Away Win)
    prob = model.predict_proba(X)  # Get probabilities for Home Win, Draw, Away Win

    # Map prediction to readable outcome
    outcomes = ['Home Win', 'Draw', 'Away Win']
    result = outcomes[outcome_prediction[0]]

    # Simulate scoreline based on the predicted outcome (replace with more sophisticated logic if needed)
    if result == "Home Win":
        home_goals = np.random.randint(2, 4)  # Random goals between 2 and 3
        away_goals = np.random.randint(0, 2)  # Random goals between 0 and 1
    elif result == "Away Win":
        home_goals = np.random.randint(0, 2)
        away_goals = np.random.randint(2, 4)
    else:  # Draw
        home_goals = np.random.randint(1, 3)
        away_goals = home_goals  # Same goals for both teams

    scoreline = f"{home_goals} - {away_goals}"

    return jsonify({
        'predicted_outcome': result,
        'scoreline': scoreline,
        'probabilities': {
            'Home Win': prob[0][0],
            'Draw': prob[0][1],
            'Away Win': prob[0][2]
        }
    })
echo "# footy_predictor_app" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/RagibAnjum10/footy_predictor_app.git
git push -u origin main
# Basic health check route
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True)
