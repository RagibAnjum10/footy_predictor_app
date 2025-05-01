
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier

from prepare_dataset import load_filtered_matches
from feature_engineering import add_elo_and_form_features

from train_model import add_head_to_head_features, add_team_strength_proxy, add_betting_odds_features

# Load and process data
df = load_filtered_matches("data/")
df = add_elo_and_form_features(df)
df = add_team_strength_proxy(df)
df = add_head_to_head_features(df)
df = add_betting_odds_features(df)
df["MatchDay"] = range(len(df))

# Feature list
base_features = [
    "HomeElo", "AwayElo",
    "HomeTeam_FormPts", "HomeTeam_AvgGF", "HomeTeam_AvgGA",
    "AwayTeam_FormPts", "AwayTeam_AvgGF", "AwayTeam_AvgGA",
    "HomeStrength", "AwayStrength",
    "HomeH2HWinRate", "AwayH2HWinRate",
    "MatchDay",
    "Home_GoalDiff3", "Away_GoalDiff3",
    "Home_WinStreak3", "Away_WinStreak3"
]

odds_prefixes = ["B365", "PS", "WH", "BW"]
odds_features = []
for prefix in odds_prefixes:
    for outcome in ["H", "D", "A"]:
        col = f"{prefix}_Prob{outcome}"
        if col in df.columns:
            odds_features.append(col)

temp_df = df.dropna(subset=base_features + odds_features + ["FTR"])
if len(temp_df) >= 100:
    print(f"Including odds features — total rows: {len(temp_df)}")
    features = base_features + odds_features
    df = temp_df
else:
    print("Skipping odds features — not enough complete rows.")
    features = base_features
    df = df.dropna(subset=features + ["FTR"])

# Prepare X and y
X = df[features]
y = df["FTR"]
le = LabelEncoder()
y = le.fit_transform(y)

# Model
clf = LGBMClassifier(
    class_weight='balanced',
    n_estimators=300,
    learning_rate=0.03,
    max_depth=6,
    num_leaves=31,
    random_state=42
)

# TimeSeries Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(clf, X, y, cv=tscv, scoring='accuracy')
print("TimeSeries Cross-Validation Accuracy Scores:", scores)
print("Mean Accuracy:", scores.mean())

# Train on full data for feature importances
clf.fit(X, y)
importances = clf.feature_importances_

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(X.columns, importances)
plt.title("Feature Importances (LightGBM)")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("Feature importance plot saved as feature_importance.png")
