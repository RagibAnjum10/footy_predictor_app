import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
import joblib

from prepare_dataset import load_filtered_matches
from feature_engineering import add_elo_and_form_features, add_team_strength_proxy, add_head_to_head_features, add_betting_odds_features

def add_head_to_head_features(df, window=5):
    df = df.copy()
    df["HomeH2HWinRate"] = 0.5
    df["AwayH2HWinRate"] = 0.5

    history = []

    for idx, row in df.iterrows():
        home = row["HomeTeam"]
        away = row["AwayTeam"]

        prev_matches = [r for r in history if (r["Home"] == home and r["Away"] == away) or
                        (r["Home"] == away and r["Away"] == home)]

        recent_matches = prev_matches[-window:]
        if recent_matches:
            home_wins = 0
            away_wins = 0
            for m in recent_matches:
                if m["Home"] == home and m["Result"] == "H":
                    home_wins += 1
                elif m["Home"] == away and m["Result"] == "A":
                    home_wins += 1
                elif m["Home"] == home and m["Result"] == "A":
                    away_wins += 1
                elif m["Home"] == away and m["Result"] == "H":
                    away_wins += 1
            total = home_wins + away_wins
            if total > 0:
                df.at[idx, "HomeH2HWinRate"] = home_wins / total
                df.at[idx, "AwayH2HWinRate"] = away_wins / total

        history.append({
            "Home": home,
            "Away": away,
            "Result": row["FTR"]
        })

    return df

def add_team_strength_proxy(df):
    avg_points = df.groupby("HomeTeam")["FTR"].apply(lambda results: (
        results == "H").sum() * 3 + (results == "D").sum()).reset_index(name="Points")
    avg_points["Strength"] = avg_points["Points"] / avg_points["Points"].max()
    strength_dict = dict(zip(avg_points["HomeTeam"], avg_points["Strength"]))

    df["HomeStrength"] = df["HomeTeam"].map(strength_dict)
    df["AwayStrength"] = df["AwayTeam"].map(strength_dict)
    return df

def add_betting_odds_features(df):
    bookies = ["B365", "PS", "WH", "BW"]
    for prefix in bookies:
        try:
            df[f"{prefix}_ProbH"] = 1 / df[f"{prefix}H"]
            df[f"{prefix}_ProbD"] = 1 / df[f"{prefix}D"]
            df[f"{prefix}_ProbA"] = 1 / df[f"{prefix}A"]
            total = df[[f"{prefix}_ProbH", f"{prefix}_ProbD", f"{prefix}_ProbA"]].sum(axis=1)
            df[f"{prefix}_ProbH"] /= total
            df[f"{prefix}_ProbD"] /= total
            df[f"{prefix}_ProbA"] /= total
        except:
            print(f"Skipping {prefix} odds due to missing columns.")
    return df

def main():
    df = load_filtered_matches("data/")  # Assuming this function loads your data
    print(f"Loaded {len(df)} matches from EPL and La Liga.")

    # Apply feature engineering
    df = add_elo_and_form_features(df)
    df = add_team_strength_proxy(df)
    df = add_head_to_head_features(df)
    df = add_betting_odds_features(df)
    df["MatchDay"] = range(len(df))

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

    # Add odds features (if available)
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

    print(f"Remaining samples after dropna: {len(df)}")
    if len(df) < 50:
        print("⚠️ Not enough data to train a model. Please check your input files.")
        return

    X = df[features]
    y = df["FTR"]

    le = LabelEncoder()
    y = le.fit_transform(y)

    # Train-test split
    split_idx = int(0.8 * len(df))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print("Training set size:", X_train.shape)
    if X_train.shape[0] == 0:
        print("❌ Training set is empty. Cannot proceed.")
        return

    print("Class distribution (before SMOTE):", Counter(y_train))
    smote = SMOTE(random_state=42)
    try:
        X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    except ValueError as e:
        print("❌ SMOTE failed:", e)
        return

    print("Class distribution (after SMOTE):", Counter(y_train_bal))

    # Train the model
    clf = LGBMClassifier(
        class_weight='balanced',
        n_estimators=300,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=31,
        random_state=42
    )
    clf.fit(X_train_bal, y_train_bal)

    # Model evaluation
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test F1 Score (macro): {f1:.4f}")

    # Save the trained model
    joblib.dump(clf, 'football_predictor_model.pkl')
    print("Model saved as 'football_predictor_model.pkl'")

if __name__ == "__main__":
    main()
