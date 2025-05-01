import pandas as pd
from collections import defaultdict

def initialize_elo(teams, base_elo=1500):
    """Initialize Elo ratings for teams."""
    return {team: base_elo for team in teams}

def update_elo(winner_elo, loser_elo, k=20):
    """Update Elo ratings after a match."""
    expected_win = 1 / (1 + 10 ** ((loser_elo - winner_elo) / 400))
    new_winner_elo = winner_elo + k * (1 - expected_win)
    new_loser_elo = loser_elo + k * (0 - (1 - expected_win))
    return new_winner_elo, new_loser_elo

def add_head_to_head_features(df, window=5):
    """Add head-to-head features based on past encounters."""
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
    """Add a team strength feature based on average points."""
    avg_points = df.groupby("HomeTeam")["FTR"].apply(lambda results: (
        results == "H").sum() * 3 + (results == "D").sum()).reset_index(name="Points")
    avg_points["Strength"] = avg_points["Points"] / avg_points["Points"].max()
    strength_dict = dict(zip(avg_points["HomeTeam"], avg_points["Strength"]))

    df["HomeStrength"] = df["HomeTeam"].map(strength_dict)
    df["AwayStrength"] = df["AwayTeam"].map(strength_dict)
    return df

def calculate_team_form(df, team_col, goals_for_col, goals_against_col, form_window=3):
    """Calculate team form over a rolling window."""
    form_pts = defaultdict(list)
    gf_hist = defaultdict(list)
    ga_hist = defaultdict(list)

    form_features = []

    for _, row in df.iterrows():
        team = row[team_col]
        gf = row[goals_for_col]
        ga = row[goals_against_col]
        pts = 3 if gf > ga else 1 if gf == ga else 0

        form_pts[team].append(pts)
        gf_hist[team].append(gf)
        ga_hist[team].append(ga)

        recent_pts = sum(form_pts[team][-form_window:])
        recent_gf = sum(gf_hist[team][-form_window:]) / min(len(gf_hist[team]), form_window)
        recent_ga = sum(ga_hist[team][-form_window:]) / min(len(ga_hist[team]), form_window)

        form_features.append((recent_pts, recent_gf, recent_ga))

    return pd.DataFrame(form_features, columns=[f"{team_col}_FormPts", f"{team_col}_AvgGF", f"{team_col}_AvgGA"])

def add_additional_form_features(df, window=3):
    """Add additional form-related features: goal difference and win streak."""
    df = df.copy()
    home_gd = defaultdict(list)
    away_gd = defaultdict(list)
    home_winstreak = defaultdict(list)
    away_winstreak = defaultdict(list)

    win_diff_home = []
    win_diff_away = []
    streak_home = []
    streak_away = []

    for _, row in df.iterrows():
        h = row["HomeTeam"]
        a = row["AwayTeam"]
        h_gd = row["FTHG"] - row["FTAG"]
        a_gd = row["FTAG"] - row["FTHG"]
        res = row["FTR"]

        h_win = 1 if res == "H" else 0
        a_win = 1 if res == "A" else 0

        home_gd[h].append(h_gd)
        away_gd[a].append(a_gd)
        home_winstreak[h].append(h_win)
        away_winstreak[a].append(a_win)

        gd_h = sum(home_gd[h][-window:])
        gd_a = sum(away_gd[a][-window:])
        streak_h = sum(home_winstreak[h][-window:])
        streak_a = sum(away_winstreak[a][-window:])

        win_diff_home.append(gd_h)
        win_diff_away.append(gd_a)
        streak_home.append(streak_h)
        streak_away.append(streak_a)

    df["Home_GoalDiff3"] = win_diff_home
    df["Away_GoalDiff3"] = win_diff_away
    df["Home_WinStreak3"] = streak_home
    df["Away_WinStreak3"] = streak_away

    return df

def add_betting_odds_features(df):
    """Add betting odds features to the dataframe."""
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


def add_elo_and_form_features(df):
    """Combine Elo ratings and form features into the dataframe."""
    df = df.copy()
    df = df.sort_values("Date")

    elo = initialize_elo(set(df["HomeTeam"]).union(df["AwayTeam"]))
    home_elo = []
    away_elo = []

    for _, row in df.iterrows():
        h = row["HomeTeam"]
        a = row["AwayTeam"]
        res = row["FTR"]

        home_elo.append(elo[h])
        away_elo.append(elo[a])

        if res == "H":
            new_h, new_a = update_elo(elo[h], elo[a])
        elif res == "A":
            new_a, new_h = update_elo(elo[a], elo[h])
        else:
            new_h = elo[h]
            new_a = elo[a]

        elo[h] = new_h
        elo[a] = new_a

    df["HomeElo"] = home_elo
    df["AwayElo"] = away_elo

    home_form = calculate_team_form(df, "HomeTeam", "FTHG", "FTAG")
    away_form = calculate_team_form(df, "AwayTeam", "FTAG", "FTHG")

    df = pd.concat([df, home_form, away_form], axis=1)
    df = add_additional_form_features(df)
    return df
