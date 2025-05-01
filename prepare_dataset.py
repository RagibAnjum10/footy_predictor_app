
import pandas as pd
import os
import glob

def load_filtered_matches(data_dir, allowed_divs=('E0', 'SP1')):
    """
    Loads match data from a directory and filters for specified divisions (EPL, La Liga).

    Parameters:
    - data_dir: str, path to folder containing match CSVs.
    - allowed_divs: tuple, division codes to keep (default: English Premier League 'E0' and La Liga 'SP1').

    Returns:
    - DataFrame with filtered matches.
    """
    all_csvs = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
    all_dfs = []
    
    for csv_path in all_csvs:
        try:
            df = pd.read_csv(csv_path, encoding="ISO-8859-1")
            if 'Div' in df.columns:
                df = df[df['Div'].isin(allowed_divs)]
                if not df.empty:
                    all_dfs.append(df)
        except Exception as e:
            print(f"Failed to load {csv_path}: {e}")
    
    if not all_dfs:
        raise ValueError("No valid CSV files found with matching divisions.")

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df['Date'] = pd.to_datetime(full_df['Date'], errors='coerce')
    full_df = full_df.dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'FTR'])
    full_df = full_df.sort_values('Date')
    
    return full_df.reset_index(drop=True)

# For testing manually
if __name__ == "__main__":
    df = load_filtered_matches("data/")
    print(df.head())
    print(f"Loaded {len(df)} matches.")
