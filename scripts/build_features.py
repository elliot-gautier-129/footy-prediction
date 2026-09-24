"""Build the model-ready feature table from the raw football-data and Understat CSVs.

Runs the SQL pipeline from notebooks/datacleaner.ipynb end to end:
xG load -> team name mapping -> per-season feature extraction -> xG join -> rolling xG.

Usage:
    python scripts/build_features.py
"""
from footy_prediction.data_loading import load_season_xg
from footy_prediction.feature_engineering import (
    build_rolling_xg_features,
    build_team_name_mapping,
    join_xg_features,
    upload_features,
)
from footy_prediction.paths import FEATURES_DIR

LEAGUE_CODES = ["E0", "SP1", "D1", "I1", "F1"]
SEASONS = [19, 27]


def main():
    load_season_xg(league_codes=LEAGUE_CODES, seasons=SEASONS)
    build_team_name_mapping()
    upload_features(league_codes=LEAGUE_CODES, seasons=SEASONS)
    join_xg_features()
    features_xg_rolling = build_rolling_xg_features()

    output_file = FEATURES_DIR / "features_data_xg_rolling.csv"
    features_xg_rolling.to_csv(output_file, index=False)
    print(f"Saved '{output_file}' ({len(features_xg_rolling)} rows)")


if __name__ == "__main__":
    main()
