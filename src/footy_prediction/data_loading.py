"""Loaders for raw football-data.co.uk results, Understat xG files and saved feature tables."""
import os

import pandas as pd

from footy_prediction.paths import FEATURES_DIR, FOOTBALL_DATA_DIR, UNDERSTAT_DIR
from footy_prediction.sql_utils import con


def load_season(league_code, season) -> pd.DataFrame:
    """Load a league-season CSV and register it as the SQL table matches."""
    season = str(season)
    if len(season) != 4 or not season.isdigit():
        raise ValueError("season must be a four-digit value such as 2526")

    season_file = FOOTBALL_DATA_DIR / f"{league_code}" / f"{season}.csv"

    if not season_file.exists():
        raise FileNotFoundError(f"Could not find {season_file}")

    season_df = pd.read_csv(season_file, encoding="latin-1")
    # Some football-data files start with a UTF-8 byte-order mark, which latin-1
    # decodes into the first header ("ï»¿Div"). Strip it so every season has "Div".
    season_df.columns = season_df.columns.str.removeprefix("\u00ef\u00bb\u00bf")

    # Normalize the league name so SQL does not depend on BOM-prefixed Div headers.
    season_df["league"] = league_code
    con.register("matches", season_df)
    return season_df


def load_season_xg(
    league_codes: list[str],
    seasons: list[int],
) -> pd.DataFrame:
    """Combine xG files for selected leagues and seasons."""
    xg_df = pd.DataFrame()
    league_prefixes = {
        "E0": "eng",
        "SP1": "esp",
        "I1": "ita",
        "D1": "ger",
        "F1": "fra"
    }

    input_dir = UNDERSTAT_DIR

    for league_code in league_codes:
        if league_code not in league_prefixes:
            raise ValueError(
                f"No Understat file prefix configured for {league_code}. "
                f"Available leagues: {list(league_prefixes)}"
            )

        prefix = league_prefixes[league_code]
        for season_start in range(seasons[0], seasons[1]):
            season_code = (
                f"{season_start % 100:02d}"
                f"{(season_start + 1) % 100:02d}"
            )
            print(f"Processing xG {league_code} season {season_code}...")

            season_file = input_dir / f"{prefix}" / f"{prefix}_{season_code}.csv"
            if not season_file.exists():
                raise FileNotFoundError(f"Could not find {season_file}")

            season_df = pd.read_csv(season_file, encoding="latin-1")
            season_df["league"] = league_code
            xg_df = pd.concat([xg_df, season_df], ignore_index=True)

    if xg_df.empty:
        raise ValueError("No xG rows were loaded")

    output_dir = input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    league_name = "_".join(league_codes)
    season_name = f"{seasons[0]}{seasons[1]}"
    output_file = output_dir / f"xg_{league_name}_{season_name}.csv"
    xg_df.to_csv(output_file, index=False)

    # Register the combined data for join_xg.sql.
    con.register("xg_df", xg_df)

    size_bytes = os.path.getsize(output_file)
    file_row_count = len(xg_df)
    for unit in ["bytes", "KB", "MB", "GB"]:
        if size_bytes < 1024.0 or unit == "GB":
            print(
                f"Saved '{output_file}' successfully.\n"
                f"Size: {size_bytes:.2f} {unit}\n"
                f"Row Count: {file_row_count}"
            )
            break
        size_bytes /= 1024.0

    return xg_df


# load the features from data/features

def load_features(file_name: str) -> pd.DataFrame:

    features_file = FEATURES_DIR / f"{file_name}.csv"

    features_df = pd.read_csv(features_file)

     # Turns catergorical columns into categorical veriables
    categorical_columns = ["HomeTeam", "AwayTeam", "Referee", "Div"]
    for column in categorical_columns:
        categories = sorted(features_df[column].dropna().unique())
        features_df[column] = pd.Categorical(features_df[column], categories=categories)

    return features_df
