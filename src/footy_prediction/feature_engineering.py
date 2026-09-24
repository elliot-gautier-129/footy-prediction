"""Pre-match feature engineering, driven by the SQL scripts in sql/.

The SQL steps share the DuckDB connection in ``footy_prediction.sql_utils`` and
must run in this order:

1. ``load_season_xg``            registers ``xg_df``
2. ``build_team_name_mapping``   names_mapping.sql   -> ``team_name_mapping``
3. ``upload_features``           feature_extraction.sql per league-season -> ``features_data``
4. ``join_xg_features``          join_xg.sql         -> ``features_xg_df``
5. ``build_rolling_xg_features`` rolling_xg.sql      -> ``features_data_xg``
"""
import os

import pandas as pd

from footy_prediction.data_loading import load_season
from footy_prediction.paths import FEATURES_DIR, FOOTBALL_DATA_DIR
from footy_prediction.sql_utils import con, load_sql, run_sql_file


def build_team_name_mapping():
    """Run names_mapping.sql to create team_name_mapping (requires xg_df to be registered)."""
    return run_sql_file("names_mapping.sql")


def upload_features(league_codes: list[str], seasons: list[int]):
    """Build one master feature table for multiple leagues and seasons."""
    sql_text = load_sql("feature_extraction.sql")
    features_df = pd.DataFrame()

    for league_code in league_codes:
        for season_start in range(seasons[0], seasons[1]):
            season_code = (
                f"{season_start % 100:02d}"
                f"{(season_start + 1) % 100:02d}"
            )
            print(f"Processing {league_code} season {season_code}...")

            season_file = (
                FOOTBALL_DATA_DIR
                / league_code
                / f"{season_code}.csv"
            )
            if not season_file.exists():
                raise FileNotFoundError(f"Could not find {season_file}")

            load_season(league_code, season_code)
            con.execute(sql_text)

            # Select the final engineered table, not the intermediate table.
            season_features = con.sql("""
                SELECT *
                FROM prematch_sql
                ORDER BY match_id
            """).df()
            # Season label (e.g. "2019/20") for season-based walk-forward CV.
            season_features["season"] = (
                f"20{season_start % 100:02d}/{(season_start + 1) % 100:02d}"
            )
            features_df = pd.concat(
                [features_df, season_features],
                ignore_index=True,
            )

    if features_df.empty:
        raise ValueError("No feature rows were generated")

    # match_id must be chronological across leagues: the holdout split and walk-forward
    # ordering sort by it. (Within a league-season, the SQL match_id breaks kickoff ties.)
    features_df = features_df.sort_values(
        ["kickoff", "league", "match_id"], kind="stable"
    ).reset_index(drop=True)
    features_df["match_id"] = range(1, len(features_df) + 1)

    output_dir = FEATURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / (
        f"{'_'.join(league_codes)}_{seasons[0]}{seasons[1]}.csv"
    )
    features_df.to_csv(output_file, index=False)
    con.register("features_data", features_df)

    size_bytes = os.path.getsize(output_file)
    for unit in ["bytes", "KB", "MB", "GB"]:
        if size_bytes < 1024.0 or unit == "GB":
            print(
                f"Saved '{output_file}' successfully.\n"
                f"Size: {size_bytes:.2f} {unit}\n"
                f"Row Count: {len(features_df)}"
            )
            break
        size_bytes /= 1024.0

    return features_df


def join_xg_features() -> pd.DataFrame:
    """Run join_xg.sql (features_data INNER JOIN xg_df) and return features_xg_df."""
    run_sql_file("join_xg.sql")
    return con.sql("""
        SELECT *
        FROM features_xg_df
    """).df()


def build_rolling_xg_features() -> pd.DataFrame:
    """Run rolling_xg.sql on features_xg_df and return the model-ready features_data_xg table."""
    run_sql_file("rolling_xg.sql")
    return con.sql("""
        SELECT *
        FROM features_data_xg
        ORDER BY match_id
    """).df()
