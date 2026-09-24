import os
import subprocess
import sys

import duckdb
import numpy as np
import pandas as pd
import pytest

from footy_prediction import paths
from footy_prediction.evaluation import (
    evaluate_final_holdout,
    evaluate_holdout_model,
    evaluate_market_baseline,
)
from footy_prediction.feature_sets import DEFAULT_FEATURES, FEATURE_GROUPINGS
from footy_prediction.modeling import (
    K_fold_training,
    _brier_score,
    _season_start_year,
    run_grouping_cross_validation,
    walk_forward_training,
)
from footy_prediction.sql_utils import load_sql, run_sql_file

SQL_FILES = ["feature_extraction.sql", "join_xg.sql", "names_mapping.sql", "rolling_xg.sql"]


def test_public_functions_are_importable():
    from footy_prediction.data_loading import load_features, load_season, load_season_xg
    from footy_prediction.feature_engineering import (
        build_rolling_xg_features,
        build_team_name_mapping,
        join_xg_features,
        upload_features,
    )

    for obj in [
        load_features, load_season, load_season_xg, upload_features,
        build_team_name_mapping, join_xg_features, build_rolling_xg_features,
        walk_forward_training, K_fold_training, run_grouping_cross_validation,
        evaluate_final_holdout, evaluate_holdout_model, evaluate_market_baseline,
    ]:
        assert callable(obj)
    assert "market_only" in FEATURE_GROUPINGS
    assert "HomeTeam" in DEFAULT_FEATURES


@pytest.mark.parametrize("filename", SQL_FILES)
def test_sql_files_load(filename):
    assert "CREATE OR REPLACE" in load_sql(filename)


def test_sql_paths_do_not_depend_on_working_directory(tmp_path):
    code = (
        "from footy_prediction.sql_utils import load_sql;"
        "from footy_prediction.paths import FOOTBALL_DATA_DIR;"
        "assert load_sql('rolling_xg.sql');"
        "assert FOOTBALL_DATA_DIR.is_dir()"
    )
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True)
    assert os.getcwd() != str(tmp_path)
    assert paths.SQL_DIR.is_dir()


def test_season_start_year():
    assert _season_start_year("2020/21") == 2020
    assert _season_start_year("2021") == 2021
    # football-data season codes such as 1920 match the four-digit pattern first;
    # the codes 1920..2627 still sort in chronological order.
    assert _season_start_year(1920) == 1920
    assert sorted(["2122", "1920", "2627"], key=_season_start_year) == ["1920", "2122", "2627"]


def test_brier_score_perfect_and_uniform():
    targets = np.array([0, 1, 2])
    assert _brier_score(np.eye(3), targets) == 0.0
    assert _brier_score(np.full((3, 3), 1 / 3), targets) == pytest.approx(2 / 3)


def _synthetic_matches(n_rounds=12):
    """Two teams playing each other alternately at home; home team always wins."""
    rows = []
    for i in range(n_rounds):
        home, away = ("Arsenal", "Chelsea") if i % 2 == 0 else ("Chelsea", "Arsenal")
        rows.append({
            "Div": "E0", "Date": f"{i + 1:02d}/08/2025", "Time": "15:00",
            "HomeTeam": home, "AwayTeam": away,
            "FTHG": 2, "FTAG": 0, "FTR": "H",
            "HS": 10, "AS": 5, "HST": 5, "AST": 2, "HF": 10, "AF": 12,
            "HC": 6, "AC": 3, "HY": 1, "AY": 2, "HR": 0, "AR": 0,
            "AvgH": 2.0, "AvgD": 3.5, "AvgA": 4.0,
            "league": "E0",
        })
    return pd.DataFrame(rows)


def test_feature_extraction_sql_uses_only_previous_matches():
    connection = duckdb.connect()
    connection.execute(
        "CREATE TABLE team_name_mapping (team_name VARCHAR, team_code VARCHAR, count BIGINT)"
    )
    connection.register("matches", _synthetic_matches())
    run_sql_file("feature_extraction.sql", connection)
    features = connection.sql("SELECT * FROM prematch_sql ORDER BY match_id").df()

    first = features.iloc[0]
    # No history exists before the first match, so its rolling features must be missing.
    assert pd.isna(first["home_points_last_5"])
    assert pd.isna(first["away_points_last_5"])
    # Arsenal's previous home matches (all wins) give 3 points; its own result is excluded.
    third = features.iloc[2]
    assert third["HomeTeam"] == "Arsenal"
    assert third["home_points_home_last_5"] == 3
    # Arsenal alternated W (home) / L (away) before match 3 -> mean 1.5 over 2 matches.
    assert third["home_points_last_5"] == pytest.approx(1.5)
    probs = features[["market_home_prob_fair", "market_draw_prob_fair", "market_away_prob_fair"]]
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_market_baseline_on_synthetic_data():
    data = pd.DataFrame({
        "match_id": range(15),
        "FTR": ["H", "D", "A", "H", "H", "A", "D", "H", "A", "H", "H", "D", "H", "D", "A"],
        "market_home_prob_fair": 0.5,
        "market_draw_prob_fair": 0.25,
        "market_away_prob_fair": 0.25,
    })
    result = evaluate_market_baseline(data, start_fraction=0.8)
    assert result["match_count"] == 3
    assert result["market_brier_score"] == pytest.approx(
        np.mean([0.25 + 0.0625 + 0.0625, 0.25 + 0.5625 + 0.0625, 0.25 + 0.0625 + 0.5625]), abs=1e-4
    )
    assert result["start_percentile"] == "80%"


def test_load_season_strips_utf8_bom_from_div_header():
    from footy_prediction.data_loading import load_season

    # E0 2526 is one of the raw files that starts with a UTF-8 byte-order mark.
    season_df = load_season("E0", 2526)
    assert season_df.columns[0] == "Div"
    assert season_df["Div"].notna().all()


def _synthetic_seasons(n_seasons=5, matches_per_season=400, seed=0):
    rng = np.random.default_rng(seed)
    n = n_seasons * matches_per_season
    home_prob = rng.uniform(0.2, 0.7, n)
    draw_prob = np.full(n, 0.25)
    away_prob = 1 - home_prob - draw_prob
    outcomes = [rng.choice(["H", "D", "A"], p=[h, d, a]) for h, d, a in zip(home_prob, draw_prob, away_prob)]
    return pd.DataFrame({
        "match_id": range(n),
        "season": [f"20{19 + i // matches_per_season}/{20 + i // matches_per_season}" for i in range(n)],
        "FTR": outcomes,
        "HomeTeam": pd.Categorical(rng.choice(["A", "B", "C", "D"], n)),
        "market_home_prob_fair": home_prob,
        "market_draw_prob_fair": draw_prob,
        "market_away_prob_fair": away_prob,
    })


def test_walk_forward_training_runs_with_categorical_features():
    data = _synthetic_seasons()
    features = ["HomeTeam", "market_home_prob_fair", "market_draw_prob_fair", "market_away_prob_fair"]
    result = walk_forward_training(data, features, n_iter=1, max_selected_features=2)
    # 5 seasons: last is the holdout, first outer fold trains on 2 seasons -> 2 outer folds.
    assert result.loc[0, "cv_fold_count"] == 2
    assert result.loc[0, "final_holdout_season"] == "2023/24"
    assert set(result.loc[0, "last_fold_features"]) <= set(features)
    records = result.attrs["feature_selection_records"]
    assert "2023/24" not in set(records["validation_season"])


def test_run_grouping_cross_validation_returns_only_lightgbm_params():
    data = _synthetic_seasons()
    groupings = {"market": ["market_home_prob_fair", "market_away_prob_fair"]}
    summary = run_grouping_cross_validation(data, groupings)
    params = summary.loc[0, "best_params"]
    assert set(params) == {
        "num_leaves", "max_depth", "learning_rate", "min_child_samples",
        "subsample", "colsample_bytree", "reg_lambda",
    }
    result = evaluate_holdout_model(data, summary.loc[0, "selected_features"], hyperparams=params)
    assert 0 < result["holdout_log_loss"]


def test_market_baseline_handles_slice_missing_a_class():
    data = pd.DataFrame({
        "match_id": range(10),
        "FTR": ["H", "D", "A", "H", "H", "A", "D", "H", "H", "A"],
        "market_home_prob_fair": 0.5,
        "market_draw_prob_fair": 0.25,
        "market_away_prob_fair": 0.25,
    })
    # The last two matches contain no draw.
    result = evaluate_market_baseline(data, start_fraction=0.8)
    assert result["match_count"] == 2


def test_results_round_trip(tmp_path):
    from footy_prediction.results import load_results, save_results

    data = _synthetic_seasons()
    data["kickoff"] = pd.date_range("2019-08-01", periods=len(data), freq="D")
    data["Div"] = "E0"
    data["AwayTeam"] = "B"
    summary = run_grouping_cross_validation(
        data, {"market": ["market_home_prob_fair", "market_away_prob_fair"]}
    )
    holdout = evaluate_holdout_model(data, summary.loc[0, "selected_features"], hyperparams=summary.loc[0, "best_params"])
    save_results(summary, holdout, data, output_dir=tmp_path)

    loaded = load_results(tmp_path)
    predictions = loaded["predictions"]
    assert len(predictions) == holdout["test_matches"]
    probs = predictions[["model_prob_H", "model_prob_D", "model_prob_A"]].to_numpy()
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert loaded["run_info"]["best_grouping"] == "market"
    assert len(loaded["grouping_cv"].loc[0, "fold_log_losses"]) == 2
    assert load_results(tmp_path / "missing") is None


def test_saved_features_have_chronological_match_ids():
    from footy_prediction.data_loading import load_features

    features = load_features("features_data_xg_rolling")
    kickoff = pd.to_datetime(features.sort_values("match_id")["kickoff"])
    # The holdout split takes the last rows by match_id, so match_id must follow kickoff.
    assert kickoff.is_monotonic_increasing
    assert features["match_id"].is_unique
