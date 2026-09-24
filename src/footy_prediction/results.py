"""Save and load the outputs of scripts/train_model.py so they can be presented without retraining."""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from footy_prediction.paths import RESULTS_DIR

GROUPING_CV_FILE = "grouping_cv.json"
HOLDOUT_PREDICTIONS_FILE = "holdout_predictions.csv"
RUN_INFO_FILE = "run_info.json"

MATCH_COLUMNS = [
    "match_id", "kickoff", "season", "Div", "HomeTeam", "AwayTeam", "FTR",
    "market_home_prob_fair", "market_draw_prob_fair", "market_away_prob_fair",
]


def _to_builtin(value):
    """Convert numpy scalars/arrays inside nested structures to JSON-friendly Python types."""
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_results(
    best_params_groupings: pd.DataFrame,
    holdout: dict,
    features_df: pd.DataFrame,
    output_dir: Path = RESULTS_DIR,
) -> Path:
    """Write the grouping CV summary, per-match holdout predictions and run metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _to_builtin(best_params_groupings.to_dict(orient="records"))
    (output_dir / GROUPING_CV_FILE).write_text(json.dumps(records, indent=2))

    predictions = pd.DataFrame(
        holdout["probabilities"], columns=["model_prob_H", "model_prob_D", "model_prob_A"]
    )
    predictions.insert(0, "match_id", holdout["test_match_ids"])
    match_info = features_df[MATCH_COLUMNS].astype({"Div": str, "HomeTeam": str, "AwayTeam": str})
    predictions = predictions.merge(match_info, on="match_id", how="left", validate="one_to_one")
    predictions.to_csv(output_dir / HOLDOUT_PREDICTIONS_FILE, index=False)

    best = best_params_groupings.iloc[0]
    run_info = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "best_grouping": best["grouping"],
        "selected_features": best["selected_features"],
        "best_params": best["best_params"],
        "split_fraction": holdout["split_fraction"],
        "train_matches": holdout["train_matches"],
        "test_matches": holdout["test_matches"],
        "holdout_log_loss": holdout["holdout_log_loss"],
        "holdout_brier_score": holdout["holdout_brier_score"],
    }
    (output_dir / RUN_INFO_FILE).write_text(json.dumps(_to_builtin(run_info), indent=2))
    return output_dir


def load_results(results_dir: Path = RESULTS_DIR) -> dict | None:
    """Load saved results, or return None if scripts/train_model.py has not been run yet."""
    results_dir = Path(results_dir)
    paths = [results_dir / name for name in (GROUPING_CV_FILE, HOLDOUT_PREDICTIONS_FILE, RUN_INFO_FILE)]
    if not all(path.exists() for path in paths):
        return None
    grouping_cv = pd.DataFrame(json.loads(paths[0].read_text()))
    predictions = pd.read_csv(paths[1], parse_dates=["kickoff"])
    run_info = json.loads(paths[2].read_text())
    return {"grouping_cv": grouping_cv, "predictions": predictions, "run_info": run_info}
