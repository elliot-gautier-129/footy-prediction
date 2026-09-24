"""Out-of-sample evaluation of LightGBM models and the bookmaker-market baseline."""
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, log_loss

from footy_prediction.modeling import _brier_score, _fit_lgbm, _ordered_seasons


def evaluate_final_holdout(
    data, feature_columns, hyperparams, target_column="FTR",
    order_column="match_id", season_column="season", random_state=42,
):
    """Fit on all pre-final seasons and evaluate the untouched final season."""
    required = [order_column, season_column, target_column, *feature_columns]
    cleaned = data.sort_values(order_column).dropna(subset=required).reset_index(drop=True)
    seasons = _ordered_seasons(cleaned, season_column)
    final_season = seasons[-1]
    train = cleaned[cleaned[season_column] != final_season]
    test = cleaned[cleaned[season_column] == final_season]
    target_map = {"H": 0, "D": 1, "A": 2}
    model = _fit_lgbm(train[feature_columns], train[target_column].map(target_map), hyperparams, random_state)
    probabilities = model.predict_proba(test[feature_columns])
    y_test = test[target_column].map(target_map).to_numpy()
    return {
        "train_seasons": seasons[:-1],
        "test_season": final_season,
        "log_loss": log_loss(y_test, probabilities, labels=[0, 1, 2]),
        "brier_score": _brier_score(probabilities, y_test),
        "accuracy": accuracy_score(y_test, probabilities.argmax(axis=1)),
        "model": model,
        "probabilities": probabilities,
    }


def evaluate_holdout_model(
    data: pd.DataFrame,
    feature_columns: list[str],
    hyperparams: dict,
    split_fraction: float = 0.8,
    target_column: str = "FTR",
    order_column: str = "match_id",
    random_state: int = 42
) -> dict:
    """Trains a LightGBM model on the initial split_fraction of the data (e.g. 0-80%)
    and evaluates Log Loss and Brier Score on the holdout slice (80-100%).
    """
    if not 0 < split_fraction < 1:
        raise ValueError("split_fraction must be between 0 and 1")
    if not feature_columns:
        raise ValueError("feature_columns is empty: no features were selected to train on")

    # 1. Clean hyperparameter types (cast floats like 15.0 or 3.0 to int where required)
    cleaned_params = hyperparams.copy()
    int_keys = ["num_leaves", "max_depth", "min_child_samples"]
    for key in int_keys:
        if key in cleaned_params and cleaned_params[key] is not None:
            cleaned_params[key] = int(cleaned_params[key])

    # Remove non-LightGBM keys if present
    cleaned_params.pop("param_id", None)

    # 2. Chronological sort and drop missing targets/features
    required_cols = [order_column, target_column, *feature_columns]
    ordered_data = data.sort_values(order_column).dropna(subset=required_cols).reset_index(drop=True)

    # Map target strings (H=0, D=1, A=2)
    target_map = {'H': 0, 'D': 1, 'A': 2}
    ordered_data['target'] = ordered_data[target_column].map(target_map)

    # 3. Temporal Train / Holdout Split
    split_idx = int(len(ordered_data) * split_fraction)
    
    train_df = ordered_data.iloc[:split_idx]
    test_df = ordered_data.iloc[split_idx:]

    X_train = train_df[feature_columns]
    y_train = train_df['target'].values

    X_test = test_df[feature_columns]
    y_test = test_df['target'].values

    # 4. Train LightGBM Model
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=150,
        random_state=random_state,
        verbosity=-1,
        **cleaned_params
    )
    model.fit(X_train, y_train)

    # 5. Predict probabilities on the holdout set (80th - 100th percentile)
    test_preds = model.predict_proba(X_test)

    # 6. Compute Log Loss
    holdout_log_loss = log_loss(y_test, test_preds, labels=[0, 1, 2])

    # 7. Compute Multi-class Brier Score
    y_test_onehot = np.eye(3)[y_test]
    holdout_brier_score = np.mean(np.sum((test_preds - y_test_onehot) ** 2, axis=1))

    return {
        "train_matches": len(X_train),
        "test_matches": len(X_test),
        "split_fraction": split_fraction,
        "holdout_log_loss": round(holdout_log_loss, 4),
        "holdout_brier_score": round(holdout_brier_score, 4),
        "model": model,
        # Per-match holdout output, columns ordered [H, D, A].
        "test_match_ids": test_df[order_column].to_numpy(),
        "probabilities": test_preds,
    }


def evaluate_market_baseline(
    data: pd.DataFrame,
    start_fraction: float = 0.8,
    end_fraction: float = 1.0,
    target_column: str = "FTR",
    order_column: str = "match_id",
    market_prob_cols: list[str] = [
        "market_home_prob_fair",
        "market_draw_prob_fair",
        "market_away_prob_fair"
    ]
) -> dict[str, float]:
    """Evaluates market-implied fair probabilities on a specific temporal slice of the dataset.
    
    Args:
        data: DataFrame containing target and market fair probabilities.
        start_fraction: Starting percentile fraction (e.g., 0.8 for 80th percentile).
        end_fraction: Ending percentile fraction (e.g., 1.0 for 100th percentile).
        target_column: Column name for target ('FTR').
        order_column: Column name for chronological ordering ('match_id').
        market_prob_cols: List of 3 probability columns [Home, Draw, Away].
        
    Returns:
        Dictionary containing Log Loss, Brier Score, and match count for the slice.
    """
    if not (0 <= start_fraction < end_fraction <= 1.0):
        raise ValueError("Must satisfy 0 <= start_fraction < end_fraction <= 1.0")

    # 1. Chronological order & drop missing values in relevant columns
    cols_to_check = [order_column, target_column, *market_prob_cols]
    ordered_data = data.sort_values(order_column).dropna(subset=cols_to_check).reset_index(drop=True)

    # 2. Slice temporal subset (e.g., 80% to 100%)
    start_idx = int(len(ordered_data) * start_fraction)
    end_idx = int(len(ordered_data) * end_fraction)
    
    test_slice = ordered_data.iloc[start_idx:end_idx].copy()
    
    if len(test_slice) == 0:
        raise ValueError(f"Slice range [{start_fraction}, {end_fraction}] resulted in zero matches.")

    # 3. Map targets (H=0, D=1, A=2)
    target_map = {'H': 0, 'D': 1, 'A': 2}
    y_true = test_slice[target_column].map(target_map).values

    # 4. Extract Market Probabilities array (N x 3)
    y_market_probs = test_slice[market_prob_cols].values

    # Normalize across rows to guarantee probabilities sum strictly to 1.0
    y_market_probs = y_market_probs / y_market_probs.sum(axis=1, keepdims=True)

    # 5. Compute Log Loss
    market_log_loss = metrics.log_loss(y_true, y_market_probs, labels=[0, 1, 2])

    # 6. Compute Multi-class Brier Score
    # Convert integer targets into one-hot binary matrix
    y_true_onehot = np.eye(3)[y_true]
    market_brier_score = np.mean(np.sum((y_market_probs - y_true_onehot) ** 2, axis=1))

    return {
        "match_count": len(test_slice),
        "start_percentile": f"{int(start_fraction * 100)}%",
        "end_percentile": f"{int(end_fraction * 100)}%",
        "market_log_loss": round(market_log_loss, 4),
        "market_brier_score": round(market_brier_score, 4)
    }
