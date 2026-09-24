"""Run the LightGBM experiment from notebooks/lightgbm.ipynb.

Compares the feature groupings with walk-forward CV, evaluates the best
hyperparameters on the final 20% of matches, and compares against the
bookmaker-market baseline.

Usage:
    python scripts/train_model.py
"""
from footy_prediction.data_loading import load_features
from footy_prediction.evaluation import evaluate_holdout_model, evaluate_market_baseline
from footy_prediction.feature_sets import FEATURE_GROUPINGS
from footy_prediction.modeling import run_grouping_cross_validation
from footy_prediction.results import save_results


def main():
    features_df = load_features("features_data_xg_rolling")

    best_params_groupings = run_grouping_cross_validation(features_df, FEATURE_GROUPINGS)
    print(best_params_groupings)

    # Evaluate the best grouping with its own CV-selected features and hyperparameters.
    best = best_params_groupings.iloc[0]
    print(f"Best grouping: {best['grouping']} -> features {best['selected_features']}")
    holdout = evaluate_holdout_model(
        features_df, best["selected_features"], hyperparams=best["best_params"]
    )
    print({
        key: value for key, value in holdout.items()
        if key not in ("model", "probabilities", "test_match_ids")
    })

    # Saved for the Streamlit app (app/streamlit_app.py).
    results_dir = save_results(best_params_groupings, holdout, features_df)
    print(f"Saved results to '{results_dir}'")

    market_benchmark = evaluate_market_baseline(
        data=features_df,
        start_fraction=0.8,
        end_fraction=1.0,
        target_column="FTR",
        order_column="match_id"
    )
    print(market_benchmark)


if __name__ == "__main__":
    main()
