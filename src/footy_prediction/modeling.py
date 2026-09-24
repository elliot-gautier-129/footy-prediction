"""LightGBM training: nested expanding-season walk-forward CV with leakage-safe feature selection."""
import itertools
import random
import re

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss


def _season_start_year(value) -> int:
    """Extract the chronological start year from values such as 2020/21 or 202021."""
    text = str(value)
    match = re.search(r"(\d{4})", text)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d{2})(\d{2})", text)
    if match:
        return 2000 + int(match.group(1))
    raise ValueError(f"Cannot determine chronological year from season {value!r}")


def _ordered_seasons(data, season_column):
    seasons = list(data[season_column].dropna().unique())
    return sorted(seasons, key=_season_start_year)


def _clean_params(params):
    cleaned = dict(params)
    for key in ["num_leaves", "max_depth", "min_child_samples", "n_estimators"]:
        if key in cleaned and cleaned[key] is not None:
            cleaned[key] = int(cleaned[key])
    cleaned.pop("param_id", None)
    return cleaned


def _brier_score(probabilities, targets):
    truth = np.eye(3)[targets]
    return float(np.mean(np.sum((probabilities - truth) ** 2, axis=1)))


def _fit_lgbm(X_train, y_train, params, random_state):
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        n_estimators=150,
        random_state=random_state,
        verbosity=-1,
        **_clean_params(params),
    )
    model.fit(X_train, y_train)
    return model


def _class_prior_probabilities(train_targets, n_rows):
    """Predict the training-set class frequencies for every row (the no-feature baseline)."""
    priors = (
        pd.Series(train_targets)
        .value_counts(normalize=True)
        .reindex([0, 1, 2], fill_value=0.0)
        .to_numpy()
    )
    return np.tile(priors, (n_rows, 1))


def _reduce_correlated_features(data, candidates, threshold):
    if threshold is None or len(candidates) < 2:
        return list(candidates)
    numeric = data[candidates].select_dtypes(include=np.number)
    if numeric.shape[1] < 2:
        return list(candidates)
    corr = numeric.corr().abs()
    keep = []
    for column in candidates:
        if column not in corr:
            keep.append(column)
            continue
        # Only compare against numeric kept columns; categorical ones have no correlation entry.
        if not any(
            corr.loc[column, previous] > threshold for previous in keep if previous in corr
        ):
            keep.append(column)
    return keep


def _inner_folds(data, season_column):
    seasons = _ordered_seasons(data, season_column)
    return [
        (seasons[:index], seasons[index])
        for index in range(1, len(seasons))
    ]


def _score_features_inner(
    data,
    selected_features,
    inner_folds,
    params,
    target_column,
    season_column,
    order_column,
    random_state,
):
    target_map = {"H": 0, "D": 1, "A": 2}
    losses = []
    for train_seasons, validation_season in inner_folds:
        train = data[data[season_column].isin(train_seasons)].sort_values(order_column)
        validation = data[data[season_column] == validation_season].sort_values(order_column)
        if not selected_features:
            # No features yet: score the training-set class frequencies as the baseline.
            probabilities = _class_prior_probabilities(
                train[target_column].map(target_map), len(validation)
            )
            losses.append(
                log_loss(
                    validation[target_column].map(target_map),
                    probabilities,
                    labels=[0, 1, 2],
                )
            )
            continue
        model = _fit_lgbm(
            train[selected_features],
            train[target_column].map(target_map).to_numpy(),
            params,
            random_state,
        )
        probabilities = model.predict_proba(validation[selected_features])
        losses.append(
            log_loss(
                validation[target_column].map(target_map),
                probabilities,
                labels=[0, 1, 2],
            )
        )
    return float(np.mean(losses))


def _select_features_nested(
    outer_train,
    candidate_features,
    base_features,
    params,
    feature_groups,
    correlation_threshold,
    max_selected_features,
    min_log_loss_improvement,
    target_column,
    season_column,
    order_column,
    random_state,
):
    # Correlation filtering uses outer-training rows only.
    candidates = [f for f in candidate_features if f not in base_features]
    candidates = _reduce_correlated_features(
        outer_train, candidates, correlation_threshold
    )
    inner_folds = _inner_folds(outer_train, season_column)
    if not inner_folds:
        return list(base_features), []

    selected = list(base_features)
    selected_records = []
    current_score = _score_features_inner(
        outer_train, selected, inner_folds, params,
        target_column, season_column, order_column, random_state,
    )

    while candidates and len(selected) < max_selected_features:
        trial_scores = []
        for candidate in candidates:
            score = _score_features_inner(
                outer_train, selected + [candidate], inner_folds, params,
                target_column, season_column, order_column, random_state,
            )
            trial_scores.append((score, candidate))
        best_score, best_feature = min(trial_scores)
        improvement = current_score - best_score
        if improvement < min_log_loss_improvement:
            break
        selected.append(best_feature)
        candidates.remove(best_feature)
        current_score = best_score
        selected_records.append({
            "feature": best_feature,
            "inner_log_loss": best_score,
            "improvement": improvement,
        })
    return selected, selected_records


def walk_forward_training(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "FTR",
    order_column: str = "match_id",
    season_column: str = "season",
    feature_groups: dict | None = None,
    base_features: list[str] | None = None,
    correlation_threshold: float | None = 0.90,
    max_selected_features: int = 8,
    min_log_loss_improvement: float = 0.001,
    n_iter: int = 40,
    random_state: int = 42,
):
    """Nested expanding season walk-forward CV with leakage-safe feature selection."""
    required = [order_column, season_column, target_column, *feature_columns]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    if max_selected_features < 1:
        raise ValueError("max_selected_features must be positive")

    cleaned = data.sort_values(order_column).dropna(subset=required).reset_index(drop=True).copy()
    seasons = _ordered_seasons(cleaned, season_column)
    if len(seasons) < 4:
        raise ValueError(
            "At least four seasons are required: two to train the first outer fold "
            "(nested selection needs an inner fold), one to validate it, and a final holdout"
        )

    final_holdout_season = seasons[-1]
    cv_seasons = seasons[:-1]
    # Each outer fold trains on >= 2 seasons so nested feature selection has an inner fold.
    outer_folds = [(cv_seasons[:i], cv_seasons[i]) for i in range(2, len(cv_seasons))]
    base_features = list(base_features or [])
    candidate_features = [f for f in feature_columns if f not in base_features]
    if feature_groups is not None:
        grouped = [feature for group in feature_groups.values() for feature in group]
        candidate_features = [f for f in candidate_features if f in grouped]

    search_space = {
        "num_leaves": [7, 15, 31],
        "max_depth": [3, 4, 6],
        "learning_rate": [0.01, 0.03, 0.05],
        "min_child_samples": [20, 30, 50],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "reg_lambda": [0.0, 1.0, 5.0],
    }
    keys, values = zip(*search_space.items())
    combinations = [dict(zip(keys, values)) for values in itertools.product(*values)]
    generator = random.Random(random_state)
    parameter_samples = generator.sample(combinations, min(n_iter, len(combinations)))
    target_map = {"H": 0, "D": 1, "A": 2}
    results = []
    selection_records = []

    for parameter_id, params in enumerate(parameter_samples):
        fold_losses = []
        fold_briers = []
        for fold_number, (train_seasons, validation_season) in enumerate(outer_folds, 1):
            outer_train = cleaned[cleaned[season_column].isin(train_seasons)].sort_values(order_column)
            validation = cleaned[cleaned[season_column] == validation_season].sort_values(order_column)
            selected, selections = _select_features_nested(
                outer_train, candidate_features, base_features, params, feature_groups,
                correlation_threshold, max_selected_features, min_log_loss_improvement,
                target_column, season_column, order_column, random_state,
            )
            if selected:
                model = _fit_lgbm(outer_train[selected], outer_train[target_column].map(target_map), params, random_state)
                probabilities = model.predict_proba(validation[selected])
            else:
                # No candidate beat the class-prior baseline in the inner folds.
                probabilities = _class_prior_probabilities(
                    outer_train[target_column].map(target_map), len(validation)
                )
            y_validation = validation[target_column].map(target_map).to_numpy()
            fold_losses.append(log_loss(y_validation, probabilities, labels=[0, 1, 2]))
            fold_briers.append(_brier_score(probabilities, y_validation))
            selection_records.append({
                "parameter_id": parameter_id,
                "fold": fold_number,
                "validation_season": validation_season,
                "selected_features": selected,
            })
        results.append({
            "param_id": parameter_id,
            "mean_log_loss": np.mean(fold_losses),
            "mean_brier_score": np.mean(fold_briers),
            "cv_fold_count": len(outer_folds),
            "final_holdout_season": final_holdout_season,
            # Selection from the last outer fold, which trains on the most seasons.
            "last_fold_features": selection_records[-1]["selected_features"],
            "validation_seasons": [validation_season for _, validation_season in outer_folds],
            "fold_log_losses": [float(loss) for loss in fold_losses],
            **params,
        })

    result_df = pd.DataFrame(results).sort_values("mean_log_loss").reset_index(drop=True)
    result_df.attrs["feature_selection_records"] = pd.DataFrame(selection_records)
    result_df.attrs["final_holdout_season"] = final_holdout_season
    return result_df


def K_fold_training(*args, **kwargs):
    """Backward-compatible wrapper for walk_forward_training."""
    return walk_forward_training(*args, **kwargs)


def run_grouping_cross_validation(features_df, feature_groupings):
    grouping_results = []

    for group_name, feature_list in feature_groupings.items():
        print(f"Running cross-validation for grouping: {group_name}...")
        
        cv_summary = K_fold_training(
            data=features_df,
            feature_columns=feature_list,
            target_column="FTR",
            order_column="match_id",
            n_iter=20  # Fast evaluation per set
        )
        
        # Extract top hyperparameter row (already sorted by mean_log_loss)
        best_row = cv_summary.iloc[0].to_dict()
        
        # Pull out metric columns
        log_loss_val = best_row.pop("mean_log_loss")
        brier_val = best_row.pop("mean_brier_score")
        best_row.pop("param_id")
        # CV metadata, not LightGBM hyperparameters.
        best_row.pop("cv_fold_count")
        best_row.pop("final_holdout_season")
        selected_features = best_row.pop("last_fold_features")
        validation_seasons = best_row.pop("validation_seasons")
        fold_log_losses = best_row.pop("fold_log_losses")
        
        # remaining key-value pairs in best_row are the hyperparameters
        grouping_results.append({
            "grouping": group_name,
            "feature_count": len(feature_list),
            "best_mean_log_loss": log_loss_val,
            "best_mean_brier_score": brier_val,
            "selected_features": selected_features,
            "validation_seasons": validation_seasons,
            "fold_log_losses": fold_log_losses,
            "best_params": best_row  # Stores hyperparameters as a dictionary
        })

    # Format into a summary comparison DataFrame
    comparison_df = pd.DataFrame(grouping_results).sort_values("best_mean_log_loss").reset_index(drop=True)
    return comparison_df
