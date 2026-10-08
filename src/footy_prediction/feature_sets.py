"""Candidate pre-match feature lists used for model comparison.

All groupings use football information only. The bookmaker probabilities
(market_*_prob_fair) are deliberately left out so the model has to learn from the
football itself; they stay in the feature table as the benchmark the model is scored
against. Referee is left out because it is only recorded for the Premier League, and a
missing value drops the whole row.
"""
# These are candidate pre-match features. The target FTR is kept separate.
# HS and AS are excluded because they are only known after the match.

FEATURE_GROUPINGS = {
    # 1. Who is playing: team identity and league only, no form at all.
    #    Baseline for "how much do long-run team strengths explain?"
    "team_identity": [
        "HomeTeam",
        "AwayTeam",
        "Div",
    ],

    # 2. Recent results: points and goals over the last 5 matches, overall and at this venue.
    "results_form": [
        "home_points_last_5",
        "away_points_last_5",
        "home_goals_for_last_5",
        "away_goals_for_last_5",
        "home_goals_against_last_5",
        "away_goals_against_last_5",
        "home_points_home_last_5",
        "away_points_away_last_5",
    ],

    # 3. Underlying performance: chance quality created and conceded (xG) and shots on target,
    #    which are less noisy than results.
    "xg_form": [
        "home_xg_last_5",
        "away_xg_last_5",
        "home_xg_against_last_5",
        "away_xg_against_last_5",
        "home_shots_on_target_for_last_5",
        "away_shots_on_target_for_last_5",
    ],

    # 4. Results and xG together, using compact difference features for results.
    "results_plus_xg": [
        "points_difference_last_5",
        "goals_difference_last_5",
        "home_points_home_last_5",
        "away_points_away_last_5",
        "home_xg_last_5",
        "away_xg_last_5",
        "home_xg_against_last_5",
        "away_xg_against_last_5",
    ],

    # 5. Team identity plus xG form, results difference and fatigue (rest days).
    "teams_plus_form": [
        "HomeTeam",
        "AwayTeam",
        "Div",
        "home_xg_last_5",
        "away_xg_last_5",
        "home_xg_against_last_5",
        "away_xg_against_last_5",
        "points_difference_last_5",
        "rest_days_diff",
    ],
}

# Every non-market candidate feature across the groupings.
DEFAULT_FEATURES = list(dict.fromkeys(
    feature for features in FEATURE_GROUPINGS.values() for feature in features
))
