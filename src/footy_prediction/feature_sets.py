"""Candidate pre-match feature lists used for model comparison."""
# These are candidate pre-match features. The target FTR is kept separate.
# HS and AS are excluded because they are only known after the match.
DEFAULT_FEATURES = [
    "HomeTeam",
    "AwayTeam",
    "home_points_last_5",
    "away_points_last_5",
    "home_points_home_last_5",
    "away_points_away_last_5",
    "market_home_prob_fair",
    "market_draw_prob_fair",
    "market_away_prob_fair",
    "goals_difference_last_5",
    "points_difference_last_5",
    "home_rest_days",
    "away_rest_days",
]

FEATURE_GROUPINGS = {
    # 1. Benchmark: Pure market consensus (fair odds probabilities)
    "market_only": [
        "HomeTeam",
        "AwayTeam",
        "market_home_prob_fair",
        "market_draw_prob_fair",
        "market_away_prob_fair",
        "Div"
    ],

    # 2. Market probabilities plus rolling xG from each team's last five matches
    "market_plus_rolling_xg": [
        "market_home_prob_fair",
        "market_draw_prob_fair",
        "market_away_prob_fair",
        "HomeTeam",
        "AwayTeam",
        "home_xg_last_5",
        "away_xg_last_5",
        "Div"
    ],

    # 3. Add referee and division/league context to market plus rolling xG
    "market_plus_rolling_xg_ref_div": [
        "HomeTeam",
        "AwayTeam",
        "market_home_prob_fair",
        "market_draw_prob_fair",
        "market_away_prob_fair",
        "home_xg_last_5",
        "away_xg_last_5",
        "Referee",
        "Div",
    ],

    # 4. Market, rolling xG, referee, venue form, and rest-day difference
    "market_plus_xg_ref_venue_rest": [
        "HomeTeam",
        "AwayTeam",
        "market_home_prob_fair",
        "market_draw_prob_fair",
        "market_away_prob_fair",
        "home_xg_last_5",
        "away_xg_last_5",
        "Referee",
        "away_points_away_last_5",
        "home_points_home_last_5",
        "rest_days_diff",
    ],

    # Retain the full existing feature set for reference.
    # "full_combined": DEFAULT_FEATURES,
}
