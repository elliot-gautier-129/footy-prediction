-- getting the xg rolling average

CREATE OR REPLACE TEMP TABLE history AS
SELECT
    match_id,
    kickoff,
    league,
    HomeTeam as team,
    'home' as venue,
    home_xg as xg,
    away_xg as xg_against   -- xG conceded: the opponent's xG in the same match
FROM features_xg_df
UNION ALL
SELECT
    match_id, kickoff, league, AwayTeam as team, 
    'away' as venue,
    away_xg as xg,
    home_xg as xg_against
FROM features_xg_df;


CREATE OR REPLACE TEMP TABLE history_features AS
WITH previous_match AS (
    SELECT
        history.*,
        LAG(kickoff) OVER (
            PARTITION BY league, team
            ORDER BY kickoff, match_id
        ) AS previous_kickoff
    FROM history
)
SELECT
    previous_match.* EXCLUDE (previous_kickoff),
    AVG(xg) OVER team_w5 AS xg_last_5,
    AVG(xg_against) OVER team_w5 AS xg_against_last_5
FROM previous_match
WINDOW team_w5 AS (
    PARTITION BY league, team
    ORDER BY kickoff, match_id
    ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING
);

CREATE OR REPLACE TABLE features_data_xg AS
SELECT
    fd.match_id,
    fd.kickoff,
    fd.HomeTeam,
    fd.AwayTeam,
    fd.Div,
    fd.season,   -- used to split walk-forward CV folds by season
    fd.FTR,   -- this will be our y variable for models
    fd.Referee,
    -- market probs from the average pre-match odds across bookmakers (AvgH/AvgD/AvgA)
    fd.market_home_prob_fair,
    fd.market_draw_prob_fair,
    fd.market_away_prob_fair,
    -- best pre-match decimal odds seen across the tracked bookmakers (football-data.co.uk
    -- Max columns): the price a bettor shopping across platforms could actually take,
    -- used for backtest PnL rather than the margin-free fair probabilities above.
    fd.MaxH AS best_odds_home,
    fd.MaxD AS best_odds_draw,
    fd.MaxA AS best_odds_away,
    -- rolling statistics for the last 5 matches (remove anything shot related and now add just xg)
    -- venue strength
    fd.home_points_home_last_5,
    fd.away_points_away_last_5,
    fd.goals_difference_last_5,
    fd.points_difference_last_5,
    fd.home_fouls_for_last_5,
    fd.away_fouls_for_last_5,
    fd.home_corners_for_last_5,
    fd.away_corners_for_last_5,
    fd.home_rest_days - fd.away_rest_days AS rest_days_diff,
    home.xg_last_5 AS home_xg_last_5,
    away.xg_last_5 AS away_xg_last_5,
    -- non-market form features for the model groupings (all use windows ending one match earlier)
    home.xg_against_last_5 AS home_xg_against_last_5,
    away.xg_against_last_5 AS away_xg_against_last_5,
    fd.home_points_last_5,
    fd.away_points_last_5,
    fd.home_goals_for_last_5,
    fd.away_goals_for_last_5,
    fd.home_goals_against_last_5,
    fd.away_goals_against_last_5,
    fd.home_shots_on_target_for_last_5,
    fd.away_shots_on_target_for_last_5,

FROM features_data AS fd
LEFT JOIN history_features AS home ON fd.match_id = home.match_id AND home.venue = 'home'
LEFT JOIN history_features AS away ON fd.match_id = away.match_id AND away.venue = 'away'