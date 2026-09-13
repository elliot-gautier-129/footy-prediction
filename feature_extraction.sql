CREATE OR REPLACE TEMP TABLE matches_data AS
WITH source_matches AS (
    SELECT ROW_NUMBER() OVER () AS source_order, *
    FROM matches
), parsed_matches AS (
    SELECT
        source_order,
        try_strptime(
            CAST("Date" AS VARCHAR) || ' ' || COALESCE(CAST("Time" AS VARCHAR), '00:00'),
            '%d/%m/%Y %H:%M'
        ) AS kickoff,
        * EXCLUDE (source_order)
    FROM source_matches
), numbered_matches AS (
    SELECT ROW_NUMBER() OVER (ORDER BY kickoff, source_order) - 1 AS match_id, *
    FROM parsed_matches
), market_odds AS (
    SELECT
        numbered_matches.*,
        1 / "AvgH" AS market_home_prob,
        1 / "AvgD" AS market_draw_prob,
        1 / "AvgA" AS market_away_prob
    FROM numbered_matches
)
SELECT
    market_odds.* EXCLUDE (source_order),
    market_home_prob / (market_home_prob + market_draw_prob + market_away_prob) AS market_home_prob_fair,
    market_draw_prob / (market_home_prob + market_draw_prob + market_away_prob) AS market_draw_prob_fair,
    market_away_prob / (market_home_prob + market_draw_prob + market_away_prob) AS market_away_prob_fair
FROM market_odds;

CREATE OR REPLACE TEMP TABLE history AS
SELECT
    match_id, kickoff, "HomeTeam" AS team, 'home' AS venue,
    "FTHG" AS goals_for, "FTAG" AS goals_against,
    "HF" AS fouls_for, "HC" AS corners_for,
    "HY" AS yellow_cards_for, "HR" AS red_cards_for,
    CASE "FTR" WHEN 'H' THEN 3 WHEN 'D' THEN 1 WHEN 'A' THEN 0 END AS points,
    "HS" AS shots_for, "HST" AS shots_on_target_for
FROM matches_data
UNION ALL
SELECT
    match_id, kickoff, "AwayTeam" AS team, 'away' AS venue,
    "FTAG" AS goals_for, "FTHG" AS goals_against,
    "AF" AS fouls_for, "AC" AS corners_for,
    "AY" AS yellow_cards_for, "AR" AS red_cards_for,
    CASE "FTR" WHEN 'H' THEN 0 WHEN 'D' THEN 1 WHEN 'A' THEN 3 END AS points,
    "AS" AS shots_for, "AST" AS shots_on_target_for
FROM matches_data;

CREATE OR REPLACE TEMP TABLE history_features AS
WITH previous_match AS (
    SELECT
        history.*,
        LAG(kickoff) OVER (PARTITION BY team ORDER BY kickoff, match_id) AS previous_kickoff
    FROM history
)
SELECT 
    previous_match.* EXCLUDE (previous_kickoff),
    EPOCH(kickoff - previous_kickoff) / 86400 AS rest_days,
    AVG(points) OVER team_w5 AS points_last_5,
    AVG(points) OVER team_w10 AS points_last_10,
    AVG(goals_for) OVER team_w5 AS goals_for_last_5,
    AVG(goals_for) OVER team_w10 AS goals_for_last_10,
    AVG(goals_against) OVER team_w5 AS goals_against_last_5,
    AVG(goals_against) OVER team_w10 AS goals_against_last_10,
    AVG(shots_for) OVER team_w5 AS shots_for_last_5,
    AVG(shots_for) OVER team_w10 AS shots_for_last_10,
    AVG(shots_on_target_for) OVER team_w5 AS shots_on_target_for_last_5,
    AVG(shots_on_target_for) OVER team_w10 AS shots_on_target_for_last_10,
    AVG(fouls_for) OVER team_w5 AS fouls_for_last_5,
    AVG(fouls_for) OVER team_w10 AS fouls_for_last_10,
    AVG(corners_for) OVER team_w5 AS corners_for_last_5,
    AVG(corners_for) OVER team_w10 AS corners_for_last_10,
    AVG(yellow_cards_for) OVER team_w5 AS yellow_cards_for_last_5,
    AVG(yellow_cards_for) OVER team_w10 AS yellow_cards_for_last_10,
    AVG(red_cards_for) OVER team_w5 AS red_cards_for_last_5,
    AVG(red_cards_for) OVER team_w10 AS red_cards_for_last_10
FROM previous_match
WINDOW
    team_w5 AS (PARTITION BY team ORDER BY kickoff, match_id ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING),
    team_w10 AS (PARTITION BY team ORDER BY kickoff, match_id ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING);

CREATE OR REPLACE TEMP TABLE venue_history_features AS
SELECT
    history.*,
    AVG(points) OVER venue_w5 AS points_venue_last_5,
    AVG(points) OVER venue_w10 AS points_venue_last_10,
    AVG(goals_for) OVER venue_w5 AS goals_for_venue_last_5,
    AVG(goals_for) OVER venue_w10 AS goals_for_venue_last_10,
    AVG(goals_against) OVER venue_w5 AS goals_against_venue_last_5,
    AVG(goals_against) OVER venue_w10 AS goals_against_venue_last_10
FROM history
WINDOW
    venue_w5 AS (PARTITION BY team, venue ORDER BY kickoff, match_id ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING),
    venue_w10 AS (PARTITION BY team, venue ORDER BY kickoff, match_id ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING);

CREATE OR REPLACE TEMP TABLE prematch_sql AS
SELECT
    m.*,
    home.rest_days AS home_rest_days,
    home.points_last_5 AS home_points_last_5,
    home.points_last_10 AS home_points_last_10,
    home.goals_for_last_5 AS home_goals_for_last_5,
    home.goals_for_last_10 AS home_goals_for_last_10,
    home.goals_against_last_5 AS home_goals_against_last_5,
    home.goals_against_last_10 AS home_goals_against_last_10,
    home.shots_for_last_5 AS home_shots_for_last_5,
    home.shots_for_last_10 AS home_shots_for_last_10,
    home.shots_on_target_for_last_5 AS home_shots_on_target_for_last_5,
    home.shots_on_target_for_last_10 AS home_shots_on_target_for_last_10,
    home.fouls_for_last_5 AS home_fouls_for_last_5,
    home.fouls_for_last_10 AS home_fouls_for_last_10,
    home.corners_for_last_5 AS home_corners_for_last_5,
    home.corners_for_last_10 AS home_corners_for_last_10,
    home.yellow_cards_for_last_5 AS home_yellow_cards_for_last_5,
    home.yellow_cards_for_last_10 AS home_yellow_cards_for_last_10,
    home.red_cards_for_last_5 AS home_red_cards_for_last_5,
    home.red_cards_for_last_10 AS home_red_cards_for_last_10,
    away.rest_days AS away_rest_days,
    away.points_last_5 AS away_points_last_5,
    away.points_last_10 AS away_points_last_10,
    away.goals_for_last_5 AS away_goals_for_last_5,
    away.goals_for_last_10 AS away_goals_for_last_10,
    away.goals_against_last_5 AS away_goals_against_last_5,
    away.goals_against_last_10 AS away_goals_against_last_10,
    away.shots_for_last_5 AS away_shots_for_last_5,
    away.shots_for_last_10 AS away_shots_for_last_10,
    away.shots_on_target_for_last_5 AS away_shots_on_target_for_last_5,
    away.shots_on_target_for_last_10 AS away_shots_on_target_for_last_10,
    away.fouls_for_last_5 AS away_fouls_for_last_5,
    away.fouls_for_last_10 AS away_fouls_for_last_10,
    away.corners_for_last_5 AS away_corners_for_last_5,
    away.corners_for_last_10 AS away_corners_for_last_10,
    away.yellow_cards_for_last_5 AS away_yellow_cards_for_last_5,
    away.yellow_cards_for_last_10 AS away_yellow_cards_for_last_10,
    away.red_cards_for_last_5 AS away_red_cards_for_last_5,
    away.red_cards_for_last_10 AS away_red_cards_for_last_10,
    home_venue.points_venue_last_5 AS home_points_home_last_5,
    home_venue.points_venue_last_10 AS home_points_home_last_10,
    home_venue.goals_for_venue_last_5 AS home_goals_for_home_last_5,
    home_venue.goals_for_venue_last_10 AS home_goals_for_home_last_10,
    home_venue.goals_against_venue_last_5 AS home_goals_against_home_last_5,
    home_venue.goals_against_venue_last_10 AS home_goals_against_home_last_10,
    away_venue.points_venue_last_5 AS away_points_away_last_5,
    away_venue.points_venue_last_10 AS away_points_away_last_10,
    away_venue.goals_for_venue_last_5 AS away_goals_for_away_last_5,
    away_venue.goals_for_venue_last_10 AS away_goals_for_away_last_10,
    away_venue.goals_against_venue_last_5 AS away_goals_against_away_last_5,
    away_venue.goals_against_venue_last_10 AS away_goals_against_away_last_10,
    home.points_last_5 - away.points_last_5 AS points_difference_last_5,
    home.goals_for_last_5 - away.goals_for_last_5 AS goals_difference_last_5
FROM matches_data AS m
LEFT JOIN history_features AS home ON m.match_id = home.match_id AND home.venue = 'home'
LEFT JOIN history_features AS away ON m.match_id = away.match_id AND away.venue = 'away'
LEFT JOIN venue_history_features AS home_venue ON m.match_id = home_venue.match_id AND home_venue.venue = 'home'
LEFT JOIN venue_history_features AS away_venue ON m.match_id = away_venue.match_id AND away_venue.venue = 'away';