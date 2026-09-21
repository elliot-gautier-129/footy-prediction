
CREATE OR REPLACE TABLE features_xg_df AS
WITH parsed_xg AS (
    SELECT
          
            CAST(TRY_CAST("date" AS DATE) AS VARCHAR)
                || '_' || home_team_code || '_' || away_team_code AS match_concat,
        home_xg,
        away_xg
    FROM xg_df 
),
joined_matches AS (
    SELECT
        md.*,
        pm.home_xg,
        pm.away_xg,
    FROM features_data md
    -- dont keep unmatching rows as they will get dropped later anyways
    INNER JOIN parsed_xg pm
        ON md.match_concat = pm.match_concat
)
SELECT *
FROM joined_matches;