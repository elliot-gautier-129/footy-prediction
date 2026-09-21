def build_prematch_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Create leakage-safe features available before each match."""
    data = matches.copy()
    data = data.rename(columns={"ï»¿Div": "Div"})

    # Combine date and time so all matches can be ordered chronologically.
    data["kickoff"] = pd.to_datetime(
        data["Date"].astype(str) + " " + data["Time"].fillna("00:00").astype(str),
        dayfirst=True,
        errors="coerce",
    )
    data = data.sort_values("kickoff").reset_index(drop=True)
    data["match_id"] = range(len(data))

    # Convert average bookmaker odds into normalized implied probabilities.
    data["market_home_prob"] = 1 / data["AvgH"]
    data["market_draw_prob"] = 1 / data["AvgD"]
    data["market_away_prob"] = 1 / data["AvgA"]
    # since bookmakers make a cut on their odds notmalise the probabilities so that they add to one 
    overround = data[
        ["market_home_prob", "market_draw_prob", "market_away_prob"]
    ].sum(axis=1)
    for outcome in ["home", "draw", "away"]:
        data[f"market_{outcome}_prob_fair"] = (
            data[f"market_{outcome}_prob"] / overround
        )

    # Represent each match once from each team's perspective.
    home_history = pd.DataFrame({
        "match_id": data["match_id"],
        "kickoff": data["kickoff"],
        "team": data["HomeTeam"],
        "venue": "home",
        "goals_for": data["FTHG"],
        "goals_against": data["FTAG"],
        "fouls_for": data["HF"],
        "corners_for": data["HC"],
        "yellow_cards_for": data["HY"],
        "red_cards_for": data["HR"],
        "points": data["FTR"].map({"H": 3, "D": 1, "A": 0}),
        "shots_for": data["HS"],
        "shots_on_target_for": data["HST"],
    })
    away_history = pd.DataFrame({
        "match_id": data["match_id"],
        "kickoff": data["kickoff"],
        "team": data["AwayTeam"],
        "venue": "away",
        "goals_for": data["FTAG"],
        "goals_against": data["FTHG"],
        "fouls_for": data["AF"],
        "corners_for": data["AC"],
        "yellow_cards_for": data["AY"],
        "red_cards_for": data["AR"],
        "points": data["FTR"].map({"H": 0, "D": 1, "A": 3}),
        "shots_for": data["AS"],
        "shots_on_target_for": data["AST"],
    })
    history = pd.concat([home_history, away_history], ignore_index=True)
    history = history.sort_values(["team", "kickoff", "match_id"])

    # Overall team form, using only matches before the current match.
    rolling_columns = [
        "points",
        "goals_for",
        "goals_against",
        "shots_for",
        "shots_on_target_for",
        "fouls_for",
        "corners_for",
        "yellow_cards_for",
        "red_cards_for",
    ]
    feature_frames = []
    # group by the teams but ignore the "team" column as it is not needed for the rolling calculations
    for _, team_history in history.groupby("team", sort=False):
        team_history = team_history.copy()
        # calculate the number of rest days between matches for each team
        team_history["rest_days"] = (
            team_history["kickoff"].diff().dt.total_seconds() / 86400
        )
        # loop over the two rolling averages and the selected columns
        for window in [5, 10]:
            for column in rolling_columns:
                # shift(1) prevents the current match result entering its features.
                team_history[f"{column}_last_{window}"] = (
                    team_history[column] # get the column values for the team
                    .shift(1) # shift them by one to avoid using the current match's data
                    .rolling(window, min_periods=1) # calc rolling mean allowing for fewer than window matches
                    .mean()
                )
        feature_frames.append(team_history) # make a list of the team history dataframes with the new rolling features added

    history_features = pd.concat(feature_frames) # concatenate the list of dataframes into a single dataframe with all teams' history features

    # Venue-specific form for the three requested columns.
    venue_columns = ["points", "goals_for", "goals_against"]
    venue_feature_frames = []
    for _, team_venue_history in history.groupby(["team", "venue"], sort=False):
        team_venue_history = team_venue_history.copy()
        venue = team_venue_history["venue"].iloc[0]
        for window in [5, 10]:
            for column in venue_columns:
                # The venue group means home form uses previous home matches only,
                # and away form uses previous away matches only.
                team_venue_history[
                    f"{column}_{venue}_last_{window}"
                ] = (
                    team_venue_history[column]
                    .shift(1)
                    .rolling(window, min_periods=1)
                    .mean()
                )
        venue_feature_frames.append(team_venue_history)

    # This concatenation is required before the venue features can be merged.
    venue_history_features = pd.concat(venue_feature_frames)

    # create column names for every rolling feature
    rolling_features = [
        f"{column}_last_{window}"
        for window in [5, 10]
        for column in rolling_columns
    ]
    selected_features = ["match_id", "venue", "rest_days", *rolling_features] # creates list of columns to keep

    # Attach overall history to the home and away team separately.
    home_features = history_features[
        history_features["venue"] == "home" # keeps only home matches
    ][selected_features].drop(columns="venue") # selects features to keep drops venue column as no longer needed
    home_features = home_features.rename(columns={
        column: f"home_{column}" # adds home to evey featurn column but match_id  
        for column in home_features.columns
        if column != "match_id"
    })

    # same for away team
    away_features = history_features[
        history_features["venue"] == "away"
    ][selected_features].drop(columns="venue")
    away_features = away_features.rename(columns={
        column: f"away_{column}"
        for column in away_features.columns
        if column != "match_id"
    })

    # Select the venue-specific features relevant to each side of the match.
    home_venue_columns = [
        "match_id",
        "points_home_last_5",
        "points_home_last_10",
        "goals_for_home_last_5",
        "goals_for_home_last_10",
        "goals_against_home_last_5",
        "goals_against_home_last_10",
    ]
    away_venue_columns = [
        "match_id",
        "points_away_last_5",
        "points_away_last_10",
        "goals_for_away_last_5",
        "goals_for_away_last_10",
        "goals_against_away_last_5",
        "goals_against_away_last_10",
    ]
    # does the same for renaming the venue specific columns for home and away teams
    home_venue_features = venue_history_features[
        venue_history_features["venue"] == "home"
    ][home_venue_columns].rename(columns={
        column: f"home_{column}"
        for column in home_venue_columns
        if column != "match_id"
    })
    away_venue_features = venue_history_features[
        venue_history_features["venue"] == "away"
    ][away_venue_columns].rename(columns={
        column: f"away_{column}"
        for column in away_venue_columns
        if column != "match_id"
    })

    # Merge overall and venue-specific history back onto each match.
    features = data.merge(home_features, on="match_id")
    features = features.merge(away_features, on="match_id")
    features = features.merge(home_venue_features, on="match_id")
    features = features.merge(away_venue_features, on="match_id")

    # Compare recent overall form between the two teams.
    features["points_difference_last_5"] = (
        features["home_points_last_5"] - features["away_points_last_5"]
    )
    features["goals_difference_last_5"] = (
        features["home_goals_for_last_5"] - features["away_goals_for_last_5"]
    )
    return features


# Build the model-ready feature table for the loaded season.
prematch_2526 = build_prematch_features(df_2526)
prematch_2526[[
    "HomeTeam",
    "AwayTeam",
    "FTR",
    "home_points_last_5",
    "away_points_last_5",
    "home_points_home_last_5",
    "away_points_away_last_5",
    "home_goals_for_home_last_5",
    "away_goals_for_away_last_5",
    "market_home_prob_fair",
    "market_draw_prob_fair",
    "market_away_prob_fair",
]]
# len(list(prematch_2526.columns))
prematch_2526.head(10)