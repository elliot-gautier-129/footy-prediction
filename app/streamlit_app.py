"""Interactive presentation of the footy-prediction results.

Run from anywhere with:
    streamlit run app/streamlit_app.py

Model results are read from results/ (written by scripts/train_model.py); set
FOOTY_RESULTS_DIR to read them from another directory.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from footy_prediction.data_loading import load_features
from footy_prediction.feature_sets import FEATURE_GROUPINGS
from footy_prediction.paths import RESULTS_DIR
from footy_prediction.results import load_results

st.set_page_config(page_title="Footy Prediction", page_icon="⚽", layout="wide")

LEAGUE_NAMES = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
}
OUTCOMES = {"H": "Home win", "D": "Draw", "A": "Away win"}
OUTCOME_INDEX = {"H": 0, "D": 1, "A": 2}
MODEL_COLUMNS = ["model_prob_H", "model_prob_D", "model_prob_A"]
MARKET_COLUMNS = ["market_home_prob_fair", "market_draw_prob_fair", "market_away_prob_fair"]

# Validated categorical palette (light / dark steps of the same hues).
PALETTES = {
    "light": {
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
        "muted": "#c3c2b7",
        "grid": "#e1e0d9",
        "ink": "#52514e",
        "background": "#ffffff",
    },
    "dark": {
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500"],
        "muted": "#383835",
        "grid": "#2c2c2a",
        "ink": "#c3c2b7",
        "background": "#0e1117",
    },
}


def _theme_type() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except AttributeError:
        return "light"


PALETTE = PALETTES[_theme_type()]
MODEL_COLOR, MARKET_COLOR = PALETTE["series"][0], PALETTE["series"][1]
OUTCOME_COLORS = dict(zip(OUTCOMES, PALETTE["series"][:3]))


# --------------------------------------------------------------------------- data


@st.cache_data
def get_features() -> pd.DataFrame:
    features = load_features("features_data_xg_rolling")
    features["kickoff"] = pd.to_datetime(features["kickoff"])
    for column in ["HomeTeam", "AwayTeam", "Div", "Referee"]:
        features[column] = features[column].astype(str)
    return features


@st.cache_data
def get_results(results_dir: str):
    return load_results(Path(results_dir))


def per_match_log_loss(probabilities: np.ndarray, outcomes: pd.Series) -> np.ndarray:
    """-log(probability given to the actual result), per match."""
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    index = outcomes.map(OUTCOME_INDEX).to_numpy()
    chosen = probabilities[np.arange(len(index)), index]
    return -np.log(np.clip(chosen, 1e-15, 1))


def per_match_brier(probabilities: np.ndarray, outcomes: pd.Series) -> np.ndarray:
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    truth = np.eye(3)[outcomes.map(OUTCOME_INDEX).to_numpy()]
    return np.sum((probabilities - truth) ** 2, axis=1)


def add_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    scored = predictions.dropna(subset=MODEL_COLUMNS + MARKET_COLUMNS + ["FTR"]).copy()
    model = scored[MODEL_COLUMNS].to_numpy()
    market = scored[MARKET_COLUMNS].to_numpy()
    scored["model_log_loss"] = per_match_log_loss(model, scored["FTR"])
    scored["market_log_loss"] = per_match_log_loss(market, scored["FTR"])
    scored["model_brier"] = per_match_brier(model, scored["FTR"])
    scored["market_brier"] = per_match_brier(market, scored["FTR"])
    scored["model_pick"] = np.array(list(OUTCOMES))[model.argmax(axis=1)]
    scored["market_pick"] = np.array(list(OUTCOMES))[market.argmax(axis=1)]
    return scored


# ------------------------------------------------------------------------- charts


def style(fig: go.Figure, height: int = 360, y_title: str = "", x_title: str = "") -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=64, r=16, t=40, b=48),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
        hoverlabel=dict(namelength=-1),
        bargap=0.25,
        bargroupgap=0.08,
    )
    fig.update_xaxes(title=x_title, showgrid=False, linecolor=PALETTE["muted"], ticks="")
    fig.update_yaxes(title=y_title, gridcolor=PALETTE["grid"], gridwidth=1, zeroline=False, automargin=True)
    return fig


def show_chart(fig: go.Figure, table: pd.DataFrame, key: str):
    """Every chart ships with a table view of the same numbers."""
    chart_tab, table_tab = st.tabs(["Chart", "Table"])
    with chart_tab:
        st.plotly_chart(fig, width="stretch", theme="streamlit", key=key)
    with table_tab:
        st.dataframe(table, width="stretch", hide_index=True)


def league_label(code: str) -> str:
    return f"{LEAGUE_NAMES.get(code, code)} ({code})"


# ------------------------------------------------------------------------- layout

features = get_features()
results_dir = os.environ.get("FOOTY_RESULTS_DIR", str(RESULTS_DIR))
results = get_results(results_dir)

st.title("Footy Prediction")
st.caption(
    "Pre-match H/D/A prediction for Europe's big-five leagues: leakage-safe rolling features "
    "built in DuckDB SQL, a LightGBM classifier chosen by season walk-forward CV, "
    "and a comparison against bookmaker-implied probabilities."
)

# One filter row above everything it scopes.
all_leagues = sorted(features["Div"].unique(), key=lambda code: list(LEAGUE_NAMES).index(code))
all_seasons = sorted(features["season"].unique())
filter_left, filter_right = st.columns([3, 2])
with filter_left:
    leagues = st.multiselect(
        "Leagues", all_leagues, default=all_leagues, format_func=league_label
    )
with filter_right:
    season_range = st.select_slider(
        "Seasons", options=all_seasons, value=(all_seasons[0], all_seasons[-1])
    )
if not leagues:
    st.info("Select at least one league.")
    st.stop()

selected_seasons = [s for s in all_seasons if season_range[0] <= s <= season_range[1]]
filtered = features[features["Div"].isin(leagues) & features["season"].isin(selected_seasons)]

if results is None:
    st.warning(
        "No saved model results found in `results/`. Run `python scripts/train_model.py` "
        "(about 35 minutes) to populate the model tabs. Data tabs below still work."
    )
    scored = pd.DataFrame()
else:
    scored_all = add_scores(results["predictions"])
    scored = scored_all[
        scored_all["Div"].isin(leagues) & scored_all["season"].isin(selected_seasons)
    ]

tab_overview, tab_models, tab_holdout, tab_matches, tab_teams = st.tabs(
    ["Overview", "Model comparison", "Model vs market", "Match explorer", "Team form"]
)

# ----------------------------------------------------------------------- overview
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Matches", f"{len(filtered):,}")
    c2.metric("Seasons", len(selected_seasons))
    c3.metric("Leagues", len(leagues))
    c4.metric("Home win rate", f"{(filtered['FTR'] == 'H').mean():.1%}")

    st.subheader("How matches end, by league")
    shares = (
        filtered.groupby("Div")["FTR"].value_counts(normalize=True).unstack(fill_value=0)
        .reindex(columns=list(OUTCOMES), fill_value=0)
        .reindex([code for code in all_leagues if code in leagues])
    )
    fig = go.Figure()
    for outcome, label in OUTCOMES.items():
        fig.add_bar(
            y=[LEAGUE_NAMES[c] for c in shares.index], x=shares[outcome], name=label,
            orientation="h",
            # 2px background-coloured gap between stacked segments.
            marker=dict(color=OUTCOME_COLORS[outcome], line=dict(width=2, color=PALETTE["background"])),
            text=[f"{v:.0%}" for v in shares[outcome]], textposition="inside",
            insidetextanchor="middle",
            hovertemplate="%{y}<br>" + label + ": %{x:.1%}<extra></extra>",
        )
    fig.update_layout(barmode="stack", legend_traceorder="normal")
    style(fig, height=300, x_title="Share of matches")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(tickformat=".0%", range=[0, 1])
    table = (shares * 100).round(1).rename(columns=OUTCOMES).reset_index()
    table["Div"] = table["Div"].map(LEAGUE_NAMES)
    show_chart(fig, table.rename(columns={"Div": "League"}), "outcome_shares")

    st.subheader("Matches per season")
    counts = filtered.groupby(["season", "Div"]).size().unstack(fill_value=0)
    counts.columns = [LEAGUE_NAMES[c] for c in counts.columns]
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.sum(axis=1), marker=dict(color=PALETTE["series"][0], cornerradius=4),
        hovertemplate="%{x}: %{y} matches<extra></extra>",
    ))
    style(fig, height=280, y_title="Matches")
    show_chart(fig, counts.assign(Total=counts.sum(axis=1)).reset_index(), "season_counts")
    st.caption(
        "The latest season is in progress. Features only use matches played before kickoff "
        "(SQL window frames ending one match earlier), so a team's first match has no history."
    )

# --------------------------------------------------------------- model comparison
with tab_models:
    if results is None:
        st.info("Run `python scripts/train_model.py` to see model results.")
    else:
        grouping_cv = results["grouping_cv"]
        run_info = results["run_info"]
        st.markdown(
            "Each feature grouping is tuned with **nested expanding-window CV**: every outer "
            "fold trains on all earlier seasons and validates on the next one, with feature "
            "selection done inside the training seasons only. Lower log loss is better."
        )
        best_name = run_info["best_grouping"]
        ordered = grouping_cv.sort_values("best_mean_log_loss")
        colors = [MODEL_COLOR if g == best_name else PALETTE["muted"] for g in ordered["grouping"]]
        # A dot plot: log-loss differences are ~0.01, which bars from zero would hide.
        fig = go.Figure(go.Scatter(
            x=ordered["best_mean_log_loss"], y=ordered["grouping"], mode="markers+text",
            marker=dict(color=colors, size=14, line=dict(width=2, color=PALETTE["background"])),
            text=[f"{v:.4f}" for v in ordered["best_mean_log_loss"]], textposition="middle right",
            hovertemplate="%{y}<br>Mean CV log loss: %{x:.4f}<extra></extra>",
        ))
        lo, hi = ordered["best_mean_log_loss"].min(), ordered["best_mean_log_loss"].max()
        pad = max((hi - lo) * 0.6, 0.004)
        style(fig, height=260, x_title="Mean CV log loss (best hyperparameters)")
        fig.update_xaxes(range=[lo - pad, hi + pad])
        fig.update_yaxes(autorange="reversed", showgrid=True)
        table = ordered[["grouping", "best_mean_log_loss", "best_mean_brier_score", "selected_features"]].copy()
        table["selected_features"] = table["selected_features"].map(", ".join)
        # Rows with any missing grouping feature are dropped before CV, so groupings can see
        # different matches (e.g. Referee is only recorded for the Premier League).
        table["matches_used"] = [
            len(features.dropna(subset=["FTR", "season", *FEATURE_GROUPINGS.get(g, [])]))
            if g in FEATURE_GROUPINGS else None
            for g in table["grouping"]
        ]
        show_chart(fig, table, "grouping_bar")
        st.caption(f"Highlighted: **{best_name}**, the grouping used for the holdout evaluation.")
        if table["matches_used"].nunique() > 1:
            st.warning(
                "Groupings were scored on different numbers of matches (see **matches_used** in "
                "the table view), because rows with a missing feature are dropped. Groupings that "
                "include `Referee` only cover the Premier League, so their log loss is not "
                "directly comparable with the others."
            )

        st.subheader("Log loss by validation season")
        fig = go.Figure()
        rows = []
        for i, row in enumerate(grouping_cv.itertuples()):
            fig.add_scatter(
                x=row.validation_seasons, y=row.fold_log_losses, name=row.grouping,
                mode="lines+markers", line=dict(width=2, color=PALETTE["series"][i % 4]),
                marker=dict(size=8),
                hovertemplate=row.grouping + "<br>%{x}: %{y:.4f}<extra></extra>",
            )
            rows += [
                {"grouping": row.grouping, "validation_season": s, "log_loss": round(v, 4)}
                for s, v in zip(row.validation_seasons, row.fold_log_losses)
            ]
        style(fig, height=340, y_title="Log loss", x_title="Validation season")
        fig.update_layout(hovermode="x unified")
        show_chart(fig, pd.DataFrame(rows), "fold_lines")

        with st.expander("Best hyperparameters per grouping"):
            params = pd.DataFrame(list(grouping_cv["best_params"]), index=grouping_cv["grouping"])
            st.dataframe(params, width="stretch")

# ---------------------------------------------------------------- model vs market
with tab_holdout:
    if results is None:
        st.info("Run `python scripts/train_model.py` to see model results.")
    elif scored.empty:
        st.info("No holdout matches in the selected leagues/seasons. The holdout is the "
                "most recent 20% of matches, so include the latest seasons.")
    else:
        run_info = results["run_info"]
        st.markdown(
            f"Holdout: the **most recent {1 - run_info['split_fraction']:.0%} of matches** "
            f"({run_info['test_matches']:,} matches), never used to fit the final model. "
            f"The model ({run_info['best_grouping']}) and the bookmaker market are scored on "
            f"**exactly the same {len(scored):,} matches** in the current filter."
        )
        model_ll, market_ll = scored["model_log_loss"].mean(), scored["market_log_loss"].mean()
        model_br, market_br = scored["model_brier"].mean(), scored["market_brier"].mean()
        model_acc = (scored["model_pick"] == scored["FTR"]).mean()
        market_acc = (scored["market_pick"] == scored["FTR"]).mean()
        c1, c2, c3 = st.columns(3)
        c1.metric("Log loss: model", f"{model_ll:.4f}",
                  f"{model_ll - market_ll:+.4f} vs market ({market_ll:.4f})", delta_color="inverse")
        c2.metric("Brier score: model", f"{model_br:.4f}",
                  f"{model_br - market_br:+.4f} vs market ({market_br:.4f})", delta_color="inverse")
        c3.metric("Accuracy: model", f"{model_acc:.1%}",
                  f"{model_acc - market_acc:+.1%} vs market ({market_acc:.1%})")

        st.subheader("Calibration")
        outcome = st.radio(
            "Outcome", list(OUTCOMES), format_func=OUTCOMES.get, horizontal=True, key="calibration_outcome"
        )
        n_bins = 10
        actual = (scored["FTR"] == outcome).astype(float)
        fig = go.Figure()
        fig.add_scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
            line=dict(color=PALETTE["muted"], width=1), hoverinfo="skip",
        )
        calibration_rows = []
        for name, column, color in [
            ("Model", MODEL_COLUMNS[OUTCOME_INDEX[outcome]], MODEL_COLOR),
            ("Market", MARKET_COLUMNS[OUTCOME_INDEX[outcome]], MARKET_COLOR),
        ]:
            bins = pd.cut(scored[column], list(np.linspace(0, 1, n_bins + 1)), include_lowest=True)
            grouped = pd.DataFrame({"p": scored[column], "y": actual, "bin": bins}).groupby(
                "bin", observed=True
            ).agg(predicted=("p", "mean"), observed=("y", "mean"), matches=("y", "size"))
            grouped = grouped[grouped["matches"] >= 10]
            fig.add_scatter(
                x=grouped["predicted"], y=grouped["observed"], name=name, mode="lines+markers",
                line=dict(width=2, color=color), marker=dict(size=8),
                customdata=grouped["matches"],
                hovertemplate=name + "<br>Predicted %{x:.1%}<br>Observed %{y:.1%}"
                "<br>%{customdata} matches<extra></extra>",
            )
            calibration_rows += [
                {"source": name, "predicted": round(float(predicted), 3),
                 "observed": round(float(observed), 3), "matches": int(matches)}
                for predicted, observed, matches in grouped[["predicted", "observed", "matches"]].to_numpy()
            ]
        style(fig, height=400, y_title="Observed frequency", x_title="Predicted probability")
        fig.update_xaxes(tickformat=".0%", range=[0, 1])
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
        show_chart(fig, pd.DataFrame(calibration_rows), "calibration")
        st.caption("Points on the diagonal mean the probabilities are honest: events predicted "
                   "at 40% happen about 40% of the time. Bins with fewer than 10 matches are hidden.")

        st.subheader("Cumulative log-loss advantage over time")
        timeline = scored.sort_values(["kickoff", "match_id"])
        advantage = (timeline["market_log_loss"] - timeline["model_log_loss"]).cumsum()
        fig = go.Figure(go.Scatter(
            x=timeline["kickoff"], y=advantage, mode="lines", name="Market − model",
            line=dict(width=2, color=MODEL_COLOR),
            hovertemplate="%{x|%d %b %Y}<br>Cumulative advantage: %{y:.2f}<extra></extra>",
        ))
        fig.add_hline(y=0, line=dict(color=PALETTE["muted"], width=1))
        style(fig, height=320, y_title="Σ (market − model) log loss")
        show_chart(
            fig,
            pd.DataFrame({"kickoff": timeline["kickoff"], "cumulative_advantage": advantage.round(3)}),
            "cumulative",
        )
        st.caption("Rising = the model is beating the market; falling = the market is ahead.")

        st.subheader("By league")
        by_league = scored.groupby("Div")[["model_log_loss", "market_log_loss"]].mean()
        by_league = by_league.reindex([c for c in all_leagues if c in by_league.index])
        fig = go.Figure()
        league_names = [LEAGUE_NAMES[c] for c in by_league.index]
        # Connector per league so the model-vs-market gap reads at a glance.
        for name, lo_v, hi_v in zip(league_names, by_league.min(axis=1), by_league.max(axis=1)):
            fig.add_scatter(x=[lo_v, hi_v], y=[name, name], mode="lines", showlegend=False,
                            line=dict(color=PALETTE["muted"], width=2), hoverinfo="skip")
        for name, column, color in [("Model", "model_log_loss", MODEL_COLOR),
                                    ("Market", "market_log_loss", MARKET_COLOR)]:
            fig.add_scatter(
                x=by_league[column], y=league_names, name=name, mode="markers",
                marker=dict(color=color, size=12, line=dict(width=2, color=PALETTE["background"])),
                hovertemplate="%{y}<br>" + name + " log loss: %{x:.4f}<extra></extra>",
            )
        style(fig, height=300, x_title="Mean log loss (lower is better)")
        fig.update_yaxes(autorange="reversed")
        table = by_league.round(4).reset_index()
        table["Div"] = table["Div"].map(LEAGUE_NAMES)
        show_chart(fig, table.rename(columns={"Div": "League"}), "by_league")

# ------------------------------------------------------------------ match explorer
with tab_matches:
    if results is None:
        st.info("Run `python scripts/train_model.py` to see model results.")
    elif scored.empty:
        st.info("No holdout matches in the selected leagues/seasons.")
    else:
        teams = sorted(set(scored["HomeTeam"]) | set(scored["AwayTeam"]))
        team = st.selectbox("Team", ["All teams"] + teams, key="match_team")
        matches = scored if team == "All teams" else scored[
            (scored["HomeTeam"] == team) | (scored["AwayTeam"] == team)
        ]
        matches = matches.sort_values("kickoff", ascending=False)
        table = pd.DataFrame({
            "Kickoff": matches["kickoff"].dt.strftime("%Y-%m-%d %H:%M"),
            "League": matches["Div"].map(LEAGUE_NAMES),
            "Match": matches["HomeTeam"] + " vs " + matches["AwayTeam"],
            "Result": matches["FTR"].map(OUTCOMES),
            "Model H": matches["model_prob_H"], "Model D": matches["model_prob_D"],
            "Model A": matches["model_prob_A"],
            "Market H": matches["market_home_prob_fair"], "Market D": matches["market_draw_prob_fair"],
            "Market A": matches["market_away_prob_fair"],
            "Model log loss": matches["model_log_loss"], "Market log loss": matches["market_log_loss"],
        })
        percent = st.column_config.NumberColumn(format="percent")
        selection = st.dataframe(
            table, width="stretch", hide_index=True, height=320,
            on_select="rerun", selection_mode="single-row", key="match_table",
            column_config={
                c: (st.column_config.NumberColumn(format="%.3f") if c.endswith("log loss") else percent)
                for c in table.columns if c.startswith(("Model ", "Market "))
            },
        )
        chosen_rows = selection.selection.rows if selection else []
        if not chosen_rows:
            st.caption("Select a row to compare the model and market probabilities for that match.")
        else:
            match = matches.iloc[chosen_rows[0]]
            st.subheader(f"{match['HomeTeam']} vs {match['AwayTeam']}")
            st.caption(f"{LEAGUE_NAMES[match['Div']]} · {match['kickoff']:%d %b %Y %H:%M} · "
                       f"Result: **{OUTCOMES[match['FTR']]}**")
            labels = [f"{OUTCOMES[o]}{' ✓' if o == match['FTR'] else ''}" for o in OUTCOMES]
            fig = go.Figure()
            for name, columns, color in [("Model", MODEL_COLUMNS, MODEL_COLOR),
                                         ("Market", MARKET_COLUMNS, MARKET_COLOR)]:
                values = match[columns].to_numpy(dtype=float)
                fig.add_bar(
                    x=labels, y=values, name=name, marker=dict(color=color, cornerradius=4),
                    text=[f"{v:.0%}" for v in values], textposition="outside",
                    hovertemplate="%{x}<br>" + name + ": %{y:.1%}<extra></extra>",
                )
            style(fig, height=320, y_title="Probability")
            fig.update_yaxes(tickformat=".0%", range=[0, 1])
            show_chart(fig, pd.DataFrame({
                "Outcome": list(OUTCOMES.values()),
                "Model": match[MODEL_COLUMNS].to_numpy(dtype=float).round(3),
                "Market": match[MARKET_COLUMNS].to_numpy(dtype=float).round(3),
            }), "match_probs")

# ---------------------------------------------------------------------- team form
with tab_teams:
    st.markdown(
        "Pre-match rolling features for one team, as the model saw them before each kickoff."
    )
    teams = sorted(set(filtered["HomeTeam"]) | set(filtered["AwayTeam"]))
    default_team = teams.index("Arsenal") if "Arsenal" in teams else 0
    team = st.selectbox("Team", teams, index=default_team, key="form_team")
    home = filtered[filtered["HomeTeam"] == team].assign(
        venue="Home", opponent=lambda d: d["AwayTeam"], xg_last_5=lambda d: d["home_xg_last_5"],
        win_prob=lambda d: d["market_home_prob_fair"],
        points_venue_last_5=lambda d: d["home_points_home_last_5"],
    )
    away = filtered[filtered["AwayTeam"] == team].assign(
        venue="Away", opponent=lambda d: d["HomeTeam"], xg_last_5=lambda d: d["away_xg_last_5"],
        win_prob=lambda d: d["market_away_prob_fair"],
        points_venue_last_5=lambda d: d["away_points_away_last_5"],
    )
    team_matches = pd.concat([home, away]).sort_values("kickoff")
    team_matches["win_prob_avg_10"] = team_matches["win_prob"].rolling(10, min_periods=3).mean()


    def season_gaps(frame: pd.DataFrame) -> pd.DataFrame:
        """Insert an empty row between seasons so lines break over the off-season."""
        parts = []
        for _, season_rows in frame.groupby("season", sort=True):
            parts += [season_rows, season_rows.tail(1).assign(
                kickoff=season_rows["kickoff"].max() + pd.Timedelta(days=1),
                xg_last_5=np.nan, win_prob=np.nan, win_prob_avg_10=np.nan,
            )]
        return pd.concat(parts) if parts else frame


    plotted = season_gaps(team_matches)
    columns = ["kickoff", "season", "venue", "opponent", "FTR", "xg_last_5", "win_prob",
               "points_venue_last_5"]

    left, right = st.columns(2)
    with left:
        st.markdown("**xG per match, average of previous 5**")
        fig = go.Figure(go.Scatter(
            x=plotted["kickoff"], y=plotted["xg_last_5"], mode="lines",
            line=dict(width=2, color=PALETTE["series"][0]), name="xG last 5", connectgaps=False,
            customdata=np.stack([plotted["venue"], plotted["opponent"]], axis=-1),
            hovertemplate="%{x|%d %b %Y} · %{customdata[0]} vs %{customdata[1]}"
            "<br>xG last 5: %{y:.2f}<extra></extra>",
        ))
        style(fig, height=300, y_title="xG")
        st.plotly_chart(fig, width="stretch", theme="streamlit", key="team_xg")
    with right:
        st.markdown("**Market win probability before kickoff**")
        fig = go.Figure()
        fig.add_scatter(
            x=plotted["kickoff"], y=plotted["win_prob"], mode="lines", name="Each match",
            line=dict(width=1, color=PALETTE["muted"]), connectgaps=False,
            customdata=np.stack([plotted["venue"], plotted["opponent"]], axis=-1),
            hovertemplate="%{x|%d %b %Y} · %{customdata[0]} vs %{customdata[1]}"
            "<br>Win probability: %{y:.0%}<extra></extra>",
        )
        fig.add_scatter(
            x=plotted["kickoff"], y=plotted["win_prob_avg_10"], mode="lines",
            name="10-match average", line=dict(width=2, color=PALETTE["series"][0]),
            connectgaps=False, hovertemplate="10-match average: %{y:.0%}<extra></extra>",
        )
        style(fig, height=300, y_title="Probability")
        fig.update_yaxes(tickformat=".0%", range=[0, 1])
        st.plotly_chart(fig, width="stretch", theme="streamlit", key="team_prob")
    with st.expander("Table view"):
        st.dataframe(team_matches[columns].sort_values("kickoff", ascending=False),
                     width="stretch", hide_index=True)
