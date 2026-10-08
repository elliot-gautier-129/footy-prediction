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

from footy_prediction.backtest import STRATEGIES, attach_best_odds, simulate_strategy, summarize_bets
from footy_prediction.data_loading import load_features
from footy_prediction.feature_sets import FEATURE_GROUPINGS
from footy_prediction.paths import PROJECT_DIR, RESULTS_DIR
from footy_prediction.results import load_results

st.set_page_config(page_title="Footy Prediction", page_icon="⚽", layout="wide")

# Tabular numerals for st.metric values: the one piece of the type system without a
# native [theme] option. Scoped to Streamlit's stable data-testid hook, not a
# generated class name. (st.dataframe's grid is canvas-rendered, not real DOM text,
# so this can't reach table cells — there's no CSS hook for that.)
st.html('<style>[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }</style>')

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

# Validated categorical palette (light / dark steps of the same hues), aligned to the
# app's design tokens in .streamlit/config.toml. series[:3] are the fixed H/D/A and
# model/market colors (home/model=blue, draw/market=slate, away=emerald); series[3:]
# are only used by the 5-grouping model-comparison chart.
PALETTES = {
    "light": {
        "series": ["#3B82F6", "#94A3B8", "#10B981", "#6366F1", "#D97706"],
        "muted": "#CBD5E1",
        "grid": "#E2E8F0",
        "ink": "#64748B",
        "background": "#FFFFFF",
    },
    "dark": {
        "series": ["#3987E5", "#9AA8BD", "#199E70", "#8B7CF0", "#C98500"],
        "muted": "#475569",
        "grid": "#334155",
        "ink": "#94A3B8",
        "background": "#161B22",
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


CHART_FONT = "Inter, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"


def style(fig: go.Figure, height: int = 360, y_title: str = "", x_title: str = "") -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=64, r=16, t=40, b=48),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None),
        hoverlabel=dict(namelength=-1, font=dict(family=CHART_FONT)),
        font=dict(family=CHART_FONT, color=PALETTE["ink"], size=13),
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

LOGO_PATH = PROJECT_DIR / "footy-pred-logo.png"

all_leagues = sorted(features["Div"].unique(), key=lambda code: list(LEAGUE_NAMES).index(code))
all_seasons = sorted(features["season"].unique())
if "leagues" not in st.session_state:
    st.session_state["leagues"] = all_leagues
if "season_range" not in st.session_state:
    st.session_state["season_range"] = (all_seasons[0], all_seasons[-1])


def _sync_leagues(suffix: str) -> None:
    st.session_state["leagues"] = st.session_state[f"leagues_{suffix}"]


def _sync_season_range(suffix: str) -> None:
    st.session_state["season_range"] = st.session_state[f"season_range_{suffix}"]


def render_filter_toolbar(suffix: str):
    """Compact filters toolbar: a 'Filters' popover plus a muted active-filters summary.

    Streamlit runs every tab's body on every rerun (it doesn't tell the script which
    tab is visually active), so this is called once per tab, right under the tab bar,
    with a tab-specific widget key. Every instance mirrors the same two canonical
    session_state values (re-seeded here, kept current by the on_change callbacks
    above), so all six stay in sync and a change made on one tab is never lost when
    switching to another.
    """
    season_range = st.session_state["season_range"]
    st.session_state[f"leagues_{suffix}"] = st.session_state["leagues"]
    st.session_state[f"season_range_{suffix}"] = season_range
    toolbar_left, toolbar_right = st.columns([1, 5], vertical_alignment="center")
    with toolbar_left:
        with st.popover("Filters"):
            st.multiselect(
                "Leagues", all_leagues, key=f"leagues_{suffix}", format_func=league_label,
                on_change=_sync_leagues, args=(suffix,),
            )
            # select_slider silently drops range mode on interaction unless `value`
            # is passed explicitly alongside `key` (a Streamlit quirk, verified against
            # this installed version) -- logs one harmless one-time server-side warning
            # about the value/session_state overlap, never shown to the user.
            st.select_slider(
                "Seasons", options=all_seasons, value=season_range, key=f"season_range_{suffix}",
                on_change=_sync_season_range, args=(suffix,),
            )
    leagues_now = st.session_state["leagues"]
    season_range_now = st.session_state["season_range"]
    with toolbar_right:
        if not leagues_now:
            summary = "No leagues selected"
        elif len(leagues_now) == len(all_leagues):
            summary = "All leagues"
        elif len(leagues_now) <= 2:
            summary = ", ".join(league_label(c) for c in leagues_now)
        else:
            summary = f"{len(leagues_now)} leagues"
        st.caption(f"{summary} · {season_range_now[0]}–{season_range_now[1]}")
    return leagues_now, season_range_now


def selected_seasons_in(season_range) -> list:
    return [s for s in all_seasons if season_range[0] <= s <= season_range[1]]


def compute_filtered(leagues, selected_seasons) -> pd.DataFrame:
    return features[features["Div"].isin(leagues) & features["season"].isin(selected_seasons)]


def compute_scored(leagues, selected_seasons) -> pd.DataFrame:
    if scored_all is None:
        return pd.DataFrame()
    return scored_all[scored_all["Div"].isin(leagues) & scored_all["season"].isin(selected_seasons)]


if results is None:
    st.warning(
        "No saved model results found in `results/`. Run `python scripts/train_model.py` "
        "(about 35 minutes) to populate the model tabs. Data tabs below still work."
    )
    scored_all = None
else:
    scored_all = add_scores(results["predictions"])

# Sticky top nav: logo + tabs in one row, pinned to the top of the scroll
# container on scroll. `position: sticky` (not `fixed`) is used deliberately --
# it reserves its own space in normal document flow, so it can never overlap
# the content below it (no manual spacer/padding-top hack needed), while still
# producing the same "stays visible as you scroll" behavior.
#
# The container itself (`.st-key-topnav`, via `st.container(key=...)` --
# Streamlit's documented, stable hook for targeting a specific container from
# CSS, not a generated/unstable class) can't be the sticky element: Streamlit
# places a tab's *content* whereever `st.tabs()` was called, not wherever the
# later `with tab_x:` block is written, so this container necessarily holds
# the full height of every tab's content, not just the bar. Instead, its two
# direct children -- the logo's element container and the tab bar's
# `[role="tablist"]` -- are each individually made sticky, which is enough:
# they're flex siblings in this `horizontal=True` container, so they still
# render side by side and stick together as one row, while the (non-sticky)
# tab content scrolls normally beneath them.
_nav_bg = "#FFFFFF" if _theme_type() == "light" else "#161B22"
_nav_border = "#E2E8F0" if _theme_type() == "light" else "#2A2F3A"
_tab_hover = "rgba(15, 23, 42, 0.05)" if _theme_type() == "light" else "rgba(226, 232, 240, 0.08)"
_tab_ink = "#475569" if _theme_type() == "light" else "#94A3B8"
st.html(f"""
<style>
/* Streamlit's default block-container gutters (6rem top, 80px each side on desktop)
leave a large, unbalanced gap around the nav bar and every section below it --
nothing to do with local vs. deployed, it's the same in both. Tightened uniformly
so the nav card and the page content both sit closer to the viewport edge. */
.block-container {{
    padding-top: 1.75rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}}

.st-key-topnav > [data-testid="stElementContainer"],
.st-key-topnav > [data-testid="stTabs"] > div > [role="tablist"] {{
    /* top: 60px, not 0 -- Streamlit's own app header (the Deploy/menu bar) is a
    fixed-position element 60px tall with a very high z-index that's always
    present at the very top of the viewport; sticking to top:0 puts this nav
    bar directly underneath it, invisible. */
    position: sticky;
    top: 60px;
    z-index: 999;
    background-color: {_nav_bg};
}}
.st-key-topnav > [data-testid="stElementContainer"] {{
    padding: 0.5rem 1rem 0.5rem 0;
    border-bottom: 1px solid {_nav_border};
}}
.st-key-topnav > [data-testid="stTabs"] > div > [role="tablist"] {{
    padding-top: 1.1rem;
    padding-bottom: 0.75rem;
    margin-bottom: 1rem;
    border-bottom: 1px solid {_nav_border};
    gap: 0.25rem;
}}
/* Each tab as its own chip: padding + a hover fill is what actually separates
them -- the flex `gap` alone reads as plain text floating in a row. */
.st-key-topnav [role="tab"] {{
    padding: 0.55rem 1rem !important;
    border-radius: 8px;
    font-size: 0.95rem;
    font-weight: 500;
    transition: background-color 0.15s ease, color 0.15s ease;
}}
.st-key-topnav [role="tab"]:not([aria-selected="true"]) {{
    color: {_tab_ink};
}}
.st-key-topnav [role="tab"]:hover {{
    background-color: {_tab_hover};
}}
.st-key-topnav [role="tab"][aria-selected="true"] {{
    font-weight: 600;
}}
/* Narrow screens: the row wraps (logo above tabs), and both pieces sticking to
the same top offset would overlap. Keep the tab bar -- the interactive part --
sticky, and let the logo scroll away normally instead. */
@media (max-width: 640px) {{
    .st-key-topnav > [data-testid="stElementContainer"] {{
        position: static;
    }}
}}
</style>
""")
with st.container(key="topnav", horizontal=True, vertical_alignment="top"):
    st.image(str(LOGO_PATH), width=170)
    tab_overview, tab_models, tab_holdout, tab_backtest, tab_matches, tab_teams = st.tabs(
        ["Overview", "Model comparison", "Model vs market", "Backtest",
         "Match explorer", "Team form"]
    )

# ----------------------------------------------------------------------- overview
with tab_overview:
    leagues, season_range = render_filter_toolbar("overview")
    if not leagues:
        st.info("Select at least one league.")
    else:
        selected_seasons = selected_seasons_in(season_range)
        filtered = compute_filtered(leagues, selected_seasons)
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Matches", f"{len(filtered):,}")
            c2.metric("Seasons", len(selected_seasons))
            c3.metric("Leagues", len(leagues))
            c4.metric("Home win rate", f"{(filtered['FTR'] == 'H').mean():.1%}")

        st.subheader("How matches end, by league")
        with st.container(border=True):
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
        with st.container(border=True):
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
    leagues, season_range = render_filter_toolbar("models")
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
        with st.container(border=True):
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
            # No horizontal gridlines: they would run through the value labels.
            fig.update_yaxes(autorange="reversed", showgrid=False)
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
            used = table["matches_used"].dropna()
            if len(used) and used.min() < 0.9 * used.max():
                st.warning(
                    "Groupings were scored on quite different numbers of matches (see "
                    "**matches_used** in the table view), because rows with a missing feature are "
                    "dropped. Compare their log loss with care."
                )
            elif used.nunique() > 1:
                st.caption(
                    "Rows with a missing feature are dropped, so groupings see slightly different "
                    "numbers of matches (early-season games have no form history yet); see "
                    "**matches_used** in the table view."
                )

        st.subheader("Log loss by validation season")
        with st.container(border=True):
            fig = go.Figure()
            rows = []
            for i, row in enumerate(grouping_cv.itertuples()):
                # Fixed palette order; never cycle colours (extra groupings fall back to grey).
                color = PALETTE["series"][i] if i < len(PALETTE["series"]) else PALETTE["muted"]
                fig.add_scatter(
                    x=row.validation_seasons, y=row.fold_log_losses, name=row.grouping,
                    mode="lines+markers", line=dict(width=2, color=color),
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
    leagues, season_range = render_filter_toolbar("holdout")
    selected_seasons = selected_seasons_in(season_range)
    scored = compute_scored(leagues, selected_seasons)
    if not leagues:
        st.info("Select at least one league.")
    elif results is None:
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
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            c1.metric("Log loss: model", f"{model_ll:.4f}",
                      f"{model_ll - market_ll:+.4f} vs market ({market_ll:.4f})", delta_color="inverse")
            c2.metric("Brier score: model", f"{model_br:.4f}",
                      f"{model_br - market_br:+.4f} vs market ({market_br:.4f})", delta_color="inverse")
            c3.metric("Accuracy: model", f"{model_acc:.1%}",
                      f"{model_acc - market_acc:+.1%} vs market ({market_acc:.1%})")

        st.subheader("Calibration")
        with st.container(border=True):
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
        with st.container(border=True):
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
        with st.container(border=True):
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

# -------------------------------------------------------------------------- backtest
with tab_backtest:
    leagues, season_range = render_filter_toolbar("backtest")
    selected_seasons = selected_seasons_in(season_range)
    scored = compute_scored(leagues, selected_seasons)
    if not leagues:
        st.info("Select at least one league.")
    elif results is None:
        st.info("Run `python scripts/train_model.py` to see backtest results.")
    elif scored.empty:
        st.info("No holdout matches in the selected leagues/seasons.")
    else:
        st.markdown(
            "Flat-stake betting strategies on the holdout matches, using the **best pre-match "
            "decimal odds seen across tracked bookmakers** (football-data.co.uk's `Max` columns) "
            "as a proxy for shopping around multiple platforms for the best price. This ignores "
            "transaction costs, stake limits and odds drift after the snapshot was taken, and the "
            "model vs market tab shows the market is ahead on average, so these strategies are not "
            "expected to be profitable yet — treat this as instrumentation to improve on, not a "
            "trading recommendation."
        )
        backtest_scored = attach_best_odds(scored, features)

        with st.container(border=True):
            control_left, control_mid, control_right = st.columns(3)
            with control_left:
                stake = st.number_input("Stake per bet", min_value=1.0, value=10.0, step=1.0)
            with control_mid:
                edge_threshold = st.slider(
                    "Value edge: minimum model − market probability gap", 0.0, 0.30, 0.05, 0.01,
                )
            with control_right:
                ev_threshold = st.slider(
                    "Positive EV: minimum model_prob × best_odds − 1", -0.20, 0.50, 0.0, 0.01,
                )
        strategy_params = {
            "Model favourite": {},
            "Value edge": {"edge_threshold": edge_threshold},
            "Positive expected value": {"ev_threshold": ev_threshold},
        }

        bets_by_strategy = {
            name: simulate_strategy(backtest_scored, name, stake=stake, **strategy_params[name])
            for name in STRATEGIES
        }

        st.subheader("Strategy summary")
        with st.container(border=True):
            summary_rows = []
            for name, bets in bets_by_strategy.items():
                summary = summarize_bets(bets)
                summary_rows.append({"Strategy": name, **summary})
            summary_table = pd.DataFrame(summary_rows).rename(columns={
                "bets": "Bets", "win_rate": "Win rate", "total_staked": "Total staked",
                "total_profit": "Total profit", "roi": "ROI", "max_drawdown": "Max drawdown",
            })
            st.dataframe(
                summary_table, width="stretch", hide_index=True,
                column_config={
                    "Bets": st.column_config.NumberColumn(alignment="right"),
                    "Win rate": st.column_config.NumberColumn(format="percent", alignment="right"),
                    "ROI": st.column_config.NumberColumn(format="percent", alignment="right"),
                    "Total staked": st.column_config.NumberColumn(format="%.2f", alignment="right"),
                    "Total profit": st.column_config.NumberColumn(format="%.2f", alignment="right"),
                    "Max drawdown": st.column_config.NumberColumn(format="%.2f", alignment="right"),
                },
            )
            with st.expander("What each strategy does"):
                for name, params in strategy_params.items():
                    if params:
                        detail = ", ".join(f"{key}={value:.0%}" for key, value in params.items())
                        st.caption(f"**{name}**: {STRATEGIES[name]['description']} ({detail})")
                    else:
                        st.caption(f"**{name}**: {STRATEGIES[name]['description']}")

        st.subheader("Cumulative P&L over time")
        with st.container(border=True):
            fig = go.Figure()
            pnl_table_parts = []
            for i, (name, bets) in enumerate(bets_by_strategy.items()):
                color = PALETTE["series"][i] if i < len(PALETTE["series"]) else PALETTE["muted"]
                if bets.empty:
                    continue
                fig.add_scatter(
                    x=bets["kickoff"], y=bets["cumulative_profit"], name=name, mode="lines",
                    line=dict(width=2, color=color),
                    hovertemplate=name + "<br>%{x|%d %b %Y}<br>Cumulative P&L: %{y:.2f}<extra></extra>",
                )
                pnl_table_parts.append(pd.DataFrame({
                    "kickoff": bets["kickoff"], "strategy": name,
                    "cumulative_profit": bets["cumulative_profit"],
                }))
            fig.add_hline(y=0, line=dict(color=PALETTE["muted"], width=1))
            style(fig, height=360, y_title=f"Cumulative profit (stake {stake:g}/bet)")
            # Long format (not pivoted on kickoff) because several matches can share a kickoff time.
            pnl_table = pd.concat(pnl_table_parts, ignore_index=True) if pnl_table_parts else pd.DataFrame()
            show_chart(fig, pnl_table, "backtest_pnl")
            st.caption(
                "Each line only advances on the dates that strategy placed a bet, so lines for "
                "stricter strategies (fewer bets) are sparser."
            )

        st.subheader("Bet log")
        with st.container(border=True):
            log_strategy = st.selectbox("Strategy", list(STRATEGIES), key="backtest_log_strategy")
            log = bets_by_strategy[log_strategy]
            if log.empty:
                st.info("No bets were placed by this strategy with the current thresholds.")
            else:
                log_table = pd.DataFrame({
                    "Kickoff": log["kickoff"].dt.strftime("%Y-%m-%d %H:%M"),
                    "League": log["Div"].map(LEAGUE_NAMES),
                    "Match": log["HomeTeam"] + " vs " + log["AwayTeam"],
                    "Pick": log["pick"].map(OUTCOMES),
                    "Result": log["FTR"].map(OUTCOMES),
                    "Odds": log["odds"],
                    "Won": log["won"],
                    "Profit": log["profit"],
                    "Cumulative P&L": log["cumulative_profit"],
                }).sort_values("Kickoff", ascending=False)
                st.dataframe(
                    log_table, width="stretch", hide_index=True, height=320,
                    column_config={
                        "Odds": st.column_config.NumberColumn(format="%.2f", alignment="right"),
                        "Profit": st.column_config.NumberColumn(format="%.2f", alignment="right"),
                        "Cumulative P&L": st.column_config.NumberColumn(format="%.2f", alignment="right"),
                    },
                )

# ------------------------------------------------------------------ match explorer
with tab_matches:
    leagues, season_range = render_filter_toolbar("matches")
    selected_seasons = selected_seasons_in(season_range)
    scored = compute_scored(leagues, selected_seasons)
    if not leagues:
        st.info("Select at least one league.")
    elif results is None:
        st.info("Run `python scripts/train_model.py` to see model results.")
    elif scored.empty:
        st.info("No holdout matches in the selected leagues/seasons.")
    else:
        teams = sorted(set(scored["HomeTeam"]) | set(scored["AwayTeam"]))
        with st.container(border=True):
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
            percent = st.column_config.NumberColumn(format="percent", alignment="right")
            selection = st.dataframe(
                table, width="stretch", hide_index=True, height=320,
                on_select="rerun", selection_mode="single-row", key="match_table",
                column_config={
                    c: (st.column_config.NumberColumn(format="%.3f", alignment="right")
                        if c.endswith("log loss") else percent)
                    for c in table.columns if c.startswith(("Model ", "Market "))
                },
            )
            chosen_rows = selection.selection.rows if selection else []
            if not chosen_rows:
                st.caption("Select a row to compare the model and market probabilities for that match.")
        if chosen_rows:
            match = matches.iloc[chosen_rows[0]]
            st.subheader(f"{match['HomeTeam']} vs {match['AwayTeam']}")
            with st.container(border=True):
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
    leagues, season_range = render_filter_toolbar("teams")
    if not leagues:
        st.info("Select at least one league.")
    else:
        selected_seasons = selected_seasons_in(season_range)
        filtered = compute_filtered(leagues, selected_seasons)
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

        with st.container(border=True):
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
