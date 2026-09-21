"""
fbref_scraper.py

A general-purpose scraper for match-level football data from FBref.com,
built on the pipeline described in:
https://medium.com/@henrik.schjoth/scraping-fbref-creating-a-pipeline-f5c9c23ba9da

Output
------
One row per TEAM per MATCH (so a single fixture produces two rows, one
from each side's perspective), with these columns:

    Season, Team, Date, Time, Comp, Round, Day, Venue, Result, GF, GA,
    Opponent, xG, xGA, Poss, Attendance, Captain, Formation,
    Opp Formation, Referee, Match Report, Notes

This mirrors FBref's own "Scores & Fixtures" match log that appears on
every team's season page, with `Season` and `Team` added so rows from
different teams/seasons can be safely concatenated.

How it works
------------
1. For a given competition + season, fetch the competition's standings
   page and collect every team's page URL from the league table.
2. For each team, fetch that team's season page, which includes an
   embedded "Scores & Fixtures" table covering ALL competitions the team
   played that season (league, cup, continental, etc.), tagged by a
   `Comp` column.
3. Concatenate every team's table, then filter down to:
     - rows belonging to the requested competition (via `Comp`)
     - rows that are actually played (a Score/Result exists)
4. Sleep between requests to respect FBref's published bot policy.

Rate limits (see https://www.sports-reference.com/bot-traffic.html)
--------------------------------------------------------------------
FBref and Stathead explicitly state that making "more often than ten
requests in a minute" gets a session temporarily blocked (up to a day).
This scraper defaults to one request every 7 seconds (~8.5 req/min) to
stay safely under that limit with margin for retries. Do not lower this
below ~6.5s unless you enjoy getting temp-banned.

How far back the data goes
---------------------------
- Basic results (teams, score, date, venue, attendance, referee) exist
  for many top leagues going back decades (English top-flight results
  go back to the 1800s on FBref's historical pages).
- The extra columns this scraper targets - xG, xGA, Poss, Formation,
  Opp Formation, Captain - are FBref's "advanced" stats. For the Big 5
  European leagues (Premier League, La Liga, Serie A, Bundesliga,
  Ligue 1) these exist from the **2017-18 season onward**. Older
  seasons, and most leagues outside the top ~20-30 competitions
  worldwide, will come back with those columns empty (NaN) - that is a
  gap in FBref's own data, not a bug in this scraper.
- There is no hard cap on how many matches you can pull other than time
  and FBref's rate limit: every additional team-season you request costs
  one HTTP request. E.g. Premier League 2017-18 through 2025-26 (9
  seasons x 20 teams) is ~180 requests -> roughly 20-25 minutes at the
  default polite delay.

Dependencies: requests, beautifulsoup4, lxml, pandas
"""

from __future__ import annotations

import io
import re
import time
import random
import logging
from dataclasses import dataclass
from typing import Iterable, Optional, Union

import requests
import pandas as pd
from bs4 import BeautifulSoup, Comment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("fbref_scraper")

BASE = "https://fbref.com"

HEADERS = {
    # A normal browser UA. FBref sits behind Cloudflare and can still
    # challenge/ block scripted clients regardless of this; see README.
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Expected column set of the per-team "Scores & Fixtures" match log.
EXPECTED_COLUMNS = [
    "Date", "Time", "Comp", "Round", "Day", "Venue", "Result", "GF", "GA",
    "Opponent", "xG", "xGA", "Poss", "Attendance", "Captain", "Formation",
    "Opp Formation", "Referee", "Match Report", "Notes",
]

# A starter map of common competitions -> (fbref_id, url_slug, season_format).
# season_format:
#   "range" -> season strings look like "2023-2024" (most European leagues)
#   "year"  -> season strings look like "2024" (e.g. Brazil, MLS, Liga MX)
# This is NOT exhaustive. Pass an explicit (id, slug, season_format) tuple,
# or use resolve_competition()/list_competitions() to look any other
# competition up dynamically from FBref's own competitions directory.
COMPETITIONS = {
    "Premier League": (9, "Premier-League", "range"),
    "Championship": (10, "Championship", "range"),
    "La Liga": (12, "La-Liga", "range"),
    "Serie A": (11, "Serie-A", "range"),                 # Italy
    "Bundesliga": (20, "Bundesliga", "range"),
    "Ligue 1": (13, "Ligue-1", "range"),
    "Primeira Liga": (32, "Primeira-Liga", "range"),
    "Eredivisie": (23, "Eredivisie", "range"),
    "Champions League": (8, "Champions-League", "range"),
    "Europa League": (19, "Europa-League", "range"),
    "MLS": (22, "Major-League-Soccer", "year"),
    "Brazilian Serie A": (24, "Serie-A", "year"),
    "Liga MX": (31, "Liga-MX", "year"),
}


class RateLimiter:
    """Simple polite-delay + jitter rate limiter for FBref requests."""

    def __init__(self, delay: float = 7.0, jitter: float = 1.5):
        self.delay = delay
        self.jitter = jitter
        self._last = 0.0

    def wait(self):
        now = time.time()
        elapsed = now - self._last
        target = self.delay + random.uniform(0, self.jitter)
        if elapsed < target:
            time.sleep(target - elapsed)
        self._last = time.time()


_session = requests.Session()
_session.headers.update(HEADERS)
_limiter = RateLimiter()


def _get(url: str, max_retries: int = 3) -> str:
    """Fetch a URL politely, retrying on 429/403 with backoff."""
    for attempt in range(1, max_retries + 1):
        _limiter.wait()
        resp = _session.get(url, timeout=30)
        if resp.status_code == 200:
            return resp.text
        if resp.status_code in (429, 403):
            backoff = 30 * attempt
            log.warning(
                "Got %s for %s (attempt %d/%d). Backing off %ds - FBref's "
                "bot policy blocks sessions making >10 req/min, and "
                "Cloudflare can also 403 scripted clients outright.",
                resp.status_code, url, attempt, max_retries, backoff,
            )
            time.sleep(backoff)
            continue
        resp.raise_for_status()
    raise RuntimeError(f"Failed to fetch {url} after {max_retries} attempts")


def _tables_including_comments(html: str) -> list:
    """
    FBref hides some tables inside HTML comments so naive parsers miss
    them. This pulls every <table> out of both the live DOM and any
    commented-out blocks, and parses them all with pandas.
    """
    soup = BeautifulSoup(html, "lxml")
    frames = []

    def _parse_tables_in(fragment_soup):
        for table in fragment_soup.find_all("table"):
            try:
                df = pd.read_html(io.StringIO(str(table)))[0]
                frames.append((table.get("id"), df))
            except ValueError:
                continue

    _parse_tables_in(soup)
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if "<table" in comment:
            _parse_tables_in(BeautifulSoup(comment, "lxml"))
    return frames


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[-1] if not str(c[-1]).startswith("Unnamed") else c[0]
                      for c in df.columns]
    return df


def resolve_competition(name_or_id: Union[str, int, tuple]) -> tuple:
    """
    Resolve a competition to (fbref_id, url_slug, season_format).

    Accepts:
      - a name already in COMPETITIONS (case-insensitive)
      - an explicit (id, slug, season_format) tuple you looked up yourself
      - a bare fbref competition id (int/str) - slug/season_format will
        be guessed by fetching the competition's own page
    """
    if isinstance(name_or_id, tuple):
        return name_or_id

    if isinstance(name_or_id, str) and name_or_id in COMPETITIONS:
        return COMPETITIONS[name_or_id]

    for key, val in COMPETITIONS.items():
        if key.lower() == str(name_or_id).lower():
            return val

    # Fall back: treat it as a raw competition id and discover the slug
    # by requesting FBref's competition page directly.
    comp_id = int(name_or_id)
    html = _get(f"{BASE}/en/comps/{comp_id}/")
    soup = BeautifulSoup(html, "lxml")
    title = soup.find("h1")
    if not title:
        raise ValueError(f"Could not resolve competition id {comp_id}")
    slug = title.get_text(strip=True).replace(" ", "-")
    # Guess season format by checking whether any season link on the page
    # looks like YYYY-YYYY (range) or plain YYYY (year).
    season_links = [a["href"] for a in soup.find_all("a", href=True)
                    if f"/comps/{comp_id}/" in a["href"]]
    season_format = "year"
    for href in season_links:
        m = re.search(r"/comps/\d+/(\d{4}-\d{4}|\d{4})/", href)
        if m:
            season_format = "range" if "-" in m.group(1) else "year"
            break
    return comp_id, slug, season_format


def list_competitions() -> pd.DataFrame:
    """
    Scrape FBref's competitions directory (https://fbref.com/en/comps/)
    and return every competition FBref lists, with id, name and region -
    useful for finding the id/slug of a competition not in COMPETITIONS.
    """
    html = _get(f"{BASE}/en/comps/")
    soup = BeautifulSoup(html, "lxml")
    rows = []
    for a in soup.select("table a[href*='/en/comps/']"):
        m = re.match(r"^/en/comps/(\d+)/(?:history/)?([\w%-]+)", a["href"])
        if not m:
            continue
        rows.append({"id": int(m.group(1)), "slug": m.group(2), "name": a.get_text(strip=True)})
    df = pd.DataFrame(rows).drop_duplicates(subset="id").reset_index(drop=True)
    return df


def _standings_url(comp_id: int, slug: str, season: str) -> str:
    return f"{BASE}/en/comps/{comp_id}/{season}/{season}-{slug}-Stats"


def _team_urls_for_season(comp_id: int, slug: str, season: str) -> list:
    """Return [(team_id, team_name, team_url), ...] for a competition/season."""
    url = _standings_url(comp_id, slug, season)
    html = _get(url)
    soup = BeautifulSoup(html, "lxml")
    teams = {}
    for a in soup.find_all("a", href=True):
        m = re.match(rf"^/en/squads/([0-9a-f]{{8}})/{season}/", a["href"])
        if m:
            team_id = m.group(1)
            team_name = a.get_text(strip=True)
            if team_name and team_id not in teams:
                teams[team_id] = (team_id, team_name, BASE + a["href"])
    if not teams:
        raise RuntimeError(
            f"No team links found on {url} - the competition/season "
            "combination may be wrong, or FBref changed its page layout."
        )
    return list(teams.values())


def _team_match_log(team_id: str, team_name: str, season: str) -> pd.DataFrame:
    """Fetch and parse one team's Scores & Fixtures table for a season."""
    url = f"{BASE}/en/squads/{team_id}/{season}/{team_name.replace(' ', '-')}-Stats"
    html = _get(url)
    for table_id, df in _tables_including_comments(html):
        df = _flatten_columns(df.copy())
        cols = set(df.columns)
        if {"Date", "Opponent", "Result"}.issubset(cols):
            df["Team"] = team_name
            return df
    raise RuntimeError(f"Could not find a Scores & Fixtures table at {url}")


def get_match_data(
    competition: Union[str, int, tuple],
    seasons: Union[str, Iterable[str]],
    delay: float = 7.0,
    played_only: bool = True,
) -> pd.DataFrame:
    """
    Pull match-level data for any FBref competition and any season(s).

    Parameters
    ----------
    competition : str | int | tuple
        A name from COMPETITIONS (e.g. "Premier League"), a raw FBref
        competition id (e.g. 9), or an explicit (id, slug, season_format)
        tuple.
    seasons : str | iterable of str
        One season (e.g. "2023-2024" or "2024") or a list of seasons.
        Use the same format FBref uses for that competition - most
        European leagues are "YYYY-YYYY"; some others (MLS, Brazil,
        Liga MX, ...) are just "YYYY".
    delay : float
        Seconds to wait between requests (default 7s, ~8.5 req/min,
        under FBref's 10 req/min bot-traffic limit). Do not go much
        lower than this.
    played_only : bool
        Drop fixtures that haven't been played yet (no result).

    Returns
    -------
    pandas.DataFrame
        One row per team per match, columns:
        Season, Team, Date, Time, Comp, Round, Day, Venue, Result, GF,
        GA, Opponent, xG, xGA, Poss, Attendance, Captain, Formation,
        Opp Formation, Referee, Match Report, Notes
    """
    global _limiter
    _limiter = RateLimiter(delay=delay)

    comp_id, slug, _season_format = resolve_competition(competition)
    if isinstance(seasons, str):
        seasons = [seasons]

    all_frames = []
    for season in seasons:
        log.info("Fetching team list for competition id=%s season=%s", comp_id, season)
        teams = _team_urls_for_season(comp_id, slug, season)
        log.info("Found %d teams for %s", len(teams), season)

        for team_id, team_name, _url in teams:
            log.info("  -> %s (%s)", team_name, season)
            try:
                df = _team_match_log(team_id, team_name, season)
            except Exception as exc:  # keep going even if one team fails
                log.error("Failed to scrape %s %s: %s", team_name, season, exc)
                continue
            df["Season"] = season
            all_frames.append(df)

    if not all_frames:
        return pd.DataFrame(columns=["Season", "Team"] + EXPECTED_COLUMNS)

    combined = pd.concat(all_frames, ignore_index=True, sort=False)

    # Keep only the expected columns that actually exist (FBref layout
    # varies slightly by competition/era), in a consistent order.
    ordered_cols = ["Season", "Team"] + [c for c in EXPECTED_COLUMNS if c in combined.columns]
    combined = combined[ordered_cols]

    # Filter to the requested competition (team pages include ALL
    # competitions that team played, e.g. cups/Europe too).
    resolved_name = slug.replace("-", " ")
    if "Comp" in combined.columns:
        mask = combined["Comp"].astype(str).str.contains(
            resolved_name.split()[0], case=False, na=False
        )
        # Fall back to no filtering if the name-matching heuristic misses
        # (competition renamed, translated name, etc.) rather than
        # silently dropping everything.
        if mask.any():
            combined = combined[mask]

    if played_only and "Result" in combined.columns:
        combined = combined[combined["Result"].notna() & (combined["Result"] != "")]

    if "Date" in combined.columns:
        combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
        combined = combined.sort_values(["Season", "Date"]).reset_index(drop=True)

    return combined


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Scrape match-level data from FBref.")
    parser.add_argument("competition", help='e.g. "Premier League" or a raw FBref competition id')
    parser.add_argument("seasons", nargs="+", help='e.g. 2022-2023 2023-2024')
    parser.add_argument("--out", default=None, help="CSV path to save the result")
    parser.add_argument("--delay", type=float, default=7.0)
    args = parser.parse_args()

    df = get_match_data(args.competition, args.seasons, delay=args.delay)
    print(df.head())
    print(f"\n{len(df)} rows scraped.")
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"Saved to {args.out}")
