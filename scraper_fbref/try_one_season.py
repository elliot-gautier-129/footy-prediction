"""
try_one_season.py

A small, cautious runner for testing fbref_scraper.get_match_data() against
the LIVE site for the first time. Run this on your VM (or wherever
fbref_scraper.py lives) — it can't be run from the Claude sandbox since
that environment's network policy blocks fbref.com outright.

It does two things on purpose:
  1. A "smoke test" first — scrapes just 2 teams to confirm the parsing
     logic still matches FBref's current HTML before you commit to a
     slow, full-season run.
  2. Then, if that works, scrapes the full season you asked for.

Edit COMPETITION / SEASON below, or pass them as CLI args.
"""

import sys
import logging

from fbref_scraper import get_match_data, _team_urls_for_season, _team_match_log, resolve_competition

log = logging.getLogger("try_one_season")

# ---- Change these, or override on the command line ----
COMPETITION = "Premier League"   # name from COMPETITIONS, or a raw FBref id
SEASON = "2023-2024"             # "YYYY-YYYY" for most European leagues, "YYYY" for others
# --------------------------------------------------------


def smoke_test(competition: str, season: str, n_teams: int = 2):
    """
    Scrape just a couple of teams to sanity-check that:
      - the standings page still yields team links
      - each team's Scores & Fixtures table still parses with the
        expected columns
    before spending 20+ minutes on a full season.
    """
    comp_id, slug, _fmt = resolve_competition(competition)
    print(f"Resolved '{competition}' -> id={comp_id}, slug={slug}")

    teams = _team_urls_for_season(comp_id, slug, season)
    print(f"Found {len(teams)} teams for {competition} {season}. "
          f"Testing the first {n_teams}...")

    for team_id, team_name, url in teams[:n_teams]:
        print(f"\n--- {team_name} ({url}) ---")
        df = _team_match_log(team_id, team_name, season)
        print(f"Parsed {len(df)} rows, columns: {list(df.columns)}")
        print(df.head(3))

    print("\nSmoke test passed: parsing still matches FBref's current layout.")


def run_full_season(competition: str, season: str, out_csv: str | None = None):
    df = get_match_data(competition, season)
    print(f"\nScraped {len(df)} rows for {competition} {season}.")
    print(df.head(10))

    if out_csv is None:
        safe_comp = str(competition).replace(" ", "_")
        out_csv = f"{safe_comp}_{season}.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved to {out_csv}")
    return df


if __name__ == "__main__":
    competition = sys.argv[1] if len(sys.argv) > 1 else COMPETITION
    season = sys.argv[2] if len(sys.argv) > 2 else SEASON

    try:
        smoke_test(competition, season, n_teams=2)
    except Exception as exc:
        print(f"\nSmoke test FAILED: {exc}")
        print(
            "If this is a 403/Cloudflare error, FBref is blocking the "
            "request outright rather than rate-limiting it — tell Claude "
            "and a Selenium-based fallback can be added. If it's a "
            "parsing error (missing columns / no teams found), FBref's "
            "HTML layout likely drifted and the scraper needs a small fix."
        )
        sys.exit(1)

    answer = input(
        f"\nSmoke test OK. Run the FULL {competition} {season} season now? "
        f"This will make ~20 more requests at the default 7s delay "
        f"(~2-3 minutes). [y/N] "
    )
    if answer.strip().lower().startswith("y"):
        run_full_season(competition, season)
    else:
        print("Skipped full run.")
