# FBref match-level scraper

A general-purpose scraper for match-level football data from [FBref.com](https://fbref.com),
built on the pipeline described in
[Scraping FBref — Creating a Pipeline](https://medium.com/@henrik.schjoth/scraping-fbref-creating-a-pipeline-f5c9c23ba9da).

## Quick start

```bash
pip install requests beautifulsoup4 lxml pandas
```

```python
from fbref_scraper import get_match_data

# One competition, one season
df = get_match_data("Premier League", "2023-2024")

# One competition, multiple seasons
df = get_match_data("Premier League", ["2021-2022", "2022-2023", "2023-2024"])

# A competition not in the built-in list: pass its raw FBref id
df = get_match_data(24, "2023")  # Brazilian Série A, id looked up automatically

df.to_csv("matches.csv", index=False)
```

Command line:

```bash
python fbref_scraper.py "Premier League" 2022-2023 2023-2024 --out matches.csv
```

## Output shape

One row per **team per match** (so every fixture produces two rows — one from
each side's perspective). Columns:

```
Season, Team, Date, Time, Comp, Round, Day, Venue, Result, GF, GA, Opponent,
xG, xGA, Poss, Attendance, Captain, Formation, Opp Formation, Referee,
Match Report, Notes
```

## How far back the data goes / how much you can pull

- **Basic results** (date, score, venue, attendance, referee) go back
  decades for major leagues — English top-flight results exist on FBref
  back into the 1800s.
- **Advanced columns this scraper targets** — `xG`, `xGA`, `Poss`,
  `Formation`, `Opp Formation`, `Captain` — only exist on FBref from the
  **2017-18 season onward**, and only for competitions FBref tracks in
  detail (the Big 5 European leagues plus most major leagues/cups
  worldwide; smaller leagues may have partial or no advanced-stat
  history). Older seasons or lesser competitions will come back with
  those specific columns empty — that's a gap in FBref's underlying
  data, not a scraper bug.
- **No hard limit** on how many matches/seasons/competitions you can
  request — it's bounded only by time and FBref's rate limit (below).
  Each team-season = 1 HTTP request. Example: Premier League, 2017-18
  through 2025-26 (9 seasons × ~20 teams) ≈ 180 requests ≈ 20-25 minutes
  at the default polite delay.

## Rate limiting — read this before running large pulls

FBref's own bot-traffic policy (sports-reference.com/bot-traffic.html)
states that **more than 10 requests per minute** gets a session
temporarily blocked, for up to a day. This scraper defaults to one
request every 7 seconds (~8.5 req/min) with jitter, retries with
backoff on 429/403, and keeps going if one team/season fails rather
than aborting the whole run. Don't drop `delay` much below ~6.5s.

Note: FBref sits behind Cloudflare, which can still challenge or block
scripted HTTP clients independent of the request-rate rule above (this
is separate from, and on top of, the stated rate limit). If you get
persistent 403s even at a slow, polite rate, that's Cloudflare bot
detection rather than the request-count policy, and the tutorial's
fallback (Selenium with a real headless browser) is the usual
workaround — this script does not include a Selenium fallback by
default, to keep it simple; ask if you want that added.

## Adding competitions

`COMPETITIONS` in `fbref_scraper.py` has a starter set (Premier League,
Championship, La Liga, Serie A, Bundesliga, Ligue 1, Primeira Liga,
Eredivisie, Champions League, Europa League, MLS, Brazilian Série A,
Liga MX). For anything else:

```python
from fbref_scraper import list_competitions, resolve_competition

all_comps = list_competitions()          # scrapes fbref.com/en/comps/
print(all_comps.head(20))

resolve_competition(20)                  # -> (20, 'Bundesliga', 'range')
```

or just pass the raw FBref competition id straight into `get_match_data`.

## Files

- `fbref_scraper.py` — the scraper (importable module + CLI)
- `test_parsing.py` — offline test proving the HTML-parsing logic
  (including FBref's habit of hiding some tables inside HTML comments)
  against a hand-built fixture, with no network calls

## A note on testing

This scraper was written and syntax/parsing-tested here, but this
sandboxed environment's network policy blocks outbound connections to
fbref.com, so I could not run it against the live site end-to-end from
here. Run it in your own Python environment and let me know if
anything breaks — FBref's HTML structure drifts occasionally and the
table-detection logic may need a small tweak for a specific
competition/season.
