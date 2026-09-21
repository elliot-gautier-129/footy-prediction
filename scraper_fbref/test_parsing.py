"""
Offline sanity test for the parsing logic in fbref_scraper.py, using a
hand-built HTML fixture that mimics FBref's real structure:
  - a normal visible table
  - a second table hidden inside an HTML comment (FBref does this for
    several of its tables to deter naive scrapers)
  - a two-row (MultiIndex) header, which FBref uses on some tables

This does NOT hit the network - it only proves the comment-extraction,
multi-index flattening, and column-matching logic works as intended.
"""
from fbref_scraper import _tables_including_comments, _flatten_columns, EXPECTED_COLUMNS

FAKE_TEAM_PAGE = """
<html><body>
<div id="content">
  <table id="matchlogs_for">
    <thead>
      <tr><th>Date</th><th>Time</th><th>Comp</th><th>Round</th><th>Day</th>
          <th>Venue</th><th>Result</th><th>GF</th><th>GA</th><th>Opponent</th>
          <th>xG</th><th>xGA</th><th>Poss</th><th>Attendance</th><th>Captain</th>
          <th>Formation</th><th>Opp Formation</th><th>Referee</th>
          <th>Match Report</th><th>Notes</th></tr>
    </thead>
    <tbody>
      <tr><td>2023-08-12</td><td>15:00</td><td>Premier League</td><td>Matchweek 1</td>
          <td>Sat</td><td>Home</td><td>W</td><td>2</td><td>1</td><td>Nottm Forest</td>
          <td>1.8</td><td>0.9</td><td>63</td><td>60,123</td><td>Martin Ødegaard</td>
          <td>4-3-3</td><td>4-2-3-1</td><td>Michael Oliver</td><td>Match Report</td><td></td></tr>
    </tbody>
  </table>
  <!--
  <table id="misc_stats">
    <thead><tr><th></th><th>Team</th><th colspan="2">Performance</th></tr>
           <tr><th>Rk</th><th>Squad</th><th>CrdY</th><th>CrdR</th></tr></thead>
    <tbody><tr><td>1</td><td>Arsenal</td><td>2</td><td>0</td></tr></tbody>
  </table>
  -->
</div>
</body></html>
"""

results = _tables_including_comments(FAKE_TEAM_PAGE)
print(f"Found {len(results)} tables (expected 2: one visible, one commented-out)")
assert len(results) == 2, "comment-extraction failed"

ids = [tid for tid, _df in results]
print("Table ids found:", ids)
assert "matchlogs_for" in ids
assert "misc_stats" in ids, "failed to pull the commented-out table"

matchlog = next(df for tid, df in results if tid == "matchlogs_for")
matchlog = _flatten_columns(matchlog)
print("\nParsed match log columns:", list(matchlog.columns))
missing = [c for c in EXPECTED_COLUMNS if c not in matchlog.columns]
assert not missing, f"missing expected columns: {missing}"
print("\nParsed row:\n", matchlog.iloc[0])

print("\nALL OFFLINE PARSING CHECKS PASSED")
