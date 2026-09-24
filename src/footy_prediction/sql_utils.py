"""Shared DuckDB connection and helpers for reading/executing the scripts in sql/."""
import duckdb

from footy_prediction.paths import SQL_DIR

# Create one DuckDB connection for the SQL cells below.
# (Shared by the loaders and SQL scripts, so tables registered by one step
# are visible to the next.)
con = duckdb.connect()


def load_sql(filename: str) -> str:
    """Return the text of a SQL script in the project's sql/ directory."""
    return (SQL_DIR / filename).read_text()


def run_sql_file(
    filename: str,
    connection: duckdb.DuckDBPyConnection | None = None,
) -> duckdb.DuckDBPyConnection:
    """Execute a SQL script from sql/ on the shared connection (or the one given)."""
    connection = con if connection is None else connection
    return connection.execute(load_sql(filename))
