"""Project directory locations, resolved from this file rather than the working directory."""
from pathlib import Path

# src/footy_prediction/paths.py -> parents[2] is the repository root.
PROJECT_DIR = Path(__file__).resolve().parents[2]

SQL_DIR = PROJECT_DIR / "sql"
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
FOOTBALL_DATA_DIR = RAW_DIR / "football_data"
UNDERSTAT_DIR = RAW_DIR / "understat"
FEATURES_DIR = DATA_DIR / "features"
RESULTS_DIR = PROJECT_DIR / "results"
