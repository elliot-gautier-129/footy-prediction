#!/usr/bin/env bash
# Resolve the repository root from this script's location so it works from any directory.
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Download football-data.co.uk files for the big-five European leagues.
# Example: download_big_five 19 27 E0 SP1
# The end year is exclusive, so 19 27 downloads seasons 1920 through 2627.
download_big_five() {
    local start_year="$1"
    local end_year="$2"
    shift 2

    local project_dir="$PROJECT_DIR"
    local raw_dir="$project_dir/data/raw/football_data"
    local base_url="https://www.football-data.co.uk/mmz4281"
    local league
    local year
    local next_year
    local season
    local url
    local output

    if [[ -z "$start_year" || -z "$end_year" ]]; then
        echo "Usage: download_big_five START_YEAR END_YEAR [E0 SP1 D1 I1 F1]"
        return 1
    fi

    # If no league codes are supplied, download all five leagues.
    if [[ "$#" -eq 0 ]]; then
        set -- E0 SP1 D1 I1 F1
    fi

    for league in "$@"; do
        case "$league" in
            E0|SP1|D1|I1|F1) ;;
            *)
                echo "Unsupported league: $league"
                echo "Use one of: E0 SP1 D1 I1 F1"
                return 1
                ;;
        esac
    done

    for ((year = 10#$start_year; year < 10#$end_year; year++)); do
        next_year=$((year + 1))
        season="$(printf '%02d%02d' "$((year % 100))" "$((next_year % 100))")"

        for league in "$@"; do
            mkdir -p "$raw_dir/$league"
            url="$base_url/$season/$league.csv"
            output="$raw_dir/$league/$season.csv"

            echo "Downloading $league $season..."
            wget --tries=3 --waitretry=5 --timeout=30 \
                -O "$output" "$url"
            echo "Saved: $output"
        done
    done

    echo "Download complete."
}

# Pass arguments straight through, e.g. `download_football_data.sh 19 27` downloads
# all five leagues from 1920 through 2627.
download_big_five "$@"
