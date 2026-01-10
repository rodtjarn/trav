# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A toolkit for scraping, processing, and predicting Swedish trotting race results using Random Forest and other ML models. The system collects data from ATG.se (race programs) and travsport.se (results), processes it into ML-ready features, and prepares it for predictive modeling.

**Legal Notice**: This is for educational/research purposes only. Always check robots.txt and Terms of Service before scraping. Rate limiting is implemented (2 second delays between requests).

## Core Commands

### Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies (use latest versions for Python 3.13+ compatibility)
pip install requests beautifulsoup4 pandas lxml
```

**Note**: The requirements.txt specifies pandas 2.1.4 which doesn't work with Python 3.13+. Use the command above instead to get compatible versions.

### Development Workflow
```bash
# 1. Inspect website structure (run this FIRST when ATG.se structure changes)
python html_inspector.py
python html_inspector.py "https://www.atg.se/spel/kalender/2025-01-15"

# 2. Scrape race data (after updating selectors)
python atg_scraper.py

# 3. Process data for ML
python data_processor.py
```

### Output Files
- `atg_races.csv` / `atg_races.json` - Raw scraped data
- `processed_trotting_data.csv` - ML-ready features
- `atg_structure_*.html` - HTML dumps for manual inspection (from inspector)

## Architecture

### Three-Stage Pipeline

**Stage 1: HTML Inspection** (`html_inspector.py`)
- Analyzes ATG.se HTML structure to identify CSS selectors
- Extracts class names, IDs, data attributes, and JSON embedded in scripts
- Checks for potential API endpoints
- **Purpose**: ATG.se structure changes frequently; this tool helps update selectors in the scraper

**Stage 2: Data Scraping** (`atg_scraper.py`)
- `SwedishTrottingScraper` class with configurable rate limiting
- Key methods:
  - `scrape_atg_program(date)` - Scrapes race program for specific date
  - `scrape_race_details(race_id)` - Gets detailed horse/driver info
  - `scrape_travsport_results(date)` - Gets historical results
- Uses regex-based CSS selector patterns for flexibility
- Saves to both CSV and JSON formats

**Stage 3: Feature Engineering** (`data_processor.py`)
- `TrottingDataProcessor` class for ML preparation
- Critical features:
  - **Temporal**: hour, day_of_week, is_weekend, month
  - **Post position**: post_position, is_inside_post (≤5), position_ratio
  - **Driver stats** (90-day rolling): starts, win_rate, avg_position, top3_rate
  - **Horse form** (last 5 races): avg_position, wins, days_since_last_race, avg_speed
  - **Speed metrics**: Converts Swedish time format (1.15,0) to km/min
  - **Track encoding**: One-hot encoded venue names
- Creates targets: `is_winner`, `is_top3`, `finish_position`

### Data Flow

```
ATG.se → html_inspector.py → [manual selector updates] → atg_scraper.py →
raw CSV/JSON → data_processor.py → processed_trotting_data.csv → ML model
```

### Swedish Trotting-Specific Logic

**Time Format Parsing** (`parse_record_time`):
- Input: "1.15,0" or "1.13,5a" (Swedish format: minutes.seconds,tenths)
- Output: Total seconds (75.0, 73.5)
- Handles "a" suffix for auto start, removes letters

**Distance Parsing** (`parse_distance`):
- Extracts meters from various formats: "2140m", "2140", "2140 meter"
- Standard distances: 1640m, 2140m, 2640m

**Speed Calculation**:
- Formula: (distance_km / time_minutes) = km/min
- Standard metric for trotting race performance
- Calculated from personal records and race times

**Critical Racing Concepts**:
- "Gallop" (galopp): Horse breaks from trotting gait → disqualification
- "Monte" vs "Auto": Standing start vs flying/rolling start
- "Tillägg": Distance handicap for stronger horses
- "Spår": Post position (inside positions 1-5 are advantageous)

### Selector Pattern System

The scraper uses `re.compile()` with flexible patterns because ATG.se frequently updates class names:

```python
# Example: finds 'race-card', 'race_card', 'raceCard', etc.
race_cards = soup.find_all('div', class_=re.compile(r'race.*card|program.*item', re.I))
```

**When selectors break**:
1. Run `html_inspector.py` to get current structure
2. Review saved HTML file and identify new class names
3. Update regex patterns in `atg_scraper.py` methods:
   - `_parse_atg_race_card()`
   - `_parse_race_metadata()`
   - `_parse_horse_entry()`

### Key Data Fields

**Race Level**:
- `track` - Venue (Solvalla, Åby, Jägersro, etc.)
- `race_number`, `distance`, `prize_money`, `race_type`
- `track_condition`, `temperature`

**Horse Level**:
- `horse_name`, `number`, `post_position`
- `driver`, `trainer`, `distance_handicap`
- `record`, `age`, `sex`, `starts`, `wins`, `earnings`
- `last_5_races`, `odds`

**Results** (for training data):
- `finish_position`, `finish_time`, `margin`
- `gallop` - Critical: indicates disqualification
- `payout`

## Machine Learning Integration

The processed data is designed for Random Forest classification with SMOTE for class balancing:

```python
# Example model setup (from README)
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Features exclude: is_winner, is_top3, finish_position,
# horse_name, driver, trainer, date, race_id, start_time, track
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1
)
```

Target variables:
- `is_winner` - Binary classification (most common use case)
- `is_top3` - Top-3 prediction for "placering" bets
- `finish_position` - Regression target

## Historical Data Collection

To collect training data, modify `atg_scraper.py` main():

```python
from datetime import timedelta
start_date = datetime.now() - timedelta(days=30)
dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d')
         for i in range(30)]

for date in dates:
    races = scraper.scrape_atg_program(date)
    results = scraper.scrape_travsport_results(date)
    # Merge and save
```

## Common Issues

**Website Structure Changed**: Most common issue. Run `html_inspector.py`, update selectors.

**Rate Limiting**: If blocked, increase `delay` parameter in `SwedishTrottingScraper(delay=5.0)`.

**Missing Data**:
- Future races may not have odds set yet
- Historical data may be incomplete
- Always validate with `df.info()` before ML training

**Time Parsing Failures**: Check for new formats in ATG's race times, update regex in `parse_record_time()`.

## Important Domains

- **ATG.se** - Main betting/program site (calendar: `/spel/kalender/YYYY-MM-DD`)
- **travsport.se** - Official results and statistics
- **V75** - Special 7-race betting format (high prize pools)
- **Elitloppet** - Sweden's premier trotting race
