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

**See MODEL_TRAINING_GUIDE.md for complete training instructions.**

### Quick Start - Model Training

```bash
# Retrain with existing data
python train_vgame_model.py --data temporal_processed_data.csv --train-end 2025-10-31

# Collect new V-game tagged data
python collect_vgame_data.py --start 2025-01-01 --end 2025-12-31 --output vgame_data.csv

# Full pipeline
python temporal_data_processor.py vgame_data.csv
python train_vgame_model.py --data temporal_processed_data.csv --train-end 2025-10-31
```

### Model Architecture

The processed data is designed for Random Forest classification with SMOTE for class balancing:

```python
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
```

### Target Variables

- `is_winner` - Binary classification (most common use case)
- `is_top3` - Top-3 prediction for "placering" bets
- `finish_position` - Regression target

### Critical: Temporal Train/Test Split

**NEVER use random train/test split for time-series data!**

```python
# ✅ CORRECT - Temporal split
train_data = df[df['date'] <= '2025-10-31']
test_data = df[df['date'] >= '2025-11-01']

# ❌ WRONG - Random split (causes data leakage)
train_test_split(X, y, test_size=0.2, random_state=42)
```

**Why temporal matters:**
- Racing is time-series data
- Must predict FUTURE races, not random samples
- Random split leaks future information into training
- Results in overly optimistic metrics

### Expected Performance

**Realistic metrics** (on held-out test set):
- Per-race accuracy: 20-25% (realistic win rate when betting)
- ROC-AUC: 0.65-0.75
- Horse-level accuracy: 75-85% (less important for betting)

**High-confidence picks** (predicted_prob >= 0.35):
- Win rate: 30-40%
- Fewer races but higher success rate

### Current Models

- `temporal_rf_model.pkl` - Current production model
- `temporal_rf_metadata.json` - Model info and performance metrics
- Trained on 2025 data with temporal validation

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

## Betting Tool

### V-Game Betting System (`create_bet.py`)
A tool for generating optimal betting slips for Swedish V-game betting systems.

**See VGAME_BETTING_RULES.md for complete rules and game type explanations.**

#### Quick Reference - V-Games

| Game | Races | Typical Day | Pool Size | Command |
|------|-------|------------|-----------|---------|
| **V75** | 7 | Saturday | 30-100M SEK | `python create_bet.py --game V75` |
| **V86** | 8 | Wed/Sat | 20-60M SEK | `python create_bet.py --game V86` |
| **V85** | 8 | Friday | 10-30M SEK | `python create_bet.py --game V85` |
| **V65** | 6 | Any day | 2-10M SEK | `python create_bet.py --game V65` |
| **V64** | 6 | Tue/Thu/Sun | 3-8M SEK | `python create_bet.py --game V64` |
| **V5** | 5 | Variable | 1-3M SEK | `python create_bet.py --game V5` |
| **V4** | 4 | Variable | 0.5-2M SEK | `python create_bet.py --game V4` |
| **V3** | 3 | Lunch | 0.2-1M SEK | `python create_bet.py --game V3` |
| **GS75** | 7 | Sat (4x/year) | 100M+ SEK | `python create_bet.py --game GS75` |

#### Betting Tool Usage

```bash
# Auto-find next race (any type)
python create_bet.py

# Specific V-game type
python create_bet.py --game V75
python create_bet.py --game V86 --total 500

# Track-based search
python create_bet.py --track solvalla

# Date-based search
python create_bet.py --date 2026-01-17
```

**Strategy**: Individual high-EV betting (NOT system/pool betting)
- Analyzes V-game races but places individual WIN bets
- Uses ML model (temporal_rf_model.pkl) for predictions
- Selects horses with highest expected value
- Better ROI than traditional system betting

#### How Our Tool Differs from Traditional V-Game Betting

**Traditional V-Game**: Pick all winners correctly → share prize pool
**Our Tool**: Individual bets on high-probability horses → profit on each win

This means you can profit even without hitting all races, with more consistent returns.

## Important Domains

- **ATG.se** - Main betting/program site (calendar: `/spel/kalender/YYYY-MM-DD`)
  - API: `https://www.atg.se/services/racinginfo/v1/api`
- **travsport.se** - Official results and statistics
- **V-Games** - Pool betting systems (V75, V86, V85, V65, V64, V5, V4, V3, GS75)
- **Elitloppet** - Sweden's premier trotting race
