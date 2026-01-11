# Backtesting Guide

A comprehensive tool for analyzing model performance on past race days.

## Quick Start

```bash
# Basic usage - analyze all races at a track
python backtest_race.py --date 2026-01-10 --track Romme

# Analyze specific V-game
python backtest_race.py --date 2026-01-10 --track Romme --game V85

# Custom budget
python backtest_race.py --date 2026-01-11 --track Östersund --game GS75 --budget 1000

# Swedish characters can be typed without special keys
python backtest_race.py --date 2026-01-11 --track Ostersund --game GS75  # å→a, ö→o
python backtest_race.py --date 2025-12-06 --track Aby                    # Matches Åby
python backtest_race.py --date 2025-12-20 --track orebro                 # Case-insensitive
```

## Command Line Options

| Option | Required | Description | Example |
|--------|----------|-------------|---------|
| `--date` | Yes | Race date in YYYY-MM-DD format | `--date 2026-01-10` |
| `--track` | Yes | Track name (case-insensitive, accent-insensitive) | `--track Romme` or `--track Ostersund` |
| `--budget` | No | Total betting budget in SEK (default: 500) | `--budget 1000` |
| `--game` | No | Filter by V-game type | `--game V85` |

**Note on Swedish Characters**: Track names with Swedish characters (å, ä, ö) can be typed with regular ASCII letters (a, o):
- `Åby` can be typed as `Aby` or `aby`
- `Östersund` can be typed as `Ostersund` or `ostersund`
- `Örebro` can be typed as `Orebro`
- `Jägersro` can be typed as `Jagersro`
- `Färjestad` can be typed as `Farjestad`
- `Gävle` can be typed as `Gavle`

All comparisons are case-insensitive, so you can use `romme`, `Romme`, or `ROMME`.

### Supported Game Types

- `V75` - 7 races (usually Saturday)
- `GS75` - 7 races (Grande Slam Saturday)
- `V86` - 8 races (Wednesday/Saturday)
- `V85` - 8 races
- `V65` - 6 races
- `V64` - 6 races
- `V5` - 5 races
- `V4` - 4 races
- `V3` - 3 races

### Common Track Names

**Swedish Tracks:**
- Solvalla
- Åby
- Jägersro
- Axevalla
- Bergsåker
- Bro Park
- Eskilstuna
- Färjestad
- Gävle
- Halmstad
- Kalmar
- Mantorp
- Romme
- Rättvik
- Solvalla
- Umåker
- Östersund
- Örebro

**Norwegian Tracks:**
- Bjerke
- Forus
- Jarlsberg
- Klosterskogen
- Momarken
- Orkla
- Sørlandet

**Danish Tracks:**
- Charlottenlund
- Aalborg
- Odense
- Skive

## Output Explained

### 1. Model Predictions
Shows the top 10 betting opportunities identified by the model:
```
🔥 STRONG - Race 4
   Horse #7: Karat River
   Win probability: 24.5%
   Actual odds: 5.8
   Expected Value: 0.41
   Bet amount: 39 SEK
    ✅ WON! Payout: 224 SEK
```

**Category Levels:**
- 🔥 **STRONG** - Win probability ≥ 30%
- ⭐ **GOOD** - Win probability 25-30%
- 💎 **DECENT** - Win probability 20-25%

**Expected Value (EV):**
- EV > 0: Positive value bet (model thinks it's undervalued)
- EV < 0: Negative value bet (overvalued)
- Formula: `(probability × odds) - 1`

### 2. Losing Bet Placement Analysis
Breaks down where losing bets finished:
```
📊 LOSING BET PLACEMENT ANALYSIS

2nd place 🥈: 1 bet(s)
  - Race 6, #8 High on Pepper (30.4% prob, 1.3 odds)
3rd place 🥉: 1 bet(s)
  - Race 10, #9 Twigs Khaleesi (26.8% prob, 2.7 odds)
DNF/Galloped: 2 bet(s)
  - Race 10, #10 Staro Roseway (28.3% prob, 38.0 odds)
```

**What to look for:**
- High % of 2nd/3rd: Near-misses, could indicate good model but bad luck
- High % of DNF: Horses breaking gait, hard to predict
- High % of 4th+: Model was wrong about these horses

### 3. Actual Race Winners
Shows what actually won each race for comparison:
```
🏆 ACTUAL RACE WINNERS

Race 1: Borlänge Kommun - STL Kallblodsdivisionen
  🏆 #2 Nytomt Amira (odds: 9.9)
Race 2: DAT AB - STL Diamantstoet
  🏆 #3 Urbina Southwind (odds: 3.4)
```

### 4. Final Results
Summary statistics:
```
📊 FINAL RESULTS

Total bet: 495 SEK
Winners: 3/10 (30.0%)
Total payout: 903 SEK
Profit: +408 SEK (+82.5% ROI)

✅ PROFITABLE DAY! 🎉
```

**Metrics:**
- **Win rate**: Percentage of bets that won
- **ROI**: Return on Investment = (Payout / Total Bet - 1) × 100%
- Model expectation: ~29.5% win rate

### 5. Key Insights
Automatically identifies patterns:
```
💡 KEY INSIGHTS:

  • 2 horse(s) galloped/DNF (20% of bets)
  • 2 horse(s) finished 2nd or 3rd (near-misses)
  • Average winner odds: 5.8
  • High EV bets (>1.0): 5 total, 0 won
```

## Example Sessions

### Example 1: Successful Day
```bash
$ python backtest_race.py --date 2026-01-10 --track Romme --game V85

Winners: 3/10 (30.0%)
Profit: +408 SEK (+82.5% ROI)
✅ PROFITABLE DAY! 🎉
```

**Analysis:** Model hit expected 30% win rate with strong ROI due to finding value bets.

### Example 2: Losing Day
```bash
$ python backtest_race.py --date 2026-01-11 --track Östersund --game GS75

Winners: 0/5 (0.0%)
Profit: -495 SEK (-100.0% ROI)
❌ TOTAL LOSS - No winners

KEY INSIGHTS:
  • 3 horse(s) galloped/DNF (60% of bets)
```

**Analysis:** Bad day - 60% of picks DNF'd. This is variance and hard to predict.

### Example 3: Near-Miss Day
```bash
$ python backtest_race.py --date 2025-11-01 --track Romme

Winners: 2/10 (20.0%)
Profit: -105 SEK (-21.5% ROI)

KEY INSIGHTS:
  • 3 horse(s) finished 2nd or 3rd (near-misses)
```

**Analysis:** Several near-misses suggest model was close but unlucky.

## Understanding Variance

Horse racing has high variance. Individual days will fluctuate:

| Scenario | Win Rate | Expected Frequency |
|----------|----------|-------------------|
| Great day | 40-50% | ~10% of days |
| Good day | 30-40% | ~30% of days |
| Average day | 20-30% | ~35% of days |
| Bad day | 10-20% | ~20% of days |
| Terrible day | 0-10% | ~5% of days |

**Key takeaway:** The model targets ~29.5% accuracy over many races. Single days can vary significantly.

## Tips for Analysis

1. **Look at multiple days** - Don't judge the model on one race day
2. **Check EV, not just wins** - High EV bets that lose are still good long-term
3. **DNF analysis** - Many DNFs suggests unpredictable conditions (weather, track)
4. **Near-misses** - Many 2nd/3rd places suggests the model is close
5. **Favorite vs longshot balance** - Mix of both is healthy
6. **Game type matters** - GS75 (kallblod/cold-blood) races are more unpredictable than V75 (warm-blood)

## Finding Past Races to Analyze

### Recent Saturdays
```bash
# Check what races are available
python -c "
from atg_api_scraper import ATGAPIScraper
from datetime import datetime, timedelta

scraper = ATGAPIScraper()
today = datetime.now()

for days_back in range(7, 30, 7):
    date = (today - timedelta(days=days_back)).strftime('%Y-%m-%d')
    if (today - timedelta(days=days_back)).weekday() == 5:  # Saturday
        print(f'Saturday: {date}')
"
```

### Check what tracks raced on a date
```bash
python -c "
from atg_api_scraper import ATGAPIScraper

scraper = ATGAPIScraper()
cal = scraper.get_calendar_for_date('2026-01-10')

for track in cal['tracks']:
    print(f\"{track['name']} - {track.get('biggestGameType', 'No V-game')}\")
"
```

## Troubleshooting

### "Track not found"
```
❌ Track 'Rommme' not found on 2026-01-10
Available tracks: Romme, Forus, Vincennes
```
**Fix:** Check spelling. Track names are case-insensitive but must be spelled correctly.

### "No races found for date"
```
❌ No races found for 2026-01-15
```
**Fix:** Verify the date. May be too far in the past (data not available) or future (not yet scheduled).

### "No betting opportunities found"
```
❌ No betting opportunities found (model didn't have high confidence picks)
```
**Fix:** Model didn't find any horses with >20% win probability. This is rare but can happen with very weak fields or data issues.

## Advanced Usage

### Compare Multiple Saturdays
```bash
# Create a simple loop
for date in 2025-11-01 2025-12-27 2026-01-10; do
    echo "=== $date ==="
    python backtest_race.py --date $date --track Romme --budget 500 | grep "Winners:\|Profit:"
    echo
done
```

### Export Results
```bash
# Save to file
python backtest_race.py --date 2026-01-10 --track Romme --game V85 > results_romme_jan10.txt
```

### Different Budget Strategies
```bash
# Conservative (300 SEK)
python backtest_race.py --date 2026-01-10 --track Romme --budget 300

# Moderate (500 SEK) - Default
python backtest_race.py --date 2026-01-10 --track Romme

# Aggressive (1000 SEK)
python backtest_race.py --date 2026-01-10 --track Romme --budget 1000
```

## Related Tools

- `create_bet.py` - Generate live betting slips for upcoming races
- `analyze_past_performance.py` - Historical analysis from local data (2025 only)
- `check_saturday_main_race.py` - Find next Saturday's main V-game

## Notes

- **Data availability**: Results must be available via ATG API (usually races from recent months)
- **Processing time**: Takes 10-30 seconds to fetch and process race data
- **Model version**: Uses `temporal_rf_model.pkl` (retrained on 2025 V-game data)
- **Feature consistency**: Some old races may have incomplete data affecting predictions
