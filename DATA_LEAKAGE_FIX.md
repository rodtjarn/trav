# Data Leakage Fix - Summary

## Problem Identified

**Unrealistic Performance:**
- Individual betting: 83.5% win rate (66/79 bets won)
- Hit longshots at 87.9, 38.0, 28.4 odds
- Four perfect 8/8 days
- +56,352 SEK profit from 10,000 SEK investment

**Root Cause: DATA LEAKAGE**

### Leakage Mechanisms

1. **Training Data from Future**
   - Model trained on Jan 3-9, 2026
   - Tested on Dec 2025 - Jan 2026
   - Training included future test data!

2. **Rolling Statistics Leakage**
   - Driver/trainer stats calculated with 90-day rolling window
   - When trained on Jan 2026 data, stats include Oct-Dec 2025
   - Testing on Dec 2025 = model already "knew" December results
   - **Example:** Alunita (87.9 odds winner) - model saw driver's good Dec stats

3. **Random Train/Test Split**
   - Used sklearn's random split instead of temporal split
   - Mixed past and future races together
   - Model learned from future races to predict past races

## The Fix

### Fix 1: Temporal Data Split ✅

**Before:**
```python
# Random split - BAD!
X_train, X_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**After:**
```python
# Temporal split - GOOD!
train_df = df[df['date'] <= '2025-10-31']  # Jan-Oct for training
test_df = df[df['date'] >= '2025-11-01']   # Nov-Dec for testing
```

### Fix 2: Forward-Looking Features Only ✅

**Before:**
```python
# Calculated driver stats from ALL data
driver_stats = df.groupby('driver_id').agg({
    'finish_place': ['count', lambda x: (x == 1).sum()]
})
# This includes FUTURE races!
```

**After:**
```python
# Calculate driver stats from PAST races only
for idx in driver_races.index:
    race_date = df.loc[idx, 'date']
    lookback_start = race_date - timedelta(days=90)

    # Get historical races BEFORE current race
    historical = driver_races[
        (driver_races['date'] >= lookback_start) &
        (driver_races['date'] < race_date)  # STRICTLY BEFORE
    ]

    # Calculate stats from historical data only
    stats = calculate_stats(historical)
```

**Key difference:** For each race at date D, we only use data from D-90 to D-1 (not including D or future).

### Fix 3: Proper Backtesting ✅

**Before:**
- Test on Dec 2025 using model trained on Jan 2026
- Features calculated from full dataset

**After:**
- For each test date, recalculate features using only prior data
- Features as they would have been known at prediction time
- True forward-testing

## New Data Collection

**Collected: Full Year 2025**
- Date range: 2025-01-01 to 2025-12-31
- Total races: 215,039
- Unique dates: 365 days
- Tracks: 182 venues
- Races with results: 141,140 (65.6%)

**Training Plan:**
- Train: Jan-Oct 2025 (~176,000 races)
- Test: Nov-Dec 2025 (~39,000 races)
- Validation: Strict temporal ordering

## Expected Realistic Performance

### Previous (with leakage)
- Per-race accuracy: 83.5%
- Top-3 accuracy: ~90%
- Perfect days: 4/10 (40%)
- ROI: +563.5%

### Expected (no leakage)
- Per-race accuracy: **15-25%**
- Top-3 accuracy: **40-60%**
- Perfect days: **0-5%**
- ROI: **-10% to +10%**

This is **REALISTIC** for horse racing. Professional handicappers typically achieve:
- 20-30% win rate on favorites
- 50-60% top-3 rate
- Slightly positive to slightly negative ROI

## Files Created

1. **scrape_full_year.py** - Collect full year of data
2. **temporal_data_processor.py** - Process data with no leakage
3. **train_temporal_model.py** - Train with temporal validation
4. **trotting_data_2025_final.csv** - Full year dataset (59 MB)

## Current Status

✅ Data collection complete (215,039 races)
⏳ Temporal processing in progress (~20-30 minutes)
⏳ Model training pending
⏳ Realistic evaluation pending

## Why This Matters

**With data leakage:**
- Results look amazing but are fake
- Cannot be reproduced in real betting
- Lose money when applied to real races

**Without data leakage:**
- Results are realistic and honest
- Can be trusted for actual betting decisions
- Model performance reflects true predictive power

## Next Steps

1. Wait for temporal processing to complete
2. Train model with temporal split
3. Evaluate on Nov-Dec 2025 (unseen future data)
4. Get realistic per-race accuracy
5. Re-run V85 backtesting with properly trained model
6. Compare realistic vs leaked results
