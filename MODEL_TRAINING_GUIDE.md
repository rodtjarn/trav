# Model Training Guide - V-Game Predictions

Complete guide for training ML models to predict Swedish trotting races across all V-game types (V75, V86, V85, V65, V64, V5, V4, V3, GS75).

## ⚠️ CRITICAL: Temporal Train/Test Split

**Why This Matters:**
- Racing is time-series data
- Models must predict FUTURE races
- Random split = data leakage = overly optimistic results
- Temporal split = realistic performance estimates

**The Rule:**
```
Training Data: Past races (e.g., Jan-Oct 2025)
Test Data:     Future races (e.g., Nov-Dec 2025)
NEVER mix temporal boundaries!
```

## Quick Start - Retrain Existing Model

If you already have `temporal_processed_data.csv`:

```bash
# Train on Jan-Oct 2025, test on Nov-Dec 2025
python train_vgame_model.py \
  --data temporal_processed_data.csv \
  --train-end 2025-10-31

# Custom parameters
python train_vgame_model.py \
  --data temporal_processed_data.csv \
  --train-end 2025-10-31 \
  --estimators 200 \
  --depth 25
```

**Output:**
- `vgame_rf_model.pkl` - Trained model
- `vgame_rf_metadata.json` - Model info & metrics
- `vgame_rf_model_test_predictions.csv` - Test set predictions

## Full Pipeline - Collect New Data

### Step 1: Collect V-Game Tagged Data

Collect data with V-game type labels:

```bash
# Collect 60 days of data
python collect_vgame_data.py \
  --start 2025-10-01 \
  --end 2025-11-30 \
  --output vgame_tagged_data.csv

# Collect full year
python collect_vgame_data.py \
  --start 2025-01-01 \
  --end 2025-12-31 \
  --output vgame_2025_full.csv \
  --delay 0.5
```

**What This Does:**
- Fetches race data from ATG API
- Tags each race with V-game type (V75, V86, V65, etc.)
- Adds `vgame_type` and `is_vgame` columns
- Saves progress every 7 days

**Expected Output:**
```
Statistics:
  Days processed: 60
  Total races: 1,200
  Total starts: 12,000
  V-game starts: 4,500 (37.5%)

V-Game breakdown:
  V65: 180 races
  V86: 16 races
  V75: 8 races
  V64: 24 races
  V5: 32 races
```

### Step 2: Process Data for ML

Use the temporal data processor:

```bash
python temporal_data_processor.py vgame_tagged_data.csv
```

This creates temporal features:
- Driver rolling stats (90-day window)
- Horse form (last 5 races)
- Track-specific features
- Time-based features (hour, day of week, month)

**Output:** `temporal_processed_data.csv`

### Step 3: Train Model with Temporal Split

```bash
# Train on first 80% of data, test on last 20%
python train_vgame_model.py \
  --data temporal_processed_data.csv \
  --train-end 2025-10-31

# View training progress
# Model will show:
# - Training set stats (past data)
# - Test set stats (future data)
# - Per-race accuracy (realistic win rate)
# - Feature importance
```

### Step 4: Validate Results

Check the output:

```
TEMPORAL SPLIT - CRITICAL FOR VALIDITY
=====================================

📊 TRAINING SET (Past Data)
   Date range: 2025-01-01 to 2025-10-31
   Samples: 45,000
   Winners: 4,500 (10.0%)
   Races: 3,750

📊 TEST SET (Future Data - Held Out)
   Date range: 2025-11-01 to 2025-12-31
   Samples: 10,000
   Winners: 1,000 (10.0%)
   Races: 833

✓ No date overlap - temporal split is valid

...

🎯 BETTING PERFORMANCE:
   Races with correct winner: 185/833 (22.2%)
   Expected win rate: 22.2%
   This is what you'd achieve betting on the model's top pick each race
```

**Key Metrics to Check:**
1. **Per-race accuracy** (20-25% is good for horse racing)
2. **No date overlap** between train/test
3. **Reasonable ROC-AUC** (>0.65)
4. **Feature importance** makes sense

## Understanding the Results

### Horse-Level Metrics (Less Important)

```
Accuracy: 0.8950
ROC-AUC:  0.7234
```

These measure individual horse predictions but don't reflect betting reality.

### Per-Race Metrics (MOST IMPORTANT)

```
Per-race accuracy: 22.2%
```

This is your **realistic win rate** when betting on the model's top pick.

**Why the difference?**
- Horse-level: "Was this horse correctly classified?"
- Per-race: "Did we pick the winner of this race?"
- Betting only cares about per-race!

### Performance by Confidence

```
Prob >= 0.20:  833 races,  22.2% win rate
Prob >= 0.25:  520 races,  25.8% win rate
Prob >= 0.30:  310 races,  29.7% win rate
Prob >= 0.35:  180 races,  33.9% win rate
Prob >= 0.40:   85 races,  38.8% win rate
```

**Key Insight:** Higher confidence = higher win rate but fewer opportunities

### ROI Estimation

If average odds = 4.5 for winners:
```
Win rate: 25%
Average payout: 4.5x bet
Expected return: 0.25 × 4.5 = 1.125 (12.5% ROI)
```

## Common Training Scenarios

### Scenario 1: You Have Recent Data (Recommended)

```bash
# Collect last 90 days
END_DATE=$(date +%Y-%m-%d)
START_DATE=$(date -d "90 days ago" +%Y-%m-%d)

python collect_vgame_data.py \
  --start $START_DATE \
  --end $END_DATE \
  --output recent_data.csv

# Process
python temporal_data_processor.py recent_data.csv

# Train (use first 70 days for training, last 20 for testing)
SPLIT_DATE=$(date -d "20 days ago" +%Y-%m-%d)
python train_vgame_model.py \
  --data temporal_processed_data.csv \
  --train-end $SPLIT_DATE
```

### Scenario 2: You Have Full Historical Data

```bash
# Use existing data
python train_vgame_model.py \
  --data temporal_processed_data.csv \
  --train-end 2025-10-31 \
  --estimators 200 \
  --depth 25
```

### Scenario 3: Focus on Specific V-Game Type

After training, filter predictions:

```python
import pandas as pd

# Load test predictions
df = pd.read_csv('vgame_rf_model_test_predictions.csv')

# Analyze V86 only
v86_df = df[df['vgame_type'] == 'V86']
print(f"V86 win rate: {v86_df['was_correct'].mean()*100:.1f}%")
```

## Train/Test Split Guidelines

### Recommended Splits

**Minimum data:** 30 days
- Train: 24 days
- Test: 6 days

**Good amount:** 90 days
- Train: 70 days
- Test: 20 days

**Ideal:** 365 days
- Train: 300 days
- Test: 65 days

### How to Choose Split Date

```bash
# Method 1: Fixed date
--train-end 2025-10-31

# Method 2: Percentage (80/20 split)
# If data spans 2025-01-01 to 2025-12-31
# 80% = 292 days from start = 2025-10-19
--train-end 2025-10-19

# Method 3: Recent validation
# Train on everything except last 2 weeks
--train-end $(date -d "14 days ago" +%Y-%m-%d)
```

## Validating No Data Leakage

The training script checks for leakage automatically:

```
✓ No date overlap - temporal split is valid
```

If you see a warning:
```
⚠️  WARNING: Date overlap detected: {2025-10-31}
   This could indicate data leakage!
```

**Fix:** Adjust `--train-end` and `--test-start` to ensure no overlap.

## Model Parameters

### Basic Parameters

```bash
--estimators 100     # Number of trees (more = slower but better)
--depth 20           # Max tree depth (higher = more complex)
```

### When to Adjust

**Underfitting** (too simple):
- Symptoms: Low train AND test accuracy
- Fix: Increase `--depth` or `--estimators`

**Overfitting** (memorizing training data):
- Symptoms: High train accuracy, low test accuracy
- Fix: Decrease `--depth`, add more training data

**Optimal** (just right):
- Train accuracy: ~75-85%
- Test per-race accuracy: 20-25%
- Similar performance on train and test

## Feature Engineering

The processor creates these features:

### Temporal Features (Critical!)
- `hour` - Race time affects form
- `day_of_week` - Weekends different from weekdays
- `is_weekend` - Major races on weekends
- `month` - Seasonal patterns

### Rolling Statistics (90-day window)
- `driver_starts_90d` - Recent activity
- `driver_win_rate_90d` - Recent performance
- `horse_days_since_last_race` - Fitness indicator

### Race-Specific
- `post_position` - Inside posts advantageous
- `distance` - Horse specialization
- `track_encoded_*` - Track-specific patterns

## Troubleshooting

### "No target column found"

**Problem:** Data doesn't have results

**Solution:**
```bash
# Check if data has finish_place column
head temporal_processed_data.csv | grep finish_place

# If missing, you need historical race RESULTS
# Future races don't have results yet
```

### "ROC-AUC too low (<0.60)"

**Problem:** Model not learning

**Solutions:**
1. Collect more data
2. Check feature quality
3. Try different parameters
4. Verify no data errors

### "Per-race accuracy too low (<15%)"

**Problem:** Model worse than random

**Solutions:**
1. Check if using temporal features
2. Verify no data leakage (reverse split)
3. Review feature importance
4. Consider feature engineering

### "Training takes too long"

**Solutions:**
```bash
# Reduce trees
--estimators 50

# Reduce depth
--depth 15

# Skip SMOTE
--no-smote

# Use less data (for testing)
head -50000 temporal_processed_data.csv > small_sample.csv
```

## Next Steps

After training:

1. **Validate predictions:**
   ```bash
   # Review test predictions
   head -20 vgame_rf_model_test_predictions.csv
   ```

2. **Update betting tool:**
   ```bash
   # Replace current model
   cp vgame_rf_model.pkl temporal_rf_model.pkl
   cp vgame_rf_metadata.json temporal_rf_metadata.json
   ```

3. **Test betting tool:**
   ```bash
   python create_bet.py --game V65
   ```

4. **Backtest strategy:**
   ```python
   # Analyze test predictions for profitability
   import pandas as pd

   df = pd.read_csv('vgame_rf_model_test_predictions.csv')

   # Simulate betting on high-confidence picks
   high_conf = df[df['predicted_prob'] >= 0.30]
   win_rate = high_conf['was_correct'].mean()

   print(f"High confidence picks: {len(high_conf)}")
   print(f"Win rate: {win_rate*100:.1f}%")
   ```

## Best Practices

1. ✅ **Always use temporal split** - Never random split
2. ✅ **Collect recent data** - Racing changes over time
3. ✅ **Validate on held-out test set** - Don't tune on test data
4. ✅ **Monitor per-race accuracy** - This predicts betting performance
5. ✅ **Review feature importance** - Ensure makes sense
6. ✅ **Test before deploying** - Use test predictions to validate
7. ✅ **Update regularly** - Retrain monthly with new data

## Model Monitoring

After deployment, track:

```python
# Track actual betting performance
actual_bets = []  # Your bets
actual_bets.append({
    'date': '2026-01-11',
    'race_id': 'xxx',
    'predicted_prob': 0.35,
    'bet_amount': 100,
    'won': True,
    'payout': 350
})

# Compare to model expectations
df = pd.DataFrame(actual_bets)
actual_win_rate = df['won'].mean()
expected_win_rate = df['predicted_prob'].mean()

print(f"Expected: {expected_win_rate*100:.1f}%")
print(f"Actual: {actual_win_rate*100:.1f}%")
print(f"Difference: {(actual_win_rate - expected_win_rate)*100:.1f} percentage points")

# If actual << expected: Time to retrain!
```

## Summary

**Quick Retrain:**
```bash
# Use existing data
python train_vgame_model.py --data temporal_processed_data.csv --train-end 2025-10-31
```

**Full Pipeline:**
```bash
# 1. Collect
python collect_vgame_data.py --start 2025-01-01 --end 2025-12-31 --output data.csv

# 2. Process
python temporal_data_processor.py data.csv

# 3. Train
python train_vgame_model.py --data temporal_processed_data.csv --train-end 2025-10-31

# 4. Deploy
cp vgame_rf_model.pkl temporal_rf_model.pkl
cp vgame_rf_metadata.json temporal_rf_metadata.json
```

**Expected Results:**
- Per-race win rate: 20-25%
- ROC-AUC: 0.65-0.75
- High-confidence win rate: 30-40%
