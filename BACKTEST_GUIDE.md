# Complete Backtesting Guide

This guide covers backtesting for both **individual betting** and **system betting** strategies.

---

## Quick Reference

| Strategy | Single Date | Batch Testing |
|----------|-------------|---------------|
| **Individual Betting** | `backtest_individual.py` | `batch_backtest_individual.py` |
| **System Betting (ML)** | `backtest_ml_system.py` | `batch_backtest_ml_system.py` |

---

## Individual Betting Backtest

### What It Tests

- **Strategy**: High-EV individual WIN bets on top horses
- **Budget**: Default 500 SEK (customizable)
- **Selection**: Top 10 horses by expected value
- **Bet sizing**: Proportional to EV using Kelly criterion

### Single Date Test

```bash
# Test on specific Saturday
python backtest_individual.py --date 2024-03-09 --budget 500 --top-n 10
```

**Example Output**:
```
Top 10 bets (Total: 495 SEK):
------------------------------------------------------------
Race 4: #7 Karat River       | 125 SEK | Prob: 0.304 | ✅ WIN
         → Payout: 280 SEK (+155 SEK)
Race 2: #6 Ultra Violet      |  75 SEK | Prob: 0.193 | ❌ LOSS
...

RESULTS for 2024-03-09:
  Total bet: 495 SEK
  Winners: 3/10 (30.0%)
  Total payout: 654 SEK
  Profit: +159 SEK (+32.1% ROI)
  ✅ PROFITABLE!
```

### Batch Test (Multiple Dates)

```bash
# Test 10 random Saturdays from 2024
python batch_backtest_individual.py --years 2024 --max-per-year 10 --budget 500

# Test multiple years
python batch_backtest_individual.py --years 2024 2026 --max-per-year 5 --budget 500

# Higher budget, fewer bets
python batch_backtest_individual.py --years 2024 --max-per-year 5 --budget 1000 --top-n 5
```

**Example Summary**:
```
BATCH BACKTEST SUMMARY
================================================================================
Total dates tested: 10
Overall Performance:
  Total bet: 4,950 SEK
  Total payout: 5,234 SEK
  Total profit: +284 SEK
  Overall ROI: +5.7%
  Winners: 28/100 (28.0%)

Profitable days: 6/10 (60.0%)
Average profit per day: +28 SEK
Best day: +312 SEK
Worst day: -245 SEK
```

---

## System Betting Backtest (ML)

### What It Tests

- **Strategy**: ML-based V-game system (picks 1-5 horses per race)
- **Budget**: Default 500 SEK (automatically adjusted)
- **Selection**: ML model decides optimal picks per race
- **Win condition**: ALL races must be correct

### Single Date Test

```bash
# Test on specific Saturday
python backtest_ml_system.py --date 2024-03-09 --game V75 --budget 500
```

**Example Output**:
```
ML Model Predictions:
------------------------------------------------------------
Race 1: 2 pick(s) | Top prob: 0.285 | Gap: 0.045
  #7: Horse Name (0.285)
  #6: Horse Name (0.240)
Race 2: 5 pick(s) | Top prob: 0.220 | Gap: 0.008
  ...

System Configuration: 2 × 5 × 3 × 2 × 4 × 3 × 2
Cost: 432 SEK (reduced from initial to fit budget)

Race-by-Race Results:
Race 1: Winner #7 → ✅ HIT
Race 2: Winner #3 → ❌ MISS
...

Correct races: 5/7 (71.4%)
System hit: ❌ NO (needs 7/7)
Loss: -432 SEK
```

### Batch Test (Multiple Dates)

```bash
# Test 5 random Saturdays from 2024
python batch_backtest_ml_system.py --years 2024 --max-per-year 5 --budget 500

# Test both 2024 and 2026
python batch_backtest_ml_system.py --years 2024 2026 --max-per-year 10 --budget 500
```

**Example Summary**:
```
BATCH BACKTEST SUMMARY
================================================================================
Total systems tested: 10
System Hits: 1/10 (10.0%)
Individual Race Accuracy: 58/70 (82.9%)
Average correct per system: 5.8/7.0

Total cost: 3,240 SEK
Estimated Payout: 50,000 SEK (1 hit)
Estimated Profit: +46,760 SEK
Estimated ROI: +1,443%
```

---

## Comparison: Individual vs System

| Aspect | Individual Betting | System Betting |
|--------|-------------------|----------------|
| **Win Rate** | 25-30% of bets | <1-10% of systems |
| **Variance** | Moderate | Extreme |
| **Profit Pattern** | Small consistent gains | Rare huge wins |
| **Budget** | 500 SEK typical | 300-500 SEK typical |
| **Complexity** | Simple | Complex (all races) |
| **Best For** | Consistent profit | Jackpot hunting |

---

## Common Test Scenarios

### Test Specific High-Stakes Race Days

```bash
# Elitloppet (usually late May)
python backtest_individual.py --date 2024-05-25 --budget 1000
python backtest_ml_system.py --date 2024-05-25 --game V75 --budget 1000

# Svenskt Travderby (usually August)
python backtest_individual.py --date 2024-08-31 --budget 1000
```

### Compare Strategies on Same Date

```bash
DATE="2024-03-09"

echo "=== INDIVIDUAL BETTING ==="
python backtest_individual.py --date $DATE --budget 500

echo ""
echo "=== SYSTEM BETTING ==="
python backtest_ml_system.py --date $DATE --game V75 --budget 500
```

### Monthly Performance Analysis

```bash
# Test first Saturday of each month in 2024
for month in 01 02 03 04 05 06 07 08 09 10 11 12; do
    # Find first Saturday of month (approximate)
    date="2024-${month}-06"
    echo "=== $date ==="
    python backtest_individual.py --date $date --budget 500 2>&1 | grep "Profit:"
done
```

### Quarter Comparison

```bash
# Q1 2024
python batch_backtest_individual.py --years 2024 --max-per-year 13 --budget 500

# Save results and analyze by quarter
python -c "
import pandas as pd
df = pd.read_csv('batch_individual_backtest_results.csv')
df['date'] = pd.to_datetime(df['date'])
df['quarter'] = df['date'].dt.quarter
print(df.groupby('quarter')['profit'].sum())
"
```

---

## Understanding Results

### Individual Betting Metrics

- **Win Rate**: Should be 25-30% with good model
- **ROI**: Positive ROI over many bets indicates profitable model
- **Profitable Days**: Expect 50-60% of days to be profitable
- **Average Profit**: Small but consistent gains

### System Betting Metrics

- **System Hit Rate**: 1-10% is realistic (need ALL races correct)
- **Race Coverage**: 70-80% of individual races covered is good
- **Cost per System**: 300-500 SEK typical with budget constraints
- **Jackpot Potential**: 50K-500K SEK per hit (game dependent)

---

## Tips for Backtesting

### 1. Test on Unseen Data

✅ **Good**: Test on 2024 (before 2025 training data)
✅ **Good**: Test on 2026 (after 2025 training data)
❌ **Bad**: Test on 2025 (same as training data)

### 2. Multiple Date Testing

- Single dates can be misleading (high variance)
- Test at least 10 dates for meaningful statistics
- Compare across different seasons

### 3. Budget Management

- Start with 500 SEK budget for testing
- Individual: Can scale up to 1000-2000 SEK
- System: Higher budgets allow more picks but rarely hit

### 4. Interpreting Losses

- Individual: Losses are normal, focus on long-term ROI
- System: Most attempts lose (that's why jackpots are big!)

### 5. Data Quality

- Older dates may have incomplete data
- Some Saturdays may not have V-games
- Recent dates have better data quality

---

## Example Workflows

### Weekly Performance Check

```bash
# Check last 4 Saturdays
python batch_backtest_individual.py --years 2024 --max-per-year 4 --budget 500
```

### Strategy Optimization

```bash
# Test different budget levels
for budget in 300 500 1000; do
    echo "=== Budget: $budget SEK ==="
    python batch_backtest_individual.py --years 2024 --max-per-year 5 --budget $budget
done
```

### Model Validation

```bash
# Test on completely unseen year
python batch_backtest_individual.py --years 2024 --max-per-year 20 --budget 500
python batch_backtest_ml_system.py --years 2024 --max-per-year 20 --budget 500
```

---

## Output Files

After batch testing, results are saved to CSV:

- `batch_individual_backtest_results.csv` - Individual betting results
- `batch_backtest_results.csv` - System betting results

Analyze with:
```bash
# View summary
python -c "
import pandas as pd
df = pd.read_csv('batch_individual_backtest_results.csv')
print(df[['date', 'profit', 'roi', 'win_rate']].to_string())
"
```

---

## Troubleshooting

### "No V-game found"
- Not all Saturdays have V-games
- Try adjacent Saturdays
- Check if date is valid

### "Failed to fetch race data"
- API may be slow or rate-limited
- Try again later
- Check internet connection

### Very low win rates
- Expected for older data (model trained on 2025)
- 20-30% is normal for individual betting
- <10% is normal for system betting

---

## Next Steps

1. **Start simple**: Test a few single dates with `backtest_individual.py`
2. **Compare strategies**: Run both individual and system on same dates
3. **Batch test**: Use batch scripts for statistical significance
4. **Analyze results**: Export to CSV and analyze patterns
5. **Optimize**: Adjust budgets and parameters based on results

---

**Remember**: Past performance doesn't guarantee future results!
