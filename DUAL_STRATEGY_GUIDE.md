# Dual Strategy Backtesting Guide

Compare individual betting vs V-game system betting strategies.

## Two Backtesting Scripts

### 1. `backtest_race.py` - Individual Betting Only

**Use for**: Analyzing individual high-EV betting performance

```bash
python backtest_race.py --date 2026-01-10 --track Romme --game V85
```

**Strategy**:
- Picks top 10 horses across all races
- Places individual WIN bets
- Budget: 500 SEK (customizable)

**Best for**:
- General performance analysis
- Understanding which horses the model identifies
- Simpler, cleaner output

---

### 2. `backtest_race_with_system.py` - Dual Strategy

**Use for**: Comparing individual betting vs V-game system betting

```bash
python backtest_race_with_system.py --date 2026-01-10 --track Romme --game V85
```

**Strategies**:
1. **Individual betting**: 500 SEK on high-EV horses (top 10 picks)
2. **System betting**: 500 SEK on V-game system (pick 2-3 horses per race)

**Total budget**: 1000 SEK (500 + 500)

**Best for**:
- Understanding risk/reward tradeoffs
- Comparing consistent small wins vs jackpot potential
- Strategy optimization

---

## Understanding V-Game System Betting

### How V-Game Systems Work

Traditional V-game betting requires picking the winner of ALL races:

| Game | Races | Prize Pool | Typical Payout |
|------|-------|------------|----------------|
| V75  | 7     | 30-100M    | 100K - 10M SEK |
| V86  | 8     | 20-60M     | 50K - 5M SEK |
| V85  | 8     | 10-30M     | 20K - 500K SEK |
| GS75 | 7     | 100M+      | 1M - 100M SEK |
| V65  | 6     | 2-10M      | 10K - 200K SEK |

### Reduced System Strategy

Instead of picking just ONE horse per race (expensive and risky), pick 2-3:

**Example V85 System (8 races)**:
```
Race 1: Pick 2 horses
Race 2: Pick 2 horses
Race 3: Pick 2 horses
Race 4: Pick 2 horses
Race 5: Pick 2 horses
Race 6: Pick 2 horses
Race 7: Pick 3 horses
Race 8: Pick 2 horses

Total combinations: 2×2×2×2×2×2×3×2 = 384 rows
Cost: 384 SEK (1 SEK per row)
```

**To win**: ALL 8 races must have a winner in your picks

**Model selection logic**:
- Top horse >35% probability → Pick 1 (banker)
- Top horse 25-35% → Pick top 2
- Top horse <25% → Pick top 3
- Auto-reduce if cost exceeds budget

---

## Example Results

### Romme V85 - January 10, 2026

```
INDIVIDUAL BETTING (500 SEK):
  Winners: 1/10 (10%)
  Payout: 201 SEK
  Profit: -293 SEK (-59% ROI)

SYSTEM BETTING (500 SEK):
  System: 384 rows
  Correct races: 1/8
  Payout: 0 SEK (needs 8/8 correct)
  Profit: -384 SEK (-100% ROI)

COMBINED:
  Total bet: 878 SEK
  Total payout: 201 SEK
  Profit: -677 SEK (-77% ROI)
  Winner: Individual betting performed better
```

### Östersund GS75 - January 11, 2026

```
INDIVIDUAL BETTING (500 SEK):
  Winners: 0/5 (0%)
  Payout: 0 SEK
  Profit: -495 SEK (-100% ROI)

SYSTEM BETTING (500 SEK):
  System: 432 rows
  Correct races: 1/7
  Payout: 0 SEK (needs 7/7 correct)
  Profit: -432 SEK (-100% ROI)

COMBINED:
  Total bet: 927 SEK
  Total payout: 0 SEK
  Profit: -927 SEK (-100% ROI)
  Winner: Both strategies lost
```

---

## Key Insights

### Individual Betting

**Pros**:
- ✅ More consistent returns
- ✅ Can profit even with 20-30% win rate
- ✅ Lower variance
- ✅ Easier to understand and track

**Cons**:
- ❌ Limited upside (max ~10x return per bet)
- ❌ Cannot win huge jackpots
- ❌ Requires many races to see long-term results

**Expected performance** (based on 29.5% model accuracy):
- Win rate: 20-30% of bets
- ROI: Variable, depends on odds found
- Profit: Small consistent gains over many races

### System Betting

**Pros**:
- ✅ Jackpot potential (100K - 10M+ SEK)
- ✅ Can win huge amounts with single bet
- ✅ Lower cost per attempt vs full system
- ✅ Exciting - all-or-nothing appeal

**Cons**:
- ❌ Very low win probability (need ALL races correct)
- ❌ High variance - mostly losses
- ❌ Cannot partially profit (all-or-nothing)
- ❌ Difficult to beat with ML (too many unknowns)

**Expected performance**:
- Win rate: <1% (getting 7-8 races all correct is rare)
- ROI: Highly negative on average, occasional jackpot
- Profit: Extreme variance - big loss streaks, rare huge wins

---

## When to Use Each Strategy

### Use Individual Betting When:
- You want consistent, measurable results
- You're testing/validating the model
- You prefer lower risk
- You have a smaller bankroll
- You want to profit from ML predictions

### Use System Betting When:
- You're OK with high risk/high reward
- You want jackpot potential
- You can afford to lose the stake
- You're playing for entertainment
- You want to compare ML vs traditional betting

### Use Dual Strategy When:
- You want both: consistent profits + jackpot shots
- You can afford 1000 SEK per race day
- You want to hedge (individual covers system losses)
- You want comprehensive performance analysis

---

## Command Examples

```bash
# Individual betting only (500 SEK)
python backtest_race.py --date 2026-01-10 --track Romme --game V85

# Dual strategy (1000 SEK total)
python backtest_race_with_system.py --date 2026-01-10 --track Romme --game V85

# Custom budgets for dual strategy
python backtest_race_with_system.py \
  --date 2026-01-10 --track Romme --game V85 \
  --individual 300 --system 700

# Compare multiple days
for date in 2025-11-01 2025-12-27 2026-01-10; do
  echo "=== $date ==="
  python backtest_race_with_system.py --date $date --track Romme --game V85 \
    | grep "Strategy Comparison" -A 3
done
```

---

## Recommendations

### For Profit Maximization:
**Use individual betting.** The model's 29.5% accuracy is good for individual bets but not enough for consistent system wins.

### For Entertainment:
**Use dual strategy.** Get regular action from individual bets + occasional jackpot excitement from system.

### For Analysis:
**Use both scripts.** Compare performance to understand risk/reward tradeoffs.

### Realistic Expectations:

**Individual betting**:
- Expect 20-30% of your picks to win
- Expect slow, steady profit accumulation (or small losses)
- Variance is moderate

**System betting**:
- Expect to lose most attempts (95%+)
- Expect occasional big wins (rare)
- Variance is extreme

**Dual strategy**:
- Individual bets cover some of system losses
- System provides jackpot upside
- More balanced risk/reward profile

---

## Technical Notes

### Payout Estimation

System payouts are **estimated** in the script based on typical dividends:

| Game  | Conservative Estimate |
|-------|-----------------------|
| V75   | 200,000 SEK          |
| V86   | 100,000 SEK          |
| V85   | 50,000 SEK           |
| GS75  | 500,000 SEK          |
| V65   | 30,000 SEK           |
| V64   | 20,000 SEK           |

**Actual payouts vary wildly** based on:
- Pool size that day
- Number of correct bettors
- Difficulty of races
- Jackpot carryover

Real V75 payouts have ranged from 10,000 SEK to 100,000,000+ SEK.

### System Cost Calculation

Cost = Number of rows × 1 SEK per row

**Example**:
- Pick 2, 2, 2, 2, 2, 2, 3, 2 horses = 2×2×2×2×2×2×3×2 = 384 rows
- Cost = 384 SEK

The script automatically reduces picks if cost exceeds budget.

---

## Related Tools

- `create_bet.py` - Generate live betting slips (individual betting)
- `backtest_race.py` - Analyze individual betting only
- `backtest_race_with_system.py` - Analyze dual strategy
- `BACKTESTING_GUIDE.md` - Full backtesting documentation
- `VGAME_BETTING_RULES.md` - V-game rules and details
