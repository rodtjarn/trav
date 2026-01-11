# Backtesting Tools Quick Reference

## Which Tool Should I Use?

```
┌─────────────────────────────────────────────────────────────┐
│                  BACKTESTING DECISION TREE                   │
└─────────────────────────────────────────────────────────────┘

Want to test individual betting only?
  → backtest_race.py
    Example: python backtest_race.py --date 2026-01-10 --track Romme --game V85

Want to compare individual vs system betting?
  → backtest_race_with_system.py
    Example: python backtest_race_with_system.py --date 2026-01-10 --track Romme --game V85

Want to understand V-game systems?
  → Read DUAL_STRATEGY_GUIDE.md

Want to learn backtesting basics?
  → Read BACKTESTING_GUIDE.md
```

---

## Script Comparison

| Feature | backtest_race.py | backtest_race_with_system.py |
|---------|------------------|------------------------------|
| **Strategies** | Individual only | Individual + System |
| **Budget** | 500 SEK (customizable) | 1000 SEK (500+500) |
| **Requires --game** | Optional | Required |
| **Output complexity** | Simple | Detailed comparison |
| **Best for** | Quick analysis | Strategy comparison |
| **Use case** | Daily analysis | Deep dive analysis |

---

## Budget Breakdown

### backtest_race.py

```
Budget: 500 SEK total
  - Top 10 individual bets across all races
  - Bet sizes vary based on confidence
  - Example: 175, 81, 61, 36, 29, 25, 25, 25, 20, 20 SEK
```

### backtest_race_with_system.py

```
Total budget: 1000 SEK
  ├─ Individual: 500 SEK
  │    └─ Top 10 high-EV bets
  └─ System: 500 SEK
       └─ Reduced V-game system (2-3 picks per race)

Example:
  Individual: 495 SEK (10 bets)
  System: 384 SEK (384 rows, 2×2×2×2×2×2×3×2 combinations)
  Total: 879 SEK
```

---

## Output Examples

### backtest_race.py Output

```
🎯 MODEL PREDICTIONS (Top 10 bets):
🔥 STRONG - Race 4
   Horse #7: Karat River
   ...
   ✅ WON! Payout: 224 SEK

📊 FINAL RESULTS:
Total bet: 495 SEK
Winners: 3/10 (30.0%)
Profit: +408 SEK (+82.5% ROI)
✅ PROFITABLE DAY!
```

### backtest_race_with_system.py Output

```
STRATEGY 1: INDIVIDUAL HIGH-EV BETTING (500 SEK)
  Total bet: 495 SEK
  Winners: 3/10 (30%)
  Profit: +408 SEK

STRATEGY 2: V85 SYSTEM BETTING (500 SEK)
  System: 384 rows
  Correct races: 6/8
  Payout: 0 SEK (needs 8/8)
  Profit: -384 SEK

COMBINED RESULTS:
  Total bet: 879 SEK
  Total payout: 854 SEK
  Profit: -25 SEK (-2.8% ROI)

Strategy Comparison:
  Individual ROI: +82.5%
  System ROI: -100.0%
  → Individual betting performed better
```

---

## Command Patterns

### backtest_race.py

```bash
# Basic usage
python backtest_race.py --date 2026-01-10 --track Romme

# With V-game filter
python backtest_race.py --date 2026-01-10 --track Romme --game V85

# Custom budget
python backtest_race.py --date 2026-01-10 --track Romme --budget 1000

# Swedish characters
python backtest_race.py --date 2026-01-11 --track Ostersund --game GS75
```

### backtest_race_with_system.py

```bash
# Basic dual strategy (requires --game)
python backtest_race_with_system.py --date 2026-01-10 --track Romme --game V85

# Custom budgets
python backtest_race_with_system.py \
  --date 2026-01-10 --track Romme --game V85 \
  --individual 300 --system 700

# Only system betting (set individual to minimum)
python backtest_race_with_system.py \
  --date 2026-01-10 --track Romme --game V85 \
  --individual 50 --system 950
```

---

## When to Use Each

### Use `backtest_race.py` When:

✅ You want quick performance analysis
✅ You're testing the model's individual betting capability
✅ You don't need system betting comparison
✅ You want simpler, cleaner output
✅ You're analyzing many days quickly

### Use `backtest_race_with_system.py` When:

✅ You want to compare both strategies
✅ You're curious about V-game system potential
✅ You want to see jackpot vs consistent profit tradeoff
✅ You're willing to invest more per analysis (1000 SEK)
✅ You want comprehensive strategy insights

---

## Performance Expectations

### Individual Betting (backtest_race.py)

Based on 29.5% model accuracy:

| Metric | Expected Range |
|--------|----------------|
| Win rate | 20-35% |
| Average ROI | -20% to +100% |
| Variance | Moderate |
| Profit pattern | Small consistent gains or losses |

### System Betting (backtest_race_with_system.py)

Reality of V-game systems:

| Metric | Expected Range |
|--------|----------------|
| Win rate | <1% (need all races correct) |
| Average ROI | -80% to -100% (occasional jackpot) |
| Variance | Extreme |
| Profit pattern | Frequent total loss, rare massive win |

### Dual Strategy

Combines both:

| Metric | Expected Range |
|--------|----------------|
| Win rate | 10-30% (individual portion) |
| Average ROI | -30% to +50% |
| Variance | Moderate-High |
| Profit pattern | Individual bets offset some system losses |

---

## Example Analysis Workflow

### Quick Daily Check
```bash
# Just check individual betting performance
python backtest_race.py --date 2026-01-10 --track Romme --game V85
```

### Deep Strategy Analysis
```bash
# Compare both strategies
python backtest_race_with_system.py --date 2026-01-10 --track Romme --game V85
```

### Multi-Day Comparison
```bash
# Check last 3 Saturdays
for date in 2025-12-27 2026-01-03 2026-01-10; do
  echo "=== $date ==="
  python backtest_race.py --date $date --track Romme --game V85 | grep "Profit:"
done
```

### Strategy Optimization
```bash
# Test different budgets
python backtest_race.py --date 2026-01-10 --track Romme --budget 300
python backtest_race.py --date 2026-01-10 --track Romme --budget 500
python backtest_race.py --date 2026-01-10 --track Romme --budget 1000
```

---

## Files Overview

```
/home/per/Work/trav/
├── backtest_race.py                    # Individual betting analysis
├── backtest_race_with_system.py        # Dual strategy analysis
├── BACKTESTING_GUIDE.md                # Comprehensive backtesting guide
├── DUAL_STRATEGY_GUIDE.md              # Strategy comparison guide
└── BACKTEST_COMPARISON.md              # This file (quick reference)
```

---

## Tips

1. **Start simple**: Use `backtest_race.py` first
2. **Understand individual betting** before system betting
3. **Don't expect system wins**: They're rare (that's why jackpots are huge)
4. **Use dual strategy** to see the contrast
5. **Focus on individual betting** for ML-based profit
6. **System betting is entertainment** with jackpot potential, not a profit strategy

---

## Key Takeaway

**For profit with ML**: Use individual betting (`backtest_race.py`)
**For entertainment + analysis**: Use dual strategy (`backtest_race_with_system.py`)

The model's 29.5% accuracy is excellent for individual betting but insufficient for consistent V-game system wins (which require 100% accuracy across 6-8 races).
