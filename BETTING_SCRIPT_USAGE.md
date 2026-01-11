# V85 Betting Script - Usage Guide

## Quick Start

The `create_bet.py` script automatically finds the next V85 race and generates an optimal betting slip based on your budget.

### Basic Usage

```bash
source venv/bin/activate

# Auto-find next V85, bet 300 SEK (default)
python create_bet.py

# Auto-find next V85, bet 1000 SEK
python create_bet.py --total 1000

# Specific date and amount
python create_bet.py --total 500 --date 2026-01-17
```

---

## Command-Line Options

### `--total SEK`
Total betting budget in SEK (default: 300)

**Examples:**
```bash
python create_bet.py --total 500   # Bet 500 SEK
python create_bet.py --total 1000  # Bet 1000 SEK
python create_bet.py --total 2000  # Bet 2000 SEK
```

**Valid range:** 50 - 10,000 SEK
- Minimum: 50 SEK
- Maximum: 10,000 SEK (with warning about risk management)

### `--date YYYY-MM-DD`
Specific V85 race date (optional)

**Examples:**
```bash
python create_bet.py --date 2026-01-17         # Use specific date
python create_bet.py --total 1000 --date 2026-01-17  # Combine options
```

**If omitted:** Script automatically searches for the next V85 race (up to 30 days ahead)

---

## How It Works

### 1. **Auto-Find Next V85** (if --date not specified)
```
🔍 Searching for next V85 race day...

✅ Found V85 on 2026-01-17
   Track: Solvalla (ID: 5)
   Number of races: 8
```

The script searches ATG.se for the next V85 race day automatically.

### 2. **Analyze All Races**
```
📊 Analyzing V85 Race 1...
   ✓ Analyzed 12 horses
📊 Analyzing V85 Race 2...
   ✓ Analyzed 11 horses
...
```

Uses the temporal model to predict win probabilities for each horse.

### 3. **Select Best Bets**
Selects 3-4 races with highest Expected Value (EV > 1.2):
- 🔥 **EXCELLENT**: >40% win prob, 2-4 odds → WIN bet
- ⭐ **GOOD**: 25-40% win prob, 4-8 odds → WIN bet
- 💎 **VALUE**: 20-30% win prob, 8-15 odds → PLACE bet
- 🎲 **LONGSHOT**: 15-25% win prob, >15 odds → PLACE bet

### 4. **Allocate Budget**
Budget is distributed proportionally:
- **Best opportunity**: 35% of budget
- **2nd-3rd best**: 25% each
- **4th-6th best**: 20% each

**Example with 1000 SEK budget:**
- Race 3 (EXCELLENT): 350 SEK
- Race 5 (GOOD): 163 SEK (25% of remaining 650 SEK)
- Race 7 (VALUE): 122 SEK (25% of remaining 488 SEK)
- Race 2 (LONGSHOT): 73 SEK (20% of remaining 366 SEK)

Total: ~708 SEK allocated

### 5. **Show Profit Scenarios**
```
💰 PROFIT SCENARIOS:

Conservative (2/4 win):
  Payout: 1,540 SEK
  Profit: +832 SEK (+118% ROI)

Expected (3/4 win):
  Payout: 2,890 SEK
  Profit: +2,182 SEK (+308% ROI)

Best case (4/4 win):
  Payout: 4,120 SEK
  Profit: +3,412 SEK (+482% ROI)
```

---

## Budget Scaling Examples

### 300 SEK Budget (Conservative)
```bash
python create_bet.py
```
- 3-4 bets
- 70-100 SEK per bet
- Good for testing strategy

### 500 SEK Budget (Moderate)
```bash
python create_bet.py --total 500
```
- 3-4 bets
- 100-175 SEK per bet
- Balanced risk/reward

### 1000 SEK Budget (Aggressive)
```bash
python create_bet.py --total 1000
```
- 3-4 bets
- 200-350 SEK per bet
- Higher profit potential, higher variance

### 2000 SEK Budget (Very Aggressive)
```bash
python create_bet.py --total 2000
```
- 3-5 bets
- 400-700 SEK per bet
- Maximum profit potential
- ⚠️ **Requires large bankroll** (20,000+ SEK recommended)

---

## Expected Results by Budget

Based on temporal model validation (21.5% win rate, +96.6% ROI):

| Budget | Conservative (40% win) | Expected (60% win) | Best Case (80% win) | Bad Day |
|--------|----------------------|-------------------|-------------------|---------|
| **300 SEK** | +100-300 SEK | +500-1,200 SEK | +1,700-3,700 SEK | -300 SEK |
| **500 SEK** | +200-500 SEK | +800-2,000 SEK | +2,800-6,200 SEK | -500 SEK |
| **1,000 SEK** | +400-1,000 SEK | +1,600-4,000 SEK | +5,600-12,400 SEK | -1,000 SEK |
| **2,000 SEK** | +800-2,000 SEK | +3,200-8,000 SEK | +11,200-24,800 SEK | -2,000 SEK |

---

## Full Output Example

```bash
$ python create_bet.py --total 1000

🔍 Searching for next V85 race day...

✅ Found V85 on 2026-01-17
   Track: Solvalla (ID: 5)
   Number of races: 8

================================================================================
🎯 OPTIMAL V85 BETTING SLIP
📅 Date: 2026-01-17
💰 Budget: 1000 SEK
📊 Strategy: Individual High-EV Betting (Balanced Profit)
================================================================================

🔍 Fetching race data and generating predictions...

✓ Found 8 V85 races

📊 Analyzing V85 Race 1...
   ✓ Analyzed 12 horses
📊 Analyzing V85 Race 2...
   ✓ Analyzed 11 horses
...

================================================================================

🎯 SELECTED BETS (Top EV opportunities):

🔥 EXCELLENT - V85 Race 3
   Horse #5: Fast Willie
   Win probability: 42.3%
   Estimated odds: 2.8
   Expected Value: 1.18
   → WIN bet: 350 SEK

⭐ GOOD - V85 Race 5
   Horse #8: Lucky Star
   Win probability: 28.1%
   Estimated odds: 6.5
   Expected Value: 1.83
   → WIN bet: 163 SEK

💎 VALUE - V85 Race 7
   Horse #11: Dark Thunder
   Win probability: 21.5%
   Estimated odds: 18.0
   Expected Value: 2.87
   → PLACE bet: 122 SEK

🎲 LONGSHOT - V85 Race 2
   Horse #9: Silent Runner
   Win probability: 17.2%
   Estimated odds: 25.0
   Expected Value: 3.30
   → PLACE bet: 73 SEK

================================================================================
📋 BETTING SLIP SUMMARY
================================================================================
Total bets: 4
Total amount: 708 SEK
Remaining budget: 292 SEK

💰 PROFIT SCENARIOS:

Conservative (2/4 win):
  Payout: 1,540 SEK
  Profit: +832 SEK (+118% ROI)

Expected (3/4 win):
  Payout: 2,890 SEK
  Profit: +2,182 SEK (+308% ROI)

Best case (3/4 win):
  Payout: 4,120 SEK
  Profit: +3,412 SEK (+482% ROI)

================================================================================

📝 NOTES:
- Odds are estimated based on model probabilities
- Check actual ATG odds before placing bets
- High EV (>1.2) indicates profitable bets long-term
- Expect variance - not all bets will win
- This is a realistic strategy based on temporal model validation

Good luck! 🍀
```

---

## Tips

### ✅ DO:
- Start with 300-500 SEK to test the strategy
- Check actual ATG odds before betting (script estimates)
- Track results to validate model performance
- Adjust budget based on your bankroll (5-10% max per V85)

### ❌ DON'T:
- Bet more than 10% of total bankroll on one V85
- Override model picks with "gut feelings"
- Increase budget after losing days (no chasing losses)
- Expect to win every time (21.5% win rate means 78.5% losing bets)

---

## Troubleshooting

### "No V85 found in next 30 days"
- V85 races typically run weekly on Saturdays
- Check ATG.se manually for V85 schedule
- Use `--date YYYY-MM-DD` to specify a known V85 date

### "No good betting opportunities found"
- Model found no bets with EV > 1.2
- **Recommendation**: SKIP this V85, wait for better opportunities
- This is normal - not every V85 is worth betting

### "Minimum budget is 50 SEK"
- Increase budget to at least 50 SEK
- Smaller amounts don't allow proper bet distribution

### Script estimates vs ATG odds differ
- Script estimates odds from win probabilities
- **Always check actual ATG odds** before betting
- Discrepancies are normal (betting market vs model predictions)

---

## Related Files

- **V85_300SEK_BETTING_STRATEGY.md** - Complete strategy guide
- **300SEK_QUICK_START.md** - Quick reference
- **temporal_rf_model.pkl** - Trained temporal model
- **temporal_rf_metadata.json** - Model metadata

---

## Strategy Background

Based on temporal model analysis:
- **21.5% win rate** (realistic for horse racing)
- **+96.6% ROI** over 10 V85 races
- **Individual betting beats V85 system** (0/10 V85 jackpots)

The script uses Expected Value (EV) to select profitable bets:
```
EV = (Win Probability × Odds) - 1

EV > 1.2 = Profitable long-term
```

Focus on high-EV individual races instead of trying to hit all 8 winners for V85 jackpot.
