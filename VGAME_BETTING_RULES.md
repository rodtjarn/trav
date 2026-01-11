# Swedish V-Game Betting Systems - Complete Guide

This document explains all the V-game betting systems available through ATG (Swedish Horse Betting).

## What are V-Games?

V-games are **pool betting systems** where you pick winners across multiple consecutive races. The "V" stands for "Vinnare" (Swedish for "winner"). All bettors who pick all correct winners share the prize pool.

## Game Types & Rules

### V75 (The Big One)
- **Races**: 7 consecutive races
- **Objective**: Pick the winner of all 7 races
- **Days**: Typically **Saturdays** (Sweden's biggest betting day)
- **Pool Size**: Often 30-100+ million SEK
- **Special**: "Jackpot" when no one wins - pool rolls over to next week
- **Testing**:
  ```bash
  python create_bet.py --game V75
  # Should find Saturday races if available
  ```

### V86 (Wednesday & Saturday)
- **Races**: 8 consecutive races (note: more races than V75!)
- **Objective**: Pick the winner of all 8 races
- **Days**: **Wednesdays** and some **Saturdays**
- **Pool Size**: 20-60 million SEK
- **Special**: Two versions - "Dubbel V86" on some days (two separate V86 games)
- **Testing**:
  ```bash
  python create_bet.py --game V86
  # Should find Wednesday or Saturday races
  ```

### V85 (Friday Special)
- **Races**: 8 consecutive races
- **Objective**: Pick the winner of all 8 races
- **Days**: **Fridays** (exclusively Friday game)
- **Pool Size**: 10-30 million SEK
- **Special**: Friday night racing, often at Solvalla or Jägersro
- **Testing**:
  ```bash
  python create_bet.py --game V85
  # Should find Friday races
  ```

### V65 (Everyday Game)
- **Races**: 6 consecutive races
- **Objective**: Pick the winner of all 6 races
- **Days**: **Any day** - most common V-game
- **Pool Size**: 2-10 million SEK
- **Special**: Most frequent game, available almost daily
- **Testing**:
  ```bash
  python create_bet.py --game V65
  # Should find races any day of the week
  ```

### V64 (Four Days/Week)
- **Races**: 6 consecutive races (same as V65)
- **Objective**: Pick the winner of all 6 races
- **Days**: **Tuesdays, Thursdays, Sundays** primarily
- **Pool Size**: 3-8 million SEK
- **Special**: Alternative to V65 on certain days
- **Testing**:
  ```bash
  python create_bet.py --game V64
  # Should find Tue/Thu/Sun races
  ```

### V5 (Small Game)
- **Races**: 5 consecutive races
- **Objective**: Pick the winner of all 5 races
- **Days**: Variable - often on smaller race days
- **Pool Size**: 1-3 million SEK
- **Special**: Easier to win, smaller pools
- **Testing**:
  ```bash
  python create_bet.py --game V5
  # Should find races on smaller racing days
  ```

### V4 (Mini Game)
- **Races**: 4 consecutive races
- **Objective**: Pick the winner of all 4 races
- **Days**: Variable - often early afternoon
- **Pool Size**: 500k-2 million SEK
- **Special**: Smallest V-game, highest win probability
- **Testing**:
  ```bash
  python create_bet.py --game V4
  # Should find early afternoon races
  ```

### V3 (Quick Game)
- **Races**: 3 consecutive races
- **Objective**: Pick the winner of all 3 races
- **Days**: Variable - lunch racing
- **Pool Size**: 200k-1 million SEK
- **Special**: Very quick, lunchtime betting
- **Testing**:
  ```bash
  python create_bet.py --game V3
  # Should find lunch races
  ```

### GS75 (Grand Slam 75)
- **Races**: 7 races across **4 different race meetings**
- **Objective**: Pick the winner of all 7 races
- **Days**: **Saturdays** (4 times per year - major events)
- **Pool Size**: 100+ million SEK (HUGE)
- **Special**: Sweden's biggest betting event, combines V75 with elite races
- **Testing**:
  ```bash
  python create_bet.py --game GS75
  # Only available ~4 times per year
  ```

## Betting Strategies

### System Betting
You can bet **multiple horses per race** to increase winning chances:

**Example V65 System:**
- Race 1: Pick 2 horses (A, B)
- Race 2: Pick 1 horse (C)
- Race 3: Pick 3 horses (D, E, F)
- Race 4: Pick 1 horse (G)
- Race 5: Pick 2 horses (H, I)
- Race 6: Pick 1 horse (J)

**Total combinations**: 2 × 1 × 3 × 1 × 2 × 1 = **12 rows**
**Cost**: 12 rows × 1 SEK = 12 SEK (minimum bet per row varies)

### Individual Win Betting (Our Tool's Strategy)
Instead of system betting, `create_bet.py` uses **individual win bets**:
- Pick high-probability horses from the V-game races
- Bet on each horse separately (not combined)
- Lower cost, focused on horses with best expected value
- Not eligible for jackpot, but better ROI on individual wins

## How to Verify Test Results

### Expected Patterns

1. **Weekly Pattern:**
   ```
   Monday:    V65
   Tuesday:   V64, V65
   Wednesday: V86, V65
   Thursday:  V64, V65
   Friday:    V85, V65
   Saturday:  V75, V86, V65
   Sunday:    V64, V65
   ```

2. **Race Count:**
   - Tool should display correct number of races (e.g., "Found 8 V86 races")
   - Game ID should match game type (e.g., "V86_2026-01-14_40_1")

3. **Label Consistency:**
   - Output should say "V86 Race 1", "V86 Race 2", etc.
   - NOT "V85 Race" when searching for V86

### Test Checklist

```bash
# Test all major game types
python create_bet.py --game V75  # Should find Saturday
python create_bet.py --game V86  # Should find Wed/Sat
python create_bet.py --game V85  # Should find Friday
python create_bet.py --game V65  # Should find any day
python create_bet.py --game V64  # Should find Tue/Thu/Sun

# Verify output
# ✓ Correct game type in title: "OPTIMAL V65 BETTING SLIP"
# ✓ Correct race count: "Found 6 V65 races" (for V65)
# ✓ Correct race labels: "V65 Race 1", "V65 Race 2", etc.
# ✓ Correct game ID: "V65_YYYY-MM-DD_XX_X"
```

## Important Notes

### Our Tool's Approach
- **Individual betting strategy** - NOT pool/system betting
- Analyzes V-game races but bets on individual winners
- Uses ML model to find high-EV horses across the game's races
- More flexible than traditional system betting
- Better for bankroll management

### Why This Matters
Traditional V-game betting requires picking ALL winners correctly to win the pool. Our tool instead:
1. Identifies the races in the V-game
2. Analyzes all horses in those races
3. Selects the best individual betting opportunities
4. Places separate WIN bets (not combined system bets)

**Result**: You can profit even if you don't hit all races, with better overall ROI.

## Common Issues

### "No VXX found in next 30 days"
- **Normal** for rare games (V75, GS75)
- V75 is only on Saturdays
- GS75 is only 4 times per year
- Try a different game type

### Wrong Race Count
- If V65 shows 8 races instead of 6 → API returned wrong game type
- Report this as a bug

### Missing Game Types
Some smaller game types may not be available in all regions or time periods.

## API Game Type Codes

ATG API uses these exact codes (case-sensitive):
```
V75, V86, V85, V65, V64, V5, V4, V3, GS75
```

Other codes may exist for international games or special events.
