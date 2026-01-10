#!/usr/bin/env python3
"""
Calculate break-even point for betting strategies
"""

import pandas as pd
from betting_system import V85BettingSystem
import logging

logging.basicConfig(level=logging.WARNING)


def analyze_break_even():
    """Calculate how much budget needed to break even or profit"""

    print("\n" + "="*80)
    print("BREAK-EVEN ANALYSIS - V85 Romme 2026-01-10")
    print("="*80)

    # Value Betting Strategy results
    print("\n📊 VALUE BETTING STRATEGY")
    print("-"*80)

    # From the actual run:
    bet_amount = 200
    total_payout = 136.35
    loss = -63.65

    # Winners in value betting strategy:
    winners = [
        {'race': 1, 'horse': 'Nytomt Amira', 'number': 2, 'bet': 2.70, 'odds': 9.9, 'payout': 26.68},
        {'race': 4, 'horse': 'Karat River', 'number': 7, 'bet': 4.44, 'odds': 5.8, 'payout': 25.55},
        {'race': 5, 'horse': 'Timotejs Messenger', 'number': 1, 'bet': 3.99, 'odds': 21.1, 'payout': 84.12},
    ]

    print(f"\nWith 200 SEK budget:")
    print(f"  Total payout: {total_payout:.2f} SEK")
    print(f"  Loss: {loss:.2f} SEK")
    print(f"  Return rate: {total_payout/bet_amount*100:.1f}%")

    # Calculate break-even budget
    return_rate = total_payout / bet_amount
    breakeven_budget = bet_amount / return_rate

    print(f"\n💰 To break even (0 SEK profit):")
    print(f"  Required budget: {breakeven_budget:.2f} SEK")
    print(f"  Extra needed: +{breakeven_budget - bet_amount:.2f} SEK ({(breakeven_budget/bet_amount - 1)*100:.1f}% more)")

    # Calculate budget for different profit targets
    print(f"\n🎯 Budget needed for profit targets:")

    for target_profit in [50, 100, 200, 500]:
        required_total = bet_amount + target_profit
        required_budget = required_total / return_rate
        print(f"  +{target_profit} SEK profit: {required_budget:.2f} SEK budget ({required_budget/bet_amount:.1f}x)")

    # Show what winnings looked like
    print(f"\n✅ Winning bets (3 out of 47):")
    for w in winners:
        roi = (w['payout'] - w['bet']) / w['bet'] * 100
        print(f"  Race {w['race']}: {w['horse']} #{w['number']}")
        print(f"    Bet: {w['bet']:.2f} SEK @ {w['odds']:.1f} odds → Payout: {w['payout']:.2f} SEK (+{roi:.0f}%)")

    # Perfect hindsight scenario
    print("\n" + "="*80)
    print("🔮 PERFECT HINDSIGHT SCENARIO")
    print("="*80)
    print("\nIf you ONLY bet on the 3 winners (impossible without time machine):")

    equal_split = 200 / 3
    total_perfect_payout = sum(equal_split * w['odds'] for w in winners)
    perfect_profit = total_perfect_payout - 200

    print(f"\nStrategy: Split 200 SEK equally across 3 winners")
    for w in winners:
        payout = equal_split * w['odds']
        print(f"  {equal_split:.2f} SEK on {w['horse']} @ {w['odds']:.1f} → {payout:.2f} SEK")

    print(f"\n  Total payout: {total_perfect_payout:.2f} SEK")
    print(f"  Profit: +{perfect_profit:.2f} SEK (+{perfect_profit/200*100:.0f}%)")

    # Optimal hindsight: all on best odds
    best_winner = max(winners, key=lambda x: x['odds'])
    all_in_payout = 200 * best_winner['odds']
    all_in_profit = all_in_payout - 200

    print(f"\nBest possible: All 200 SEK on {best_winner['horse']} @ {best_winner['odds']:.1f} odds")
    print(f"  Payout: {all_in_payout:.2f} SEK")
    print(f"  Profit: +{all_in_profit:.2f} SEK (+{all_in_profit/200*100:.0f}%)")

    # Top Pick Per Race strategy
    print("\n" + "="*80)
    print("📊 TOP PICK PER RACE STRATEGY")
    print("="*80)

    top_pick_payout = 66.50  # Only Fahrenheit won
    top_pick_loss = -133.50
    top_pick_return_rate = top_pick_payout / 200
    top_pick_breakeven = 200 / top_pick_return_rate

    print(f"\nWith 200 SEK budget (25 SEK per race):")
    print(f"  Wins: 1/8 (Fahrenheit @ 2.7 odds)")
    print(f"  Total payout: {top_pick_payout:.2f} SEK")
    print(f"  Loss: {top_pick_loss:.2f} SEK")
    print(f"  Return rate: {top_pick_return_rate*100:.1f}%")

    print(f"\n💰 To break even:")
    print(f"  Required budget: {top_pick_breakeven:.2f} SEK")
    print(f"  Extra needed: +{top_pick_breakeven - 200:.2f} SEK ({(top_pick_breakeven/200 - 1)*100:.1f}% more)")

    print(f"\n⚠️  This strategy would need {top_pick_breakeven/200:.1f}x the budget just to break even!")
    print(f"     Not viable unless win rate improves significantly.")

    # Minimum wins needed
    print("\n" + "="*80)
    print("📈 WHAT WOULD IT TAKE TO PROFIT?")
    print("="*80)

    print("\nValue Betting Strategy (47 bets, currently 3 wins):")
    # Current: 3 wins got us 136.35 SEK from 200 SEK
    # Winners paid: 26.68, 25.55, 84.12 = 136.35
    # If we had 4 wins, adding an average winner
    avg_winner_payout = sum(w['payout'] for w in winners) / len(winners)

    for extra_wins in range(1, 6):
        simulated_payout = total_payout + (extra_wins * avg_winner_payout)
        simulated_profit = simulated_payout - 200

        print(f"  {3 + extra_wins} wins (instead of 3): {simulated_payout:.2f} SEK payout, {simulated_profit:+.2f} SEK")
        if simulated_profit > 0 and extra_wins == 1:
            print(f"    ✅ Just 1 more average winner would have made profit!")

    print("\nTop Pick Per Race (8 bets, currently 1 win):")
    # Won on Fahrenheit at 2.7 odds = 66.50 payout from 25 SEK bet
    # Average odds of our top picks: let's estimate
    avg_odds = 10.9  # Rough average of the favorites

    for total_wins in range(2, 5):
        simulated_payout = total_wins * 25 * 2.7  # Assuming similar odds
        simulated_profit = simulated_payout - 200

        print(f"  {total_wins}/8 wins: {simulated_payout:.2f} SEK payout, {simulated_profit:+.2f} SEK")
        if simulated_profit > 0:
            print(f"    ✅ Need at least {total_wins} winners to profit")
            break

    print("\n" + "="*80)
    print("💡 KEY INSIGHTS")
    print("="*80)
    print("""
1. Value Betting needed 293 SEK (46% more) to break even
   - Just one more average winner would have made it profitable
   - 4 wins out of 47 bets = 8.5% hit rate for profit

2. Top Pick Per Race needed 601 SEK (3x more!) to break even
   - Would need 3/8 winners (37.5% hit rate) to profit
   - Model only got 1/8 winners correct (12.5%)

3. The problem: Model is overconfident in favorites
   - Highest probability picks (>50%) went 0/5
   - Best value was in 4th-ranked horse (Timotejs Messenger)

4. Best approach: More selective value betting
   - Focus on edge >20% (not 15%)
   - Reduce number of bets to increase hit rate
   - May improve from 6.4% (3/47) to 8.5% (4/47) needed for profit
""")

    print("="*80)


if __name__ == "__main__":
    analyze_break_even()
