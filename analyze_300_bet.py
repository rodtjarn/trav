#!/usr/bin/env python3
"""
Analyze results with 300 SEK budget on today's V85
"""

import pandas as pd
from predict_v85 import V85Predictor
from atg_api_scraper import ATGAPIScraper
import logging

logging.basicConfig(level=logging.WARNING)


def analyze_300_sek_bet():
    """Analyze betting with 300 SEK budget"""

    budget = 300
    predictor = V85Predictor()
    scraper = ATGAPIScraper(delay=0.5)

    print("\n" + "="*80)
    print(f"V85 BETTING ANALYSIS - 300 SEK BUDGET")
    print(f"Date: 2026-01-10, Track: Romme")
    print("="*80)

    # Get predictions and results
    predictions_df = predictor.predict_v85('2026-01-10')
    v85_info = scraper.get_v85_info('2026-01-10')
    race_data = scraper.scrape_date('2026-01-10')
    df_results = pd.DataFrame(race_data)
    df_results = df_results[df_results['race_id'].isin(v85_info['race_ids'])]
    df_results['v85_race_number'] = df_results['race_id'].map(v85_info['v85_race_mapping'])

    # VALUE BETTING STRATEGY with 300 SEK
    print("\n" + "="*80)
    print("💰 VALUE BETTING STRATEGY - 300 SEK")
    print("="*80)

    value_bets = []
    for race_num in sorted(predictions_df.keys()):
        race_pred = predictions_df[race_num]
        for idx, row in race_pred.iterrows():
            model_prob = row['win_probability']
            odds = row['final_odds']
            if pd.isna(odds) or odds == 0:
                continue
            implied_prob = 1 / odds
            value = model_prob - implied_prob
            if value > 0.15:
                value_bets.append({
                    'race': race_num,
                    'horse': row['horse_name'],
                    'number': int(row['start_number']),
                    'model_prob': model_prob,
                    'implied_prob': implied_prob,
                    'value': value,
                    'odds': odds
                })

    total_value = sum(bet['value'] for bet in value_bets)
    total_payout = 0
    wins = 0
    winning_bets = []

    print(f"\nFound {len(value_bets)} value bets (edge >15%)")
    print(f"Distributing 300 SEK proportionally to value edge...\n")

    for bet in value_bets:
        bet_amount = (bet['value'] / total_value) * budget

        # Check result
        race_results = df_results[df_results['v85_race_number'] == bet['race']]
        actual_winner = race_results[race_results['finish_place'] == 1].iloc[0]

        won = actual_winner['start_number'] == bet['number']

        if won:
            payout = bet_amount * bet['odds']
            total_payout += payout
            wins += 1
            winning_bets.append({
                'race': bet['race'],
                'horse': bet['horse'],
                'number': bet['number'],
                'bet': bet_amount,
                'odds': bet['odds'],
                'payout': payout,
                'profit': payout - bet_amount
            })

    print("✅ WINNING BETS:")
    print("-"*80)
    total_bet_on_winners = 0
    for w in winning_bets:
        total_bet_on_winners += w['bet']
        roi = (w['profit'] / w['bet']) * 100
        print(f"\n  V85 Race {w['race']}: {w['horse']} (#{w['number']})")
        print(f"  Bet: {w['bet']:.2f} SEK @ {w['odds']:.1f} odds")
        print(f"  Payout: {w['payout']:.2f} SEK")
        print(f"  Profit: +{w['profit']:.2f} SEK (+{roi:.0f}%)")

    profit = total_payout - budget
    roi = (profit / budget) * 100

    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Total budget: {budget:.2f} SEK")
    print(f"Winning bets: {wins}/{len(value_bets)} ({wins/len(value_bets)*100:.1f}%)")
    print(f"Amount on winners: {total_bet_on_winners:.2f} SEK")
    print(f"Total payout: {total_payout:.2f} SEK")
    print(f"Profit/Loss: {profit:+.2f} SEK")
    print(f"ROI: {roi:+.1f}%")

    if profit >= 0:
        print(f"\n🎉 PROFIT! You would have won {profit:.2f} SEK!")
        print(f"   Final balance: {budget + profit:.2f} SEK")
    else:
        print(f"\n❌ Loss of {-profit:.2f} SEK")
        print(f"   Final balance: {budget + profit:.2f} SEK")
        print(f"   Needed {-profit:.2f} SEK more in winnings to break even")

    # Break down by race
    print("\n" + "="*80)
    print("BREAKDOWN BY RACE")
    print("="*80)

    for race_num in sorted(predictions_df.keys()):
        race_results = df_results[df_results['v85_race_number'] == race_num]
        actual_winner = race_results[race_results['finish_place'] == 1].iloc[0]

        race_bets = [b for b in value_bets if b['race'] == race_num]
        race_winners = [w for w in winning_bets if w['race'] == race_num]

        race_bet_total = sum((b['value'] / total_value) * budget for b in race_bets)
        race_payout_total = sum(w['payout'] for w in race_winners)
        race_profit = race_payout_total - race_bet_total

        print(f"\n  V85 Race {race_num} (Track Race {race_results.iloc[0]['race_number']})")
        print(f"  Winner: #{int(actual_winner['start_number'])} {actual_winner['horse_name']} @ {actual_winner['final_odds']:.1f} odds")
        print(f"  Your bets: {len(race_bets)} horses, {race_bet_total:.2f} SEK total")

        if race_winners:
            print(f"  ✅ HIT! Payout: {race_payout_total:.2f} SEK, Profit: {race_profit:+.2f} SEK")
        else:
            print(f"  ❌ MISS - Lost {race_bet_total:.2f} SEK")

    print("\n" + "="*80)

    # Compare to 200 SEK budget
    print("\n📊 COMPARISON: 300 SEK vs 200 SEK")
    print("="*80)

    budget_200 = 200
    payout_200 = 136.35
    profit_200 = -63.65
    roi_200 = -31.8

    print(f"\n  200 SEK Budget:")
    print(f"    Payout: {payout_200:.2f} SEK")
    print(f"    Profit: {profit_200:+.2f} SEK")
    print(f"    ROI: {roi_200:+.1f}%")

    print(f"\n  300 SEK Budget:")
    print(f"    Payout: {total_payout:.2f} SEK")
    print(f"    Profit: {profit:+.2f} SEK")
    print(f"    ROI: {roi:+.1f}%")

    improvement = profit - profit_200
    print(f"\n  Improvement: {improvement:+.2f} SEK")

    if profit >= 0:
        print(f"  ✅ The extra 100 SEK turned loss into PROFIT!")
    else:
        still_needed = -profit
        print(f"  ⚠️  Still need {still_needed:.2f} SEK more to break even")

    # What if we had bet 293 SEK (calculated break-even)?
    print("\n" + "="*80)
    print("🎯 AT BREAK-EVEN BUDGET (293.36 SEK)")
    print("="*80)

    breakeven_budget = 293.36
    # Same return rate applies
    return_rate = total_payout / budget
    breakeven_payout = breakeven_budget * return_rate
    breakeven_profit = breakeven_payout - breakeven_budget

    print(f"\n  Budget: {breakeven_budget:.2f} SEK")
    print(f"  Expected payout: {breakeven_payout:.2f} SEK")
    print(f"  Expected profit: {breakeven_profit:+.2f} SEK")
    print(f"  Expected ROI: {(breakeven_profit/breakeven_budget)*100:+.1f}%")

    print("\n" + "="*80)


if __name__ == "__main__":
    analyze_300_sek_bet()
