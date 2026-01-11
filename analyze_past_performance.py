#!/usr/bin/env python3
"""
Analyze model performance on past race day
Shows predictions vs actual results
"""

import pandas as pd
import pickle
import sys
from datetime import datetime

def analyze_race_day(date_str, track_name=None, budget=500):
    """Analyze how the model would have performed on a specific date"""

    # Load model
    with open('temporal_rf_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Load data
    df = pd.read_csv('temporal_processed_data_vgame.csv')

    # Filter for date and track
    if track_name:
        race_data = df[(df['date'] == date_str) & (df['track_name'] == track_name)].copy()
    else:
        race_data = df[df['date'] == date_str].copy()

    if len(race_data) == 0:
        print(f"No data found for {date_str}" + (f" at {track_name}" if track_name else ""))
        return

    track = race_data['track_name'].iloc[0]
    num_races = len(race_data['race_number'].unique())

    print("="*80)
    print(f"MODEL PERFORMANCE ANALYSIS - {track.upper()}")
    print(f"Date: {date_str}")
    print(f"Total Budget: {budget} SEK")
    print("="*80)
    print()

    # Get feature columns from the model
    feature_cols = model.feature_names_in_

    # Make sure all features are present
    missing_features = set(feature_cols) - set(race_data.columns)
    if missing_features:
        print(f"Warning: Missing features: {missing_features}")
        # Add missing features with default values
        for feat in missing_features:
            race_data[feat] = 0

    # Make predictions
    X = race_data[feature_cols]
    predictions = model.predict_proba(X)[:, 1]

    race_data['predicted_prob'] = predictions
    race_data['estimated_odds'] = race_data['predicted_prob'].apply(
        lambda p: max(1.5, min(99.0, 1.0 / p)) if p > 0.01 else 50.0
    )

    # Analyze each race
    all_opportunities = []

    for race_num in sorted(race_data['race_number'].unique()):
        race = race_data[race_data['race_number'] == race_num].copy()
        race = race.sort_values('predicted_prob', ascending=False)

        for idx, row in race.iterrows():
            prob = row['predicted_prob']
            est_odds = row['estimated_odds']
            actual_odds = row['final_odds'] if pd.notna(row['final_odds']) else est_odds

            # Calculate EV with actual odds
            ev = (prob * actual_odds) - 1

            # Categorize bet (same logic as betting tool)
            if prob >= 0.30:
                category = "STRONG"
                bet_pct = 0.35
            elif prob >= 0.25:
                category = "GOOD"
                bet_pct = 0.25
            elif prob >= 0.20:
                category = "DECENT"
                bet_pct = 0.15
            else:
                category = "NONE"
                bet_pct = 0

            if category != "NONE":
                all_opportunities.append({
                    'race': race_num,
                    'horse_name': row['horse_name'],
                    'start_num': int(row['start_number']),
                    'probability': prob,
                    'estimated_odds': est_odds,
                    'actual_odds': actual_odds,
                    'ev': ev,
                    'category': category,
                    'bet_pct': bet_pct,
                    'is_winner': row['is_winner'],
                    'finish_place': int(row['finish_place']) if pd.notna(row['finish_place']) and row['finish_place'] > 0 else None
                })

    # Sort by probability and select top bets
    all_opportunities.sort(key=lambda x: x['probability'], reverse=True)

    # Allocate budget
    total_pct = sum(opp['bet_pct'] for opp in all_opportunities[:10])

    selected_bets = []
    for opp in all_opportunities[:10]:
        if total_pct > 0:
            bet_amount = int((opp['bet_pct'] / total_pct) * budget)
            if bet_amount >= 20:  # Minimum bet
                opp['bet_amount'] = bet_amount
                selected_bets.append(opp)

    # Show predictions
    print(f"🎯 MODEL PREDICTIONS (Top {len(selected_bets)} bets):")
    print()

    total_bet = 0
    total_payout = 0
    wins = 0

    for i, bet in enumerate(selected_bets, 1):
        category_emoji = {"STRONG": "🔥", "GOOD": "⭐", "DECENT": "💎"}
        emoji = category_emoji.get(bet['category'], "")

        result_str = ""
        if bet['is_winner']:
            payout = bet['bet_amount'] * bet['actual_odds']
            total_payout += payout
            wins += 1
            result_str = f" ✅ WON! Payout: {payout:.0f} SEK"
        elif bet['finish_place']:
            result_str = f" ❌ Lost (finished {bet['finish_place']})"
        else:
            result_str = " ❌ Lost (DNF/galloped)"

        print(f"{emoji} {bet['category']} - Race {bet['race']}")
        print(f"   Horse #{bet['start_num']}: {bet['horse_name']}")
        print(f"   Win probability: {bet['probability']:.1%}")
        print(f"   Actual odds: {bet['actual_odds']:.1f}")
        print(f"   Bet amount: {bet['bet_amount']} SEK")
        print(f"   {result_str}")
        print()

        total_bet += bet['bet_amount']

    # Summary
    print("="*80)
    print("📊 ACTUAL RESULTS")
    print("="*80)
    print(f"Total bet: {total_bet} SEK")
    print(f"Winners: {wins}/{len(selected_bets)} ({wins/len(selected_bets)*100:.1f}%)")
    print(f"Total payout: {total_payout:.0f} SEK")
    print(f"Profit: {total_payout - total_bet:+.0f} SEK ({(total_payout/total_bet - 1)*100:+.1f}% ROI)")
    print()

    if total_payout > total_bet:
        print("✅ PROFITABLE DAY! 🎉")
    else:
        print("❌ Loss for the day")

    print("="*80)
    print()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_past_performance.py YYYY-MM-DD [track_name] [budget]")
        print("Example: python analyze_past_performance.py 2025-11-01 Romme 500")
        sys.exit(1)

    date = sys.argv[1]
    track = sys.argv[2] if len(sys.argv) > 2 else None
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else 500

    analyze_race_day(date, track, budget)
