#!/usr/bin/env python3
"""
Analyze model performance on recent race day by fetching live data from ATG API
"""

import pickle
import sys
from datetime import datetime
from atg_api_scraper import ATGAPIScraper
from temporal_data_processor import TemporalDataProcessor

def analyze_live_race_day(date_str, track_name=None, budget=500):
    """Analyze how the model performed on a recent date by fetching live data"""

    # Load model
    print("Loading model...")
    with open('temporal_rf_model.pkl', 'rb') as f:
        model = pickle.load(f)

    feature_cols = list(model.feature_names_in_)

    # Initialize scraper and processor
    scraper = ATGAPIScraper()
    processor = TemporalDataProcessor()

    print("="*80)
    print(f"LIVE MODEL PERFORMANCE ANALYSIS")
    print(f"Date: {date_str}")
    print(f"Total Budget: {budget} SEK")
    print("="*80)
    print()

    # Get calendar for the date
    print(f"Fetching race calendar for {date_str}...")
    cal = scraper.get_calendar_for_date(date_str)

    if not cal or 'tracks' not in cal:
        print(f"No races found for {date_str}")
        return

    # Find the track
    target_track = None
    if track_name:
        for track in cal['tracks']:
            if track.get('name', '').lower() == track_name.lower():
                target_track = track
                break
        if not target_track:
            print(f"Track '{track_name}' not found on {date_str}")
            print(f"Available tracks: {', '.join([t.get('name', '') for t in cal['tracks']])}")
            return
    else:
        # Use first track
        target_track = cal['tracks'][0]

    track_name_actual = target_track.get('name', 'Unknown')
    races = target_track.get('races', [])

    print(f"Track: {track_name_actual}")
    print(f"Number of races: {len(races)}")
    print()

    # Fetch all race details
    all_race_data = []
    for race in races:
        race_id = race.get('id')
        if not race_id:
            continue

        print(f"Fetching race {race.get('number')}...")
        race_details = scraper.get_race_details(race_id)
        if race_details:
            all_race_data.append(race_details)

    if not all_race_data:
        print("No race details available")
        return

    print(f"\nProcessing {len(all_race_data)} races...")

    # Process through data processor
    df = processor.process_race_data(all_race_data, feature_cols)

    if df.empty:
        print("Failed to process race data")
        return

    print(f"Processed {len(df)} horses")
    print()

    # Make predictions
    X = df[feature_cols]
    predictions = model.predict_proba(X)[:, 1]

    df['predicted_prob'] = predictions
    df['estimated_odds'] = df['predicted_prob'].apply(
        lambda p: max(1.5, min(99.0, 1.0 / p)) if p > 0.01 else 50.0
    )

    # Analyze each race and create betting opportunities
    all_opportunities = []

    for race_num in sorted(df['race_number'].unique()):
        race = df[df['race_number'] == race_num].copy()
        race = race.sort_values('predicted_prob', ascending=False)

        for idx, row in race.iterrows():
            prob = row['predicted_prob']
            est_odds = row['estimated_odds']
            actual_odds = row.get('final_odds', est_odds)

            # Calculate EV with actual odds
            if actual_odds and actual_odds > 0:
                ev = (prob * actual_odds) - 1
            else:
                ev = -1

            # Categorize bet
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
                    'horse_name': row.get('horse_name', 'Unknown'),
                    'start_num': int(row.get('start_number', 0)),
                    'probability': prob,
                    'estimated_odds': est_odds,
                    'actual_odds': actual_odds if actual_odds and actual_odds > 0 else None,
                    'ev': ev,
                    'category': category,
                    'bet_pct': bet_pct,
                    'is_winner': row.get('is_winner', False),
                    'finish_place': int(row.get('finish_place', 0)) if row.get('finish_place', 0) > 0 else None
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

    # Show predictions and results
    print(f"🎯 MODEL PREDICTIONS (Top {len(selected_bets)} bets):")
    print()

    total_bet = 0
    total_payout = 0
    wins = 0

    for i, bet in enumerate(selected_bets, 1):
        category_emoji = {"STRONG": "🔥", "GOOD": "⭐", "DECENT": "💎"}
        emoji = category_emoji.get(bet['category'], "")

        result_str = ""
        if bet['actual_odds'] is None:
            result_str = " ⏳ No results available yet"
        elif bet['is_winner']:
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
        if bet['actual_odds']:
            print(f"   Actual odds: {bet['actual_odds']:.1f}")
        else:
            print(f"   Estimated odds: {bet['estimated_odds']:.1f}")
        print(f"   Bet amount: {bet['bet_amount']} SEK")
        print(f"   {result_str}")
        print()

        total_bet += bet['bet_amount']

    # Summary
    print("="*80)
    print("📊 ACTUAL RESULTS")
    print("="*80)
    print(f"Total bet: {total_bet} SEK")

    if wins > 0 or any(not bet['actual_odds'] for bet in selected_bets):
        print(f"Winners: {wins}/{len(selected_bets)} ({wins/len(selected_bets)*100:.1f}%)")
        print(f"Total payout: {total_payout:.0f} SEK")
        print(f"Profit: {total_payout - total_bet:+.0f} SEK ({(total_payout/total_bet - 1)*100:+.1f}% ROI)")
        print()

        if total_payout > total_bet:
            print("✅ PROFITABLE DAY! 🎉")
        elif total_payout > 0:
            print("❌ Loss for the day")
        else:
            print("⏳ Results pending")
    else:
        print("No results available yet")

    print("="*80)
    print()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_live_results.py YYYY-MM-DD [track_name] [budget]")
        print("Example: python analyze_live_results.py 2026-01-10 Romme 500")
        sys.exit(1)

    date = sys.argv[1]
    track = sys.argv[2] if len(sys.argv) > 2 else None
    budget = int(sys.argv[3]) if len(sys.argv) > 3 else 500

    analyze_live_race_day(date, track, budget)
