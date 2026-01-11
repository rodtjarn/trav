#!/usr/bin/env python3
"""
Generic backtesting tool for any past race day
Analyzes model predictions vs actual results with detailed placement stats

Usage:
  python backtest_race.py --date 2026-01-10 --track Romme
  python backtest_race.py --date 2025-11-01 --track Romme --budget 1000
  python backtest_race.py --date 2026-01-11 --track Östersund --game GS75
"""

import pickle
import argparse
import sys
import unicodedata
from atg_api_scraper import ATGAPIScraper
from temporal_data_processor import TemporalDataProcessor

def normalize_track_name(name):
    """
    Normalize track name for comparison by removing accents and converting to lowercase

    Swedish character mapping:
    å/Å → a
    ä/Ä → a
    ö/Ö → o

    This allows users to type 'Aby' to match 'Åby', 'Ostersund' to match 'Östersund', etc.
    """
    # Manual mapping for Swedish characters (more reliable than unicodedata)
    replacements = {
        'å': 'a', 'Å': 'a',
        'ä': 'a', 'Ä': 'a',
        'ö': 'o', 'Ö': 'o'
    }

    normalized = name
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)

    return normalized.lower()

def analyze_race_day(date_str, track_name, budget=500, game_type=None):
    """
    Backtest model performance on a specific race day

    Args:
        date_str: Date in YYYY-MM-DD format
        track_name: Track name (e.g., 'Romme', 'Solvalla', 'Östersund')
        budget: Total betting budget in SEK
        game_type: Optional V-game type to filter (e.g., 'V75', 'GS75', 'V85')
    """

    print("="*80)
    print(f"BACKTESTING ANALYSIS - {track_name.upper()}")
    print(f"Date: {date_str}")
    if game_type:
        print(f"Game Type: {game_type}")
    print(f"Budget: {budget} SEK")
    print("="*80)
    print()

    # Load model
    print("Loading model...")
    with open('temporal_rf_model.pkl', 'rb') as f:
        model = pickle.load(f)

    feature_cols = list(model.feature_names_in_)

    # Initialize scraper and processor
    scraper = ATGAPIScraper()
    processor = TemporalDataProcessor()

    # Get calendar for the date
    print(f"Fetching race calendar for {date_str}...")
    cal = scraper.get_calendar_for_date(date_str)

    if not cal or 'tracks' not in cal:
        print(f"❌ No races found for {date_str}")
        return

    # Find the track (normalize to handle Swedish characters: å→a, ä→a, ö→o)
    normalized_input = normalize_track_name(track_name)
    target_track = None

    for track in cal['tracks']:
        track_name_from_api = track.get('name', '')
        if normalize_track_name(track_name_from_api) == normalized_input:
            target_track = track
            break

    if not target_track:
        print(f"❌ Track '{track_name}' not found on {date_str}")
        available_tracks = [t.get('name', '') for t in cal['tracks']]
        print(f"Available tracks: {', '.join(available_tracks)}")
        print(f"\nTip: Swedish characters å/ä/ö can be typed as a/a/o")
        print(f"     (e.g., 'Aby' matches 'Åby', 'Ostersund' matches 'Östersund')")
        return

    track_name_actual = target_track.get('name', 'Unknown')
    all_races = target_track.get('races', [])

    print(f"✓ Found track: {track_name_actual}")
    print(f"✓ Total races at track: {len(all_races)}")
    print()

    # Filter by game type if specified
    race_ids_to_analyze = []

    if game_type:
        game_info = scraper.get_game_info(date_str, game_type)
        if game_info and game_info.get('race_ids'):
            race_ids_to_analyze = game_info.get('race_ids', [])
            print(f"✓ Found {game_type} game with {len(race_ids_to_analyze)} races")
        else:
            print(f"❌ No {game_type} game found on {date_str}")
            return
    else:
        # Use all races from the track
        race_ids_to_analyze = [race.get('id') for race in all_races if race.get('id')]
        print(f"✓ Analyzing all {len(race_ids_to_analyze)} races")

    print()

    # Fetch race data and results
    print("Fetching race details and results...")
    all_race_data = []
    actual_results = {}  # race_num -> {horse_num -> {place, odds, name}}
    race_names = {}  # race_num -> race_name

    for i, race_id in enumerate(race_ids_to_analyze, 1):
        # Fetch race details for processing
        race_details = scraper.get_race_details(race_id)
        if race_details:
            all_race_data.append(race_details)

        # Fetch actual results from API
        response = scraper.session.get(f'https://www.atg.se/services/racinginfo/v1/api/races/{race_id}')
        if response.status_code == 200:
            race = response.json()
            actual_results[i] = {}
            race_names[i] = race.get('name', f'Race {i}')

            for start in race.get('starts', []):
                horse_num = start.get('number')
                result_data = start.get('result', {})
                place = result_data.get('place')
                odds = result_data.get('finalOdds')

                if horse_num:
                    actual_results[i][horse_num] = {
                        'place': place if place else 999,
                        'odds': odds if odds and odds > 0 else 0,
                        'name': start.get('horse', {}).get('name', 'Unknown')
                    }

    if not all_race_data:
        print("❌ Failed to fetch race data")
        return

    print(f"✓ Fetched {len(all_race_data)} races")
    print()

    # Process through model
    print("Processing races through model...")
    df = processor.process_race_data(all_race_data, feature_cols)

    if df.empty:
        print("❌ Failed to process race data")
        return

    print(f"✓ Processed {len(df)} horses")
    print()

    # Make predictions
    X = df[feature_cols]
    predictions = model.predict_proba(X)[:, 1]
    df['predicted_prob'] = predictions

    # Create betting opportunities
    all_opportunities = []

    for race_num in sorted(df['race_number'].unique()):
        race = df[df['race_number'] == race_num].copy()

        for idx, row in race.iterrows():
            prob = row['predicted_prob']
            horse_num = int(row.get('start_number', 0))

            # Get actual result
            actual = actual_results.get(race_num, {}).get(horse_num, {})
            actual_odds = actual.get('odds', 0)
            actual_place = actual.get('place', 999)

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

            if category != "NONE" and actual_odds > 0:
                ev = (prob * actual_odds) - 1

                all_opportunities.append({
                    'race': race_num,
                    'horse_name': row.get('horse_name', 'Unknown'),
                    'start_num': horse_num,
                    'probability': prob,
                    'actual_odds': actual_odds,
                    'actual_place': actual_place,
                    'ev': ev,
                    'category': category,
                    'bet_pct': bet_pct,
                    'is_winner': actual_place == 1
                })

    if not all_opportunities:
        print("❌ No betting opportunities found (model didn't have high confidence picks)")
        return

    # Sort and select top bets
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

    # Display predictions and results
    print("="*80)
    print(f"🎯 MODEL PREDICTIONS (Top {len(selected_bets)} bets)")
    print("="*80)
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
        else:
            if bet['actual_place'] == 999:
                place_str = "DNF/galloped"
            elif bet['actual_place'] == 2:
                place_str = "2nd 🥈"
            elif bet['actual_place'] == 3:
                place_str = "3rd 🥉"
            else:
                place_str = f"{bet['actual_place']}th"
            result_str = f" ❌ Lost (finished {place_str})"

        print(f"{emoji} {bet['category']} - Race {bet['race']}")
        print(f"   Horse #{bet['start_num']}: {bet['horse_name']}")
        print(f"   Win probability: {bet['probability']:.1%}")
        print(f"   Actual odds: {bet['actual_odds']:.1f}")
        print(f"   Expected Value: {bet['ev']:.2f}")
        print(f"   Bet amount: {bet['bet_amount']} SEK")
        print(f"   {result_str}")
        print()

        total_bet += bet['bet_amount']

    # Show losing bet placement analysis
    losing_bets = [bet for bet in selected_bets if not bet['is_winner']]

    if losing_bets:
        print("="*80)
        print("📊 LOSING BET PLACEMENT ANALYSIS")
        print("="*80)
        print()

        # Group by placement
        placement_counts = {}
        for bet in losing_bets:
            place = bet['actual_place']
            if place == 999:
                place_key = "DNF/Galloped"
            elif place == 2:
                place_key = "2nd place 🥈"
            elif place == 3:
                place_key = "3rd place 🥉"
            elif place == 4:
                place_key = "4th place"
            else:
                place_key = f"{place}th place"

            if place_key not in placement_counts:
                placement_counts[place_key] = []
            placement_counts[place_key].append(bet)

        # Sort by placement (DNF last)
        sorted_placements = sorted(placement_counts.keys(),
                                   key=lambda x: 999 if 'DNF' in x else int(x.split()[0].replace('nd','').replace('rd','').replace('th','')))

        for place_key in sorted_placements:
            bets = placement_counts[place_key]
            print(f"{place_key}: {len(bets)} bet(s)")
            for bet in bets:
                print(f"  - Race {bet['race']}, #{bet['start_num']} {bet['horse_name']} "
                      f"({bet['probability']:.1%} prob, {bet['actual_odds']:.1f} odds)")

        print()

    # Show actual winners for comparison
    print("="*80)
    print("🏆 ACTUAL RACE WINNERS")
    print("="*80)
    print()

    for race_num in sorted(actual_results.keys()):
        race_name = race_names.get(race_num, f'Race {race_num}')

        # Find winner(s)
        winners = []
        for horse_num, data in actual_results[race_num].items():
            if data['place'] == 1:
                winners.append((horse_num, data['name'], data['odds']))

        if winners:
            print(f"Race {race_num}: {race_name}")
            for num, name, odds in winners:
                odds_str = f"{odds:.1f}" if odds > 0 else "N/A"
                print(f"  🏆 #{num} {name} (odds: {odds_str})")
        else:
            print(f"Race {race_num}: No winner data available")

    print()

    # Summary
    print("="*80)
    print("📊 FINAL RESULTS")
    print("="*80)
    print(f"Total bet: {total_bet} SEK")
    print(f"Winners: {wins}/{len(selected_bets)} ({wins/len(selected_bets)*100:.1f}%)")
    print(f"Total payout: {total_payout:.0f} SEK")
    print(f"Profit: {total_payout - total_bet:+.0f} SEK ({(total_payout/total_bet - 1)*100:+.1f}% ROI)")
    print()

    if total_payout > total_bet:
        profit_pct = (total_payout/total_bet - 1)*100
        print(f"✅ PROFITABLE DAY! 🎉 (+{profit_pct:.1f}% ROI)")
    elif total_payout == 0:
        print("❌ TOTAL LOSS - No winners")
    else:
        print("❌ Loss for the day")

    # Show comparison to expected model performance
    print()
    print("="*80)
    print("📈 PERFORMANCE vs MODEL EXPECTATIONS")
    print("="*80)
    print(f"Actual win rate: {wins/len(selected_bets)*100:.1f}%")
    print(f"Model training win rate: 29.5% (per-race accuracy)")
    print()

    if wins/len(selected_bets) >= 0.295:
        print("✅ Performance meets or exceeds model expectations!")
    elif wins/len(selected_bets) >= 0.20:
        print("⚠️  Performance slightly below model expectations (variance expected)")
    else:
        print("❌ Performance significantly below expectations (bad day/variance)")

    print("="*80)
    print()

    # Show key insights
    print("💡 KEY INSIGHTS:")
    print()

    # Count DNFs
    dnf_count = sum(1 for bet in losing_bets if bet['actual_place'] == 999)
    if dnf_count > 0:
        print(f"  • {dnf_count} horse(s) galloped/DNF ({dnf_count/len(selected_bets)*100:.0f}% of bets)")

    # Count near-misses (2nd-3rd)
    near_miss_count = sum(1 for bet in losing_bets if 2 <= bet['actual_place'] <= 3)
    if near_miss_count > 0:
        print(f"  • {near_miss_count} horse(s) finished 2nd or 3rd (near-misses)")

    # Average odds of winners vs losers
    if wins > 0:
        avg_winner_odds = sum(bet['actual_odds'] for bet in selected_bets if bet['is_winner']) / wins
        print(f"  • Average winner odds: {avg_winner_odds:.1f}")

    # High EV bets
    high_ev_bets = [bet for bet in selected_bets if bet['ev'] > 1.0]
    if high_ev_bets:
        high_ev_wins = sum(1 for bet in high_ev_bets if bet['is_winner'])
        print(f"  • High EV bets (>1.0): {len(high_ev_bets)} total, {high_ev_wins} won")

    print()
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Backtest model performance on any past race day',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --date 2026-01-10 --track Romme
  %(prog)s --date 2025-11-01 --track Romme --budget 1000
  %(prog)s --date 2026-01-11 --track Östersund --game GS75
  %(prog)s --date 2025-12-27 --track Gävle --game V85 --budget 500

This tool shows:
  • Model predictions vs actual results
  • Win rate and ROI
  • Losing bet placement analysis
  • Actual race winners
  • Performance vs model expectations
        """
    )

    parser.add_argument(
        '--date',
        type=str,
        required=True,
        metavar='YYYY-MM-DD',
        help='Race date (required)'
    )

    parser.add_argument(
        '--track',
        type=str,
        required=True,
        metavar='TRACK',
        help='Track name (required, e.g., Romme, Solvalla, Östersund)'
    )

    parser.add_argument(
        '--budget',
        type=int,
        default=500,
        metavar='SEK',
        help='Total betting budget in SEK (default: 500)'
    )

    parser.add_argument(
        '--game',
        type=str,
        metavar='TYPE',
        help='Filter by game type: V75, GS75, V86, V85, V65, etc. (optional)'
    )

    args = parser.parse_args()

    # Validate budget
    if args.budget < 50:
        print("❌ Minimum budget is 50 SEK")
        sys.exit(1)

    # Run analysis
    analyze_race_day(args.date, args.track, args.budget, args.game)


if __name__ == '__main__':
    main()
