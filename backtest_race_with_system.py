#!/usr/bin/env python3
"""
Generic backtesting tool with BOTH individual and V-game system betting
Analyzes model predictions vs actual results with dual strategy comparison

Usage:
  python backtest_race_with_system.py --date 2026-01-10 --track Romme --game V85
  python backtest_race_with_system.py --date 2026-01-11 --track Östersund --game GS75

Budget: 500 SEK individual bets + 500 SEK system bet = 1000 SEK total
"""

import pickle
import argparse
import sys
from itertools import product
from atg_api_scraper import ATGAPIScraper
from temporal_data_processor import TemporalDataProcessor

def normalize_track_name(name):
    """Normalize track name for Swedish character matching (å→a, ä→a, ö→o)"""
    replacements = {
        'å': 'a', 'Å': 'a',
        'ä': 'a', 'Ä': 'a',
        'ö': 'o', 'Ö': 'o'
    }
    normalized = name
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    return normalized.lower()


def generate_reduced_system(df, num_races, budget=500, max_picks_per_race=3):
    """
    Generate a reduced V-game system within budget

    Strategy:
    - Pick top horses per race based on predicted probability
    - If top horse has >35% prob → pick 1 (banker)
    - If top horse has 25-35% → pick top 2
    - If top horse has <25% → pick top 3
    - Reduce picks if total cost exceeds budget

    Args:
        df: DataFrame with predictions (must have 'vgame_race_num' column)
        num_races: Number of races in the V-game
        budget: Maximum cost in SEK
        max_picks_per_race: Maximum horses to pick per race

    Returns:
        dict with selections, cost, and coverage info
    """

    system_selections = {}

    # Select horses for each race
    for race_num in range(1, num_races + 1):
        race_horses = df[df['vgame_race_num'] == race_num].copy()

        if len(race_horses) == 0:
            continue

        race_horses = race_horses.sort_values('predicted_prob', ascending=False)

        # Determine number of picks based on confidence
        top_prob = race_horses.iloc[0]['predicted_prob']

        if top_prob >= 0.35:
            num_picks = 1  # High confidence - banker
        elif top_prob >= 0.25:
            num_picks = 2  # Medium confidence
        else:
            num_picks = min(3, max_picks_per_race)  # Low confidence

        # Ensure we don't exceed available horses
        num_picks = min(num_picks, len(race_horses))

        selected = race_horses.head(num_picks)

        system_selections[race_num] = []
        for idx, row in selected.iterrows():
            system_selections[race_num].append({
                'start_num': int(row.get('start_number', 0)),
                'horse_name': row.get('horse_name', 'Unknown'),
                'probability': row['predicted_prob']
            })

    # Calculate total combinations and cost
    num_picks_per_race = [len(picks) for picks in system_selections.values()]
    total_combinations = 1
    for num in num_picks_per_race:
        total_combinations *= num

    base_cost_per_row = 1  # 1 SEK per row is standard
    total_cost = total_combinations * base_cost_per_row

    # If cost exceeds budget, reduce selections
    reduction_attempts = 0
    max_reductions = 20

    while total_cost > budget and reduction_attempts < max_reductions:
        # Find race with most picks (and >1 pick) and reduce by 1
        max_race = None
        max_picks = 0

        for race_num, picks in system_selections.items():
            if len(picks) > max_picks and len(picks) > 1:
                max_picks = len(picks)
                max_race = race_num

        if max_race is None:
            break  # Can't reduce further

        # Remove lowest probability pick from that race
        system_selections[max_race] = system_selections[max_race][:-1]

        # Recalculate cost
        num_picks_per_race = [len(picks) for picks in system_selections.values()]
        total_combinations = 1
        for num in num_picks_per_race:
            total_combinations *= num
        total_cost = total_combinations * base_cost_per_row

        reduction_attempts += 1

    return {
        'selections': system_selections,
        'total_rows': total_combinations,
        'total_cost': total_cost,
        'picks_per_race': num_picks_per_race
    }


def check_system_result(system_selections, actual_results):
    """
    Check if the V-game system won

    Args:
        system_selections: Dict of {race_num: [picks]}
        actual_results: Dict of {race_num: {horse_num: {place, odds, name}}}

    Returns:
        dict with hit status, correct races, and winning combination
    """

    winning_horses = {}
    correct_races = []

    for race_num in sorted(system_selections.keys()):
        picks = system_selections[race_num]
        race_results = actual_results.get(race_num, {})

        # Find actual winner
        winner_num = None
        winner_name = None

        for horse_num, data in race_results.items():
            if data.get('place') == 1:
                winner_num = horse_num
                winner_name = data.get('name')
                break

        # Check if we picked the winner
        picked_nums = [p['start_num'] for p in picks]

        if winner_num in picked_nums:
            correct_races.append(race_num)
            winning_horses[race_num] = {
                'num': winner_num,
                'name': winner_name,
                'picked': True
            }
        else:
            winning_horses[race_num] = {
                'num': winner_num,
                'name': winner_name,
                'picked': False
            }

    all_correct = len(correct_races) == len(system_selections)

    return {
        'hit': all_correct,
        'correct_races': correct_races,
        'total_races': len(system_selections),
        'winning_horses': winning_horses
    }


def estimate_vgame_payout(game_type, num_rows, all_correct=False):
    """
    Estimate V-game payout based on typical dividends

    This is a rough estimation since actual payouts vary significantly
    based on pool size and number of winners.

    For a perfect system (all correct), typical payouts are:
    - V75: 100,000 - 10,000,000 SEK (varies wildly)
    - V86: 50,000 - 5,000,000 SEK
    - V85: 20,000 - 500,000 SEK
    - V65: 10,000 - 200,000 SEK
    - V64: 5,000 - 100,000 SEK
    - GS75: 1,000,000 - 100,000,000 SEK (jackpot)

    We'll use conservative estimates divided by number of rows
    """

    if not all_correct:
        return 0

    # Conservative payout estimates for a single correct row
    base_payouts = {
        'V75': 200000,
        'V86': 100000,
        'V85': 50000,
        'V65': 30000,
        'V64': 20000,
        'V5': 5000,
        'V4': 2000,
        'V3': 1000,
        'GS75': 500000  # Very conservative for GS75
    }

    base_payout = base_payouts.get(game_type, 10000)

    # Payout is divided among all correct rows in your system
    # (simplified - actual ATG pools are more complex)
    payout_per_row = base_payout / num_rows if num_rows > 0 else 0

    return payout_per_row * num_rows


def analyze_race_day(date_str, track_name, individual_budget=500, system_budget=500, game_type=None):
    """
    Backtest with BOTH individual and system betting strategies

    Args:
        date_str: Date in YYYY-MM-DD format
        track_name: Track name
        individual_budget: Budget for individual high-EV bets (default: 500 SEK)
        system_budget: Budget for V-game system bet (default: 500 SEK)
        game_type: V-game type (required for system betting)
    """

    total_budget = individual_budget + system_budget

    print("="*80)
    print(f"DUAL STRATEGY BACKTESTING - {track_name.upper()}")
    print(f"Date: {date_str}")
    if game_type:
        print(f"Game Type: {game_type}")
    print(f"Individual Betting Budget: {individual_budget} SEK")
    print(f"System Betting Budget: {system_budget} SEK")
    print(f"Total Budget: {total_budget} SEK")
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

    # Get calendar
    print(f"Fetching race calendar for {date_str}...")
    cal = scraper.get_calendar_for_date(date_str)

    if not cal or 'tracks' not in cal:
        print(f"❌ No races found for {date_str}")
        return

    # Find track
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
        return

    track_name_actual = target_track.get('name', 'Unknown')

    # Get race IDs
    if game_type:
        game_info = scraper.get_game_info(date_str, game_type)
        if not game_info or not game_info.get('race_ids'):
            print(f"❌ No {game_type} game found on {date_str}")
            return
        race_ids_to_analyze = game_info.get('race_ids', [])
        print(f"✓ Found {game_type} game with {len(race_ids_to_analyze)} races")
    else:
        print("❌ --game parameter required for dual strategy analysis")
        print("   System betting requires a V-game type (V75, V86, V85, etc.)")
        return

    print()

    # Fetch race data and results
    print("Fetching race details and results...")
    all_race_data = []
    actual_results = {}
    race_names = {}

    for i, race_id in enumerate(race_ids_to_analyze, 1):
        race_details = scraper.get_race_details(race_id)
        if race_details:
            all_race_data.append(race_details)

        # Fetch results
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

    # Map track race numbers to V-game sequential numbers (1, 2, 3, ...)
    race_id_to_seq = {}
    for seq_num, race_id in enumerate(race_ids_to_analyze, 1):
        # Extract track race number from race_id (format: 2026-01-11_33_4)
        track_race_num = int(race_id.split('_')[-1])
        race_id_to_seq[track_race_num] = seq_num

    # Add sequential race number to dataframe
    df['vgame_race_num'] = df['race_number'].map(race_id_to_seq)

    # ===================================================================
    # STRATEGY 1: INDIVIDUAL HIGH-EV BETTING
    # ===================================================================

    print("="*80)
    print(f"STRATEGY 1: INDIVIDUAL HIGH-EV BETTING ({individual_budget} SEK)")
    print("="*80)
    print()

    individual_opportunities = []

    for vgame_race_num in sorted(df['vgame_race_num'].unique()):
        race = df[df['vgame_race_num'] == vgame_race_num].copy()

        for idx, row in race.iterrows():
            prob = row['predicted_prob']
            horse_num = int(row.get('start_number', 0))

            actual = actual_results.get(vgame_race_num, {}).get(horse_num, {})
            actual_odds = actual.get('odds', 0)
            actual_place = actual.get('place', 999)

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

                individual_opportunities.append({
                    'race': vgame_race_num,
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

    # Allocate individual budget
    individual_opportunities.sort(key=lambda x: x['probability'], reverse=True)
    total_pct = sum(opp['bet_pct'] for opp in individual_opportunities[:10])

    individual_bets = []
    for opp in individual_opportunities[:10]:
        if total_pct > 0:
            bet_amount = int((opp['bet_pct'] / total_pct) * individual_budget)
            if bet_amount >= 20:
                opp['bet_amount'] = bet_amount
                individual_bets.append(opp)

    # Calculate individual results
    individual_total_bet = sum(bet['bet_amount'] for bet in individual_bets)
    individual_payout = 0
    individual_wins = 0

    for bet in individual_bets:
        if bet['is_winner']:
            individual_payout += bet['bet_amount'] * bet['actual_odds']
            individual_wins += 1

    print(f"Top {len(individual_bets)} bets selected")
    print(f"Total bet: {individual_total_bet} SEK")
    print(f"Winners: {individual_wins}/{len(individual_bets)} ({individual_wins/len(individual_bets)*100:.1f}%)")
    print(f"Payout: {individual_payout:.0f} SEK")
    print(f"Profit: {individual_payout - individual_total_bet:+.0f} SEK")
    print()

    # ===================================================================
    # STRATEGY 2: V-GAME SYSTEM BETTING
    # ===================================================================

    print("="*80)
    print(f"STRATEGY 2: {game_type} SYSTEM BETTING ({system_budget} SEK)")
    print("="*80)
    print()

    system = generate_reduced_system(df, len(race_ids_to_analyze), budget=system_budget)

    print(f"System configuration:")
    print(f"  Total rows: {system['total_rows']}")
    print(f"  Cost: {system['total_cost']} SEK")
    print(f"  Picks per race: {system['picks_per_race']}")
    print()

    print("System selections:")
    for race_num in sorted(system['selections'].keys()):
        picks = system['selections'][race_num]
        print(f"  Race {race_num}: {len(picks)} pick(s)")
        for pick in picks:
            print(f"    #{pick['start_num']} {pick['horse_name']} ({pick['probability']:.1%})")
    print()

    # Check system result
    system_result = check_system_result(system['selections'], actual_results)

    print("System result:")
    print(f"  Correct races: {len(system_result['correct_races'])}/{system_result['total_races']}")

    for race_num in sorted(system_result['winning_horses'].keys()):
        winner = system_result['winning_horses'][race_num]
        status = "✅" if winner['picked'] else "❌"
        print(f"  {status} Race {race_num}: #{winner['num']} {winner['name']}")
    print()

    system_payout = 0
    if system_result['hit']:
        system_payout = estimate_vgame_payout(game_type, system['total_rows'], all_correct=True)
        print(f"🎉 SYSTEM HIT! All {system_result['total_races']} races correct!")
        print(f"  Estimated payout: {system_payout:.0f} SEK")
        print(f"  (Note: Actual {game_type} payouts vary significantly)")
    else:
        print(f"❌ System missed - {len(system_result['correct_races'])}/{system_result['total_races']} races correct")
        print(f"  Payout: 0 SEK")

    system_profit = system_payout - system['total_cost']
    print(f"  Profit: {system_profit:+.0f} SEK")
    print()

    # ===================================================================
    # COMBINED RESULTS
    # ===================================================================

    print("="*80)
    print("📊 COMBINED RESULTS - DUAL STRATEGY")
    print("="*80)
    print()

    print("Individual Betting:")
    print(f"  Bet: {individual_total_bet} SEK")
    print(f"  Payout: {individual_payout:.0f} SEK")
    print(f"  Profit: {individual_payout - individual_total_bet:+.0f} SEK")
    print()

    print("System Betting:")
    print(f"  Bet: {system['total_cost']} SEK")
    print(f"  Payout: {system_payout:.0f} SEK")
    print(f"  Profit: {system_profit:+.0f} SEK")
    print()

    total_bet = individual_total_bet + system['total_cost']
    total_payout = individual_payout + system_payout
    total_profit = total_payout - total_bet

    print("="*40)
    print("TOTAL:")
    print(f"  Total bet: {total_bet} SEK")
    print(f"  Total payout: {total_payout:.0f} SEK")
    print(f"  Total profit: {total_profit:+.0f} SEK ({(total_profit/total_bet)*100:+.1f}% ROI)")
    print("="*40)
    print()

    if total_profit > 0:
        print("✅ PROFITABLE DAY! 🎉")
    else:
        print("❌ Loss for the day")

    # Show which strategy performed better
    individual_roi = ((individual_payout - individual_total_bet) / individual_total_bet * 100) if individual_total_bet > 0 else 0
    system_roi = (system_profit / system['total_cost'] * 100) if system['total_cost'] > 0 else 0

    print()
    print("Strategy Comparison:")
    print(f"  Individual ROI: {individual_roi:+.1f}%")
    print(f"  System ROI: {system_roi:+.1f}%")

    if system_roi > individual_roi:
        print(f"  → System betting performed better (+{system_roi - individual_roi:.1f}% advantage)")
    elif individual_roi > system_roi:
        print(f"  → Individual betting performed better (+{individual_roi - system_roi:.1f}% advantage)")
    else:
        print("  → Both strategies performed equally")

    print()
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Backtest with BOTH individual and V-game system betting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --date 2026-01-10 --track Romme --game V85
  %(prog)s --date 2026-01-11 --track Östersund --game GS75
  %(prog)s --date 2025-11-01 --track Romme --game V85 --individual 500 --system 500

Budgets:
  --individual: Budget for high-EV individual bets (default: 500 SEK)
  --system: Budget for V-game system bet (default: 500 SEK)

  Total budget = individual + system (default: 1000 SEK)

This tool compares:
  1. Individual betting on high-probability horses
  2. V-game system betting (pick multiple horses per race)

Shows which strategy performs better on historical data.
        """
    )

    parser.add_argument('--date', type=str, required=True, metavar='YYYY-MM-DD',
                       help='Race date (required)')
    parser.add_argument('--track', type=str, required=True, metavar='TRACK',
                       help='Track name (required)')
    parser.add_argument('--game', type=str, required=True, metavar='TYPE',
                       help='V-game type (required): V75, V86, V85, V65, V64, etc.')
    parser.add_argument('--individual', type=int, default=500, metavar='SEK',
                       help='Individual betting budget (default: 500 SEK)')
    parser.add_argument('--system', type=int, default=500, metavar='SEK',
                       help='System betting budget (default: 500 SEK)')

    args = parser.parse_args()

    # Validate budgets
    if args.individual < 50:
        print("❌ Minimum individual budget is 50 SEK")
        sys.exit(1)

    if args.system < 50:
        print("❌ Minimum system budget is 50 SEK")
        sys.exit(1)

    # Run dual strategy analysis
    analyze_race_day(args.date, args.track, args.individual, args.system, args.game)


if __name__ == '__main__':
    main()
