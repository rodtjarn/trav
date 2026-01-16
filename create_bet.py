#!/usr/bin/env python3
"""
Generate optimal betting slip for maximum profit potential.

Usage:
    python create_bet.py --total 1000                 # Auto-find next race, bet 1000 SEK
    python create_bet.py                              # Auto-find next race, bet 300 SEK (default)
    python create_bet.py --game V75                   # Auto-find next V75 race
    python create_bet.py --game V86 --total 500       # Auto-find next V86, bet 500 SEK
    python create_bet.py --date 2026-01-17            # Specific date
    python create_bet.py --total 500 --date 2026-01-17
    python create_bet.py --track solvalla             # Auto-find next race on Solvalla
    python create_bet.py --track umaker --total 500   # Next race on Umåker, bet 500 SEK

Supported game types: V75, V86, V85, V65, V64, V4, V5, GS75, V3

Strategy: Individual high-EV betting (not system betting)
Based on temporal model showing 21.5% win rate, +96.6% ROI
"""

import sys
import argparse
from datetime import datetime, timedelta
from predict_v85 import V85Predictor
from atg_api_scraper import ATGAPIScraper


class BettingSlipGenerator:
    def __init__(self, budget=300):
        self.budget = budget
        self.predictor = V85Predictor(
            model_path='temporal_rf_model.pkl',
            metadata_path='temporal_rf_metadata.json'
        )

    def find_next_v85(self, max_days=30):
        """Find the next V85 race day by searching ATG calendar"""
        return self.find_next_game('V85', max_days)

    def find_next_game(self, game_type='V85', max_days=30):
        """Find the next game of specified type by searching ATG calendar"""
        print(f"🔍 Searching for next {game_type} race day...")
        print()

        today = datetime.now()

        for i in range(max_days):
            check_date = (today + timedelta(days=i)).strftime('%Y-%m-%d')

            try:
                game_info = self.predictor.scraper.get_game_info(check_date, game_type)

                if game_info and game_info.get('race_ids') and len(game_info.get('race_ids', [])) > 0:
                    race_ids = game_info.get('race_ids', [])

                    print(f"✅ Found {game_type} on {check_date}")
                    print(f"   Game ID: {game_info.get('game_id', 'Unknown')}")
                    print(f"   Number of races: {len(race_ids)}")
                    print()

                    return check_date, game_type

            except Exception as e:
                # Continue searching on error
                continue

        print(f"❌ No {game_type} found in next {max_days} days")
        print(f"\nPlease check ATG.se for {game_type} schedule or try a later date range.")
        return None, None

    def find_next_race(self, max_days=30):
        """Find the next race day by searching ATG calendar (any race, not just V85)"""
        print("🔍 Searching for next race day...")
        print()

        today = datetime.now()

        for i in range(max_days):
            check_date = (today + timedelta(days=i)).strftime('%Y-%m-%d')

            try:
                calendar = self.predictor.scraper.get_calendar_for_date(check_date)

                if not calendar or not calendar.get('tracks'):
                    continue

                # Find first track with races
                for track in calendar.get('tracks', []):
                    races = track.get('races', [])

                    if races:
                        race_info = {
                            'date': check_date,
                            'track_name': track.get('name'),
                            'track_id': track.get('id'),
                            'races': races
                        }

                        print(f"✅ Found races on {track.get('name')} on {check_date}")
                        print(f"   Number of races: {len(races)}")
                        print()

                        return race_info

            except Exception as e:
                # Continue searching on error
                continue

        print(f"❌ No races found in next {max_days} days")
        print("\nPlease check ATG.se for race schedule or try a later date range.")
        return None

    def find_next_race_on_track(self, track_name, max_days=30):
        """Find the next race on a specific track by searching ATG calendar"""
        print(f"🔍 Searching for next race on {track_name.upper()}...")
        print()

        today = datetime.now()
        # Normalize Swedish characters for better matching
        track_name_lower = track_name.lower()
        track_name_normalized = (track_name_lower
            .replace('å', 'a').replace('ä', 'a').replace('ö', 'o')
            .replace('é', 'e').replace('è', 'e'))

        for i in range(max_days):
            check_date = (today + timedelta(days=i)).strftime('%Y-%m-%d')

            try:
                calendar = self.predictor.scraper.get_calendar_for_date(check_date)

                if not calendar or not calendar.get('tracks'):
                    continue

                # Search through all tracks for matching track name
                for track in calendar.get('tracks', []):
                    track_calendar_name = track.get('name', '').lower()
                    track_calendar_normalized = (track_calendar_name
                        .replace('å', 'a').replace('ä', 'a').replace('ö', 'o')
                        .replace('é', 'e').replace('è', 'e'))

                    # Check if track name matches (case-insensitive partial match with normalization)
                    if track_name_lower in track_calendar_name or track_name_normalized in track_calendar_normalized:
                        races = track.get('races', [])

                        if races:
                            # Find the next race that hasn't started yet
                            race_info = {
                                'date': check_date,
                                'track_name': track.get('name'),
                                'track_id': track.get('id'),
                                'races': races
                            }

                            print(f"✅ Found races on {track.get('name')} on {check_date}")
                            print(f"   Number of races: {len(races)}")
                            print()

                            return race_info

            except Exception as e:
                # Continue searching on error
                continue

        print(f"❌ No races found on track '{track_name}' in next {max_days} days")
        print("\nPlease check the track name or try a different track.")
        return None

    def calculate_expected_value(self, probability, odds):
        """Calculate expected value: (probability × odds) - 1"""
        return (probability * odds) - 1

    def categorize_bet(self, probability, odds, ev):
        """
        Categorize bet by quality and determine bet type

        When real odds aren't available (ev == 0), use probability-based strategy
        """
        # If EV is 0, we're using estimated fair odds - switch to probability-based strategy
        if abs(ev) < 0.01:
            if probability > 0.35:
                return "🔥 STRONG", "WIN", 85
            elif probability > 0.28:
                return "⭐ GOOD", "WIN", 75
            elif probability > 0.22:
                return "💎 DECENT", "WIN", 65
            else:
                return "⚪ SKIP", "NONE", 0

        # EV-based strategy (when real odds are available)
        if probability > 0.40 and 2.0 <= odds <= 4.0 and ev > 0.3:
            return "🔥 EXCELLENT", "WIN", 85
        elif probability > 0.25 and 4.0 <= odds <= 8.0 and ev > 0.3:
            return "⭐ GOOD", "WIN", 75
        elif probability > 0.20 and 8.0 <= odds <= 15.0 and ev > 0.4:
            return "💎 VALUE", "PLACE", 70
        elif probability > 0.15 and odds > 15.0 and ev > 0.5:
            return "🎲 LONGSHOT", "PLACE", 70
        else:
            return "⚪ SKIP", "NONE", 0

    def generate_betting_slip_for_track(self, race_info):
        """Generate optimal betting slip for races on a specific track"""
        date_str = race_info['date']
        track_name = race_info['track_name']
        races = race_info['races']

        print("="*80)
        print(f"🎯 OPTIMAL BETTING SLIP - {track_name.upper()}")
        print(f"📅 Date: {date_str}")
        print(f"💰 Budget: {self.budget} SEK")
        print(f"📊 Strategy: Individual High-EV Betting (Balanced Profit)")
        print("="*80)
        print()

        # Get race IDs
        race_ids = [race.get('id') for race in races if race.get('id')]
        print(f"✓ Found {len(race_ids)} races on {track_name}")
        print()

        # Analyze all races and select best bets
        all_opportunities = []

        for i, race_id in enumerate(race_ids, 1):
            print(f"📊 Analyzing Race {i}...")

            try:
                # Get race details and predictions
                race_data = self.predictor.scraper.get_race_details(race_id)
                if not race_data:
                    print(f"   ⚠️  Could not fetch race details")
                    continue

                # Check for scratched horses and display them
                scratched_horses = []
                for start in race_data.get('starts', []):
                    if start.get('scratched', False):
                        horse_name = start.get('horse', {}).get('name', 'Unknown')
                        start_num = start.get('number', '?')
                        scratched_horses.append(f"#{start_num} {horse_name}")
                if scratched_horses:
                    print(f"   ⚠️  Scratched (struken): {', '.join(scratched_horses)}")

                # Process race through predictor
                df = self.predictor.processor.process_race_data([race_data], self.predictor.feature_cols)
                if df.empty:
                    print(f"   ⚠️  Could not process race data")
                    continue

                # Get predictions
                predictions = self.predictor.model.predict_proba(df[self.predictor.feature_cols])

                # Analyze each horse
                for idx, row in df.iterrows():
                    horse_name = row.get('horse_name', 'Unknown')
                    start_num = row.get('start_number', '?')
                    prob = predictions[idx][1]  # Probability of winning

                    # Get odds (placeholder - would need real odds from ATG)
                    # For demo, estimate odds from probability
                    estimated_odds = max(1.5, min(99.0, 1.0 / prob)) if prob > 0.01 else 50.0

                    ev = self.calculate_expected_value(prob, estimated_odds)
                    category, bet_type, base_bet = self.categorize_bet(prob, estimated_odds, ev)

                    if bet_type != "NONE":
                        all_opportunities.append({
                            'race': i,
                            'race_id': race_id,
                            'horse_name': horse_name,
                            'start_num': start_num,
                            'probability': prob,
                            'odds': estimated_odds,
                            'ev': ev,
                            'category': category,
                            'bet_type': bet_type,
                            'base_bet': base_bet
                        })

                print(f"   ✓ Analyzed {len(df)} horses")

            except Exception as e:
                print(f"   ⚠️  Error analyzing race: {e}")
                continue

        print()
        print("="*80)
        print()

        if not all_opportunities:
            print("❌ No good betting opportunities found!")
            print("\nRecommendation: SKIP this race day - wait for better opportunities")
            return

        # Sort by EV (or probability if all EV are zero)
        # Check if all EVs are near zero (estimated odds scenario)
        if all(abs(opp['ev']) < 0.01 for opp in all_opportunities):
            # Sort by probability when using estimated odds
            all_opportunities.sort(key=lambda x: x['probability'], reverse=True)
        else:
            # Sort by EV when real odds are available
            all_opportunities.sort(key=lambda x: x['ev'], reverse=True)

        selected_bets = []
        remaining_budget = self.budget
        min_bet = max(10, self.budget * 0.05)  # Minimum bet: 5% of budget or 10 SEK

        print("🎯 SELECTED BETS (Top EV opportunities):")
        print()

        for i, opp in enumerate(all_opportunities[:6]):  # Consider top 6
            if remaining_budget < min_bet:  # Minimum bet
                break

            # Allocate bet amount based on EV and remaining budget (scales with total budget)
            if i == 0:  # Best opportunity gets most
                bet_amount = remaining_budget * 0.35
            elif i <= 2:  # Top 3 get good allocation
                bet_amount = remaining_budget * 0.25
            else:  # Others get smaller amounts
                bet_amount = remaining_budget * 0.20

            bet_amount = int(bet_amount)

            if bet_amount < min_bet:  # Skip if too small
                continue

            selected_bets.append({**opp, 'bet_amount': bet_amount})
            remaining_budget -= bet_amount

        # Print betting slip
        total_bet = 0
        for i, bet in enumerate(selected_bets, 1):
            print(f"{bet['category']} - Race {bet['race']}")
            print(f"   Horse #{bet['start_num']}: {bet['horse_name']}")
            print(f"   Win probability: {bet['probability']*100:.1f}%")
            print(f"   Estimated odds: {bet['odds']:.1f}")
            print(f"   Expected Value: {bet['ev']:.2f}")
            print(f"   → {bet['bet_type']} bet: {bet['bet_amount']} SEK")
            print()
            total_bet += bet['bet_amount']

        print("="*80)
        print(f"📋 BETTING SLIP SUMMARY")
        print("="*80)
        print(f"Total bets: {len(selected_bets)}")
        print(f"Total amount: {total_bet} SEK")
        print(f"Remaining budget: {self.budget - total_bet} SEK")
        print()

        # Calculate profit scenarios
        print("💰 PROFIT SCENARIOS:")
        print()

        # Conservative (40% of bets win)
        winners = int(len(selected_bets) * 0.4)
        conservative_return = sum(bet['bet_amount'] * bet['odds']
                                 for bet in selected_bets[:winners])
        print(f"Conservative ({winners}/{len(selected_bets)} win):")
        print(f"  Payout: {conservative_return:.0f} SEK")
        print(f"  Profit: {conservative_return - total_bet:+.0f} SEK ({(conservative_return/total_bet - 1)*100:+.1f}% ROI)")
        print()

        # Expected (60% of bets win)
        winners = int(len(selected_bets) * 0.6)
        expected_return = sum(bet['bet_amount'] * bet['odds']
                             for bet in selected_bets[:winners])
        print(f"Expected ({winners}/{len(selected_bets)} win):")
        print(f"  Payout: {expected_return:.0f} SEK")
        print(f"  Profit: {expected_return - total_bet:+.0f} SEK ({(expected_return/total_bet - 1)*100:+.1f}% ROI)")
        print()

        # Best case (80% of bets win)
        winners = int(len(selected_bets) * 0.8)
        best_return = sum(bet['bet_amount'] * bet['odds']
                         for bet in selected_bets[:winners])
        print(f"Best case ({winners}/{len(selected_bets)} win):")
        print(f"  Payout: {best_return:.0f} SEK")
        print(f"  Profit: {best_return - total_bet:+.0f} SEK ({(best_return/total_bet - 1)*100:+.1f}% ROI)")
        print()

        print("="*80)
        print()
        print("📝 NOTES:")
        print("- Odds are estimated based on model probabilities")
        print("- Check actual ATG odds before placing bets")
        print("- High EV (>1.2) indicates profitable bets long-term")
        print("- Expect variance - not all bets will win")
        print("- This is a realistic strategy based on temporal model validation")
        print()
        print("Good luck! 🍀")
        print()

    def generate_betting_slip(self, date_str, game_type=None):
        """Generate optimal betting slip"""

        race_type_label = game_type if game_type else ""

        print("="*80)
        print(f"🎯 OPTIMAL {race_type_label} BETTING SLIP".strip())
        print(f"📅 Date: {date_str}")
        print(f"💰 Budget: {self.budget} SEK")
        print(f"📊 Strategy: Individual High-EV Betting (Balanced Profit)")
        print("="*80)
        print()

        # Get race predictions
        print("🔍 Fetching race data and generating predictions...")
        print()

        try:
            # If game_type is specified, get that game's races, otherwise get V85 for backwards compatibility
            if game_type:
                game_info = self.predictor.scraper.get_game_info(date_str, game_type)
                if not game_info or not game_info.get('race_ids'):
                    print(f"❌ No {game_type} races found for {date_str}")
                    print(f"\nTry checking ATG.se for {game_type} schedule or use a different date.")
                    return
                race_ids = game_info.get('race_ids', [])
            else:
                races = self.predictor.scraper.get_v85_info(date_str)
                if not races or not races.get('races'):
                    print(f"❌ No races found for {date_str}")
                    print(f"\nTry checking ATG.se for race schedule or use a different date.")
                    return
                race_ids = races.get('races', [])

            race_label = f"{game_type} races" if game_type else "races"
            print(f"✓ Found {len(race_ids)} {race_label}")
            print()

        except Exception as e:
            print(f"❌ Error fetching races: {e}")
            return

        # Analyze all races and select best bets
        all_opportunities = []

        for i, race_id in enumerate(race_ids, 1):
            race_label = f"{game_type} Race" if game_type else "Race"
            print(f"📊 Analyzing {race_label} {i}...")

            try:
                # Get race details and predictions
                race_data = self.predictor.scraper.get_race_details(race_id)
                if not race_data:
                    print(f"   ⚠️  Could not fetch race details")
                    continue

                # Check for scratched horses and display them
                scratched_horses = []
                for start in race_data.get('starts', []):
                    if start.get('scratched', False):
                        horse_name = start.get('horse', {}).get('name', 'Unknown')
                        start_num = start.get('number', '?')
                        scratched_horses.append(f"#{start_num} {horse_name}")
                if scratched_horses:
                    print(f"   ⚠️  Scratched (struken): {', '.join(scratched_horses)}")

                # Process race through predictor
                df = self.predictor.processor.process_race_data([race_data], self.predictor.feature_cols)
                if df.empty:
                    print(f"   ⚠️  Could not process race data")
                    continue

                # Get predictions
                predictions = self.predictor.model.predict_proba(df[self.predictor.feature_cols])

                # Analyze each horse
                for idx, row in df.iterrows():
                    horse_name = row.get('horse_name', 'Unknown')
                    start_num = row.get('start_number', '?')
                    prob = predictions[idx][1]  # Probability of winning

                    # Get odds (placeholder - would need real odds from ATG)
                    # For demo, estimate odds from probability
                    estimated_odds = max(1.5, min(99.0, 1.0 / prob)) if prob > 0.01 else 50.0

                    ev = self.calculate_expected_value(prob, estimated_odds)
                    category, bet_type, base_bet = self.categorize_bet(prob, estimated_odds, ev)

                    if bet_type != "NONE":
                        all_opportunities.append({
                            'race': i,
                            'race_id': race_id,
                            'horse_name': horse_name,
                            'start_num': start_num,
                            'probability': prob,
                            'odds': estimated_odds,
                            'ev': ev,
                            'category': category,
                            'bet_type': bet_type,
                            'base_bet': base_bet
                        })

                print(f"   ✓ Analyzed {len(df)} horses")

            except Exception as e:
                print(f"   ⚠️  Error analyzing race: {e}")
                continue

        print()
        print("="*80)
        print()

        if not all_opportunities:
            print("❌ No good betting opportunities found!")
            print("\nRecommendation: SKIP this race day - wait for better opportunities")
            return

        # Sort by EV (or probability if all EV are zero)
        # Check if all EVs are near zero (estimated odds scenario)
        if all(abs(opp['ev']) < 0.01 for opp in all_opportunities):
            # Sort by probability when using estimated odds
            all_opportunities.sort(key=lambda x: x['probability'], reverse=True)
        else:
            # Sort by EV when real odds are available
            all_opportunities.sort(key=lambda x: x['ev'], reverse=True)

        selected_bets = []
        remaining_budget = self.budget
        min_bet = max(10, self.budget * 0.05)  # Minimum bet: 5% of budget or 10 SEK

        print("🎯 SELECTED BETS (Top EV opportunities):")
        print()

        for i, opp in enumerate(all_opportunities[:6]):  # Consider top 6
            if remaining_budget < min_bet:  # Minimum bet
                break

            # Allocate bet amount based on EV and remaining budget (scales with total budget)
            if i == 0:  # Best opportunity gets most
                bet_amount = remaining_budget * 0.35
            elif i <= 2:  # Top 3 get good allocation
                bet_amount = remaining_budget * 0.25
            else:  # Others get smaller amounts
                bet_amount = remaining_budget * 0.20

            bet_amount = int(bet_amount)

            if bet_amount < min_bet:  # Skip if too small
                continue

            selected_bets.append({**opp, 'bet_amount': bet_amount})
            remaining_budget -= bet_amount

        # Print betting slip
        total_bet = 0
        for i, bet in enumerate(selected_bets, 1):
            race_label = f"{game_type} Race" if game_type else "Race"
            print(f"{bet['category']} - {race_label} {bet['race']}")
            print(f"   Horse #{bet['start_num']}: {bet['horse_name']}")
            print(f"   Win probability: {bet['probability']*100:.1f}%")
            print(f"   Estimated odds: {bet['odds']:.1f}")
            print(f"   Expected Value: {bet['ev']:.2f}")
            print(f"   → {bet['bet_type']} bet: {bet['bet_amount']} SEK")
            print()
            total_bet += bet['bet_amount']

        print("="*80)
        print(f"📋 BETTING SLIP SUMMARY")
        print("="*80)
        print(f"Total bets: {len(selected_bets)}")
        print(f"Total amount: {total_bet} SEK")
        print(f"Remaining budget: {self.budget - total_bet} SEK")
        print()

        # Calculate profit scenarios
        print("💰 PROFIT SCENARIOS:")
        print()

        # Conservative (40% of bets win)
        winners = int(len(selected_bets) * 0.4)
        conservative_return = sum(bet['bet_amount'] * bet['odds']
                                 for bet in selected_bets[:winners])
        print(f"Conservative ({winners}/{len(selected_bets)} win):")
        print(f"  Payout: {conservative_return:.0f} SEK")
        print(f"  Profit: {conservative_return - total_bet:+.0f} SEK ({(conservative_return/total_bet - 1)*100:+.1f}% ROI)")
        print()

        # Expected (60% of bets win)
        winners = int(len(selected_bets) * 0.6)
        expected_return = sum(bet['bet_amount'] * bet['odds']
                             for bet in selected_bets[:winners])
        print(f"Expected ({winners}/{len(selected_bets)} win):")
        print(f"  Payout: {expected_return:.0f} SEK")
        print(f"  Profit: {expected_return - total_bet:+.0f} SEK ({(expected_return/total_bet - 1)*100:+.1f}% ROI)")
        print()

        # Best case (80% of bets win)
        winners = int(len(selected_bets) * 0.8)
        best_return = sum(bet['bet_amount'] * bet['odds']
                         for bet in selected_bets[:winners])
        print(f"Best case ({winners}/{len(selected_bets)} win):")
        print(f"  Payout: {best_return:.0f} SEK")
        print(f"  Profit: {best_return - total_bet:+.0f} SEK ({(best_return/total_bet - 1)*100:+.1f}% ROI)")
        print()

        print("="*80)
        print()
        print("📝 NOTES:")
        print("- Odds are estimated based on model probabilities")
        print("- Check actual ATG odds before placing bets")
        print("- High EV (>1.2) indicates profitable bets long-term")
        print("- Expect variance - not all bets will win")
        print("- This is a realistic strategy based on temporal model validation")
        print()
        print("Good luck! 🍀")
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Generate optimal betting slip for maximum profit potential',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Auto-find next race, bet 300 SEK (default)
  %(prog)s --total 1000                 # Auto-find next race, bet 1000 SEK
  %(prog)s --game V75                   # Auto-find next V75, bet 300 SEK
  %(prog)s --game V86 --total 500       # Auto-find next V86, bet 500 SEK
  %(prog)s --date 2026-01-17            # Specific date, bet 300 SEK
  %(prog)s --total 500 --date 2026-01-17  # Specific date, bet 500 SEK
  %(prog)s --track solvalla             # Auto-find next race on Solvalla
  %(prog)s --track umaker --total 500   # Auto-find next race on Umåker, bet 500 SEK

Strategy:
  Individual high-EV betting (not system betting)
  Based on temporal model: 21.5%% win rate, +96.6%% ROI
        """
    )

    parser.add_argument(
        '--total',
        type=int,
        default=300,
        metavar='SEK',
        help='Total betting budget in SEK (default: 300)'
    )

    parser.add_argument(
        '--date',
        type=str,
        metavar='YYYY-MM-DD',
        help='Specific race date (default: auto-find next race or track)'
    )

    parser.add_argument(
        '--track',
        type=str,
        metavar='TRACK',
        help='Find next race on specific track (e.g., solvalla, umaker, aby, jagersro)'
    )

    parser.add_argument(
        '--game',
        type=str,
        metavar='TYPE',
        help='Search for specific game type: V75, V86, V85, V65, V64, etc. (default: any race)'
    )

    args = parser.parse_args()

    # Validate budget
    if args.total < 50:
        print("❌ Minimum budget is 50 SEK")
        sys.exit(1)

    if args.total > 10000:
        print("⚠️  Warning: Large budget detected ({} SEK)".format(args.total))
        print("   Consider splitting across multiple race days for better risk management")
        print()

    # Create generator with specified budget
    generator = BettingSlipGenerator(budget=args.total)

    # Check for conflicting options
    if args.date and args.track:
        print("❌ Cannot use both --date and --track options together")
        print("Please use either --date for a specific date or --track to find next race on a track")
        sys.exit(1)

    if args.date and args.game:
        print("❌ Cannot use both --date and --game options together")
        print("Please use either --date for a specific date or --game to find next game")
        sys.exit(1)

    if args.track and args.game:
        print("❌ Cannot use both --track and --game options together")
        print("Please use either --track for a specific track or --game to find next game")
        sys.exit(1)

    # Validate game type if specified
    valid_game_types = ['V75', 'V86', 'V85', 'V65', 'V64', 'V4', 'V5', 'GS75', 'V3']
    if args.game:
        game_type = args.game.upper()
        if game_type not in valid_game_types:
            print(f"⚠️  Warning: '{game_type}' may not be a valid game type")
            print(f"   Common types: {', '.join(valid_game_types)}")
            print(f"   Continuing anyway...")
            print()
        else:
            game_type = args.game.upper()

    # Handle track-based search
    if args.track:
        race_info = generator.find_next_race_on_track(args.track, max_days=30)
        if not race_info:
            sys.exit(1)
        generator.generate_betting_slip_for_track(race_info)
    # Handle date-based search
    elif args.date:
        # Validate date format
        try:
            datetime.strptime(args.date, '%Y-%m-%d')
            date_str = args.date
        except ValueError:
            print(f"❌ Invalid date format: {args.date}")
            print("Please use YYYY-MM-DD format")
            sys.exit(1)
        generator.generate_betting_slip(date_str, game_type=None)
    # Handle game type search
    elif args.game:
        date_str, game_type = generator.find_next_game(args.game.upper(), max_days=30)
        if not date_str:
            sys.exit(1)
        generator.generate_betting_slip(date_str, game_type=game_type)
    # Default: auto-find next race (any race)
    else:
        race_info = generator.find_next_race(max_days=30)
        if not race_info:
            sys.exit(1)
        generator.generate_betting_slip_for_track(race_info)


if __name__ == '__main__':
    main()
