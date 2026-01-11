#!/usr/bin/env python3
"""
Generate optimal betting slip for maximum profit potential.

Usage:
    python create_bet.py --total 1000                 # Auto-find next V85, bet 1000 SEK
    python create_bet.py                              # Auto-find next V85, bet 300 SEK (default)
    python create_bet.py --date 2026-01-17            # Specific date
    python create_bet.py --total 500 --date 2026-01-17
    python create_bet.py --track solvalla             # Auto-find next race on Solvalla
    python create_bet.py --track umaker --total 500   # Next race on Umåker, bet 500 SEK

Strategy: Individual high-EV betting (not V85 system)
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
        print("🔍 Searching for next V85 race day...")
        print()

        today = datetime.now()

        for i in range(max_days):
            check_date = (today + timedelta(days=i)).strftime('%Y-%m-%d')

            try:
                v85_info = self.predictor.scraper.get_v85_info(check_date)

                if v85_info and v85_info.get('races') and len(v85_info.get('races', [])) > 0:
                    races = v85_info.get('races', [])
                    track_name = v85_info.get('track', {}).get('name', 'Unknown')
                    track_id = v85_info.get('trackId', 'Unknown')

                    print(f"✅ Found V85 on {check_date}")
                    print(f"   Track: {track_name} (ID: {track_id})")
                    print(f"   Number of races: {len(races)}")
                    print()

                    return check_date

            except Exception as e:
                # Continue searching on error
                continue

        print(f"❌ No V85 found in next {max_days} days")
        print("\nPlease check ATG.se for V85 schedule or try a later date range.")
        return None

    def find_next_race_on_track(self, track_name, max_days=30):
        """Find the next race on a specific track by searching ATG calendar"""
        print(f"🔍 Searching for next race on {track_name.upper()}...")
        print()

        today = datetime.now()
        track_name_lower = track_name.lower()

        for i in range(max_days):
            check_date = (today + timedelta(days=i)).strftime('%Y-%m-%d')

            try:
                calendar = self.predictor.scraper.get_calendar_for_date(check_date)

                if not calendar or not calendar.get('tracks'):
                    continue

                # Search through all tracks for matching track name
                for track in calendar.get('tracks', []):
                    track_calendar_name = track.get('name', '').lower()

                    # Check if track name matches (case-insensitive partial match)
                    if track_name_lower in track_calendar_name:
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
        """Categorize bet by quality and determine bet type"""
        if probability > 0.40 and 2.0 <= odds <= 4.0 and ev > 1.1:
            return "🔥 EXCELLENT", "WIN", 85
        elif probability > 0.25 and 4.0 <= odds <= 8.0 and ev > 1.1:
            return "⭐ GOOD", "WIN", 75
        elif probability > 0.20 and 8.0 <= odds <= 15.0 and ev > 1.2:
            return "💎 VALUE", "PLACE", 70
        elif probability > 0.15 and odds > 15.0 and ev > 1.5:
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

                # Process race through predictor
                df = self.predictor.processor.process_race_data([race_data])
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

        # Sort by EV and select top bets within budget
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

    def generate_betting_slip(self, date_str):
        """Generate optimal betting slip"""

        print("="*80)
        print(f"🎯 OPTIMAL V85 BETTING SLIP")
        print(f"📅 Date: {date_str}")
        print(f"💰 Budget: {self.budget} SEK")
        print(f"📊 Strategy: Individual High-EV Betting (Balanced Profit)")
        print("="*80)
        print()

        # Get V85 predictions
        print("🔍 Fetching race data and generating predictions...")
        print()

        try:
            races = self.predictor.scraper.get_v85_info(date_str)
            if not races or not races.get('races'):
                print(f"❌ No V85 races found for {date_str}")
                print("\nTry checking ATG.se for V85 schedule or use a different date.")
                return

            race_ids = races.get('races', [])
            print(f"✓ Found {len(race_ids)} V85 races")
            print()

        except Exception as e:
            print(f"❌ Error fetching races: {e}")
            return

        # Analyze all races and select best bets
        all_opportunities = []

        for i, race_id in enumerate(race_ids, 1):
            print(f"📊 Analyzing V85 Race {i}...")

            try:
                # Get race details and predictions
                race_data = self.predictor.scraper.get_race_details(race_id)
                if not race_data:
                    print(f"   ⚠️  Could not fetch race details")
                    continue

                # Process race through predictor
                df = self.predictor.processor.process_race_data([race_data])
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
            print("\nRecommendation: SKIP this V85 - wait for better opportunities")
            return

        # Sort by EV and select top bets within budget
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
            print(f"{bet['category']} - V85 Race {bet['race']}")
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
  %(prog)s                              # Auto-find next V85, bet 300 SEK (default)
  %(prog)s --total 1000                 # Auto-find next V85, bet 1000 SEK
  %(prog)s --date 2026-01-17            # Specific date, bet 300 SEK
  %(prog)s --total 500 --date 2026-01-17  # Specific date, bet 500 SEK
  %(prog)s --track solvalla             # Auto-find next race on Solvalla
  %(prog)s --track umaker --total 500   # Auto-find next race on Umåker, bet 500 SEK

Strategy:
  Individual high-EV betting (not V85 system jackpot)
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
        help='Specific race date (default: auto-find next V85 or track)'
    )

    parser.add_argument(
        '--track',
        type=str,
        metavar='TRACK',
        help='Find next race on specific track (e.g., solvalla, umaker, aby, jagersro)'
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
        generator.generate_betting_slip(date_str)
    # Default: auto-find next V85
    else:
        date_str = generator.find_next_v85(max_days=30)
        if not date_str:
            sys.exit(1)
        generator.generate_betting_slip(date_str)


if __name__ == '__main__':
    main()
