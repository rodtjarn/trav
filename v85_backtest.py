#!/usr/bin/env python3
"""
V85 Backtesting - Test 1000 SEK betting strategy over last 10 V85 races
"""

import pandas as pd
from predict_v85 import V85Predictor
from atg_api_scraper import ATGAPIScraper
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.WARNING)


class V85Backtester:
    """Backtest V85 betting strategies"""

    def __init__(self, budget=1000):
        self.budget = budget
        self.predictor = V85Predictor()
        self.scraper = ATGAPIScraper(delay=0.5)
        self.cost_per_row = 1  # SEK

    def find_v85_dates(self, days_back=60):
        """
        Find V85 race dates by checking calendar backwards

        Args:
            days_back: Number of days to search backwards

        Returns:
            List of dates with V85 races
        """
        print(f"\n🔍 Searching for V85 races in last {days_back} days...")

        v85_dates = []
        today = datetime.now()

        for i in range(days_back):
            date = (today - timedelta(days=i)).strftime('%Y-%m-%d')

            # Get calendar
            calendar = self.scraper.get_calendar_for_date(date)

            if calendar and 'games' in calendar:
                games = calendar.get('games', {})
                if 'V85' in games and games['V85']:
                    v85_game = games['V85'][0]
                    v85_dates.append({
                        'date': date,
                        'game_id': v85_game.get('id'),
                        'status': v85_game.get('status')
                    })
                    print(f"  ✓ Found V85 on {date} ({v85_game.get('status')})")

            self.scraper._sleep()

        return v85_dates

    def get_actual_winners(self, date):
        """Get actual winning combination for a V85 date"""

        v85_info = self.scraper.get_v85_info(date)

        if not v85_info:
            return None

        race_data = self.scraper.scrape_date(date)
        df_results = pd.DataFrame(race_data)
        df_results = df_results[df_results['race_id'].isin(v85_info['race_ids'])]
        df_results['v85_race_number'] = df_results['race_id'].map(v85_info['v85_race_mapping'])

        winners = []
        for race_num in sorted(df_results['v85_race_number'].unique()):
            race_df = df_results[df_results['v85_race_number'] == race_num]
            winner = race_df[race_df['finish_place'] == 1]

            if len(winner) > 0:
                winners.append(int(winner.iloc[0]['start_number']))
            else:
                return None  # Race not completed

        return winners if len(winners) == 8 else None

    def create_system(self, predictions_df, structure):
        """
        Create V85 system from predictions

        Args:
            predictions_df: Model predictions
            structure: List of picks per race (e.g., [2,2,2,2,2,2,2,2])

        Returns:
            Dict mapping race number to list of horse numbers
        """
        system = {}

        for i, race_num in enumerate(sorted(predictions_df.keys())):
            num_picks = structure[i]
            race_pred = predictions_df[race_num]
            top_picks = race_pred.head(num_picks)
            system[race_num] = [int(row['start_number']) for _, row in top_picks.iterrows()]

        return system

    def check_system_covers_winners(self, system, winners):
        """Check if system covers winning combination"""
        for i, race_num in enumerate(sorted(system.keys())):
            if winners[i] not in system[race_num]:
                return False
        return True

    def calculate_combinations(self, structure):
        """Calculate total combinations"""
        total = 1
        for picks in structure:
            total *= picks
        return total

    def test_strategies_for_date(self, date):
        """Test different strategies for a specific V85 date"""

        print(f"\n{'='*80}")
        print(f"Testing {date}")
        print(f"{'='*80}")

        # Get predictions
        try:
            predictions_df = self.predictor.predict_v85(date)
        except Exception as e:
            print(f"❌ Error getting predictions: {e}")
            return None

        if not predictions_df:
            print(f"❌ No predictions available")
            return None

        # Get actual winners
        winners = self.get_actual_winners(date)

        if not winners:
            print(f"❌ Race not completed or results unavailable")
            return None

        print(f"Winning combination: {'-'.join(map(str, winners))}")

        # Test different system structures
        strategies = {
            'Conservative': [2, 2, 2, 2, 2, 2, 2, 2],      # 256 combinations
            'Moderate': [3, 2, 2, 2, 2, 2, 2, 2],           # 384 combinations
            'Aggressive': [3, 3, 2, 2, 2, 2, 2, 2],         # 576 combinations
            'Wide Coverage': [3, 3, 3, 2, 2, 2, 2, 2],      # 864 combinations
            'Model-Driven': self._get_model_driven_structure(predictions_df),
        }

        results = {}

        for strategy_name, structure in strategies.items():
            combos = self.calculate_combinations(structure)

            if combos > self.budget:
                results[strategy_name] = {
                    'structure': '-'.join(map(str, structure)),
                    'combos': combos,
                    'cost': combos,
                    'covers': False,
                    'skipped': True
                }
                continue

            system = self.create_system(predictions_df, structure)
            covers = self.check_system_covers_winners(system, winners)

            results[strategy_name] = {
                'structure': '-'.join(map(str, structure)),
                'combos': combos,
                'cost': combos * self.cost_per_row,
                'covers': covers,
                'skipped': False
            }

        return {
            'date': date,
            'winners': winners,
            'strategies': results
        }

    def _get_model_driven_structure(self, predictions_df):
        """Generate model-driven structure based on race difficulty"""
        structure = []

        for race_num in sorted(predictions_df.keys()):
            race_pred = predictions_df[race_num]
            top_prob = race_pred.iloc[0]['win_probability']

            # Check spread between top 3
            top3 = race_pred.head(3)
            spread = top3.iloc[0]['win_probability'] - top3.iloc[2]['win_probability']

            # Clear favorite
            if top_prob > 0.50 and spread > 0.15:
                structure.append(2)
            # Very uncertain
            elif spread < 0.08 or top_prob < 0.35:
                structure.append(3)
            else:
                structure.append(2)

        return structure

    def run_backtest(self, num_races=10):
        """Run backtest over multiple V85 races"""

        print("\n" + "="*80)
        print(f"V85 BACKTESTING - {self.budget} SEK BUDGET")
        print(f"Testing last {num_races} V85 races")
        print("="*80)

        # Find V85 dates
        v85_dates = self.find_v85_dates(days_back=90)

        if len(v85_dates) < num_races:
            print(f"\n⚠️  Only found {len(v85_dates)} V85 races, using all available")
            num_races = len(v85_dates)

        # Filter to completed races
        completed_dates = [d for d in v85_dates if d['status'] == 'results']

        if len(completed_dates) < num_races:
            print(f"⚠️  Only {len(completed_dates)} completed races available")
            num_races = len(completed_dates)

        # Test most recent completed races
        test_dates = completed_dates[:num_races]

        print(f"\nTesting {num_races} V85 races:")
        for d in test_dates:
            print(f"  - {d['date']}")

        # Run tests
        all_results = []

        for v85_date in test_dates:
            date = v85_date['date']
            result = self.test_strategies_for_date(date)

            if result:
                all_results.append(result)

        # Summary
        self._print_summary(all_results)

        return all_results

    def _print_summary(self, all_results):
        """Print summary of backtest results"""

        print("\n" + "="*80)
        print("BACKTEST SUMMARY")
        print("="*80)

        if not all_results:
            print("No results to analyze")
            return

        # Aggregate by strategy
        strategy_names = list(all_results[0]['strategies'].keys())
        strategy_stats = {name: {'wins': 0, 'total': 0, 'total_cost': 0} for name in strategy_names}

        print("\n📊 RACE-BY-RACE RESULTS:")
        print("-"*80)

        for result in all_results:
            date = result['date']
            winners = '-'.join(map(str, result['winners']))

            print(f"\n{date}: {winners}")

            for strategy_name, strategy_result in result['strategies'].items():
                if strategy_result['skipped']:
                    print(f"  {strategy_name:.<25} SKIPPED (>{self.budget} SEK)")
                    continue

                strategy_stats[strategy_name]['total'] += 1
                strategy_stats[strategy_name]['total_cost'] += strategy_result['cost']

                if strategy_result['covers']:
                    strategy_stats[strategy_name]['wins'] += 1
                    print(f"  {strategy_name:.<25} ✅ WIN ({strategy_result['structure']}, {strategy_result['cost']} SEK)")
                else:
                    print(f"  {strategy_name:.<25} ❌ ({strategy_result['structure']}, {strategy_result['cost']} SEK)")

        # Overall statistics
        print("\n" + "="*80)
        print("OVERALL STATISTICS")
        print("="*80)
        print(f"{'Strategy':<25} {'Wins':<10} {'Win Rate':<12} {'Avg Cost':<12} {'Total Cost':<12}")
        print("-"*80)

        for strategy_name in strategy_names:
            stats = strategy_stats[strategy_name]

            if stats['total'] == 0:
                print(f"{strategy_name:<25} SKIPPED")
                continue

            win_rate = (stats['wins'] / stats['total']) * 100
            avg_cost = stats['total_cost'] / stats['total']
            total_cost = stats['total_cost']

            print(f"{strategy_name:<25} {stats['wins']}/{stats['total']:<7} "
                  f"{win_rate:>6.1f}%      {avg_cost:>7.0f} SEK   {total_cost:>8.0f} SEK")

        # Calculate theoretical ROI
        print("\n" + "="*80)
        print("THEORETICAL PROFIT ANALYSIS")
        print("="*80)
        print("\nAssuming different average V85 dividends:")

        # V85 dividends vary wildly - show different scenarios
        example_dividends = [50000, 100000, 500000, 1000000, 5000000]

        for dividend in example_dividends:
            print(f"\nIf average dividend = {dividend:,} SEK:")

            for strategy_name in strategy_names:
                stats = strategy_stats[strategy_name]

                if stats['total'] == 0:
                    continue

                wins = stats['wins']
                total_spent = stats['total_cost']
                total_payout = wins * dividend
                profit = total_payout - total_spent
                roi = (profit / total_spent) * 100 if total_spent > 0 else 0

                if wins > 0:
                    result_emoji = "✅" if profit > 0 else "⚠️"
                    print(f"  {result_emoji} {strategy_name:.<25} "
                          f"{total_payout:>10,} SEK payout, {profit:>+10,} SEK profit ({roi:>+6.1f}% ROI)")

        print("\n" + "="*80)
        print("💡 KEY INSIGHTS")
        print("="*80)

        best_strategy = max(strategy_stats.items(),
                           key=lambda x: x[1]['wins'] if x[1]['total'] > 0 else 0)

        if best_strategy[1]['total'] > 0:
            best_name = best_strategy[0]
            best_stats = best_strategy[1]

            print(f"""
Best strategy: {best_name}
  - Win rate: {best_stats['wins']}/{best_stats['total']} ({best_stats['wins']/best_stats['total']*100:.1f}%)
  - Average cost: {best_stats['total_cost']/best_stats['total']:.0f} SEK per race
  - Total invested: {best_stats['total_cost']:,} SEK

V85 Reality Check:
  - Getting all 8 winners is EXTREMELY difficult
  - Even best models rarely exceed 20-30% win rate
  - V85 dividends vary from 10,000 to 10,000,000+ SEK
  - Your {best_stats['wins']} wins need avg dividend of {best_stats['total_cost']/best_stats['wins']:,.0f} SEK to break even

Recommendation:
  - V85 is high risk, high reward
  - Budget should be "entertainment money" you can afford to lose
  - Consider smaller pools (V64, V65) for better odds
  - Use model for place/trifecta betting instead
""")

        print("="*80)


def main():
    """Run V85 backtest"""

    backtester = V85Backtester(budget=1000)
    results = backtester.run_backtest(num_races=10)


if __name__ == "__main__":
    main()
