#!/usr/bin/env python3
"""
Compare V85 system betting vs individual race betting
Using same budget on last 10 V85 races
"""

import pandas as pd
from predict_v85 import V85Predictor
from atg_api_scraper import ATGAPIScraper
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.WARNING)


class V85BettingComparison:
    """Compare V85 system vs individual race betting"""

    def __init__(self, budget_per_v85=1000):
        self.budget_per_v85 = budget_per_v85
        self.predictor = V85Predictor()
        self.scraper = ATGAPIScraper(delay=0.5)

    def find_v85_dates(self, days_back=90):
        """Find V85 race dates"""
        print(f"\n🔍 Searching for V85 races in last {days_back} days...")

        v85_dates = []
        today = datetime.now()

        for i in range(days_back):
            date = (today - timedelta(days=i)).strftime('%Y-%m-%d')
            calendar = self.scraper.get_calendar_for_date(date)

            if calendar and 'games' in calendar:
                games = calendar.get('games', {})
                if 'V85' in games and games['V85']:
                    v85_game = games['V85'][0]
                    if v85_game.get('status') == 'results':
                        v85_dates.append({
                            'date': date,
                            'game_id': v85_game.get('id'),
                            'status': v85_game.get('status')
                        })
                        print(f"  ✓ Found V85 on {date} (results)")

            self.scraper._sleep()

        return v85_dates

    def get_model_driven_structure(self, predictions_df, max_budget):
        """Generate model-driven structure that fits budget"""
        structure = []

        for race_num in sorted(predictions_df.keys()):
            race_pred = predictions_df[race_num]
            top_prob = race_pred.iloc[0]['win_probability']
            top3 = race_pred.head(3)
            spread = top3.iloc[0]['win_probability'] - top3.iloc[2]['win_probability']

            # Clear favorite
            if top_prob > 0.50 and spread > 0.15:
                structure.append(1)
            # Very uncertain
            elif spread < 0.10 or top_prob < 0.35:
                structure.append(3)
            else:
                structure.append(2)

        # Adjust to fit budget
        while self._calc_combos(structure) > max_budget:
            max_idx = max((i for i in range(len(structure)) if structure[i] > 1),
                         key=lambda i: structure[i], default=None)
            if max_idx is not None:
                structure[max_idx] -= 1
            else:
                break

        return structure

    def _calc_combos(self, picks_list):
        """Calculate combinations"""
        total = 1
        for picks in picks_list:
            total *= picks
        return total

    def create_system(self, predictions_df, structure):
        """Create V85 system"""
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

    def get_actual_winners(self, date):
        """Get actual winning combination"""
        v85_info = self.scraper.get_v85_info(date)
        if not v85_info:
            return None

        race_data = self.scraper.scrape_date(date)
        df_results = pd.DataFrame(race_data)
        df_results = df_results[df_results['race_id'].isin(v85_info['race_ids'])]
        df_results['v85_race_number'] = df_results['race_id'].map(v85_info['v85_race_mapping'])

        winners = []
        top3_by_race = {}

        for race_num in sorted(df_results['v85_race_number'].unique()):
            race_df = df_results[df_results['v85_race_number'] == race_num]
            winner = race_df[race_df['finish_place'] == 1]

            if len(winner) > 0:
                winners.append(int(winner.iloc[0]['start_number']))
                # Get top 3
                top3 = race_df[race_df['finish_place'].isin([1, 2, 3])].sort_values('finish_place')
                top3_by_race[race_num] = {
                    'winners': [int(h['start_number']) for _, h in top3.iterrows()],
                    'odds': {int(h['start_number']): h['final_odds'] for _, h in race_df.iterrows()}
                }
            else:
                return None, None

        return winners if len(winners) == 8 else None, top3_by_race

    def individual_betting_strategy(self, predictions_df, df_results, budget):
        """
        Individual race betting strategy:
        - Bet on winner in high-confidence races (>40% win probability)
        - Bet on top-3 place in medium-confidence races (25-40% win probability)
        """
        bets = []
        total_bet = 0

        # Identify betting opportunities
        opportunities = []

        for race_num in sorted(predictions_df.keys()):
            race_pred = predictions_df[race_num]
            top_pick = race_pred.iloc[0]
            top_prob = top_pick['win_probability']

            race_results = df_results[df_results['v85_race_number'] == race_num]
            actual_winner = race_results[race_results['finish_place'] == 1].iloc[0]
            actual_top3 = race_results[race_results['finish_place'].isin([1, 2, 3])]['start_number'].tolist()

            # High confidence - bet on winner
            if top_prob > 0.40:
                opportunities.append({
                    'race': race_num,
                    'type': 'win',
                    'horse': int(top_pick['start_number']),
                    'horse_name': top_pick['horse_name'],
                    'confidence': top_prob,
                    'odds': top_pick['final_odds'],
                    'winner': int(actual_winner['start_number']),
                    'top3': [int(x) for x in actual_top3]
                })
            # Medium confidence - bet on place (top 3)
            elif top_prob > 0.25:
                opportunities.append({
                    'race': race_num,
                    'type': 'place',
                    'horse': int(top_pick['start_number']),
                    'horse_name': top_pick['horse_name'],
                    'confidence': top_prob,
                    'odds': top_pick['final_odds'],
                    'winner': int(actual_winner['start_number']),
                    'top3': [int(x) for x in actual_top3]
                })

        if not opportunities:
            return [], 0, 0

        # Distribute budget proportionally to confidence
        total_confidence = sum(op['confidence'] for op in opportunities)

        for op in opportunities:
            bet_amount = (op['confidence'] / total_confidence) * budget
            total_bet += bet_amount

            # Check if won
            if op['type'] == 'win':
                won = op['horse'] == op['winner']
                payout = bet_amount * op['odds'] if won else 0
            else:  # place
                won = op['horse'] in op['top3']
                # Place odds typically 1/3 to 1/2 of win odds
                place_odds = 1 + (op['odds'] - 1) * 0.4
                payout = bet_amount * place_odds if won else 0

            bets.append({
                'race': op['race'],
                'type': op['type'],
                'horse': op['horse'],
                'horse_name': op['horse_name'],
                'confidence': op['confidence'],
                'bet': bet_amount,
                'odds': op['odds'],
                'won': won,
                'payout': payout
            })

        total_payout = sum(b['payout'] for b in bets)
        return bets, total_bet, total_payout

    def compare_for_date(self, date):
        """Compare V85 system vs individual betting for one date"""
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

        # Get actual results
        winners, top3_data = self.get_actual_winners(date)
        if not winners:
            print(f"❌ Race not completed or results unavailable")
            return None

        v85_info = self.scraper.get_v85_info(date)
        race_data = self.scraper.scrape_date(date)
        df_results = pd.DataFrame(race_data)
        df_results = df_results[df_results['race_id'].isin(v85_info['race_ids'])]
        df_results['v85_race_number'] = df_results['race_id'].map(v85_info['v85_race_mapping'])

        print(f"\n🎯 Winning combination: {'-'.join(map(str, winners))}")

        # STRATEGY 1: V85 System Betting
        print(f"\n{'='*80}")
        print(f"STRATEGY 1: V85 SYSTEM BETTING (Budget: {self.budget_per_v85} SEK)")
        print(f"{'='*80}")

        structure = self.get_model_driven_structure(predictions_df, self.budget_per_v85)
        system = self.create_system(predictions_df, structure)
        combos = self._calc_combos(structure)
        cost = combos
        covers = self.check_system_covers_winners(system, winners)

        print(f"\nModel-Driven structure: {'-'.join(map(str, structure))}")
        print(f"Total combinations: {combos}")
        print(f"Cost: {cost} SEK")
        print(f"\nResult: {'✅ COVERS winning combo!' if covers else '❌ MISS'}")

        v85_result = {
            'structure': structure,
            'cost': cost,
            'covers': covers
        }

        # STRATEGY 2: Individual Race Betting
        print(f"\n{'='*80}")
        print(f"STRATEGY 2: INDIVIDUAL RACE BETTING (Budget: {self.budget_per_v85} SEK)")
        print(f"{'='*80}")

        bets, total_bet, total_payout = self.individual_betting_strategy(
            predictions_df, df_results, self.budget_per_v85
        )

        print(f"\nPlaced {len(bets)} bets:")
        wins = 0
        for bet in bets:
            if bet['won']:
                wins += 1
                print(f"  ✅ V85 Race {bet['race']}: {bet['type'].upper()} on {bet['horse_name']} "
                      f"({bet['bet']:.0f} SEK @ {bet['odds']:.1f} odds) → {bet['payout']:.0f} SEK")
            else:
                print(f"  ❌ V85 Race {bet['race']}: {bet['type'].upper()} on {bet['horse_name']} "
                      f"({bet['bet']:.0f} SEK @ {bet['odds']:.1f} odds)")

        profit = total_payout - total_bet
        roi = (profit / total_bet * 100) if total_bet > 0 else 0

        print(f"\nTotal bet: {total_bet:.0f} SEK")
        print(f"Total payout: {total_payout:.0f} SEK")
        print(f"Profit: {profit:+.0f} SEK ({roi:+.1f}% ROI)")
        print(f"Win rate: {wins}/{len(bets)} ({wins/len(bets)*100:.1f}%)" if bets else "No bets")

        individual_result = {
            'bets': bets,
            'total_bet': total_bet,
            'total_payout': total_payout,
            'profit': profit,
            'roi': roi,
            'wins': wins
        }

        return {
            'date': date,
            'winners': winners,
            'v85_system': v85_result,
            'individual': individual_result
        }

    def run_comparison(self, num_races=10):
        """Run comparison over multiple V85 races"""
        print("\n" + "="*80)
        print(f"V85 SYSTEM vs INDIVIDUAL BETTING COMPARISON")
        print(f"Budget: {self.budget_per_v85} SEK per V85 race")
        print(f"Testing last {num_races} V85 races")
        print("="*80)

        # Find V85 dates
        v85_dates = self.find_v85_dates(days_back=90)

        if len(v85_dates) < num_races:
            print(f"\n⚠️  Only found {len(v85_dates)} V85 races, using all available")
            num_races = len(v85_dates)

        test_dates = v85_dates[:num_races]

        print(f"\nTesting {num_races} V85 races:")
        for d in test_dates:
            print(f"  - {d['date']}")

        # Run comparisons
        all_results = []

        for v85_date in test_dates:
            date = v85_date['date']
            result = self.compare_for_date(date)

            if result:
                all_results.append(result)

        # Summary
        self._print_summary(all_results)

        return all_results

    def _print_summary(self, all_results):
        """Print comparison summary"""
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)

        if not all_results:
            print("No results to analyze")
            return

        # V85 System stats
        v85_wins = sum(1 for r in all_results if r['v85_system']['covers'])
        v85_total_cost = sum(r['v85_system']['cost'] for r in all_results)
        v85_avg_cost = v85_total_cost / len(all_results)

        # Individual betting stats
        ind_total_bet = sum(r['individual']['total_bet'] for r in all_results)
        ind_total_payout = sum(r['individual']['total_payout'] for r in all_results)
        ind_profit = ind_total_payout - ind_total_bet
        ind_roi = (ind_profit / ind_total_bet * 100) if ind_total_bet > 0 else 0
        ind_total_wins = sum(r['individual']['wins'] for r in all_results)
        ind_total_bets = sum(len(r['individual']['bets']) for r in all_results)

        print(f"\n{'='*80}")
        print(f"STRATEGY 1: V85 SYSTEM BETTING")
        print(f"{'='*80}")
        print(f"Races won: {v85_wins}/{len(all_results)} ({v85_wins/len(all_results)*100:.1f}%)")
        print(f"Total invested: {v85_total_cost:.0f} SEK")
        print(f"Average cost per race: {v85_avg_cost:.0f} SEK")

        # Theoretical V85 payouts
        print(f"\nTheoretical profit (if V85 dividend per win):")
        example_dividends = [50000, 100000, 500000, 1000000]
        for dividend in example_dividends:
            total_payout_v85 = v85_wins * dividend
            profit_v85 = total_payout_v85 - v85_total_cost
            roi_v85 = (profit_v85 / v85_total_cost * 100) if v85_total_cost > 0 else 0
            print(f"  {dividend:>9,} SEK: {total_payout_v85:>12,} SEK payout, "
                  f"{profit_v85:>+12,} SEK profit ({roi_v85:>+8.1f}% ROI)")

        print(f"\n{'='*80}")
        print(f"STRATEGY 2: INDIVIDUAL RACE BETTING")
        print(f"{'='*80}")
        print(f"Total bets: {ind_total_bets}")
        print(f"Bets won: {ind_total_wins}/{ind_total_bets} ({ind_total_wins/ind_total_bets*100:.1f}%)" if ind_total_bets > 0 else "No bets")
        print(f"Total invested: {ind_total_bet:.0f} SEK")
        print(f"Total payout: {ind_total_payout:.0f} SEK")
        print(f"ACTUAL Profit: {ind_profit:+.0f} SEK ({ind_roi:+.1f}% ROI)")

        # Break-even analysis for V85
        if v85_wins > 0:
            breakeven_dividend = v85_total_cost / v85_wins
            print(f"\n{'='*80}")
            print(f"BREAK-EVEN ANALYSIS")
            print(f"{'='*80}")
            print(f"\nV85 System needs avg dividend of {breakeven_dividend:,.0f} SEK per win to match individual betting profit")
            print(f"That's {breakeven_dividend/1000:.1f}k SEK per V85 win")

        # Recommendation
        print(f"\n{'='*80}")
        print(f"💡 RECOMMENDATION")
        print(f"{'='*80}")

        if ind_profit > 0:
            print(f"""
Individual Race Betting WINNER!
  ✅ Actual profit: {ind_profit:+,.0f} SEK ({ind_roi:+.1f}% ROI)
  ✅ Win rate: {ind_total_wins/ind_total_bets*100:.1f}% on individual bets
  ✅ Consistent returns, less risk

V85 System Betting:
  - Won {v85_wins}/{len(all_results)} V85 races ({v85_wins/len(all_results)*100:.1f}%)
  - Needs avg {breakeven_dividend:,.0f} SEK dividend to beat individual betting
  - High risk, high reward strategy
  - Typical V85 dividends: 10,000 - 10,000,000 SEK

Conclusion:
  → Individual betting provides REAL, CONSISTENT profits
  → V85 system betting is for jackpot hunting
  → Your model is excellent for individual race predictions!
""")
        else:
            print(f"""
Neither strategy profitable on this sample:
  ❌ Individual betting: {ind_profit:+,.0f} SEK ({ind_roi:+.1f}% ROI)
  ❌ V85 system: Needs {breakeven_dividend:,.0f} SEK avg dividend

However:
  - Individual betting gives ACTUAL results
  - V85 system has theoretical profit potential
  - Model shows {v85_wins/len(all_results)*100:.1f}% win rate on V85

Recommendation:
  → Individual betting more predictable
  → V85 for lottery-style jackpot hunting
  → Consider place betting (top-3) for better odds
""")

        print("="*80)


def main():
    """Run comparison"""
    comparator = V85BettingComparison(budget_per_v85=1000)
    results = comparator.run_comparison(num_races=10)


if __name__ == "__main__":
    main()
