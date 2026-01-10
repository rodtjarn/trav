#!/usr/bin/env python3
"""
V85 System Betting Analysis - Proper pari-mutuel pool betting
"""

import pandas as pd
from predict_v85 import V85Predictor
from atg_api_scraper import ATGAPIScraper
import logging
import itertools

logging.basicConfig(level=logging.WARNING)


class V85SystemAnalyzer:
    """Analyze V85 system betting strategies"""

    def __init__(self):
        self.predictor = V85Predictor()
        self.scraper = ATGAPIScraper(delay=0.5)

    def get_v85_data(self, date):
        """Get predictions and actual results"""

        # Get predictions
        predictions_df = self.predictor.predict_v85(date)

        # Get V85 info
        v85_info = self.scraper.get_v85_info(date)

        # Get actual results
        race_data = self.scraper.scrape_date(date)
        df_results = pd.DataFrame(race_data)
        df_results = df_results[df_results['race_id'].isin(v85_info['race_ids'])]
        df_results['v85_race_number'] = df_results['race_id'].map(v85_info['v85_race_mapping'])

        return predictions_df, df_results, v85_info

    def get_actual_winners(self, df_results):
        """Get the actual winning combination"""
        winners = []
        for race_num in sorted(df_results['v85_race_number'].unique()):
            race_df = df_results[df_results['v85_race_number'] == race_num]
            winner = race_df[race_df['finish_place'] == 1].iloc[0]
            winners.append(int(winner['start_number']))
        return winners

    def create_system_from_predictions(self, predictions_df, picks_per_race):
        """
        Create a V85 system from model predictions

        Args:
            predictions_df: Model predictions
            picks_per_race: List of how many horses to pick in each race (e.g., [2,2,1,2,1,1,1,1])

        Returns:
            Dictionary mapping race number to list of horse numbers
        """
        system = {}

        for i, race_num in enumerate(sorted(predictions_df.keys())):
            num_picks = picks_per_race[i]
            race_pred = predictions_df[race_num]

            # Take top N horses by probability
            top_picks = race_pred.head(num_picks)
            system[race_num] = [int(row['start_number']) for _, row in top_picks.iterrows()]

        return system

    def calculate_combinations(self, system):
        """Calculate total number of combinations in system"""
        total = 1
        for race_num, picks in system.items():
            total *= len(picks)
        return total

    def check_if_system_covers_winner(self, system, winners):
        """Check if system covers the winning combination"""
        for i, race_num in enumerate(sorted(system.keys())):
            if winners[i] not in system[race_num]:
                return False
        return True

    def analyze_systems(self, date, budget):
        """Analyze different V85 system structures with given budget"""

        print("\n" + "="*80)
        print(f"V85 SYSTEM BETTING ANALYSIS - {budget} SEK BUDGET")
        print(f"Date: {date}")
        print("="*80)

        # Get data
        predictions_df, df_results, v85_info = self.get_v85_data(date)
        actual_winners = self.get_actual_winners(df_results)

        print(f"\nV85 Game: {v85_info['game_id']}")
        print(f"Jackpot: {v85_info.get('jackpot_amount', 'Unknown')} SEK")

        print(f"\n🎯 ACTUAL WINNING COMBINATION")
        print("-"*80)
        for i, winner in enumerate(actual_winners, 1):
            race_results = df_results[df_results['v85_race_number'] == i]
            winner_horse = race_results[race_results['start_number'] == winner].iloc[0]
            print(f"  Race {i}: #{winner} {winner_horse['horse_name']} @ {winner_horse['final_odds']:.1f} odds")

        print(f"\nWinning combination: {'-'.join(map(str, actual_winners))}")

        # Try different system structures
        cost_per_row = 1  # SEK

        print("\n" + "="*80)
        print(f"SYSTEM BETTING STRATEGIES ({budget} SEK budget, {cost_per_row} SEK/row)")
        print("="*80)

        # Strategy 1: Banker system - 1 horse in strong races, 2-3 in weak races
        print("\n📋 STRATEGY 1: BANKER SYSTEM (Bank strong races, cover weak)")
        print("-"*80)

        # Identify which races had high confidence (>45% top pick)
        bankers = []
        coverage = []

        for race_num in sorted(predictions_df.keys()):
            race_pred = predictions_df[race_num]
            top_prob = race_pred.iloc[0]['win_probability']

            if top_prob > 0.45:
                bankers.append(1)  # Bank this race
            else:
                # Calculate how many to cover based on budget
                bankers.append(2)  # Cover with 2

        # Adjust coverage to fit budget
        while self._calc_combos(bankers) > budget:
            # Reduce coverage in race with lowest top probability
            max_idx = max((i for i in range(len(bankers)) if bankers[i] > 1),
                         key=lambda i: -predictions_df[i+1].iloc[0]['win_probability'])
            if bankers[max_idx] > 1:
                bankers[max_idx] -= 1

        banker_system = self.create_system_from_predictions(predictions_df, bankers)
        banker_combos = self.calculate_combinations(banker_system)
        banker_covers = self.check_if_system_covers_winner(banker_system, actual_winners)

        print(f"System structure: {'-'.join(map(str, bankers))}")
        print(f"Total combinations: {banker_combos}")
        print(f"Cost: {banker_combos * cost_per_row} SEK")
        print(f"\nYour system:")
        for race_num, picks in banker_system.items():
            race_pred = predictions_df[race_num]
            print(f"  Race {race_num}: {picks} ", end="")
            if len(picks) == 1:
                print("(BANKER)")
            else:
                print(f"({len(picks)} horses)")

        print(f"\n{'✅ WINNER!' if banker_covers else '❌ MISS'} - ", end="")
        if banker_covers:
            print("System covers winning combination!")
        else:
            # Show which races missed
            misses = []
            for i, race_num in enumerate(sorted(banker_system.keys())):
                if actual_winners[i] not in banker_system[race_num]:
                    misses.append(race_num)
            print(f"Missed races: {misses}")

        # Strategy 2: Balanced system - 2 in most races
        print("\n📋 STRATEGY 2: BALANCED SYSTEM (2 horses in most races)")
        print("-"*80)

        balanced = [2] * 8
        # Adjust to fit budget
        while self._calc_combos(balanced) > budget:
            # Bank the strongest race
            max_prob_race = max(range(8), key=lambda i: predictions_df[i+1].iloc[0]['win_probability'])
            if balanced[max_prob_race] > 1:
                balanced[max_prob_race] = 1

        balanced_system = self.create_system_from_predictions(predictions_df, balanced)
        balanced_combos = self.calculate_combinations(balanced_system)
        balanced_covers = self.check_if_system_covers_winner(balanced_system, actual_winners)

        print(f"System structure: {'-'.join(map(str, balanced))}")
        print(f"Total combinations: {balanced_combos}")
        print(f"Cost: {balanced_combos * cost_per_row} SEK")
        print(f"{'✅ WINNER!' if balanced_covers else '❌ MISS'}")

        # Strategy 3: Top heavy - 3 horses in uncertain races, 1 in strong
        print("\n📋 STRATEGY 3: TOP-HEAVY (3 in uncertain, 1 in confident)")
        print("-"*80)

        topheavy = []
        for race_num in sorted(predictions_df.keys()):
            race_pred = predictions_df[race_num]
            top_prob = race_pred.iloc[0]['win_probability']

            if top_prob > 0.48:
                topheavy.append(1)
            elif top_prob < 0.35:
                topheavy.append(3)
            else:
                topheavy.append(2)

        # Adjust to fit budget
        while self._calc_combos(topheavy) > budget:
            max_idx = max((i for i in range(len(topheavy)) if topheavy[i] > 1),
                         key=lambda i: topheavy[i], default=None)
            if max_idx is not None:
                topheavy[max_idx] -= 1

        topheavy_system = self.create_system_from_predictions(predictions_df, topheavy)
        topheavy_combos = self.calculate_combinations(topheavy_system)
        topheavy_covers = self.check_if_system_covers_winner(topheavy_system, actual_winners)

        print(f"System structure: {'-'.join(map(str, topheavy))}")
        print(f"Total combinations: {topheavy_combos}")
        print(f"Cost: {topheavy_combos * cost_per_row} SEK")
        print(f"{'✅ WINNER!' if topheavy_covers else '❌ MISS'}")

        # Strategy 4: Model-driven - allocate based on uncertainty
        print("\n📋 STRATEGY 4: MODEL-DRIVEN (Allocate by race difficulty)")
        print("-"*80)

        # Calculate uncertainty for each race (variance in top 3 probabilities)
        model_driven = []
        for race_num in sorted(predictions_df.keys()):
            race_pred = predictions_df[race_num]
            top3 = race_pred.head(3)
            top_prob = race_pred.iloc[0]['win_probability']
            prob_spread = top3.iloc[0]['win_probability'] - top3.iloc[2]['win_probability']

            # Clear favorite
            if top_prob > 0.50 and prob_spread > 0.15:
                model_driven.append(1)
            # Very uncertain
            elif prob_spread < 0.10:
                model_driven.append(3)
            else:
                model_driven.append(2)

        # Adjust to budget
        while self._calc_combos(model_driven) > budget:
            max_idx = max((i for i in range(len(model_driven)) if model_driven[i] > 1),
                         key=lambda i: model_driven[i], default=None)
            if max_idx is not None:
                model_driven[max_idx] -= 1

        modeldriven_system = self.create_system_from_predictions(predictions_df, model_driven)
        modeldriven_combos = self.calculate_combinations(modeldriven_system)
        modeldriven_covers = self.check_if_system_covers_winner(modeldriven_system, actual_winners)

        print(f"System structure: {'-'.join(map(str, model_driven))}")
        print(f"Total combinations: {modeldriven_combos}")
        print(f"Cost: {modeldriven_combos * cost_per_row} SEK")

        print(f"\nRace difficulty analysis:")
        for i, race_num in enumerate(sorted(predictions_df.keys())):
            race_pred = predictions_df[race_num]
            top_prob = race_pred.iloc[0]['win_probability']
            top3 = race_pred.head(3)
            spread = top3.iloc[0]['win_probability'] - top3.iloc[2]['win_probability']

            print(f"  Race {race_num}: Top pick {top_prob*100:.1f}%, spread {spread*100:.1f}% → {model_driven[i]} horses")

        print(f"\n{'✅ WINNER!' if modeldriven_covers else '❌ MISS'}")

        # Summary table
        print("\n" + "="*80)
        print("STRATEGY COMPARISON")
        print("="*80)
        print(f"{'Strategy':<25} {'Structure':<20} {'Combos':<10} {'Cost':<10} {'Result':<10}")
        print("-"*80)

        strategies = [
            ("Banker System", '-'.join(map(str, bankers)), banker_combos, banker_covers),
            ("Balanced System", '-'.join(map(str, balanced)), balanced_combos, balanced_covers),
            ("Top-Heavy", '-'.join(map(str, topheavy)), topheavy_combos, topheavy_covers),
            ("Model-Driven", '-'.join(map(str, model_driven)), modeldriven_combos, modeldriven_covers),
        ]

        for name, structure, combos, covers in strategies:
            cost = combos * cost_per_row
            result = "✅ WIN" if covers else "❌ MISS"
            print(f"{name:<25} {structure:<20} {combos:<10} {cost:<10} {result:<10}")

        # Check actual V85 dividend
        print("\n" + "="*80)
        print("PAYOUT INFORMATION")
        print("="*80)

        print("\n⚠️  V85 Dividend Information:")
        print("The actual payout depends on:")
        print("  - Total pool size")
        print("  - Number of winning tickets")
        print("  - ATG's takeout (typically 25%)")
        print("\nTo get actual dividend, check: https://www.atg.se/spel/v85")
        print("Look for 'Utdelning' for the winning combination")

        # Estimate based on jackpot
        print("\n📊 THEORETICAL PAYOUT ESTIMATE:")
        winning_strategies = [s for s in strategies if s[3]]

        if winning_strategies:
            print(f"\nIf V85 dividend was (example scenarios):")
            example_dividends = [50000, 100000, 500000, 1000000, 5000000]

            for dividend in example_dividends:
                print(f"\n  If dividend = {dividend:,} SEK:")
                for name, structure, combos, covers in winning_strategies:
                    if covers:
                        # You win dividend / your winning combos
                        payout = dividend
                        cost = combos * cost_per_row
                        profit = payout - cost
                        roi = (profit / cost) * 100
                        print(f"    {name}: {payout:,} SEK payout, {profit:+,} SEK profit ({roi:+.0f}% ROI)")
        else:
            print("\n❌ None of the systems covered the winning combination")
            print("   No payout for these strategies")

        print("\n" + "="*80)
        print("💡 KEY INSIGHTS")
        print("="*80)
        print("""
V85 System Betting vs Individual Bets:
  ✓ Fixed cost (1-2 SEK per combination)
  ✓ Potential for massive payouts (millions if you win)
  ✓ Need ALL 8 races correct to win main prize
  ✓ Payout divided among all winners (pari-mutuel)

Your 200-300 SEK buys:
  ✓ 200-300 full combinations
  ✓ Can cover multiple horses per race with systems
  ✓ Example: 2×2×2×2×2×2×2×2 = 256 combinations = 256 SEK

Model strengths:
  ✓ Good at identifying top-3 finishers (87.5% accuracy)
  ✓ Less accurate at picking exact winners (12.5%)
  ✓ Best strategy: Cover 2-3 horses in each race

Recommendation:
  → Use model-driven or banker system approach
  → Bank 1 horse in races with >50% top pick confidence
  → Cover 2-3 horses in uncertain races
  → Budget for 200-300 combinations
""")

        print("="*80)

    def _calc_combos(self, picks_list):
        """Helper to calculate combinations"""
        total = 1
        for picks in picks_list:
            total *= picks
        return total


def main():
    """Run V85 system analysis"""

    analyzer = V85SystemAnalyzer()

    # Analyze with different budgets
    for budget in [200, 300]:
        analyzer.analyze_systems(date='2026-01-10', budget=budget)
        print("\n\n")


if __name__ == "__main__":
    main()
