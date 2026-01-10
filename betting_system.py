#!/usr/bin/env python3
"""
V85 Betting System with 200 SEK budget
Implements various betting strategies and validates against actual results
"""

import pandas as pd
from predict_v85 import V85Predictor
from atg_api_scraper import ATGAPIScraper
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class V85BettingSystem:
    """Betting system for V85 races"""

    def __init__(self, budget=200):
        self.budget = budget
        self.predictor = V85Predictor()
        self.scraper = ATGAPIScraper(delay=0.5)

    def get_predictions_and_results(self, date):
        """Get predictions and actual results for a date"""

        # Get predictions
        logger.info(f"Generating predictions for {date}")
        predictions_df = self.predictor.predict_v85(date)

        if not predictions_df:
            logger.error("Failed to generate predictions")
            return None, None

        # Get V85 info
        v85_info = self.scraper.get_v85_info(date)

        if not v85_info:
            logger.error("No V85 game found")
            return None, None

        # Get actual results
        logger.info("Fetching actual results")
        race_data = self.scraper.scrape_date(date)

        if not race_data:
            logger.error("No race data found")
            return None, None

        df_results = pd.DataFrame(race_data)
        df_results = df_results[df_results['race_id'].isin(v85_info['race_ids'])]
        df_results['v85_race_number'] = df_results['race_id'].map(v85_info['v85_race_mapping'])

        return predictions_df, df_results

    def strategy_top_pick_single(self, predictions_df, df_results):
        """
        Strategy 1: Single bet on highest probability horse across all races
        Place entire 200 SEK on the horse with highest win probability
        """

        logger.info("\n=== STRATEGY 1: Single Bet on Top Pick ===")

        # Find horse with highest probability across all races
        best_pick = None
        best_prob = 0
        best_race = 0

        for race_num in sorted(predictions_df.keys()):
            race_pred = predictions_df[race_num]
            top_horse = race_pred.iloc[0]

            if top_horse['win_probability'] > best_prob:
                best_prob = top_horse['win_probability']
                best_pick = top_horse
                best_race = race_num

        print(f"\nBest pick: V85 Race {best_race}")
        print(f"  Horse: {best_pick['horse_name']} (#{int(best_pick['start_number'])})")
        print(f"  Win probability: {best_pick['win_probability']*100:.1f}%")
        print(f"  Odds: {best_pick['final_odds']:.1f}")
        print(f"  Bet: {self.budget} SEK")

        # Check result
        race_results = df_results[df_results['v85_race_number'] == best_race]
        actual_winner = race_results[race_results['finish_place'] == 1].iloc[0]

        if actual_winner['start_number'] == best_pick['start_number']:
            payout = self.budget * best_pick['final_odds']
            profit = payout - self.budget
            print(f"\n✅ WON! Payout: {payout:.2f} SEK, Profit: {profit:.2f} SEK")
            return profit
        else:
            print(f"\n❌ LOST! Winner was #{int(actual_winner['start_number'])} {actual_winner['horse_name']}")
            print(f"  Loss: -{self.budget} SEK")
            return -self.budget

    def strategy_value_bets(self, predictions_df, df_results):
        """
        Strategy 2: Value betting - bet on horses where model probability > implied odds probability
        Distribute budget across value bets proportionally
        """

        logger.info("\n=== STRATEGY 2: Value Betting ===")

        value_bets = []

        for race_num in sorted(predictions_df.keys()):
            race_pred = predictions_df[race_num]

            for idx, row in race_pred.iterrows():
                model_prob = row['win_probability']
                odds = row['final_odds']

                if pd.isna(odds) or odds == 0:
                    continue

                implied_prob = 1 / odds

                # Value = model probability - implied probability
                value = model_prob - implied_prob

                if value > 0.15:  # Significant edge (15%+)
                    value_bets.append({
                        'race': race_num,
                        'horse': row['horse_name'],
                        'number': int(row['start_number']),
                        'model_prob': model_prob,
                        'implied_prob': implied_prob,
                        'value': value,
                        'odds': odds
                    })

        if not value_bets:
            print("\nNo value bets found (no horses with 15%+ edge)")
            return 0

        print(f"\nFound {len(value_bets)} value bets:")

        # Distribute budget proportionally to value
        total_value = sum(bet['value'] for bet in value_bets)

        total_payout = 0

        for bet in value_bets:
            bet_amount = (bet['value'] / total_value) * self.budget

            print(f"\nV85 Race {bet['race']}: {bet['horse']} (#{bet['number']})")
            print(f"  Model prob: {bet['model_prob']*100:.1f}%, Implied prob: {bet['implied_prob']*100:.1f}%")
            print(f"  Edge: {bet['value']*100:.1f}%, Odds: {bet['odds']:.1f}")
            print(f"  Bet: {bet_amount:.2f} SEK")

            # Check result
            race_results = df_results[df_results['v85_race_number'] == bet['race']]
            actual_winner = race_results[race_results['finish_place'] == 1].iloc[0]

            if actual_winner['start_number'] == bet['number']:
                payout = bet_amount * bet['odds']
                print(f"  ✅ WON! Payout: {payout:.2f} SEK")
                total_payout += payout
            else:
                print(f"  ❌ Lost")

        profit = total_payout - self.budget
        print(f"\nTotal payout: {total_payout:.2f} SEK")
        print(f"Profit: {profit:.2f} SEK")

        return profit

    def strategy_v85_system(self, predictions_df, df_results):
        """
        Strategy 3: V85 system bet with 2-3 horses per race
        Budget covers multiple combinations
        """

        logger.info("\n=== STRATEGY 3: V85 System Bet ===")

        # For each race, select top 2-3 horses based on probability
        system = {}

        for race_num in sorted(predictions_df.keys()):
            race_pred = predictions_df[race_num]

            # Take top 2 if top pick has >45% probability, otherwise top 3
            top_prob = race_pred.iloc[0]['win_probability']

            if top_prob > 0.45:
                num_picks = 2
            else:
                num_picks = 3

            picks = []
            for idx, row in race_pred.head(num_picks).iterrows():
                picks.append({
                    'number': int(row['start_number']),
                    'horse': row['horse_name'],
                    'prob': row['win_probability']
                })

            system[race_num] = picks

        # Calculate number of combinations
        total_combinations = 1
        for race_num, picks in system.items():
            total_combinations *= len(picks)

        bet_per_combination = self.budget / total_combinations

        print(f"\nSystem configuration:")
        for race_num, picks in system.items():
            print(f"  V85 Race {race_num}: {len(picks)} horses")
            for pick in picks:
                print(f"    #{pick['number']} {pick['horse']} ({pick['prob']*100:.1f}%)")

        print(f"\nTotal combinations: {total_combinations}")
        print(f"Bet per combination: {bet_per_combination:.2f} SEK")

        # Check if we got all winners correct
        winners = []
        for race_num in sorted(system.keys()):
            race_results = df_results[df_results['v85_race_number'] == race_num]
            actual_winner = race_results[race_results['finish_place'] == 1].iloc[0]
            winners.append(int(actual_winner['start_number']))

        # Check if our system covers the winning combination
        covers_winner = True
        for race_num, actual_winner_num in zip(sorted(system.keys()), winners):
            race_picks = [p['number'] for p in system[race_num]]
            if actual_winner_num not in race_picks:
                covers_winner = False
                break

        print(f"\nActual V85 winners: {winners}")

        if covers_winner:
            # V85 payout is complex, but let's estimate based on difficulty
            # Simplified: assume 1000x return for 8 correct (very rare)
            # In reality, check actual V85 payout from ATG
            estimated_payout = self.budget * 50  # Conservative estimate
            profit = estimated_payout - self.budget
            print(f"\n✅ SYSTEM COVERED ALL WINNERS!")
            print(f"  Estimated payout: {estimated_payout:.2f} SEK")
            print(f"  Estimated profit: {profit:.2f} SEK")
            print(f"  (Note: Actual V85 payout varies based on dividend)")
            return profit
        else:
            print(f"\n❌ SYSTEM DID NOT COVER ALL WINNERS")
            print(f"  Loss: -{self.budget} SEK")
            return -self.budget

    def strategy_top_pick_per_race(self, predictions_df, df_results):
        """
        Strategy 4: Spread budget across top pick in each race
        25 SEK on each of 8 races
        """

        logger.info("\n=== STRATEGY 4: Top Pick Per Race (Win Bets) ===")

        bet_per_race = self.budget / len(predictions_df)
        total_payout = 0
        wins = 0

        print(f"\nBetting {bet_per_race:.2f} SEK on top pick in each race:")

        for race_num in sorted(predictions_df.keys()):
            race_pred = predictions_df[race_num]
            top_pick = race_pred.iloc[0]

            print(f"\nV85 Race {race_num}: {top_pick['horse_name']} (#{int(top_pick['start_number'])})")
            print(f"  Probability: {top_pick['win_probability']*100:.1f}%, Odds: {top_pick['final_odds']:.1f}")

            # Check result
            race_results = df_results[df_results['v85_race_number'] == race_num]
            actual_winner = race_results[race_results['finish_place'] == 1].iloc[0]

            if actual_winner['start_number'] == top_pick['start_number']:
                payout = bet_per_race * top_pick['final_odds']
                total_payout += payout
                wins += 1
                print(f"  ✅ WON! Payout: {payout:.2f} SEK")
            else:
                print(f"  ❌ Lost (winner: #{int(actual_winner['start_number'])} {actual_winner['horse_name']})")

        profit = total_payout - self.budget
        print(f"\n{'='*60}")
        print(f"Wins: {wins}/{len(predictions_df)}")
        print(f"Total payout: {total_payout:.2f} SEK")
        print(f"Profit: {profit:.2f} SEK")

        return profit

    def run_all_strategies(self, date='2026-01-10'):
        """Run all betting strategies and compare results"""

        predictions_df, df_results = self.get_predictions_and_results(date)

        if predictions_df is None or df_results is None:
            return

        print("\n" + "="*80)
        print(f"V85 BETTING SYSTEM - {date}")
        print(f"Budget: {self.budget} SEK")
        print("="*80)

        results = {}

        # Strategy 1: Single bet on best pick
        results['Single Best Pick'] = self.strategy_top_pick_single(predictions_df, df_results)

        # Strategy 2: Value betting
        results['Value Betting'] = self.strategy_value_bets(predictions_df, df_results)

        # Strategy 3: V85 system
        results['V85 System'] = self.strategy_v85_system(predictions_df, df_results)

        # Strategy 4: Top pick per race
        results['Top Pick Per Race'] = self.strategy_top_pick_per_race(predictions_df, df_results)

        # Summary
        print("\n" + "="*80)
        print("STRATEGY COMPARISON")
        print("="*80)
        print(f"{'Strategy':<25} {'Profit/Loss':<15} {'ROI':<10}")
        print("-"*80)

        for strategy, profit in sorted(results.items(), key=lambda x: x[1], reverse=True):
            roi = (profit / self.budget) * 100
            sign = "+" if profit >= 0 else ""
            print(f"{strategy:<25} {sign}{profit:>8.2f} SEK    {sign}{roi:>6.1f}%")

        print("="*80)

        best_strategy = max(results, key=results.get)
        print(f"\n🏆 Best strategy: {best_strategy}")
        print(f"   Profit: {results[best_strategy]:.2f} SEK ({results[best_strategy]/self.budget*100:+.1f}%)")

        print("\n" + "="*80)
        print("⚠️  DISCLAIMER")
        print("="*80)
        print("This is a retrospective analysis for educational purposes only.")
        print("Past performance does not guarantee future results.")
        print("Gambling involves risk. Only bet what you can afford to lose.")
        print("="*80)


def main():
    """Run betting system analysis"""

    system = V85BettingSystem(budget=200)
    system.run_all_strategies(date='2026-01-10')


if __name__ == "__main__":
    main()
