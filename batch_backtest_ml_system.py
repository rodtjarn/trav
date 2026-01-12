#!/usr/bin/env python3
"""
Batch backtest ML system on multiple dates from 2024 and 2026

This ensures we test on data completely outside the 2025 training set.
"""

import sys
from datetime import datetime, timedelta
import pandas as pd
import logging

# Import the single backtest function
from backtest_ml_system import backtest_ml_system, verify_no_data_leakage
from atg_api_scraper import ATGAPIScraper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_all_saturdays(year):
    """Get all Saturday dates in a given year"""
    saturdays = []
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)

    current = start_date
    while current <= end_date:
        if current.weekday() == 5:  # Saturday
            saturdays.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)

    return saturdays


def find_vgame_for_date(date_str, scraper):
    """Find main V-game for a given date"""
    priority_order = ['V75', 'GS75', 'V86', 'V85']

    for game_type in priority_order:
        game_info = scraper.get_game_info(date_str, game_type)
        if game_info and game_info.get('race_ids'):
            return game_type

    return None


def batch_backtest(years, max_per_year=10, budget=500):
    """
    Backtest on multiple Saturdays from specified years

    Args:
        years: List of years to test (e.g., [2024, 2026])
        max_per_year: Maximum number of Saturdays to test per year
        budget: Budget per system in SEK
    """

    logger.info("="*80)
    logger.info("BATCH BACKTEST - ML SYSTEM SELECTION MODEL")
    logger.info("="*80)
    logger.info(f"Years to test: {years}")
    logger.info(f"Max dates per year: {max_per_year}")
    logger.info(f"Budget per system: {budget} SEK")
    logger.info("")

    # Verify training data dates
    logger.info("Loading training data to verify no leakage...")
    try:
        df_train = pd.read_csv('system_selection_training_data.csv')
        training_dates = set(df_train['date'].unique())
        logger.info(f"Training data: {len(training_dates)} dates from {min(training_dates)} to {max(training_dates)}")
    except FileNotFoundError:
        logger.error("Training data file not found!")
        return

    # Collect all test dates
    scraper = ATGAPIScraper()
    all_results = []

    for year in years:
        logger.info(f"\n{'='*80}")
        logger.info(f"TESTING YEAR {year}")
        logger.info(f"{'='*80}")

        saturdays = get_all_saturdays(year)
        logger.info(f"Found {len(saturdays)} Saturdays in {year}")

        # Sample dates to test
        import random
        random.seed(42)  # For reproducibility
        test_dates = random.sample(saturdays, min(max_per_year, len(saturdays)))
        test_dates.sort()

        logger.info(f"Testing {len(test_dates)} dates: {test_dates[0]} to {test_dates[-1]}")
        logger.info("")

        for date_str in test_dates:
            # Verify not in training data
            if date_str in training_dates:
                logger.warning(f"⚠️  Skipping {date_str} - in training data!")
                continue

            # Find V-game
            logger.info(f"\n--- {date_str} ---")
            game_type = find_vgame_for_date(date_str, scraper)

            if not game_type:
                logger.info(f"No V-game found for {date_str}")
                continue

            logger.info(f"Found {game_type} - backtesting...")

            # Run backtest
            try:
                result = backtest_ml_system(date_str, game_type, budget)

                if result:
                    all_results.append(result)

                    # Quick summary
                    status = "✅ HIT" if result['system_hit'] else "❌ MISS"
                    logger.info(f"Result: {status} - {result['correct_races']}/{result['total_races']} races | Cost: {result['cost']} SEK")
            except Exception as e:
                logger.error(f"Error backtesting {date_str}: {e}")
                continue

    # Summary statistics
    logger.info("\n" + "="*80)
    logger.info("BATCH BACKTEST SUMMARY")
    logger.info("="*80)

    if not all_results:
        logger.error("No successful backtests!")
        return

    df_results = pd.DataFrame(all_results)

    logger.info(f"\nTotal systems tested: {len(df_results)}")
    logger.info(f"Date range: {df_results['date'].min()} to {df_results['date'].max()}")
    logger.info("")

    # System hit rate
    system_hits = df_results['system_hit'].sum()
    system_hit_rate = system_hits / len(df_results) * 100

    logger.info(f"System Hits: {system_hits}/{len(df_results)} ({system_hit_rate:.1f}%)")
    logger.info("")

    # Race accuracy
    total_correct = df_results['correct_races'].sum()
    total_races = df_results['total_races'].sum()
    race_accuracy = total_correct / total_races * 100

    logger.info(f"Individual Race Accuracy: {total_correct}/{total_races} ({race_accuracy:.1f}%)")
    logger.info(f"Average correct per system: {df_results['correct_races'].mean():.2f}/{df_results['total_races'].mean():.2f}")
    logger.info("")

    # Cost analysis
    total_cost = df_results['cost'].sum()
    avg_cost = df_results['cost'].mean()

    logger.info(f"Total cost: {total_cost:,} SEK")
    logger.info(f"Average cost per system: {avg_cost:.0f} SEK")
    logger.info("")

    # Profit/Loss (rough estimate)
    # Assume 50,000 SEK average payout for a hit
    avg_payout_estimate = 50000
    total_payout = system_hits * avg_payout_estimate
    total_profit = total_payout - total_cost
    roi = (total_profit / total_cost * 100) if total_cost > 0 else 0

    logger.info(f"Estimated Payout (hits only): {total_payout:,} SEK")
    logger.info(f"Estimated Profit/Loss: {total_profit:,} SEK")
    logger.info(f"Estimated ROI: {roi:+.1f}%")
    logger.info("")

    # By game type
    logger.info("Results by Game Type:")
    logger.info("-"*80)
    for game_type in sorted(df_results['game_type'].unique()):
        game_data = df_results[df_results['game_type'] == game_type]
        hits = game_data['system_hit'].sum()
        total = len(game_data)
        hit_rate = hits / total * 100 if total > 0 else 0

        logger.info(f"{game_type}: {hits}/{total} hits ({hit_rate:.1f}%) | "
                   f"Avg correct: {game_data['correct_races'].mean():.1f}/{game_data['total_races'].mean():.1f}")

    logger.info("")

    # By year
    logger.info("Results by Year:")
    logger.info("-"*80)
    df_results['year'] = pd.to_datetime(df_results['date']).dt.year
    for year in sorted(df_results['year'].unique()):
        year_data = df_results[df_results['year'] == year]
        hits = year_data['system_hit'].sum()
        total = len(year_data)
        hit_rate = hits / total * 100 if total > 0 else 0

        logger.info(f"{year}: {hits}/{total} hits ({hit_rate:.1f}%) | "
                   f"Avg correct: {year_data['correct_races'].mean():.1f}/{year_data['total_races'].mean():.1f}")

    logger.info("")

    # Distribution of correct races
    logger.info("Distribution of Correct Races:")
    logger.info("-"*80)
    for correct in sorted(df_results['correct_races'].unique()):
        count = (df_results['correct_races'] == correct).sum()
        pct = count / len(df_results) * 100
        logger.info(f"  {correct} correct: {count} systems ({pct:.1f}%)")

    # Save results
    output_file = 'batch_backtest_results.csv'
    df_results.to_csv(output_file, index=False)
    logger.info(f"\nResults saved to: {output_file}")

    return df_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Batch backtest ML system on multiple dates')
    parser.add_argument('--years', nargs='+', type=int, required=True,
                       help='Years to test (e.g., --years 2024 2026)')
    parser.add_argument('--max-per-year', type=int, default=10,
                       help='Max dates to test per year (default: 10)')
    parser.add_argument('--budget', type=int, default=500,
                       help='Budget per system in SEK (default: 500)')

    args = parser.parse_args()

    results = batch_backtest(args.years, args.max_per_year, args.budget)

    if results is not None:
        logger.info("\n✅ Batch backtest complete!")
    else:
        logger.error("\n❌ Batch backtest failed!")
