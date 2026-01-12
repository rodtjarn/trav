#!/usr/bin/env python3
"""
Batch backtest individual betting strategy on multiple dates

Tests the base prediction model's individual high-EV betting performance
across multiple dates from specified years.
"""

import sys
import pickle
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging
from atg_api_scraper import ATGAPIScraper
from temporal_data_processor import TemporalDataProcessor

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
            return game_type, game_info

    return None, None


def get_race_results(race_id, scraper):
    """Get actual results for a race"""
    response = scraper.session.get(f'https://www.atg.se/services/racinginfo/v1/api/races/{race_id}')

    if response.status_code != 200:
        return None

    race = response.json()
    results = {}

    for start in race.get('starts', []):
        horse_num = start.get('number')
        result_data = start.get('result', {})

        if horse_num:
            results[horse_num] = {
                'place': result_data.get('place', 999),
                'odds': result_data.get('finalOdds', 0),
                'name': start.get('horse', {}).get('name', 'Unknown')
            }

    return results


def calculate_ev(prob, odds, kelly_fraction=0.25):
    """
    Calculate expected value and bet size using Kelly criterion

    Args:
        prob: Win probability (0-1)
        odds: Decimal odds
        kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly)

    Returns:
        (ev, bet_fraction)
    """
    if odds <= 1.0 or prob <= 0:
        return 0, 0

    # EV = (prob * (odds - 1)) - (1 - prob)
    ev = (prob * (odds - 1)) - (1 - prob)

    # Kelly fraction = (prob * odds - 1) / (odds - 1)
    kelly = (prob * odds - 1) / (odds - 1)

    # Use fractional Kelly for safety
    bet_fraction = max(0, kelly * kelly_fraction)

    return ev, bet_fraction


def backtest_individual_betting(date_str, model, feature_cols, scraper, processor, budget=500, top_n=10):
    """
    Backtest individual betting on a specific date

    Args:
        date_str: Date in YYYY-MM-DD format
        model: Trained prediction model
        feature_cols: Feature columns for model
        scraper: ATGAPIScraper instance
        processor: TemporalDataProcessor instance
        budget: Total budget in SEK
        top_n: Number of top bets to place

    Returns:
        dict with results
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {date_str}")
    logger.info(f"{'='*60}")

    # Find V-game
    game_type, game_info = find_vgame_for_date(date_str, scraper)

    if not game_type or not game_info:
        logger.warning(f"No V-game found for {date_str}")
        return None

    race_ids = game_info.get('race_ids', [])
    logger.info(f"Found {game_type} with {len(race_ids)} races")

    # Fetch race data
    all_race_data = []
    race_results_dict = {}

    for race_id in race_ids:
        race_details = scraper.get_race_details(race_id)
        if race_details:
            all_race_data.append(race_details)

        results = get_race_results(race_id, scraper)
        if results:
            race_results_dict[race_id] = results

    if not all_race_data:
        logger.warning(f"Failed to fetch race data for {date_str}")
        return None

    # Process data
    df = processor.process_race_data(all_race_data, feature_cols)
    if df.empty:
        logger.warning(f"Failed to process race data for {date_str}")
        return None

    # Make predictions
    X = df[feature_cols]
    predictions = model.predict_proba(X)[:, 1]
    df['predicted_prob'] = predictions

    # Get odds (simulate with implied probability for now - in real scenario use actual odds)
    df['estimated_odds'] = 1 / df['predicted_prob']  # Simple estimate

    # Calculate EV for each horse
    df['ev'] = df.apply(lambda row: calculate_ev(row['predicted_prob'], row['estimated_odds'])[0], axis=1)
    df['bet_fraction'] = df.apply(lambda row: calculate_ev(row['predicted_prob'], row['estimated_odds'])[1], axis=1)

    # Filter positive EV bets and sort
    positive_ev = df[df['ev'] > 0].copy()
    positive_ev = positive_ev.sort_values('ev', ascending=False)

    # Select top N bets
    top_bets = positive_ev.head(top_n)

    if len(top_bets) == 0:
        logger.info("No positive EV bets found")
        return {
            'date': date_str,
            'game_type': game_type,
            'num_races': len(race_ids),
            'num_bets': 0,
            'total_bet': 0,
            'winners': 0,
            'total_payout': 0,
            'profit': 0,
            'roi': 0
        }

    # Allocate budget proportionally based on EV
    total_ev = top_bets['ev'].sum()
    top_bets['bet_amount'] = (top_bets['ev'] / total_ev * budget).round(0)

    # Ensure at least minimum bet
    min_bet = 10
    top_bets['bet_amount'] = top_bets['bet_amount'].clip(lower=min_bet)

    # Scale down if over budget
    total_bet = top_bets['bet_amount'].sum()
    if total_bet > budget:
        top_bets['bet_amount'] = (top_bets['bet_amount'] / total_bet * budget).round(0)

    total_bet = top_bets['bet_amount'].sum()

    logger.info(f"\nTop {len(top_bets)} bets (Total: {total_bet:.0f} SEK):")
    logger.info("-"*60)

    # Check results
    winners = 0
    total_payout = 0

    bet_results = []

    for idx, bet in top_bets.iterrows():
        race_id = bet['race_id']
        horse_num = int(bet['start_number'])
        bet_amount = bet['bet_amount']
        prob = bet['predicted_prob']

        # Get actual result
        results = race_results_dict.get(race_id, {})

        winner = None
        actual_odds = None
        for h_num, h_data in results.items():
            if h_data['place'] == 1:
                winner = h_num
                actual_odds = h_data['odds']
                break

        won = (winner == horse_num)
        payout = 0

        if won:
            winners += 1
            # Use actual odds if available, otherwise estimate
            if actual_odds and actual_odds > 0:
                payout = bet_amount * actual_odds
            else:
                payout = bet_amount * bet['estimated_odds']
            total_payout += payout

        status = "✅ WIN" if won else "❌ LOSS"

        horse_name = bet.get('horse_name', 'Unknown')
        logger.info(f"Race {bet.get('race_number', '?')}: #{horse_num} {horse_name:20s} | "
                   f"{bet_amount:3.0f} SEK | Prob: {prob:.3f} | {status}")

        if won:
            logger.info(f"         → Payout: {payout:.0f} SEK (+{payout-bet_amount:.0f} SEK)")

        bet_results.append({
            'race_id': race_id,
            'horse_num': horse_num,
            'horse_name': horse_name,
            'bet_amount': bet_amount,
            'prob': prob,
            'won': won,
            'payout': payout
        })

    profit = total_payout - total_bet
    roi = (profit / total_bet * 100) if total_bet > 0 else 0

    logger.info(f"\n{'='*60}")
    logger.info(f"RESULTS for {date_str}:")
    logger.info(f"  Total bet: {total_bet:.0f} SEK")
    logger.info(f"  Winners: {winners}/{len(top_bets)} ({winners/len(top_bets)*100:.1f}%)")
    logger.info(f"  Total payout: {total_payout:.0f} SEK")
    logger.info(f"  Profit: {profit:+.0f} SEK ({roi:+.1f}% ROI)")

    if profit > 0:
        logger.info(f"  ✅ PROFITABLE!")
    else:
        logger.info(f"  ❌ LOSS")

    return {
        'date': date_str,
        'game_type': game_type,
        'num_races': len(race_ids),
        'num_bets': len(top_bets),
        'total_bet': total_bet,
        'winners': winners,
        'win_rate': winners / len(top_bets) if len(top_bets) > 0 else 0,
        'total_payout': total_payout,
        'profit': profit,
        'roi': roi,
        'bet_results': bet_results
    }


def batch_backtest_individual(years, max_per_year=10, budget=500, top_n=10):
    """
    Batch backtest individual betting across multiple dates
    """
    logger.info("="*80)
    logger.info("BATCH BACKTEST - INDIVIDUAL BETTING")
    logger.info("="*80)
    logger.info(f"Years to test: {years}")
    logger.info(f"Max dates per year: {max_per_year}")
    logger.info(f"Budget per date: {budget} SEK")
    logger.info(f"Top bets per date: {top_n}")
    logger.info("")

    # Load model
    logger.info("Loading prediction model...")
    with open('temporal_rf_model.pkl', 'rb') as f:
        model = pickle.load(f)

    feature_cols = list(model.feature_names_in_)

    scraper = ATGAPIScraper()
    processor = TemporalDataProcessor()

    all_results = []

    for year in years:
        logger.info(f"\n{'='*80}")
        logger.info(f"TESTING YEAR {year}")
        logger.info(f"{'='*80}")

        saturdays = get_all_saturdays(year)
        logger.info(f"Found {len(saturdays)} Saturdays in {year}")

        # Sample dates
        import random
        random.seed(42)
        test_dates = random.sample(saturdays, min(max_per_year, len(saturdays)))
        test_dates.sort()

        logger.info(f"Testing {len(test_dates)} dates")

        for date_str in test_dates:
            try:
                result = backtest_individual_betting(
                    date_str, model, feature_cols, scraper, processor, budget, top_n
                )

                if result:
                    all_results.append(result)
            except Exception as e:
                logger.error(f"Error backtesting {date_str}: {e}")
                continue

    # Summary
    logger.info("\n" + "="*80)
    logger.info("BATCH BACKTEST SUMMARY")
    logger.info("="*80)

    if not all_results:
        logger.error("No successful backtests!")
        return None

    df_results = pd.DataFrame(all_results)

    logger.info(f"\nTotal dates tested: {len(df_results)}")
    logger.info(f"Date range: {df_results['date'].min()} to {df_results['date'].max()}")
    logger.info("")

    # Overall statistics
    total_bet = df_results['total_bet'].sum()
    total_payout = df_results['total_payout'].sum()
    total_profit = total_payout - total_bet
    overall_roi = (total_profit / total_bet * 100) if total_bet > 0 else 0

    total_bets = df_results['num_bets'].sum()
    total_winners = df_results['winners'].sum()
    overall_win_rate = (total_winners / total_bets * 100) if total_bets > 0 else 0

    logger.info(f"Overall Performance:")
    logger.info(f"  Total bet: {total_bet:,.0f} SEK")
    logger.info(f"  Total payout: {total_payout:,.0f} SEK")
    logger.info(f"  Total profit: {total_profit:+,.0f} SEK")
    logger.info(f"  Overall ROI: {overall_roi:+.1f}%")
    logger.info(f"  Total bets placed: {total_bets}")
    logger.info(f"  Winners: {total_winners}/{total_bets} ({overall_win_rate:.1f}%)")
    logger.info("")

    # Profitable days
    profitable_days = (df_results['profit'] > 0).sum()
    logger.info(f"Profitable days: {profitable_days}/{len(df_results)} ({profitable_days/len(df_results)*100:.1f}%)")
    logger.info(f"Average profit per day: {df_results['profit'].mean():+.0f} SEK")
    logger.info(f"Best day: {df_results['profit'].max():+.0f} SEK")
    logger.info(f"Worst day: {df_results['profit'].min():+.0f} SEK")
    logger.info("")

    # By year
    logger.info("Results by Year:")
    logger.info("-"*80)
    df_results['year'] = pd.to_datetime(df_results['date']).dt.year
    for year in sorted(df_results['year'].unique()):
        year_data = df_results[df_results['year'] == year]
        year_bet = year_data['total_bet'].sum()
        year_payout = year_data['total_payout'].sum()
        year_profit = year_payout - year_bet
        year_roi = (year_profit / year_bet * 100) if year_bet > 0 else 0
        year_winners = year_data['winners'].sum()
        year_bets = year_data['num_bets'].sum()
        year_win_rate = (year_winners / year_bets * 100) if year_bets > 0 else 0

        logger.info(f"{year}: ROI: {year_roi:+.1f}% | "
                   f"Profit: {year_profit:+,.0f} SEK | "
                   f"Win rate: {year_win_rate:.1f}% ({year_winners}/{year_bets})")

    logger.info("")

    # Save results
    output_file = 'batch_individual_backtest_results.csv'
    df_results_save = df_results.drop('bet_results', axis=1)  # Drop nested list
    df_results_save.to_csv(output_file, index=False)
    logger.info(f"Results saved to: {output_file}")

    return df_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Batch backtest individual betting strategy')
    parser.add_argument('--years', nargs='+', type=int, required=True,
                       help='Years to test (e.g., --years 2024 2026)')
    parser.add_argument('--max-per-year', type=int, default=10,
                       help='Max dates to test per year (default: 10)')
    parser.add_argument('--budget', type=int, default=500,
                       help='Budget per date in SEK (default: 500)')
    parser.add_argument('--top-n', type=int, default=10,
                       help='Number of top bets per date (default: 10)')

    args = parser.parse_args()

    results = batch_backtest_individual(args.years, args.max_per_year, args.budget, args.top_n)

    if results is not None:
        logger.info("\n✅ Batch backtest complete!")
    else:
        logger.error("\n❌ Batch backtest failed!")
