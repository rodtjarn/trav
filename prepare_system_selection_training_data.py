#!/usr/bin/env python3
"""
Prepare training data for system selection meta-model

This script processes Saturday system data to create race-level training data
where the target is: "How many horses should we have picked?"

The label is derived retrospectively:
- If winner was our top prediction: optimal = 1
- If winner was our 2nd prediction: optimal = 2
- If winner was our 3rd prediction: optimal = 3
- etc.

Features include:
- Top prediction probability and confidence spreads
- Race characteristics (num horses, game type, etc.)
- Prediction distribution metrics
"""

import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from atg_api_scraper import ATGAPIScraper
from temporal_data_processor import TemporalDataProcessor
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


def calculate_race_features(race_horses_df):
    """
    Calculate features for a race based on model predictions

    Returns features that help decide how many horses to pick
    """
    sorted_horses = race_horses_df.sort_values('predicted_prob', ascending=False)

    # Top predictions
    top1_prob = sorted_horses.iloc[0]['predicted_prob'] if len(sorted_horses) > 0 else 0
    top2_prob = sorted_horses.iloc[1]['predicted_prob'] if len(sorted_horses) > 1 else 0
    top3_prob = sorted_horses.iloc[2]['predicted_prob'] if len(sorted_horses) > 2 else 0
    top4_prob = sorted_horses.iloc[3]['predicted_prob'] if len(sorted_horses) > 3 else 0
    top5_prob = sorted_horses.iloc[4]['predicted_prob'] if len(sorted_horses) > 4 else 0

    # Confidence spreads (gaps between predictions)
    gap_1_2 = top1_prob - top2_prob
    gap_2_3 = top2_prob - top3_prob
    gap_3_4 = top3_prob - top4_prob
    gap_1_3 = top1_prob - top3_prob

    # Distribution metrics
    mean_prob = sorted_horses['predicted_prob'].mean()
    std_prob = sorted_horses['predicted_prob'].std()
    max_prob = sorted_horses['predicted_prob'].max()
    min_prob = sorted_horses['predicted_prob'].min()

    # Top concentration (sum of top 3 vs rest)
    top3_sum = top1_prob + top2_prob + top3_prob

    # Entropy-like measure (uncertainty)
    probs = sorted_horses['predicted_prob'].values
    probs = probs / probs.sum()  # Normalize
    entropy = -np.sum(probs * np.log(probs + 1e-10))

    # Number of horses
    num_horses = len(sorted_horses)

    # Number of horses above certain thresholds
    horses_above_20 = (sorted_horses['predicted_prob'] >= 0.20).sum()
    horses_above_15 = (sorted_horses['predicted_prob'] >= 0.15).sum()
    horses_above_10 = (sorted_horses['predicted_prob'] >= 0.10).sum()

    features = {
        'top1_prob': top1_prob,
        'top2_prob': top2_prob,
        'top3_prob': top3_prob,
        'top4_prob': top4_prob,
        'top5_prob': top5_prob,
        'gap_1_2': gap_1_2,
        'gap_2_3': gap_2_3,
        'gap_3_4': gap_3_4,
        'gap_1_3': gap_1_3,
        'mean_prob': mean_prob,
        'std_prob': std_prob,
        'max_prob': max_prob,
        'min_prob': min_prob,
        'top3_sum': top3_sum,
        'entropy': entropy,
        'num_horses': num_horses,
        'horses_above_20': horses_above_20,
        'horses_above_15': horses_above_15,
        'horses_above_10': horses_above_10,
    }

    return features


def determine_optimal_picks(race_horses_df, winner_num):
    """
    Determine optimal number of picks based on where winner ranked

    Args:
        race_horses_df: DataFrame with predictions for this race
        winner_num: Actual winner horse number

    Returns:
        optimal_picks: int (1-5+, or -1 if winner not in field)
    """
    sorted_horses = race_horses_df.sort_values('predicted_prob', ascending=False).reset_index(drop=True)

    # Find where winner ranked in our predictions
    winner_rank = None
    for idx, row in sorted_horses.iterrows():
        if int(row['start_number']) == winner_num:
            winner_rank = idx + 1  # 1-indexed
            break

    if winner_rank is None:
        # Winner not found in our predictions (scratched or data issue)
        return -1

    # Optimal picks = minimum needed to include the winner
    # Cap at 5 to keep system costs reasonable
    optimal = min(winner_rank, 5)

    return optimal


def prepare_training_data():
    """
    Process Saturday data to create race-level training data
    """
    logger.info("Loading model...")
    with open('temporal_rf_model.pkl', 'rb') as f:
        model = pickle.load(f)

    feature_cols = list(model.feature_names_in_)

    scraper = ATGAPIScraper()
    processor = TemporalDataProcessor()

    # Get Saturdays from 2025
    saturdays = []
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 12, 31)
    current = start_date
    while current <= end_date:
        if current.weekday() == 5:  # Saturday
            saturdays.append(current.strftime('%Y-%m-%d'))
        current += pd.Timedelta(days=1)

    logger.info(f"Processing {len(saturdays)} Saturdays")

    training_records = []

    for idx, date_str in enumerate(saturdays, 1):
        logger.info(f"Processing {date_str} ({idx}/{len(saturdays)})...")

        # Find main V-game
        priority_order = ['V75', 'GS75', 'V86', 'V85']
        game_type = None
        game_info = None

        for g in priority_order:
            info = scraper.get_game_info(date_str, g)
            if info and info.get('race_ids'):
                game_type = g
                game_info = info
                break

        if not game_type:
            logger.info(f"  No V-game found")
            continue

        race_ids = game_info.get('race_ids', [])
        logger.info(f"  Found {game_type} with {len(race_ids)} races")

        # Fetch race data
        all_race_data = []
        race_results_dict = {}

        for race_seq_num, race_id in enumerate(race_ids, 1):
            race_details = scraper.get_race_details(race_id)
            if race_details:
                all_race_data.append(race_details)

            results = get_race_results(race_id, scraper)
            if results:
                race_results_dict[race_seq_num] = results

        if not all_race_data:
            logger.warning(f"  No race data")
            continue

        # Process and predict
        df = processor.process_race_data(all_race_data, feature_cols)
        if df.empty:
            logger.warning(f"  Failed to process")
            continue

        X = df[feature_cols]
        predictions = model.predict_proba(X)[:, 1]
        df['predicted_prob'] = predictions

        # Map to sequential race numbers
        race_id_to_seq = {}
        for seq_num, race_id in enumerate(race_ids, 1):
            track_race_num = int(race_id.split('_')[-1])
            race_id_to_seq[track_race_num] = seq_num

        df['vgame_race_num'] = df['race_number'].map(race_id_to_seq)

        # Create training record for each race
        for race_num in range(1, len(race_ids) + 1):
            race_horses = df[df['vgame_race_num'] == race_num].copy()

            if len(race_horses) == 0:
                continue

            # Get winner
            results = race_results_dict.get(race_num, {})
            winner = None
            for horse_num, data in results.items():
                if data['place'] == 1:
                    winner = horse_num
                    break

            if winner is None:
                continue

            # Calculate features
            features = calculate_race_features(race_horses)

            # Determine optimal picks
            optimal = determine_optimal_picks(race_horses, winner)

            if optimal < 1:
                continue  # Skip if winner not found

            # Create record
            record = {
                'date': date_str,
                'game_type': game_type,
                'race_num': race_num,
                'winner': winner,
                'optimal_picks': optimal,
                **features
            }

            training_records.append(record)

        logger.info(f"  Collected {len([r for r in training_records if r['date'] == date_str])} race records")

    # Convert to DataFrame
    df_training = pd.DataFrame(training_records)

    # Save
    output_file = 'system_selection_training_data.csv'
    df_training.to_csv(output_file, index=False)

    logger.info("")
    logger.info("="*80)
    logger.info("TRAINING DATA PREPARATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Output: {output_file}")
    logger.info(f"Total races: {len(df_training)}")
    logger.info(f"Date range: {df_training['date'].min()} to {df_training['date'].max()}")
    logger.info("")

    # Distribution of optimal picks
    logger.info("OPTIMAL PICKS DISTRIBUTION:")
    logger.info("-"*80)
    for picks in sorted(df_training['optimal_picks'].unique()):
        count = (df_training['optimal_picks'] == picks).sum()
        pct = count / len(df_training) * 100
        logger.info(f"  {picks} pick(s): {count:4d} races ({pct:5.1f}%)")

    logger.info("")
    logger.info("DISTRIBUTION BY GAME TYPE:")
    logger.info("-"*80)
    for game_type in df_training['game_type'].unique():
        game_data = df_training[df_training['game_type'] == game_type]
        logger.info(f"{game_type}: {len(game_data)} races")
        for picks in sorted(game_data['optimal_picks'].unique()):
            count = (game_data['optimal_picks'] == picks).sum()
            pct = count / len(game_data) * 100
            logger.info(f"  {picks} pick(s): {count:4d} ({pct:5.1f}%)")

    return df_training


if __name__ == '__main__':
    df = prepare_training_data()
