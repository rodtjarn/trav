#!/usr/bin/env python3
"""
Generate V-game system bets using the ML-based selection meta-model

This uses a trained model to decide how many horses to pick in each race,
rather than using fixed rules. The model can select 1-5+ horses per race
based on race characteristics and prediction confidence.
"""

import pickle
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
from atg_api_scraper import ATGAPIScraper
from temporal_data_processor import TemporalDataProcessor
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_race_features(race_horses_df):
    """Calculate features for the meta-model"""
    sorted_horses = race_horses_df.sort_values('predicted_prob', ascending=False)

    # Top predictions
    top1_prob = sorted_horses.iloc[0]['predicted_prob'] if len(sorted_horses) > 0 else 0
    top2_prob = sorted_horses.iloc[1]['predicted_prob'] if len(sorted_horses) > 1 else 0
    top3_prob = sorted_horses.iloc[2]['predicted_prob'] if len(sorted_horses) > 2 else 0
    top4_prob = sorted_horses.iloc[3]['predicted_prob'] if len(sorted_horses) > 3 else 0
    top5_prob = sorted_horses.iloc[4]['predicted_prob'] if len(sorted_horses) > 4 else 0

    # Confidence spreads
    gap_1_2 = top1_prob - top2_prob
    gap_2_3 = top2_prob - top3_prob
    gap_3_4 = top3_prob - top4_prob
    gap_1_3 = top1_prob - top3_prob

    # Distribution metrics
    mean_prob = sorted_horses['predicted_prob'].mean()
    std_prob = sorted_horses['predicted_prob'].std()
    max_prob = sorted_horses['predicted_prob'].max()
    min_prob = sorted_horses['predicted_prob'].min()

    # Top concentration
    top3_sum = top1_prob + top2_prob + top3_prob

    # Entropy
    probs = sorted_horses['predicted_prob'].values
    probs = probs / probs.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-10))

    # Counts
    num_horses = len(sorted_horses)
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


def generate_ml_system(df, num_races, game_type, selection_model, max_budget=500):
    """
    Generate system picks using the trained meta-model

    Args:
        df: DataFrame with predictions for all horses
        num_races: Number of races in the system
        game_type: V75, V86, etc.
        selection_model: Dict with 'model' and 'feature_cols'
        max_budget: Maximum cost in SEK

    Returns:
        selections: dict of {race_num: [horse_numbers]}
        cost: Total system cost
    """
    model = selection_model['model']
    feature_cols = selection_model['feature_cols']

    selections = {}
    race_features_list = []

    # First pass: get features and predictions for each race
    for race_num in range(1, num_races + 1):
        race_horses = df[df['vgame_race_num'] == race_num].copy()

        if len(race_horses) == 0:
            logger.warning(f"No horses found for race {race_num}")
            continue

        # Calculate features
        features = calculate_race_features(race_horses)

        # Add game type encoding
        for col in feature_cols:
            if col.startswith('game_'):
                features[col] = 1 if col == f'game_{game_type}' else 0

        race_features_list.append({
            'race_num': race_num,
            'features': features,
            'horses': race_horses
        })

    # Create feature DataFrame for prediction
    X_pred = pd.DataFrame([r['features'] for r in race_features_list])

    # Ensure all required features are present
    for col in feature_cols:
        if col not in X_pred.columns:
            X_pred[col] = 0

    X_pred = X_pred[feature_cols]

    # Predict optimal picks for each race
    predicted_picks = model.predict(X_pred)

    logger.info("\nML Model Predictions:")
    logger.info("-"*60)

    # Generate selections
    for idx, race_data in enumerate(race_features_list):
        race_num = race_data['race_num']
        race_horses = race_data['horses']
        num_picks = int(predicted_picks[idx])

        # Sort by prediction probability
        sorted_horses = race_horses.sort_values('predicted_prob', ascending=False)

        # Select top N horses (cap at available horses)
        num_picks = min(num_picks, len(sorted_horses))
        selected = sorted_horses.head(num_picks)

        selections[race_num] = [int(row['start_number']) for _, row in selected.iterrows()]

        # Log
        top_prob = sorted_horses.iloc[0]['predicted_prob']
        gap = sorted_horses.iloc[0]['predicted_prob'] - sorted_horses.iloc[1]['predicted_prob'] if len(sorted_horses) > 1 else 0

        logger.info(f"Race {race_num}: {num_picks} pick(s) | Top prob: {top_prob:.3f} | Gap: {gap:.3f}")
        for _, horse in selected.iterrows():
            logger.info(f"  #{int(horse['start_number'])}: {horse.get('horse_name', 'Unknown'):25s} ({horse['predicted_prob']:.3f})")

    # Calculate cost
    if selections:
        num_picks_per_race = [len(picks) for picks in selections.values()]
        total_combinations = 1
        for num in num_picks_per_race:
            total_combinations *= num

        cost = total_combinations

        # Check if over budget - reduce highest picks first
        while cost > max_budget and max(num_picks_per_race) > 1:
            # Find race with most picks and reduce by 1
            max_picks_race = None
            max_picks_count = 0

            for race_num, picks in selections.items():
                if len(picks) > max_picks_count:
                    max_picks_count = len(picks)
                    max_picks_race = race_num

            if max_picks_race:
                selections[max_picks_race] = selections[max_picks_race][:-1]
                logger.info(f"Reduced race {max_picks_race} to {len(selections[max_picks_race])} picks (budget constraint)")

                # Recalculate cost
                num_picks_per_race = [len(picks) for picks in selections.values()]
                total_combinations = 1
                for num in num_picks_per_race:
                    total_combinations *= num
                cost = total_combinations
            else:
                break

        return selections, cost
    else:
        return {}, 0


def create_ml_system_bet(date_str, game_type, max_budget=500):
    """
    Main function to create ML-based system bet

    Args:
        date_str: Date in YYYY-MM-DD format
        game_type: V75, V86, V85, GS75, etc.
        max_budget: Maximum system cost in SEK
    """

    logger.info("="*80)
    logger.info(f"ML SYSTEM BET GENERATOR")
    logger.info("="*80)
    logger.info(f"Date: {date_str}")
    logger.info(f"Game: {game_type}")
    logger.info(f"Max budget: {max_budget} SEK")
    logger.info("")

    # Load prediction model
    logger.info("Loading prediction model...")
    with open('temporal_rf_model.pkl', 'rb') as f:
        prediction_model = pickle.load(f)

    feature_cols = list(prediction_model.feature_names_in_)

    # Load selection meta-model
    logger.info("Loading selection meta-model...")
    with open('system_selection_model.pkl', 'rb') as f:
        selection_model = pickle.load(f)

    # Initialize scraper and processor
    scraper = ATGAPIScraper()
    processor = TemporalDataProcessor()

    # Get game info
    logger.info(f"\nFetching {game_type} races for {date_str}...")
    game_info = scraper.get_game_info(date_str, game_type)

    if not game_info or not game_info.get('race_ids'):
        logger.error(f"No {game_type} races found for {date_str}")
        return None

    race_ids = game_info['race_ids']
    logger.info(f"Found {len(race_ids)} races")

    # Fetch race details
    all_race_data = []
    for race_id in race_ids:
        logger.info(f"  Fetching race {race_id}...")
        race_details = scraper.get_race_details(race_id)
        if race_details:
            all_race_data.append(race_details)

    if not all_race_data:
        logger.error("Failed to fetch race data")
        return None

    # Process data
    logger.info("\nProcessing race data...")
    df = processor.process_race_data(all_race_data, feature_cols)

    if df.empty:
        logger.error("Failed to process race data")
        return None

    # Make predictions
    logger.info("Generating predictions...")
    X = df[feature_cols]
    predictions = prediction_model.predict_proba(X)[:, 1]
    df['predicted_prob'] = predictions

    # Map to sequential race numbers
    race_id_to_seq = {}
    for seq_num, race_id in enumerate(race_ids, 1):
        track_race_num = int(race_id.split('_')[-1])
        race_id_to_seq[track_race_num] = seq_num

    df['vgame_race_num'] = df['race_number'].map(race_id_to_seq)

    # Generate system using ML
    logger.info("\nGenerating ML-based system...")
    selections, cost = generate_ml_system(df, len(race_ids), game_type, selection_model, max_budget)

    if not selections:
        logger.error("Failed to generate system")
        return None

    # Summary
    logger.info("\n" + "="*80)
    logger.info("SYSTEM SUMMARY")
    logger.info("="*80)

    num_picks_per_race = [len(picks) for picks in selections.values()]

    logger.info(f"Configuration: {' × '.join(map(str, num_picks_per_race))}")
    logger.info(f"Total rows: {cost}")
    logger.info(f"Total cost: {cost} SEK")
    logger.info("")

    logger.info("Selected horses:")
    for race_num in sorted(selections.keys()):
        picks = selections[race_num]
        logger.info(f"  Race {race_num}: {picks}")

    logger.info("")
    logger.info("System ready for betting!")

    return {
        'date': date_str,
        'game_type': game_type,
        'selections': selections,
        'cost': cost,
        'picks_per_race': num_picks_per_race
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate ML-based V-game system bet')
    parser.add_argument('--date', required=True, help='Date (YYYY-MM-DD)')
    parser.add_argument('--game', required=True, help='Game type (V75, V86, V85, GS75, etc.)')
    parser.add_argument('--budget', type=int, default=500, help='Max budget in SEK (default: 500)')

    args = parser.parse_args()

    result = create_ml_system_bet(args.date, args.game, args.budget)

    if result:
        logger.info("\n✅ System generation successful!")
    else:
        logger.error("\n❌ System generation failed")
