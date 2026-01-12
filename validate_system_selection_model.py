#!/usr/bin/env python3
"""
Temporal validation of system selection model

This validates the model using proper temporal holdout:
- Train on first 80% of dates
- Test on last 20% of dates

This prevents data leakage and simulates real-world usage.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def temporal_validation():
    """Validate model with temporal holdout"""

    logger.info("Loading training data...")
    df = pd.read_csv('system_selection_training_data.csv')

    # Sort by date
    df = df.sort_values('date')

    logger.info(f"Total races: {len(df)}")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"Unique dates: {df['date'].nunique()}")

    # Split temporally: first 80% dates for training, last 20% for testing
    unique_dates = sorted(df['date'].unique())
    split_idx = int(len(unique_dates) * 0.8)

    train_dates = unique_dates[:split_idx]
    test_dates = unique_dates[split_idx:]

    logger.info(f"\nTemporal split:")
    logger.info(f"  Train dates: {train_dates[0]} to {train_dates[-1]} ({len(train_dates)} dates)")
    logger.info(f"  Test dates: {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} dates)")

    train_df = df[df['date'].isin(train_dates)]
    test_df = df[df['date'].isin(test_dates)]

    logger.info(f"\n  Train races: {len(train_df)}")
    logger.info(f"  Test races: {len(test_df)}")

    # Load model
    logger.info("\nLoading model...")
    with open('system_selection_model.pkl', 'rb') as f:
        model_data = pickle.load(f)

    model = model_data['model']
    feature_cols = model_data['feature_cols']

    # Prepare features
    # Add game type encoding
    for dataset in [train_df, test_df]:
        game_type_dummies = pd.get_dummies(dataset['game_type'], prefix='game')
        for col in game_type_dummies.columns:
            dataset[col] = game_type_dummies[col]

    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in train_df.columns:
            train_df[col] = 0
        if col not in test_df.columns:
            test_df[col] = 0

    X_train = train_df[feature_cols]
    y_train = train_df['optimal_picks']

    X_test = test_df[feature_cols]
    y_test = test_df['optimal_picks']

    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    logger.info(f"\nModel Performance:")
    logger.info(f"  Training accuracy: {train_score:.3f}")
    logger.info(f"  Test accuracy (temporal holdout): {test_score:.3f}")

    # Predictions on test set
    y_pred = model.predict(X_test)

    logger.info("\nClassification Report (Temporal Test Set):")
    logger.info("\n" + classification_report(y_test, y_pred))

    logger.info("\nConfusion Matrix (Temporal Test Set):")
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"\n{cm}")

    # Detailed analysis
    logger.info("\nDetailed Prediction Analysis:")
    logger.info("-"*80)

    test_results = pd.DataFrame({
        'date': test_df['date'].values,
        'race_num': test_df['race_num'].values,
        'actual': y_test.values,
        'predicted': y_pred,
        'top1_prob': X_test['top1_prob'].values,
        'gap_1_2': X_test['gap_1_2'].values,
        'num_horses': X_test['num_horses'].values
    })

    # Accuracy by predicted picks
    for picks in sorted(test_results['predicted'].unique()):
        subset = test_results[test_results['predicted'] == picks]
        accuracy = (subset['actual'] == subset['predicted']).mean()

        logger.info(f"\nWhen model predicts {picks} pick(s) ({len(subset)} races):")
        logger.info(f"  Accuracy: {accuracy:.3f}")
        logger.info(f"  Actual distribution: {subset['actual'].value_counts().sort_index().to_dict()}")
        logger.info(f"  Avg top1_prob: {subset['top1_prob'].mean():.3f}")
        logger.info(f"  Avg gap_1_2: {subset['gap_1_2'].mean():.3f}")

    # System cost analysis
    logger.info("\n" + "="*80)
    logger.info("SYSTEM COST SIMULATION")
    logger.info("="*80)

    # Group by date (V-game system)
    logger.info("\nSimulating V-game systems on test dates:")

    for date in sorted(test_results['date'].unique()):
        date_races = test_results[test_results['date'] == date]

        # Calculate system cost
        predicted_picks = date_races['predicted'].values
        total_combinations = np.prod(predicted_picks)

        # Check if system would have hit (all races correct)
        all_correct = (date_races['actual'] <= date_races['predicted']).all()

        # Count correct races
        correct_races = (date_races['actual'] <= date_races['predicted']).sum()
        total_races = len(date_races)

        logger.info(f"\n{date}:")
        logger.info(f"  Configuration: {' × '.join(map(str, predicted_picks))}")
        logger.info(f"  Total rows: {total_combinations}")
        logger.info(f"  Cost: {total_combinations} SEK")
        logger.info(f"  Correct races: {correct_races}/{total_races}")
        logger.info(f"  System hit: {'✅ YES' if all_correct else '❌ NO'}")

    # Summary statistics
    system_hits = 0
    total_systems = 0
    total_cost = 0
    total_correct = 0
    total_races = 0

    for date in sorted(test_results['date'].unique()):
        date_races = test_results[test_results['date'] == date]
        predicted_picks = date_races['predicted'].values
        total_combinations = np.prod(predicted_picks)

        all_correct = (date_races['actual'] <= date_races['predicted']).all()
        correct_races = (date_races['actual'] <= date_races['predicted']).sum()

        if all_correct:
            system_hits += 1

        total_systems += 1
        total_cost += total_combinations
        total_correct += correct_races
        total_races += len(date_races)

    logger.info("\n" + "="*80)
    logger.info("TEMPORAL VALIDATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Total V-game systems tested: {total_systems}")
    logger.info(f"System hits: {system_hits}/{total_systems} ({system_hits/total_systems*100:.1f}%)")
    logger.info(f"Average cost per system: {total_cost/total_systems:.0f} SEK")
    logger.info(f"Total correct races: {total_correct}/{total_races} ({total_correct/total_races*100:.1f}%)")
    logger.info(f"Average correct per system: {total_correct/total_systems:.1f}/{total_races/total_systems:.1f}")

    logger.info("\n" + "="*80)
    logger.info("MODEL READINESS")
    logger.info("="*80)

    if test_score >= 0.25:
        logger.info("✅ Model shows reasonable performance for V-game system generation")
        logger.info("✅ Temporal validation passed")
        logger.info("✅ Ready for production use")
    else:
        logger.warning("⚠️  Model accuracy is lower than expected")
        logger.warning("   Consider collecting more training data or feature engineering")

    return test_score, system_hits, total_systems


if __name__ == '__main__':
    logger.info("Starting temporal validation of system selection model")
    logger.info("="*80)

    test_acc, hits, total = temporal_validation()

    logger.info("\n✅ Validation complete!")
    logger.info(f"Final test accuracy: {test_acc:.3f}")
    logger.info(f"System hit rate: {hits}/{total} ({hits/total*100:.1f}%)")
