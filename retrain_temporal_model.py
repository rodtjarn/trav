#!/usr/bin/env python3
"""
Retrain temporal model with proper train/test split on 2025 data

Training: 2025-01-01 to 2025-10-31
Testing:  2025-11-01 to 2025-12-31
"""

import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_feature_columns(df):
    """Get feature columns (exclude results and metadata)"""
    exclude_cols = [
        # Results (NEVER available at prediction time)
        'is_winner', 'is_top3', 'is_placed', 'finish_position',
        'finish_place', 'finish_order', 'finish_time', 'finish_speed_kmmin',
        'prize_money', 'galloped',

        # Odds (exclude to prevent leakage)
        'final_odds',

        # Metadata
        'date', 'race_date', 'race_id', 'race_number', 'start_time',

        # IDs
        'horse_id', 'driver_id', 'trainer_id', 'track_id',

        # Names
        'horse_name', 'horse_color', 'horse_sex',
        'driver_first_name', 'driver_last_name', 'driver_short_name',
        'trainer_first_name', 'trainer_last_name', 'trainer_short_name',
        'track_name', 'track', 'track_encoded', 'track_country',

        # Other metadata
        'start_method', 'track_condition', 'record_code',
        'driver_license', 'trainer_license',
        'shoes_front', 'shoes_back', 'sulky_type',
    ]

    # Also exclude any remaining non-numeric columns
    feature_cols = []
    for col in df.columns:
        if col not in exclude_cols and df[col].dtype in ['int64', 'float64', 'bool']:
            feature_cols.append(col)

    return feature_cols

def main():
    logger.info("=" * 80)
    logger.info("RETRAINING TEMPORAL MODEL WITH PROPER TRAIN/TEST SPLIT")
    logger.info("=" * 80)

    # Load full 2025 data
    logger.info("Loading temporal_processed_data.csv...")
    df = pd.read_csv('temporal_processed_data.csv')
    logger.info(f"Loaded {len(df)} samples")

    # Extract date from race_id
    df['date'] = pd.to_datetime(df['race_id'].str.extract(r'(\d{4}-\d{2}-\d{2})')[0])

    # Define split date
    split_date = '2025-10-31'
    logger.info(f"\nTrain/Test Split Date: {split_date}")
    logger.info(f"Training: 2025-01-01 to {split_date}")
    logger.info(f"Testing:  2025-11-01 to 2025-12-31")

    # Split data
    train_df = df[df['date'] <= split_date].copy()
    test_df = df[df['date'] > split_date].copy()

    logger.info(f"\nTraining samples: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    logger.info(f"Testing samples:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

    # Verify no date overlap
    assert train_df['date'].max() < test_df['date'].min(), "Date overlap detected!"
    logger.info("✓ No date overlap - clean temporal split")

    # Check target distribution
    logger.info(f"\nTraining set:")
    logger.info(f"  Winners: {train_df['is_winner'].sum()} ({train_df['is_winner'].mean()*100:.2f}%)")
    logger.info(f"  Date range: {train_df['date'].min()} to {train_df['date'].max()}")

    logger.info(f"\nTest set:")
    logger.info(f"  Winners: {test_df['is_winner'].sum()} ({test_df['is_winner'].mean()*100:.2f}%)")
    logger.info(f"  Date range: {test_df['date'].min()} to {test_df['date'].max()}")

    # Get feature columns
    feature_cols = get_feature_columns(df)
    logger.info(f"\nUsing {len(feature_cols)} features")

    # Prepare training data
    X_train = train_df[feature_cols]
    y_train = train_df['is_winner']

    X_test = test_df[feature_cols]
    y_test = test_df['is_winner']

    # Handle any missing values and infinities
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)

    logger.info("\n" + "=" * 80)
    logger.info("TRAINING RANDOM FOREST MODEL")
    logger.info("=" * 80)

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
        verbose=1
    )

    logger.info("Training model...")
    model.fit(X_train, y_train)
    logger.info("✓ Training complete")

    # Evaluate on training set
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING SET PERFORMANCE")
    logger.info("=" * 80)

    train_pred = model.predict(X_train)
    train_prob = model.predict_proba(X_train)[:, 1]

    train_acc = accuracy_score(y_train, train_pred)
    train_auc = roc_auc_score(y_train, train_prob)

    logger.info(f"Accuracy: {train_acc*100:.2f}%")
    logger.info(f"ROC-AUC: {train_auc:.4f}")

    # Evaluate on test set
    logger.info("\n" + "=" * 80)
    logger.info("TEST SET PERFORMANCE (HELD-OUT 2025 DATA)")
    logger.info("=" * 80)

    test_pred = model.predict(X_test)
    test_prob = model.predict_proba(X_test)[:, 1]

    test_acc = accuracy_score(y_test, test_pred)
    test_auc = roc_auc_score(y_test, test_prob)

    logger.info(f"Accuracy: {test_acc*100:.2f}%")
    logger.info(f"ROC-AUC: {test_auc:.4f}")

    # Per-race analysis (top prediction per race)
    logger.info("\n" + "=" * 80)
    logger.info("PER-RACE ACCURACY (TOP PREDICTION PER RACE)")
    logger.info("=" * 80)

    # Test set per-race accuracy
    test_df_eval = test_df.copy()
    test_df_eval['predicted_prob'] = test_prob

    # Group by race and get top prediction
    race_predictions = []
    for race_id, race_group in test_df_eval.groupby('race_id'):
        top_horse = race_group.loc[race_group['predicted_prob'].idxmax()]
        race_predictions.append({
            'race_id': race_id,
            'predicted_winner': top_horse['start_number'] if 'start_number' in top_horse else 'N/A',
            'actual_winner': race_group[race_group['is_winner'] == True]['start_number'].values[0] if (race_group['is_winner'] == True).any() else None,
            'correct': top_horse['is_winner']
        })

    race_results = pd.DataFrame(race_predictions)
    per_race_acc = race_results['correct'].mean()

    logger.info(f"Per-race accuracy: {per_race_acc*100:.2f}% ({race_results['correct'].sum()}/{len(race_results)} races)")
    logger.info(f"Expected random: ~{1/train_df.groupby('race_id').size().mean()*100:.1f}%")

    # Feature importances
    logger.info("\n" + "=" * 80)
    logger.info("TOP 20 FEATURE IMPORTANCES")
    logger.info("=" * 80)

    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    for idx, row in feature_importance.head(20).iterrows():
        logger.info(f"{row['feature']:40s} {row['importance']:.4f}")

    # Save model
    logger.info("\n" + "=" * 80)
    logger.info("SAVING MODEL")
    logger.info("=" * 80)

    model_path = 'temporal_rf_model_retrained.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"✓ Model saved to {model_path}")

    # Save metadata
    metadata = {
        'train_date_range': f"{train_df['date'].min()} to {train_df['date'].max()}",
        'test_date_range': f"{test_df['date'].min()} to {test_df['date'].max()}",
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'num_features': len(feature_cols),
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'train_auc': float(train_auc),
        'test_auc': float(test_auc),
        'per_race_accuracy': float(per_race_acc),
        'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    import json
    metadata_path = 'temporal_rf_model_retrained_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✓ Metadata saved to {metadata_path}")

    logger.info("\n" + "=" * 80)
    logger.info("RETRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nModel: {model_path}")
    logger.info(f"Metadata: {metadata_path}")
    logger.info(f"\nTest Set Performance:")
    logger.info(f"  Accuracy: {test_acc*100:.2f}%")
    logger.info(f"  ROC-AUC: {test_auc:.4f}")
    logger.info(f"  Per-race: {per_race_acc*100:.2f}%")

if __name__ == '__main__':
    main()
