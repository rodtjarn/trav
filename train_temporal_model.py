#!/usr/bin/env python3
"""
Temporal Model Training - NO DATA LEAKAGE
Trains model with proper temporal split and validation
"""

import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score
)
from imblearn.over_sampling import SMOTE
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TemporalModelTrainer:
    """Train model with temporal awareness"""

    def __init__(self, target='is_winner', random_state=42):
        self.target = target
        self.random_state = random_state
        self.model = None
        self.feature_cols = None
        self.metadata = {}

    def prepare_features(self, df: pd.DataFrame):
        """
        Prepare feature columns

        CRITICAL: Exclude ALL result-related columns that wouldn't be known at prediction time
        """
        # Columns to EXCLUDE (results, metadata, IDs)
        exclude_cols = [
            # Results (NEVER available at prediction time)
            'is_winner', 'is_top3', 'is_placed', 'finish_position',
            'finish_place', 'finish_order', 'finish_time', 'finish_speed_kmmin',
            'prize_money', 'galloped',

            # Odds (can be included but be careful - final odds set close to race time)
            'final_odds',

            # Metadata
            'date', 'race_date', 'race_id', 'race_number', 'start_time',

            # IDs
            'horse_id', 'driver_id', 'trainer_id', 'track_id',

            # Names
            'horse_name', 'horse_color',
            'driver_first_name', 'driver_last_name', 'driver_short_name',
            'trainer_first_name', 'trainer_last_name', 'trainer_short_name',
            'track_name', 'track', 'track_encoded',

            # Other metadata
            'start_method', 'track_condition', 'record_code',
            'driver_license', 'trainer_license',
            'shoes_front', 'shoes_back', 'sulky_type',
        ]

        # Get numeric feature columns
        all_cols = set(df.columns)
        potential_features = all_cols - set(exclude_cols)

        numeric_features = []
        for col in sorted(potential_features):
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_features.append(col)

        self.feature_cols = numeric_features

        logger.info(f"\nSelected {len(self.feature_cols)} features:")
        for col in self.feature_cols[:10]:
            logger.info(f"  - {col}")
        if len(self.feature_cols) > 10:
            logger.info(f"  ... and {len(self.feature_cols) - 10} more")

        return self.feature_cols

    def temporal_train_test_split(self, df: pd.DataFrame, train_end_date: str):
        """
        Split data temporally

        Args:
            df: Full dataset
            train_end_date: Last date for training (inclusive)

        Returns:
            X_train, X_test, y_train, y_test
        """
        train_end = pd.to_datetime(train_end_date)
        test_start = train_end + timedelta(days=1)

        # Split by date
        train_mask = pd.to_datetime(df['date']) <= train_end
        test_mask = pd.to_datetime(df['date']) >= test_start

        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()

        # Remove rows without target
        train_df = train_df[train_df[self.target].notna()]
        test_df = test_df[test_df[self.target].notna()]

        # Prepare features and targets
        X_train = train_df[self.feature_cols]
        y_train = train_df[self.target].astype(int)

        X_test = test_df[self.feature_cols]
        y_test = test_df[self.target].astype(int)

        logger.info(f"\n{'='*80}")
        logger.info(f"TEMPORAL SPLIT")
        logger.info(f"{'='*80}")
        logger.info(f"Training:   {train_df['date'].min()} to {train_df['date'].max()}")
        logger.info(f"            {len(X_train):,} samples, {y_train.sum():,} winners ({y_train.mean()*100:.1f}%)")
        logger.info(f"\nTesting:    {test_df['date'].min()} to {test_df['date'].max()}")
        logger.info(f"            {len(X_test):,} samples, {y_test.sum():,} winners ({y_test.mean()*100:.1f}%)")

        return X_train, X_test, y_train, y_test, train_df, test_df

    def apply_smote(self, X_train, y_train):
        """Apply SMOTE for class balancing"""
        logger.info(f"\n{'='*80}")
        logger.info("APPLYING SMOTE")
        logger.info(f"{'='*80}")
        logger.info(f"Before SMOTE: {len(y_train):,} samples")
        logger.info(f"  Winners: {y_train.sum():,} ({y_train.mean()*100:.1f}%)")

        # Handle NaN and infinity values
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_train = X_train.fillna(X_train.median()).fillna(0)

        smote = SMOTE(random_state=self.random_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        logger.info(f"\nAfter SMOTE: {len(y_train_balanced):,} samples")
        logger.info(f"  Winners: {y_train_balanced.sum():,} ({y_train_balanced.mean()*100:.1f}%)")

        return X_train_balanced, y_train_balanced

    def train_model(self, X_train, y_train, **rf_params):
        """Train Random Forest model"""
        default_params = {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
            'random_state': self.random_state,
            'n_jobs': -1,
            'class_weight': 'balanced'
        }
        default_params.update(rf_params)

        logger.info(f"\n{'='*80}")
        logger.info("TRAINING RANDOM FOREST")
        logger.info(f"{'='*80}")
        logger.info(f"Parameters: {default_params}")

        self.model = RandomForestClassifier(**default_params)
        self.model.fit(X_train, y_train)

        logger.info("✓ Training complete!")

        return self.model

    def evaluate_model(self, X_test, y_test, test_df):
        """Evaluate model on temporal test set"""
        logger.info(f"\n{'='*80}")
        logger.info("MODEL EVALUATION - TEMPORAL TEST SET")
        logger.info(f"{'='*80}")

        # Handle NaN and infinity values in test set
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.fillna(X_test.median()).fillna(0)

        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        logger.info(f"\nAccuracy: {accuracy:.4f}")
        logger.info(f"ROC-AUC:  {roc_auc:.4f}")

        # Classification report
        logger.info("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Not Winner', 'Winner']))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  True Negatives:  {cm[0, 0]:,}")
        logger.info(f"  False Positives: {cm[0, 1]:,}")
        logger.info(f"  False Negatives: {cm[1, 0]:,}")
        logger.info(f"  True Positives:  {cm[1, 1]:,}")

        # Per-race analysis
        logger.info(f"\n{'='*80}")
        logger.info("PER-RACE PREDICTION ACCURACY")
        logger.info(f"{'='*80}")

        test_df = test_df.copy()
        test_df['predicted_prob'] = y_pred_proba

        # Group by race and find top prediction
        races_correct = 0
        total_races = 0

        track_col = 'track_name' if 'track_name' in test_df.columns else 'track'
        for (date, track, race_num), race_df in test_df.groupby(['date', track_col, 'race_number']):
            if len(race_df) < 2:  # Need at least 2 horses
                continue

            # Find actual winner
            actual_winner = race_df[race_df[self.target] == 1]
            if len(actual_winner) == 0:
                continue

            # Find predicted winner (highest probability)
            predicted_winner = race_df.loc[race_df['predicted_prob'].idxmax()]

            if predicted_winner[self.target] == 1:
                races_correct += 1

            total_races += 1

        per_race_accuracy = races_correct / total_races if total_races > 0 else 0

        logger.info(f"\nRaces with correct winner prediction: {races_correct}/{total_races} ({per_race_accuracy*100:.1f}%)")
        logger.info(f"This is the REALISTIC win rate for betting on top model picks")

        metrics = {
            'accuracy': float(accuracy),
            'roc_auc': float(roc_auc),
            'per_race_accuracy': float(per_race_accuracy),
            'races_correct': int(races_correct),
            'total_races': int(total_races),
            'confusion_matrix': cm.tolist()
        }

        return metrics

    def show_feature_importance(self, top_n=20):
        """Display feature importance"""
        logger.info(f"\n{'='*80}")
        logger.info(f"TOP {top_n} MOST IMPORTANT FEATURES")
        logger.info(f"{'='*80}")

        importances = self.model.feature_importances_
        feature_imp_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)

        for idx, row in feature_imp_df.head(top_n).iterrows():
            logger.info(f"  {row['feature']:35s}: {row['importance']:.4f}")

        return feature_imp_df

    def save_model(self, model_path='temporal_rf_model.pkl', metadata_path='temporal_rf_metadata.json'):
        """Save model and metadata"""
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"\n✓ Model saved to {model_path}")

        metadata = {
            'model_type': 'RandomForestClassifier',
            'target': self.target,
            'feature_columns': self.feature_cols,
            'num_features': len(self.feature_cols),
            'training_date': datetime.now().isoformat(),
            'random_state': self.random_state,
            'temporal_validation': True,
            **self.metadata
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✓ Metadata saved to {metadata_path}")


def main():
    """Main training pipeline with temporal validation"""

    logger.info("="*80)
    logger.info("TEMPORAL MODEL TRAINING - NO DATA LEAKAGE")
    logger.info("="*80)

    # Load temporal processed data
    logger.info("\nLoading temporal processed data...")
    df = pd.read_csv('temporal_processed_data.csv')

    # Initialize trainer
    trainer = TemporalModelTrainer(target='is_winner', random_state=42)

    # Prepare features
    trainer.prepare_features(df)

    # Temporal split: Train on Jan-Oct 2025, Test on Nov-Dec 2025
    X_train, X_test, y_train, y_test, train_df, test_df = trainer.temporal_train_test_split(
        df,
        train_end_date='2025-10-31'
    )

    # Apply SMOTE
    X_train_balanced, y_train_balanced = trainer.apply_smote(X_train, y_train)

    # Train model
    model = trainer.train_model(
        X_train_balanced,
        y_train_balanced,
        n_estimators=100,
        max_depth=20
    )

    # Evaluate on temporal test set
    metrics = trainer.evaluate_model(X_test, y_test, test_df)

    # Show feature importance
    feature_importance = trainer.show_feature_importance(top_n=20)

    # Save model
    trainer.metadata = {
        'train_start_date': str(train_df['date'].min()),
        'train_end_date': str(train_df['date'].max()),
        'test_start_date': str(test_df['date'].min()),
        'test_end_date': str(test_df['date'].max()),
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        **metrics
    }

    trainer.save_model()

    logger.info(f"\n{'='*80}")
    logger.info("✓ TRAINING COMPLETE!")
    logger.info(f"{'='*80}")
    logger.info(f"\nModel trained with PROPER temporal validation")
    logger.info(f"Per-race accuracy (realistic win rate): {metrics['per_race_accuracy']*100:.1f}%")


if __name__ == "__main__":
    main()
