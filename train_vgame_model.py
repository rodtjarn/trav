#!/usr/bin/env python3
"""
V-Game Model Training with Proper Train/Test Split

CRITICAL: Ensures temporal split with NO data leakage
- Training data: Earlier time period
- Test data: Later time period (held-out, never seen during training)
- Validation: Per-race accuracy on test set (realistic win rate)
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
    accuracy_score,
    precision_recall_curve,
    average_precision_score
)
from imblearn.over_sampling import SMOTE
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VGameModelTrainer:
    """Train model for V-game predictions with proper temporal validation"""

    def __init__(self, target='is_winner', random_state=42):
        self.target = target
        self.random_state = random_state
        self.model = None
        self.feature_cols = None
        self.metadata = {}

    def load_and_prepare_data(self, data_file):
        """
        Load data and ensure proper format

        Args:
            data_file: Path to CSV data file
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"LOADING DATA")
        logger.info(f"{'='*80}")
        logger.info(f"File: {data_file}")

        df = pd.read_csv(data_file)

        logger.info(f"Loaded {len(df):,} starts")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"Unique races: {df['race_id'].nunique():,}")

        # Check for V-game data
        if 'vgame_type' in df.columns:
            vgame_count = df['is_vgame'].sum()
            logger.info(f"V-game starts: {vgame_count:,} ({vgame_count/len(df)*100:.1f}%)")

            if vgame_count > 0:
                logger.info(f"\nV-game breakdown:")
                for game_type, count in df[df['is_vgame']]['vgame_type'].value_counts().items():
                    logger.info(f"  {game_type}: {count:,} starts")

        # Check target availability
        if self.target in df.columns:
            target_available = df[self.target].notna().sum()
            logger.info(f"\nTarget ({self.target}) available: {target_available:,} ({target_available/len(df)*100:.1f}%)")
        else:
            logger.error(f"\n❌ Target column '{self.target}' not found!")
            logger.info(f"Available columns: {', '.join(df.columns[:20])}")
            sys.exit(1)

        return df

    def prepare_features(self, df):
        """
        Prepare feature columns - CRITICAL: Exclude all future information

        Only use data that would be available BEFORE the race starts
        """
        # Columns to ABSOLUTELY EXCLUDE (results, future information)
        exclude_cols = [
            # Results (NEVER available at prediction time)
            'is_winner', 'is_top3', 'is_placed', 'finish_position',
            'finish_place', 'finish_order', 'finish_time', 'finish_speed_kmmin',
            'prize_money', 'galloped', 'km_time_minutes', 'km_time_seconds', 'km_time_tenths',
            'win_time_minutes', 'win_time_seconds', 'win_time_tenths',

            # Final odds (set too close to race time - can include if available)
            # 'final_odds',  # Uncomment to exclude

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
                # Check for constant columns or columns with too many NaNs
                if df[col].nunique() > 1 and df[col].notna().sum() > len(df) * 0.1:
                    numeric_features.append(col)

        self.feature_cols = numeric_features

        logger.info(f"\n{'='*80}")
        logger.info(f"FEATURE SELECTION")
        logger.info(f"{'='*80}")
        logger.info(f"Selected {len(self.feature_cols)} features")
        logger.info(f"\nTop features:")
        for col in self.feature_cols[:15]:
            non_null = df[col].notna().sum()
            logger.info(f"  - {col:40s} ({non_null:,}/{len(df):,} non-null)")
        if len(self.feature_cols) > 15:
            logger.info(f"  ... and {len(self.feature_cols) - 15} more")

        return self.feature_cols

    def temporal_train_test_split(self, df, train_end_date, test_start_date=None):
        """
        Split data temporally - CRITICAL for time-series data

        Args:
            df: Full dataset
            train_end_date: Last date for training (inclusive)
            test_start_date: First date for testing (if None, day after train_end_date)

        Returns:
            X_train, X_test, y_train, y_test, train_df, test_df
        """
        train_end = pd.to_datetime(train_end_date)

        if test_start_date:
            test_start = pd.to_datetime(test_start_date)
        else:
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
        logger.info(f"TEMPORAL SPLIT - CRITICAL FOR VALIDITY")
        logger.info(f"{'='*80}")
        logger.info(f"\n📊 TRAINING SET (Past Data)")
        logger.info(f"   Date range: {train_df['date'].min()} to {train_df['date'].max()}")
        logger.info(f"   Samples: {len(X_train):,}")
        logger.info(f"   Winners: {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
        logger.info(f"   Races: {train_df['race_id'].nunique():,}")

        logger.info(f"\n📊 TEST SET (Future Data - Held Out)")
        logger.info(f"   Date range: {test_df['date'].min()} to {test_df['date'].max()}")
        logger.info(f"   Samples: {len(X_test):,}")
        logger.info(f"   Winners: {y_test.sum():,} ({y_test.mean()*100:.1f}%)")
        logger.info(f"   Races: {test_df['race_id'].nunique():,}")

        # Check for data leakage
        train_dates = set(train_df['date'])
        test_dates = set(test_df['date'])
        overlap = train_dates & test_dates

        if overlap:
            logger.warning(f"\n⚠️  WARNING: Date overlap detected: {overlap}")
            logger.warning(f"   This could indicate data leakage!")
        else:
            logger.info(f"\n✓ No date overlap - temporal split is valid")

        return X_train, X_test, y_train, y_test, train_df, test_df

    def apply_smote(self, X_train, y_train):
        """Apply SMOTE for class balancing"""
        logger.info(f"\n{'='*80}")
        logger.info("APPLYING SMOTE (Class Balancing)")
        logger.info(f"{'='*80}")
        logger.info(f"Before SMOTE:")
        logger.info(f"  Total samples: {len(y_train):,}")
        logger.info(f"  Winners: {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
        logger.info(f"  Non-winners: {(~y_train.astype(bool)).sum():,} ({(1-y_train.mean())*100:.1f}%)")

        # Handle NaN and infinity values
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_train = X_train.fillna(X_train.median()).fillna(0)

        smote = SMOTE(random_state=self.random_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        logger.info(f"\nAfter SMOTE:")
        logger.info(f"  Total samples: {len(y_train_balanced):,}")
        logger.info(f"  Winners: {y_train_balanced.sum():,} ({y_train_balanced.mean()*100:.1f}%)")
        logger.info(f"  Non-winners: {(~y_train_balanced.astype(bool)).sum():,} ({(1-y_train_balanced.mean())*100:.1f}%)")

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
        logger.info(f"Parameters:")
        for k, v in default_params.items():
            logger.info(f"  {k}: {v}")

        self.model = RandomForestClassifier(**default_params)
        self.model.fit(X_train, y_train)

        logger.info("\n✓ Training complete!")

        return self.model

    def evaluate_model(self, X_test, y_test, test_df):
        """Evaluate model on held-out test set"""
        logger.info(f"\n{'='*80}")
        logger.info("MODEL EVALUATION - HELD-OUT TEST SET")
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

        logger.info(f"\n📊 Horse-Level Metrics:")
        logger.info(f"   Accuracy: {accuracy:.4f}")
        logger.info(f"   ROC-AUC:  {roc_auc:.4f}")

        # Classification report
        logger.info(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Not Winner', 'Winner']))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"\n📉 Confusion Matrix:")
        logger.info(f"   True Negatives:  {cm[0, 0]:,} (correctly predicted non-winners)")
        logger.info(f"   False Positives: {cm[0, 1]:,} (predicted winner but lost)")
        logger.info(f"   False Negatives: {cm[1, 0]:,} (predicted loser but won)")
        logger.info(f"   True Positives:  {cm[1, 1]:,} (correctly predicted winners)")

        # Per-race analysis (MOST IMPORTANT FOR BETTING)
        logger.info(f"\n{'='*80}")
        logger.info("PER-RACE PREDICTION ACCURACY")
        logger.info("(This is the REALISTIC win rate for betting)")
        logger.info(f"{'='*80}")

        test_df = test_df.copy()
        test_df['predicted_prob'] = y_pred_proba

        # Group by race and find top prediction
        races_correct = 0
        total_races = 0
        top_picks = []

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

            top_picks.append({
                'date': date,
                'track': track,
                'race_number': race_num,
                'predicted_winner': predicted_winner.get('horse_name', 'Unknown'),
                'predicted_prob': predicted_winner['predicted_prob'],
                'was_correct': predicted_winner[self.target] == 1
            })

            if predicted_winner[self.target] == 1:
                races_correct += 1

            total_races += 1

        per_race_accuracy = races_correct / total_races if total_races > 0 else 0

        logger.info(f"\n🎯 BETTING PERFORMANCE:")
        logger.info(f"   Races with correct winner: {races_correct}/{total_races} ({per_race_accuracy*100:.1f}%)")
        logger.info(f"   Expected win rate: {per_race_accuracy*100:.1f}%")
        logger.info(f"   This is what you'd achieve betting on the model's top pick each race")

        # Analyze by probability threshold
        logger.info(f"\n📊 Performance by Confidence Level:")
        top_picks_df = pd.DataFrame(top_picks)

        for threshold in [0.2, 0.25, 0.3, 0.35, 0.4]:
            high_conf = top_picks_df[top_picks_df['predicted_prob'] >= threshold]
            if len(high_conf) > 0:
                win_rate = high_conf['was_correct'].mean()
                logger.info(f"   Prob >= {threshold:.2f}: {len(high_conf):4d} races, {win_rate*100:5.1f}% win rate")

        metrics = {
            'accuracy': float(accuracy),
            'roc_auc': float(roc_auc),
            'per_race_accuracy': float(per_race_accuracy),
            'races_correct': int(races_correct),
            'total_races': int(total_races),
            'confusion_matrix': cm.tolist()
        }

        return metrics, top_picks_df

    def show_feature_importance(self, top_n=25):
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
            logger.info(f"  {row['feature']:40s}: {row['importance']:.4f}")

        return feature_imp_df

    def save_model(self, model_path='vgame_rf_model.pkl', metadata_path='vgame_rf_metadata.json'):
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
    """Main training pipeline"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Train V-game prediction model with proper train/test split',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use existing temporal data
  %(prog)s --data temporal_processed_data.csv --train-end 2025-10-31

  # Use V-game tagged data
  %(prog)s --data vgame_tagged_data.csv --train-end 2025-11-30

  # Custom model parameters
  %(prog)s --data temporal_processed_data.csv --train-end 2025-10-31 --estimators 200 --depth 25
        """
    )

    parser.add_argument('--data', required=True, help='Input CSV file with race data')
    parser.add_argument('--train-end', required=True, help='Last training date (YYYY-MM-DD)')
    parser.add_argument('--test-start', help='First test date (YYYY-MM-DD, default: day after train-end)')
    parser.add_argument('--estimators', type=int, default=100, help='Number of trees (default: 100)')
    parser.add_argument('--depth', type=int, default=20, help='Max tree depth (default: 20)')
    parser.add_argument('--no-smote', action='store_true', help='Skip SMOTE balancing')
    parser.add_argument('--output', default='vgame_rf_model.pkl', help='Output model file')

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("V-GAME MODEL TRAINING - TEMPORAL VALIDATION")
    logger.info("="*80)

    # Initialize trainer
    trainer = VGameModelTrainer(target='is_winner', random_state=42)

    # Load data
    df = trainer.load_and_prepare_data(args.data)

    # Prepare features
    trainer.prepare_features(df)

    # Temporal split
    X_train, X_test, y_train, y_test, train_df, test_df = trainer.temporal_train_test_split(
        df,
        train_end_date=args.train_end,
        test_start_date=args.test_start
    )

    # Apply SMOTE (optional)
    if not args.no_smote:
        X_train_balanced, y_train_balanced = trainer.apply_smote(X_train, y_train)
    else:
        X_train_balanced, y_train_balanced = X_train, y_train
        logger.info("\n⚠️  Skipping SMOTE - training on imbalanced data")

    # Train model
    model = trainer.train_model(
        X_train_balanced,
        y_train_balanced,
        n_estimators=args.estimators,
        max_depth=args.depth
    )

    # Evaluate on test set
    metrics, top_picks = trainer.evaluate_model(X_test, y_test, test_df)

    # Show feature importance
    feature_importance = trainer.show_feature_importance(top_n=25)

    # Save top picks for analysis
    top_picks_file = args.output.replace('.pkl', '_test_predictions.csv')
    top_picks.to_csv(top_picks_file, index=False)
    logger.info(f"\n✓ Test predictions saved to {top_picks_file}")

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

    metadata_file = args.output.replace('.pkl', '_metadata.json')
    trainer.save_model(model_path=args.output, metadata_path=metadata_file)

    logger.info(f"\n{'='*80}")
    logger.info("✓ TRAINING COMPLETE!")
    logger.info(f"{'='*80}")
    logger.info(f"\nModel Performance:")
    logger.info(f"  Per-race win rate: {metrics['per_race_accuracy']*100:.1f}%")
    logger.info(f"  ROC-AUC: {metrics['roc_auc']:.3f}")
    logger.info(f"\nThis model was trained on past data and tested on future data.")
    logger.info(f"The win rate above is what you can realistically expect when betting.")


if __name__ == "__main__":
    main()
