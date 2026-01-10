#!/usr/bin/env python3
"""
Random Forest Model Training for Swedish Trotting Predictions
"""

import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support
)
from imblearn.over_sampling import SMOTE
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TrottingRFTrainer:
    """Random Forest trainer for trotting race predictions"""

    def __init__(self, target='is_winner', test_size=0.2, random_state=42):
        """
        Initialize trainer

        Args:
            target: Target variable ('is_winner', 'is_top3', or 'is_placed')
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
        """
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.feature_cols = None
        self.metadata = {}

    def load_data(self, filepath: str) -> tuple:
        """Load processed data"""
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)

        # Define feature columns (exclude targets and metadata)
        exclude_cols = [
            'race_id', 'race_date', 'race_number', 'start_time',
            'horse_id', 'horse_name', 'horse_color',
            'driver_id', 'driver_first_name', 'driver_last_name', 'driver_short_name',
            'trainer_id', 'trainer_first_name', 'trainer_last_name', 'trainer_short_name',
            'track_name', 'track_encoded', 'start_method', 'track_condition',
            'is_winner', 'is_top3', 'is_placed', 'finish_position',
            'finish_place', 'finish_order', 'finish_time', 'finish_speed_kmmin',
            'prize_money', 'final_odds', 'record_code', 'driver_license', 'trainer_license',
            'shoes_front', 'shoes_back', 'sulky_type',
        ]

        # Get numeric feature columns
        all_cols = set(df.columns)
        potential_features = all_cols - set(exclude_cols)

        # Keep only numeric columns
        numeric_features = []
        for col in sorted(potential_features):
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_features.append(col)

        self.feature_cols = numeric_features

        # Prepare X and y
        X = df[self.feature_cols]
        y = df[self.target]

        # Store metadata
        self.metadata = {
            'total_samples': len(df),
            'num_features': len(self.feature_cols),
            'target': self.target,
            'positive_class': int(y.sum()),
            'negative_class': int(len(y) - y.sum()),
            'class_ratio': float(y.mean()),
        }

        logger.info(f"Loaded {len(df)} samples with {len(self.feature_cols)} features")
        logger.info(f"Target '{self.target}': {y.sum()} positive ({y.mean()*100:.1f}%), "
                   f"{len(y) - y.sum()} negative ({(1-y.mean())*100:.1f}%)")

        return X, y, df

    def split_data(self, X, y):
        """Split data into train/test sets"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")

        return X_train, X_test, y_train, y_test

    def apply_smote(self, X_train, y_train):
        """Apply SMOTE for class balancing"""
        logger.info("Applying SMOTE for class balancing...")

        smote = SMOTE(random_state=self.random_state)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        logger.info(f"After SMOTE: {len(X_train_balanced)} samples")
        logger.info(f"  Positive class: {y_train_balanced.sum()} ({y_train_balanced.mean()*100:.1f}%)")
        logger.info(f"  Negative class: {len(y_train_balanced) - y_train_balanced.sum()} "
                   f"({(1-y_train_balanced.mean())*100:.1f}%)")

        return X_train_balanced, y_train_balanced

    def train_model(self, X_train, y_train, **rf_params):
        """
        Train Random Forest model

        Args:
            X_train: Training features
            y_train: Training labels
            **rf_params: Additional parameters for RandomForestClassifier
        """
        # Default RF parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
            'random_state': self.random_state,
            'n_jobs': -1,
            'class_weight': 'balanced'
        }

        # Merge with user params
        default_params.update(rf_params)

        logger.info(f"Training Random Forest with params: {default_params}")

        self.model = RandomForestClassifier(**default_params)
        self.model.fit(X_train, y_train)

        logger.info("Training complete!")

        return self.model

    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        logger.info("\n" + "="*80)
        logger.info("MODEL EVALUATION")
        logger.info("="*80)

        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        # Basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        logger.info(f"\nAccuracy: {accuracy:.4f}")
        logger.info(f"ROC-AUC Score: {roc_auc:.4f}")

        # Classification report
        logger.info("\nClassification Report:")
        print(classification_report(y_test, y_pred,
                                   target_names=['Not Winner', 'Winner']))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"\nConfusion Matrix:")
        logger.info(f"  True Negatives:  {cm[0, 0]}")
        logger.info(f"  False Positives: {cm[0, 1]}")
        logger.info(f"  False Negatives: {cm[1, 0]}")
        logger.info(f"  True Positives:  {cm[1, 1]}")

        # Store metrics
        metrics = {
            'accuracy': float(accuracy),
            'roc_auc': float(roc_auc),
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

        return metrics

    def show_feature_importance(self, top_n=20):
        """Display feature importance"""
        logger.info("\n" + "="*80)
        logger.info(f"TOP {top_n} MOST IMPORTANT FEATURES")
        logger.info("="*80)

        # Get feature importance
        importances = self.model.feature_importances_
        feature_imp_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)

        # Display top features
        for idx, row in feature_imp_df.head(top_n).iterrows():
            logger.info(f"  {row['feature']:30s}: {row['importance']:.4f}")

        return feature_imp_df

    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation"""
        logger.info(f"\nPerforming {cv}-fold cross-validation...")

        scores = cross_val_score(self.model, X, y, cv=cv, scoring='roc_auc')

        logger.info(f"Cross-validation ROC-AUC scores: {scores}")
        logger.info(f"Mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

        return scores

    def save_model(self, model_path='trotting_rf_model.pkl',
                   metadata_path='trotting_rf_metadata.json'):
        """Save trained model and metadata"""
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"Model saved to {model_path}")

        # Save metadata
        metadata = {
            'model_type': 'RandomForestClassifier',
            'target': self.target,
            'feature_columns': self.feature_cols,
            'num_features': len(self.feature_cols),
            'training_date': datetime.now().isoformat(),
            'random_state': self.random_state,
            **self.metadata
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to {metadata_path}")

    def predict_race(self, df_race):
        """
        Predict winners for a specific race

        Args:
            df_race: DataFrame with horses in the race (same format as training data)

        Returns:
            DataFrame with predictions and probabilities
        """
        X = df_race[self.feature_cols]

        # Predict probabilities
        probabilities = self.model.predict_proba(X)[:, 1]
        predictions = self.model.predict(X)

        # Create results DataFrame
        results = df_race[['horse_name', 'post_position']].copy()
        results['win_probability'] = probabilities
        results['predicted_winner'] = predictions
        results = results.sort_values('win_probability', ascending=False)

        return results


def main():
    """Main training pipeline"""

    # Initialize trainer
    logger.info("="*80)
    logger.info("SWEDISH TROTTING RACE PREDICTION - RF MODEL TRAINING")
    logger.info("="*80)

    # You can change target here: 'is_winner', 'is_top3', or 'is_placed'
    trainer = TrottingRFTrainer(target='is_winner', test_size=0.2, random_state=42)

    # Load data
    X, y, df = trainer.load_data('processed_ml_data.csv')

    # Split data
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)

    # Apply SMOTE
    X_train_balanced, y_train_balanced = trainer.apply_smote(X_train, y_train)

    # Train model
    model = trainer.train_model(
        X_train_balanced,
        y_train_balanced,
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
    )

    # Evaluate
    metrics = trainer.evaluate_model(X_test, y_test)

    # Show feature importance
    feature_importance = trainer.show_feature_importance(top_n=20)

    # Cross-validation (optional - takes time)
    # trainer.cross_validate(X_train_balanced, y_train_balanced, cv=5)

    # Save model
    trainer.save_model()

    # Example prediction
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE PREDICTION - First Race in Test Set")
    logger.info("="*80)

    # Get first race from test set
    test_indices = X_test.index[:12]  # First 12 horses (typical race size)
    df_sample_race = df.loc[test_indices]

    if len(df_sample_race) > 0:
        predictions = trainer.predict_race(df_sample_race)
        print(predictions.to_string(index=False))

    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
