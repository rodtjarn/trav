#!/usr/bin/env python3
"""
Train meta-model for V-game system selection

This model learns to predict the optimal number of horses to select
in each race based on race features and prediction confidence.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_system_selection_model():
    """Train the meta-model"""

    logger.info("Loading training data...")
    df = pd.read_csv('system_selection_training_data.csv')

    logger.info(f"Loaded {len(df)} race records")
    logger.info(f"Target distribution:\n{df['optimal_picks'].value_counts().sort_index()}")

    # Features for the model
    feature_cols = [
        'top1_prob', 'top2_prob', 'top3_prob', 'top4_prob', 'top5_prob',
        'gap_1_2', 'gap_2_3', 'gap_3_4', 'gap_1_3',
        'mean_prob', 'std_prob', 'max_prob', 'min_prob',
        'top3_sum', 'entropy', 'num_horses',
        'horses_above_20', 'horses_above_15', 'horses_above_10'
    ]

    # Encode game_type
    game_type_dummies = pd.get_dummies(df['game_type'], prefix='game')
    df = pd.concat([df, game_type_dummies], axis=1)
    feature_cols.extend(game_type_dummies.columns.tolist())

    X = df[feature_cols]
    y = df['optimal_picks']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logger.info(f"Training set: {len(X_train)} races")
    logger.info(f"Test set: {len(X_test)} races")

    # Train Random Forest classifier
    logger.info("\nTraining Random Forest classifier...")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # Handle imbalanced classes
    )

    model.fit(X_train, y_train)

    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    logger.info(f"\nTraining accuracy: {train_score:.3f}")
    logger.info(f"Test accuracy: {test_score:.3f}")

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1)
    logger.info(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

    # Predictions
    y_pred = model.predict(X_test)

    # Classification report
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y_test, y_pred))

    # Confusion matrix
    logger.info("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"\n{cm}")

    # Feature importance
    logger.info("\nTop 15 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    for idx, row in feature_importance.head(15).iterrows():
        logger.info(f"  {row['feature']:20s}: {row['importance']:.4f}")

    # Save model
    model_file = 'system_selection_model.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_cols': feature_cols,
            'classes': model.classes_
        }, f)

    logger.info(f"\nModel saved to {model_file}")

    # Analyze prediction patterns
    logger.info("\nPrediction Analysis:")
    logger.info("-"*80)

    test_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred,
        'top1_prob': X_test['top1_prob'].values,
        'gap_1_2': X_test['gap_1_2'].values,
        'num_horses': X_test['num_horses'].values
    })

    # Group by predicted picks
    for picks in sorted(test_df['predicted'].unique()):
        subset = test_df[test_df['predicted'] == picks]
        logger.info(f"\nWhen model predicts {picks} pick(s) ({len(subset)} races):")
        logger.info(f"  Actual distribution: {subset['actual'].value_counts().sort_index().to_dict()}")
        logger.info(f"  Avg top1_prob: {subset['top1_prob'].mean():.3f}")
        logger.info(f"  Avg gap_1_2: {subset['gap_1_2'].mean():.3f}")
        logger.info(f"  Avg num_horses: {subset['num_horses'].mean():.1f}")

    return model, feature_cols


if __name__ == '__main__':
    logger.info("Starting system selection model training")
    logger.info("="*80)
    model, features = train_system_selection_model()
    logger.info("\nTraining complete!")
