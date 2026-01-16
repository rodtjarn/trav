#!/usr/bin/env python3
"""
Analyze what the model learned and identify potential issues
"""

import pickle
import pandas as pd
import numpy as np

# Load model
with open('temporal_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

print("=" * 80)
print("MODEL ANALYSIS")
print("=" * 80)
print()

# Feature importances
feature_names = model.feature_names_in_
importances = model.feature_importances_

# Sort by importance
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)

print("TOP 20 FEATURE IMPORTANCES:")
print("-" * 80)
for idx, row in feature_importance.head(20).iterrows():
    print(f"{row['feature']:40s} {row['importance']:.4f}")

print()
print("=" * 80)
print("MODEL PARAMETERS:")
print("-" * 80)
print(f"Number of estimators: {model.n_estimators}")
print(f"Max depth: {model.max_depth}")
print(f"Min samples split: {model.min_samples_split}")
print(f"Min samples leaf: {model.min_samples_leaf}")
print(f"Number of features: {len(feature_names)}")

print()
print("=" * 80)
print("CHECKING TRAINING DATA DISTRIBUTION")
print("-" * 80)

# Load training data to check target distribution
try:
    train_df = pd.read_csv('processed_trotting_data.csv')
    print(f"Total training samples: {len(train_df)}")
    print(f"Winners: {train_df['is_winner'].sum()} ({train_df['is_winner'].mean()*100:.1f}%)")
    print(f"Losers: {(~train_df['is_winner']).sum()} ({(~train_df['is_winner']).mean()*100:.1f}%)")

    # Check if there are temporal features that might cause overfitting
    if 'date' in train_df.columns:
        train_df['date'] = pd.to_datetime(train_df['date'])
        print(f"Date range: {train_df['date'].min()} to {train_df['date'].max()}")

        # Check year distribution
        train_df['year'] = train_df['date'].dt.year
        print("\nTraining data by year:")
        print(train_df['year'].value_counts().sort_index())

    # Check average predicted probability
    print("\n" + "=" * 80)
    print("PREDICTION PROBABILITY ANALYSIS")
    print("-" * 80)

    # Make predictions on training data
    X = train_df[feature_names]
    train_preds = model.predict_proba(X)[:, 1]

    print(f"Mean predicted probability: {train_preds.mean():.4f}")
    print(f"Median predicted probability: {np.median(train_preds):.4f}")
    print(f"Min predicted probability: {train_preds.min():.4f}")
    print(f"Max predicted probability: {train_preds.max():.4f}")

    # Compare winners vs losers
    winner_preds = train_preds[train_df['is_winner'] == True]
    loser_preds = train_preds[train_df['is_winner'] == False]

    print(f"\nWinners - Mean prob: {winner_preds.mean():.4f}, Median: {np.median(winner_preds):.4f}")
    print(f"Losers  - Mean prob: {loser_preds.mean():.4f}, Median: {np.median(loser_preds):.4f}")
    print(f"Separation: {winner_preds.mean() - loser_preds.mean():.4f}")

except Exception as e:
    print(f"Error analyzing training data: {e}")

print()
print("=" * 80)
print("POTENTIAL ISSUES TO CHECK:")
print("-" * 80)
print("1. Is the model overfitting to 2025-specific patterns?")
print("2. Are the estimated odds (1/prob) realistic? They should range from ~2-50")
print("3. Is the Kelly criterion bet sizing too aggressive?")
print("4. Do we have enough diversity in training data?")
print("5. Are there track-specific features that don't generalize?")
