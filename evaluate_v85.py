#!/usr/bin/env python3
"""
Complete V85 evaluation: Generate predictions and validate against results
"""

import sys
from predict_v85 import V85Predictor
from validate_predictions import validate_v85_predictions
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_v85(date='2026-01-10'):
    """
    Generate predictions and validate against results

    Args:
        date: Date to evaluate (YYYY-MM-DD)
    """

    # Generate predictions
    logger.info(f"Generating V85 predictions for {date}")
    predictor = V85Predictor()

    predictions_df = predictor.predict_v85(date)

    if not predictions_df:
        logger.error("Failed to generate predictions")
        return

    # Display predictions
    predictor.display_v85_predictions(predictions_df, show_top_n=5)

    # Convert predictions DataFrame to dict format for validation
    predictions_dict = {}
    for v85_race_num in sorted(predictions_df.keys()):
        race_pred = predictions_df[v85_race_num]
        top_pick = race_pred.iloc[0]

        predictions_dict[v85_race_num] = {
            'horse': top_pick['horse_name'],
            'number': int(top_pick['start_number']),
            'probability': top_pick['win_probability'] * 100
        }

    # Validate predictions
    logger.info("\n\nValidating predictions against actual results...")
    validate_v85_predictions(date=date, predictions=predictions_dict)


if __name__ == "__main__":
    date = sys.argv[1] if len(sys.argv) > 1 else '2026-01-10'
    evaluate_v85(date)
