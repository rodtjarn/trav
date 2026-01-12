#!/usr/bin/env python3
"""
Backtest individual betting strategy on a specific date

Simple wrapper around batch_backtest_individual for single-date testing
"""

import sys
import argparse
from batch_backtest_individual import backtest_individual_betting
from atg_api_scraper import ATGAPIScraper
from temporal_data_processor import TemporalDataProcessor
import pickle
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backtest individual betting on a specific date')
    parser.add_argument('--date', required=True, help='Date (YYYY-MM-DD)')
    parser.add_argument('--budget', type=int, default=500, help='Budget in SEK (default: 500)')
    parser.add_argument('--top-n', type=int, default=10, help='Number of top bets (default: 10)')

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("INDIVIDUAL BETTING BACKTEST")
    logger.info("="*80)
    logger.info(f"Date: {args.date}")
    logger.info(f"Budget: {args.budget} SEK")
    logger.info(f"Top bets: {args.top_n}")
    logger.info("")

    # Load model
    logger.info("Loading prediction model...")
    with open('temporal_rf_model.pkl', 'rb') as f:
        model = pickle.load(f)

    feature_cols = list(model.feature_names_in_)

    scraper = ATGAPIScraper()
    processor = TemporalDataProcessor()

    # Run backtest
    result = backtest_individual_betting(
        args.date, model, feature_cols, scraper, processor, args.budget, args.top_n
    )

    if result:
        logger.info("\n✅ Backtest complete!")
        if result['profit'] > 0:
            logger.info(f"🎉 PROFITABLE: +{result['profit']:.0f} SEK ({result['roi']:+.1f}% ROI)")
        else:
            logger.info(f"📉 Loss: {result['profit']:.0f} SEK ({result['roi']:.1f}% ROI)")
    else:
        logger.error("\n❌ Backtest failed!")
