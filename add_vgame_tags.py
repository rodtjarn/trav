#!/usr/bin/env python3
"""
Add V-game tags to existing temporal processed data
"""

import pandas as pd
from datetime import datetime
from atg_api_scraper import ATGAPIScraper
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def add_vgame_tags(input_file, output_file, delay=0.5):
    """
    Add V-game tags to existing processed data

    Args:
        input_file: Input CSV file (already processed)
        output_file: Output CSV file (with V-game tags)
        delay: Delay between API requests
    """
    logger.info("="*80)
    logger.info("ADDING V-GAME TAGS TO PROCESSED DATA")
    logger.info("="*80)

    # Load existing data
    logger.info(f"Loading {input_file}...")
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df):,} rows")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Get unique dates
    unique_dates = sorted(df['date'].unique())
    logger.info(f"Unique dates: {len(unique_dates)}")

    # Initialize scraper
    scraper = ATGAPIScraper(delay=delay)

    # Build race_id to V-game mapping
    race_game_mapping = {}
    game_types = ['V75', 'V86', 'V85', 'V65', 'V64', 'V5', 'V4', 'V3', 'GS75']

    logger.info("\nScanning for V-games...")

    for i, date_str in enumerate(unique_dates, 1):
        if i % 10 == 0:
            logger.info(f"  Progress: {i}/{len(unique_dates)} dates")

        try:
            calendar = scraper.get_calendar_for_date(date_str)

            if not calendar:
                continue

            games = calendar.get('games', {})
            for game_type in game_types:
                game_list = games.get(game_type, [])
                for game in game_list:
                    race_ids = game.get('races', [])
                    for race_id in race_ids:
                        race_game_mapping[race_id] = game_type

        except Exception as e:
            logger.warning(f"Error processing {date_str}: {e}")
            continue

    logger.info(f"\nFound {len(race_game_mapping)} V-game races")

    # Count by type
    from collections import Counter
    game_counts = Counter(race_game_mapping.values())
    logger.info("\nV-game breakdown:")
    for game_type, count in sorted(game_counts.items()):
        logger.info(f"  {game_type}: {count} races")

    # Add V-game tags to dataframe
    logger.info("\nAdding tags to dataframe...")
    df['vgame_type'] = df['race_id'].map(race_game_mapping)
    df['is_vgame'] = df['vgame_type'].notna()

    vgame_count = df['is_vgame'].sum()
    logger.info(f"Tagged {vgame_count:,} V-game starts ({vgame_count/len(df)*100:.1f}%)")

    # Save tagged data
    logger.info(f"\nSaving to {output_file}...")
    df.to_csv(output_file, index=False)

    logger.info("="*80)
    logger.info("✓ COMPLETE")
    logger.info("="*80)
    logger.info(f"Output: {output_file}")
    logger.info(f"Total rows: {len(df):,}")
    logger.info(f"V-game rows: {vgame_count:,}")

    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Add V-game tags to processed data')
    parser.add_argument('--input', default='temporal_processed_data.csv', help='Input CSV file')
    parser.add_argument('--output', default='temporal_processed_data_vgame.csv', help='Output CSV file')
    parser.add_argument('--delay', type=float, default=0.5, help='API delay')

    args = parser.parse_args()

    add_vgame_tags(args.input, args.output, args.delay)
