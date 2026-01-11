#!/usr/bin/env python3
"""
Collect V-Game Tagged Data for Training
Tags races with their V-game type (V75, V86, V65, etc.)
"""

import pandas as pd
from datetime import datetime, timedelta
from atg_api_scraper import ATGAPIScraper
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def collect_vgame_data(start_date_str, end_date_str, delay=0.5, output_file='vgame_tagged_data.csv'):
    """
    Collect data with V-game tags

    Args:
        start_date_str: Start date (YYYY-MM-DD)
        end_date_str: End date (YYYY-MM-DD)
        delay: Delay between API requests (seconds)
        output_file: Output CSV file
    """
    scraper = ATGAPIScraper(delay=delay)

    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    total_days = (end_date - start_date).days + 1

    logger.info("="*80)
    logger.info("V-GAME TAGGED DATA COLLECTION")
    logger.info("="*80)
    logger.info(f"Date range: {start_date_str} to {end_date_str}")
    logger.info(f"Total days: {total_days}")
    logger.info(f"Delay between requests: {delay}s")
    logger.info("="*80)

    all_data = []
    stats = {
        'days_processed': 0,
        'total_races': 0,
        'total_starts': 0,
        'vgame_races': 0,
        'vgame_types': {}
    }

    # Game types to check
    game_types = ['V75', 'V86', 'V85', 'V65', 'V64', 'V5', 'V4', 'V3', 'GS75']

    # Process each day
    current_date = start_date
    day_num = 0

    while current_date <= end_date:
        day_num += 1
        date_str = current_date.strftime('%Y-%m-%d')

        logger.info(f"\n{'='*60}")
        logger.info(f"Day {day_num}/{total_days}: {date_str}")
        logger.info(f"{'='*60}")

        try:
            # Get calendar to find V-games for this date
            calendar = scraper.get_calendar_for_date(date_str)

            if not calendar:
                logger.warning(f"No calendar data for {date_str}")
                current_date += timedelta(days=1)
                continue

            # Build mapping of race_id -> game_type
            race_game_mapping = {}

            games = calendar.get('games', {})
            for game_type in game_types:
                game_list = games.get(game_type, [])
                for game in game_list:
                    race_ids = game.get('races', [])
                    for race_id in race_ids:
                        race_game_mapping[race_id] = game_type
                        stats['vgame_types'][game_type] = stats['vgame_types'].get(game_type, 0) + 1

            if race_game_mapping:
                logger.info(f"  Found V-games: {', '.join(set(race_game_mapping.values()))}")

            # Scrape all race data for this day
            day_data = scraper.scrape_date(date_str)

            if day_data:
                # Add V-game tags
                for start_data in day_data:
                    race_id = start_data.get('race_id')
                    if race_id in race_game_mapping:
                        start_data['vgame_type'] = race_game_mapping[race_id]
                        start_data['is_vgame'] = True
                        stats['vgame_races'] += 1
                    else:
                        start_data['vgame_type'] = None
                        start_data['is_vgame'] = False

                stats['days_processed'] += 1
                stats['total_starts'] += len(day_data)

                # Count unique races
                df_day = pd.DataFrame(day_data)
                stats['total_races'] += df_day['race_id'].nunique()

                all_data.extend(day_data)

                logger.info(f"  Collected: {len(day_data)} starts from {df_day['race_id'].nunique()} races")
                logger.info(f"  V-game starts: {df_day['is_vgame'].sum()}")

                # Show V-game breakdown
                if df_day['is_vgame'].sum() > 0:
                    vgame_counts = df_day[df_day['is_vgame']]['vgame_type'].value_counts()
                    logger.info(f"  V-game breakdown: {', '.join([f'{t}({c})' for t, c in vgame_counts.items()])}")

            else:
                logger.warning(f"  No data collected for {date_str}")

            # Save progress every 7 days
            if day_num % 7 == 0 and all_data:
                temp_file = f"{output_file}.progress"
                pd.DataFrame(all_data).to_csv(temp_file, index=False)
                logger.info(f"\n  💾 Progress saved to {temp_file}")
                logger.info(f"  Total collected so far: {len(all_data):,} starts")

        except Exception as e:
            logger.error(f"Error processing {date_str}: {e}")

        current_date += timedelta(days=1)

    # Save final data
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(output_file, index=False)

        logger.info("\n" + "="*80)
        logger.info("COLLECTION COMPLETE")
        logger.info("="*80)
        logger.info(f"✓ Saved to {output_file}")
        logger.info(f"\nStatistics:")
        logger.info(f"  Days processed: {stats['days_processed']}")
        logger.info(f"  Total races: {stats['total_races']}")
        logger.info(f"  Total starts: {stats['total_starts']:,}")
        logger.info(f"  V-game starts: {stats['vgame_races']:,} ({stats['vgame_races']/stats['total_starts']*100:.1f}%)")
        logger.info(f"\nV-Game breakdown:")
        for game_type, count in sorted(stats['vgame_types'].items()):
            logger.info(f"  {game_type}: {count} races")

        # Show data quality
        logger.info(f"\nData quality:")
        logger.info(f"  Completed races (have results): {df['finish_place'].notna().sum():,}")
        logger.info(f"  Winners identified: {df[df['finish_place'] == 1].shape[0]:,}")
        logger.info(f"  Swedish tracks: {df[df['track_country'] == 'SE'].shape[0]:,}")

        return df
    else:
        logger.error("No data collected!")
        return None


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description='Collect V-game tagged training data')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output', default='vgame_tagged_data.csv', help='Output file')
    parser.add_argument('--delay', type=float, default=0.5, help='Delay between requests (seconds)')

    args = parser.parse_args()

    # Collect data
    collect_vgame_data(
        start_date_str=args.start,
        end_date_str=args.end,
        delay=args.delay,
        output_file=args.output
    )


if __name__ == '__main__':
    main()
