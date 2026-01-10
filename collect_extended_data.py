#!/usr/bin/env python3
"""
Extended Data Collection for Swedish Trotting
Collects 30-60 days of historical data for better model training
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


def collect_extended_data(days_back=30, delay=0.5, output_file='atg_extended_data.csv'):
    """
    Collect extended historical data

    Args:
        days_back: Number of days to go back in history
        delay: Delay between API requests (seconds)
        output_file: Output CSV file
    """
    scraper = ATGAPIScraper(delay=delay)

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    logger.info("="*80)
    logger.info("EXTENDED DATA COLLECTION")
    logger.info("="*80)
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Total days: {days_back}")
    logger.info(f"Delay between requests: {delay}s")
    logger.info("="*80)

    all_data = []
    stats = {
        'days_processed': 0,
        'total_races': 0,
        'total_starts': 0,
        'completed_races': 0,
        'swedish_tracks': 0,
        'major_tracks': 0,
    }

    major_tracks = ['Solvalla', 'Jägersro', 'Åby', 'Gävle', 'Eskilstuna',
                    'Bergsåker', 'Färjestad', 'Bollnäs', 'Boden', 'Axevalla']

    # Process each day
    for day_offset in range(days_back):
        current_date = (start_date + timedelta(days=day_offset)).strftime('%Y-%m-%d')

        logger.info(f"\n{'='*60}")
        logger.info(f"Day {day_offset + 1}/{days_back}: {current_date}")
        logger.info(f"{'='*60}")

        try:
            # Scrape data for this day
            day_data = scraper.scrape_date(current_date)

            if day_data:
                # Count statistics
                df_day = pd.DataFrame(day_data)

                completed = df_day['finish_place'].notna().sum()
                swedish = df_day['track_country'].eq('SE').sum()
                major = df_day['track_name'].isin(major_tracks).sum()

                stats['days_processed'] += 1
                stats['total_races'] += df_day['race_id'].nunique()
                stats['total_starts'] += len(df_day)
                stats['completed_races'] += df_day[df_day['finish_place'].notna()]['race_id'].nunique()
                stats['swedish_tracks'] += swedish
                stats['major_tracks'] += major

                all_data.extend(day_data)

                logger.info(f"Collected: {len(day_data)} starts from {df_day['race_id'].nunique()} races")
                logger.info(f"  Completed: {completed} / {len(df_day)}")
                logger.info(f"  Swedish tracks: {swedish}")
                logger.info(f"  Major tracks: {major}")

                # Show top tracks for this day
                track_counts = df_day['track_name'].value_counts().head(3)
                logger.info(f"  Top tracks: {', '.join([f'{t}({c})' for t, c in track_counts.items()])}")

            else:
                logger.warning(f"No data collected for {current_date}")

            # Save progress every 5 days
            if (day_offset + 1) % 5 == 0 and all_data:
                temp_file = f"{output_file}.temp"
                pd.DataFrame(all_data).to_csv(temp_file, index=False)
                logger.info(f"\n💾 Progress saved to {temp_file}")
                logger.info(f"   Total collected so far: {len(all_data)} starts")

        except KeyboardInterrupt:
            logger.warning("\n⚠️  Collection interrupted by user!")
            logger.info("Saving collected data before exit...")
            break
        except Exception as e:
            logger.error(f"Error processing {current_date}: {e}")
            continue

    # Final save
    if all_data:
        df_final = pd.DataFrame(all_data)
        df_final.to_csv(output_file, index=False)

        logger.info("\n" + "="*80)
        logger.info("COLLECTION COMPLETE!")
        logger.info("="*80)
        logger.info(f"\n📊 FINAL STATISTICS:")
        logger.info(f"  Days processed: {stats['days_processed']} / {days_back}")
        logger.info(f"  Total races: {stats['total_races']}")
        logger.info(f"  Total starts: {stats['total_starts']}")
        logger.info(f"  Completed races: {stats['completed_races']}")
        logger.info(f"  Swedish track starts: {stats['swedish_tracks']} ({stats['swedish_tracks']/stats['total_starts']*100:.1f}%)")
        logger.info(f"  Major track starts: {stats['major_tracks']} ({stats['major_tracks']/stats['total_starts']*100:.1f}%)")

        # Show track distribution
        logger.info(f"\n📍 TRACK DISTRIBUTION (Top 15):")
        track_dist = df_final['track_name'].value_counts().head(15)
        for track, count in track_dist.items():
            marker = '⭐' if track in major_tracks else ''
            logger.info(f"  {track:20s}: {count:4d} starts {marker}")

        # Show date coverage
        logger.info(f"\n📅 DATE COVERAGE:")
        logger.info(f"  Start: {df_final['race_date'].min()}")
        logger.info(f"  End: {df_final['race_date'].max()}")
        logger.info(f"  Unique dates: {df_final['race_date'].nunique()}")

        # Show completion stats
        completed_df = df_final[df_final['finish_place'].notna()]
        logger.info(f"\n✅ COMPLETED RACES (with results):")
        logger.info(f"  Total: {len(completed_df)} starts")
        logger.info(f"  Winners: {completed_df['finish_place'].eq(1).sum()} ({completed_df['finish_place'].eq(1).sum()/len(completed_df)*100:.1f}%)")
        logger.info(f"  Galloped: {completed_df['galloped'].sum()} ({completed_df['galloped'].sum()/len(completed_df)*100:.1f}%)")

        logger.info(f"\n💾 Saved to: {output_file}")
        logger.info(f"   File size: {len(df_final):,} rows × {len(df_final.columns)} columns")

        return df_final
    else:
        logger.error("No data collected!")
        return None


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description='Collect extended historical trotting data')
    parser.add_argument('--days', type=int, default=30,
                       help='Number of days to collect (default: 30)')
    parser.add_argument('--delay', type=float, default=0.5,
                       help='Delay between API requests in seconds (default: 0.5)')
    parser.add_argument('--output', type=str, default='atg_extended_data.csv',
                       help='Output CSV file (default: atg_extended_data.csv)')

    args = parser.parse_args()

    # Collect data
    df = collect_extended_data(
        days_back=args.days,
        delay=args.delay,
        output_file=args.output
    )

    if df is not None:
        logger.info("\n✨ Collection successful! Ready for processing.")
        logger.info(f"\nNext steps:")
        logger.info(f"  1. Process the data: python api_data_processor.py")
        logger.info(f"  2. Retrain the model: python train_rf_model.py")
    else:
        logger.error("\n❌ Collection failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
