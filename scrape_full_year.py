#!/usr/bin/env python3
"""
Scrape full year of trotting data for proper model training
"""

import pandas as pd
from atg_api_scraper import ATGAPIScraper
from datetime import datetime, timedelta
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def scrape_year(year=2025, save_frequency=30):
    """
    Scrape entire year of race data

    Args:
        year: Year to scrape
        save_frequency: Save progress every N days
    """
    scraper = ATGAPIScraper(delay=0.5)

    # Date range
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)

    total_days = (end_date - start_date).days + 1

    logger.info("="*80)
    logger.info(f"SCRAPING FULL YEAR {year}")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Total days: {total_days}")
    logger.info("="*80)

    all_races = []
    dates_scraped = []
    dates_with_data = []
    dates_failed = []

    for day_num in range(total_days):
        current_date = start_date + timedelta(days=day_num)
        date_str = current_date.strftime('%Y-%m-%d')

        # Progress
        if day_num % 10 == 0:
            logger.info(f"\nProgress: {day_num}/{total_days} days ({day_num/total_days*100:.1f}%)")
            logger.info(f"Scraped {len(all_races)} races so far from {len(dates_with_data)} days")

        try:
            # Scrape this date
            logger.info(f"Scraping {date_str}...")
            race_data = scraper.scrape_date(date_str)

            if race_data:
                all_races.extend(race_data)
                dates_with_data.append(date_str)
                logger.info(f"  ✓ Found {len(race_data)} races")
            else:
                logger.info(f"  - No races")

            dates_scraped.append(date_str)

            # Save progress periodically
            if (day_num + 1) % save_frequency == 0:
                save_progress(all_races, dates_scraped, dates_with_data, year)

        except Exception as e:
            logger.error(f"  ✗ Error scraping {date_str}: {e}")
            dates_failed.append(date_str)
            continue

    # Final save
    logger.info("\n" + "="*80)
    logger.info("SCRAPING COMPLETE - Saving final data")
    logger.info("="*80)

    save_progress(all_races, dates_scraped, dates_with_data, year, final=True)

    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Total days scraped: {len(dates_scraped)}/{total_days}")
    logger.info(f"Days with race data: {len(dates_with_data)}")
    logger.info(f"Days failed: {len(dates_failed)}")
    logger.info(f"Total races collected: {len(all_races)}")

    if dates_failed:
        logger.info(f"\nFailed dates: {dates_failed[:10]}..." if len(dates_failed) > 10 else f"\nFailed dates: {dates_failed}")

    # Stats
    if all_races:
        df = pd.DataFrame(all_races)
        logger.info(f"\nDataset stats:")
        logger.info(f"  Total horses: {len(df)}")
        logger.info(f"  Unique tracks: {df['track'].nunique()}")
        logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"  Races with results: {df['finish_place'].notna().sum()}")

        # Monthly breakdown
        logger.info(f"\nMonthly breakdown:")
        df['month'] = pd.to_datetime(df['date']).dt.month
        monthly = df.groupby('month').size()
        for month, count in monthly.items():
            logger.info(f"  {datetime(year, month, 1).strftime('%B'):>10}: {count:>5} horses")

    return all_races


def save_progress(all_races, dates_scraped, dates_with_data, year, final=False):
    """Save progress to files"""
    suffix = "_final" if final else "_progress"

    # Save races as CSV
    if all_races:
        df = pd.DataFrame(all_races)
        csv_file = f"trotting_data_{year}{suffix}.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"  Saved {len(df)} races to {csv_file}")

        # Also save as JSON
        json_file = f"trotting_data_{year}{suffix}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(all_races, f, ensure_ascii=False, indent=2)
        logger.info(f"  Saved JSON to {json_file}")

    # Save metadata
    metadata = {
        'year': year,
        'total_races': len(all_races),
        'dates_scraped': len(dates_scraped),
        'dates_with_data': len(dates_with_data),
        'last_scraped_date': dates_scraped[-1] if dates_scraped else None,
        'scrape_timestamp': datetime.now().isoformat(),
        'status': 'complete' if final else 'in_progress'
    }

    meta_file = f"trotting_data_{year}_metadata{suffix}.json"
    with open(meta_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  Saved metadata to {meta_file}")


def main():
    """Main scraping function"""

    # Scrape full year 2025
    all_races = scrape_year(year=2025, save_frequency=30)

    logger.info("\n" + "="*80)
    logger.info("✓ YEAR SCRAPING COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nCollected {len(all_races)} total races from 2025")
    logger.info("\nFiles created:")
    logger.info("  - trotting_data_2025_final.csv")
    logger.info("  - trotting_data_2025_final.json")
    logger.info("  - trotting_data_2025_metadata_final.json")
    logger.info("\nNext steps:")
    logger.info("  1. Process data with temporal awareness")
    logger.info("  2. Train model with proper temporal split")
    logger.info("  3. Validate on future data only")


if __name__ == "__main__":
    main()
