#!/usr/bin/env python3
"""
Modern ATG API Scraper
Uses ATG's JSON API instead of HTML scraping for reliable data collection
"""

import requests
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ATGAPIScraper:
    """Scraper using ATG's official JSON API"""

    BASE_URL = "https://www.atg.se/services/racinginfo/v1/api"

    def __init__(self, delay: float = 1.0):
        """
        Initialize scraper

        Args:
            delay: Delay between requests in seconds
        """
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def _sleep(self):
        """Polite delay between requests"""
        time.sleep(self.delay)

    def get_calendar_for_date(self, date: str) -> Dict:
        """
        Get race calendar for a specific date

        Args:
            date: Date string in YYYY-MM-DD format

        Returns:
            Calendar data with tracks and races
        """
        url = f"{self.BASE_URL}/calendar/day/{date}"
        logger.info(f"Fetching calendar for {date}")

        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            logger.info(f"Found {len(data.get('tracks', []))} tracks for {date}")
            return data
        except Exception as e:
            logger.error(f"Error fetching calendar for {date}: {e}")
            return {}

    def get_game_info(self, date: str, game_type: str) -> Optional[Dict]:
        """
        Get game information for a specific date and game type

        Args:
            date: Date string in YYYY-MM-DD format
            game_type: Game type (e.g., 'V85', 'V86', 'V75', 'V65', 'V64')

        Returns:
            Dict with game info including race_ids and mapping to race numbers
            Format: {
                'game_id': 'V85_...',
                'game_type': 'V85',
                'track_id': 23,
                'race_ids': ['2026-01-10_23_3', ...],
                'race_mapping': {'2026-01-10_23_3': 1, ...}  # race_id -> game race number
            }
        """
        calendar = self.get_calendar_for_date(date)

        if not calendar:
            return None

        # Check if there's a game of the specified type
        games = calendar.get('games', {})
        game_list = games.get(game_type.upper(), [])

        if not game_list:
            logger.info(f"No {game_type} game found for {date}")
            return None

        # Usually only one game of each type per day
        game = game_list[0]
        race_ids = game.get('races', [])

        # Create mapping: race_id -> game race number
        race_mapping = {race_id: i + 1 for i, race_id in enumerate(race_ids)}

        info = {
            'game_id': game.get('id'),
            'game_type': game_type.upper(),
            'status': game.get('status'),
            'track_id': game.get('tracks', [None])[0],
            'race_ids': race_ids,
            'race_mapping': race_mapping,
            'start_time': game.get('startTime'),
        }

        logger.info(f"Found {game_type} game: {info['game_id']} with {len(race_ids)} races")
        logger.info(f"{game_type} races: track races {[rid.split('_')[-1] for rid in race_ids]} → {game_type} races 1-{len(race_ids)}")

        return info

    def get_v85_info(self, date: str) -> Optional[Dict]:
        """
        Get V85 game information for a specific date

        Args:
            date: Date string in YYYY-MM-DD format

        Returns:
            Dict with V85 info including race_ids and mapping to V85 race numbers
            Format: {
                'game_id': 'V85_...',
                'track_id': 23,
                'race_ids': ['2026-01-10_23_3', ...],
                'v85_race_mapping': {'2026-01-10_23_3': 1, ...}  # race_id -> v85_number
            }
        """
        info = self.get_game_info(date, 'V85')
        if not info:
            return None

        # Convert to old format for backwards compatibility
        info['v85_race_mapping'] = info.pop('race_mapping')
        return info

    def get_race_details(self, race_id: str) -> Optional[Dict]:
        """
        Get detailed race information including horses, drivers, results

        Args:
            race_id: Race identifier (e.g., '2026-01-10_23_1')

        Returns:
            Detailed race data
        """
        url = f"{self.BASE_URL}/races/{race_id}"

        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            logger.debug(f"Fetched details for race {race_id}")
            return data
        except Exception as e:
            logger.error(f"Error fetching race {race_id}: {e}")
            return None

    def extract_horse_data(self, start: Dict, race_info: Dict) -> Dict:
        """
        Extract all relevant data for one horse/start

        Args:
            start: Start data from API
            race_info: Race metadata

        Returns:
            Flattened dictionary with all horse/race data
        """
        horse = start.get('horse', {})
        driver = start.get('driver', {})
        trainer = horse.get('trainer', {})
        result = start.get('result', {})
        track = race_info.get('track', {})

        # Extract record time
        record_time = horse.get('record', {}).get('time', {})
        record_seconds = (
            record_time.get('minutes', 0) * 60 +
            record_time.get('seconds', 0) +
            record_time.get('tenths', 0) / 10
        )

        # Extract finish time
        km_time = result.get('kmTime', {})
        finish_seconds = (
            km_time.get('minutes', 0) * 60 +
            km_time.get('seconds', 0) +
            km_time.get('tenths', 0) / 10
        )

        # Get driver statistics (use most recent year)
        driver_stats = driver.get('statistics', {}).get('years', {})
        driver_year = max(driver_stats.keys()) if driver_stats else None
        driver_year_stats = driver_stats.get(driver_year, {}) if driver_year else {}

        # Get trainer statistics
        trainer_stats = trainer.get('statistics', {}).get('years', {})
        trainer_year = max(trainer_stats.keys()) if trainer_stats else None
        trainer_year_stats = trainer_stats.get(trainer_year, {}) if trainer_year else {}

        # Build comprehensive data dict
        data = {
            # Race info
            'race_id': race_info.get('id'),
            'race_date': race_info.get('date'),
            'race_number': race_info.get('number'),
            'track_name': track.get('name'),
            'track_id': track.get('id'),
            'track_condition': track.get('condition'),
            'track_country': track.get('countryCode'),

            # Race parameters
            'distance': race_info.get('distance'),
            'start_method': race_info.get('startMethod'),  # auto/monte
            'start_time': race_info.get('startTime'),

            # Horse info
            'horse_id': horse.get('id'),
            'horse_name': horse.get('name'),
            'horse_age': horse.get('age'),
            'horse_sex': horse.get('sex'),
            'horse_money': horse.get('money'),  # career earnings
            'horse_color': horse.get('color'),

            # Start info
            'start_number': start.get('number'),
            'post_position': start.get('postPosition'),
            'distance_handicap': start.get('distance', 0) - race_info.get('distance', 0),

            # Record/performance
            'record_time': record_seconds,
            'record_code': horse.get('record', {}).get('code'),

            # Driver info
            'driver_id': driver.get('id'),
            'driver_first_name': driver.get('firstName'),
            'driver_last_name': driver.get('lastName'),
            'driver_short_name': driver.get('shortName'),
            'driver_license': driver.get('license'),

            # Driver statistics (last complete year)
            'driver_starts': driver_year_stats.get('starts', 0),
            'driver_earnings': driver_year_stats.get('earnings', 0),
            'driver_wins': driver_year_stats.get('placement', {}).get('1', 0),
            'driver_second': driver_year_stats.get('placement', {}).get('2', 0),
            'driver_third': driver_year_stats.get('placement', {}).get('3', 0),
            'driver_win_percentage': driver_year_stats.get('winPercentage', 0),

            # Trainer info
            'trainer_id': trainer.get('id'),
            'trainer_first_name': trainer.get('firstName'),
            'trainer_last_name': trainer.get('lastName'),
            'trainer_short_name': trainer.get('shortName'),
            'trainer_license': trainer.get('license'),

            # Trainer statistics
            'trainer_starts': trainer_year_stats.get('starts', 0),
            'trainer_earnings': trainer_year_stats.get('earnings', 0),
            'trainer_wins': trainer_year_stats.get('placement', {}).get('1', 0),
            'trainer_second': trainer_year_stats.get('placement', {}).get('2', 0),
            'trainer_third': trainer_year_stats.get('placement', {}).get('3', 0),
            'trainer_win_percentage': trainer_year_stats.get('winPercentage', 0),

            # Results (if race is completed)
            'finish_place': result.get('place', None),
            'finish_order': result.get('finishOrder', None),
            'finish_time': finish_seconds if finish_seconds > 0 else None,
            'galloped': result.get('galloped', False),
            'prize_money': result.get('prizeMoney', 0),
            'final_odds': result.get('finalOdds', None),

            # Equipment
            'shoes_front': horse.get('shoes', {}).get('front', {}).get('hasShoe'),
            'shoes_back': horse.get('shoes', {}).get('back', {}).get('hasShoe'),
            'sulky_type': horse.get('sulky', {}).get('type', {}).get('code'),
        }

        return data

    def scrape_date(self, date: str) -> List[Dict]:
        """
        Scrape all races for a specific date

        Args:
            date: Date string in YYYY-MM-DD format

        Returns:
            List of horse/start dictionaries
        """
        all_data = []

        # Get calendar
        calendar = self.get_calendar_for_date(date)
        self._sleep()

        if not calendar:
            return all_data

        # Process each track
        for track in calendar.get('tracks', []):
            track_name = track.get('name')
            logger.info(f"Processing track: {track_name}")

            # Process each race
            for race_summary in track.get('races', []):
                race_id = race_summary.get('id')
                race_status = race_summary.get('status')

                # Get detailed race data
                race_details = self.get_race_details(race_id)
                self._sleep()

                if not race_details:
                    continue

                # Extract data for each horse/start
                for start in race_details.get('starts', []):
                    horse_data = self.extract_horse_data(start, race_details)
                    all_data.append(horse_data)

                logger.info(f"  Race {race_id}: {len(race_details.get('starts', []))} starts ({race_status})")

        return all_data

    def scrape_date_range(self, start_date: str, days: int = 7) -> List[Dict]:
        """
        Scrape multiple days of race data

        Args:
            start_date: Starting date in YYYY-MM-DD format
            days: Number of days to scrape

        Returns:
            List of all horse/start dictionaries
        """
        all_data = []
        start = datetime.strptime(start_date, '%Y-%m-%d')

        for i in range(days):
            date = (start + timedelta(days=i)).strftime('%Y-%m-%d')
            logger.info(f"\n{'='*60}")
            logger.info(f"Scraping date: {date}")
            logger.info(f"{'='*60}")

            day_data = self.scrape_date(date)
            all_data.extend(day_data)

            logger.info(f"Total starts collected for {date}: {len(day_data)}")

        return all_data

    def save_to_csv(self, data: List[Dict], filename: str):
        """Save data to CSV"""
        if not data:
            logger.warning(f"No data to save to {filename}")
            return

        df = pd.DataFrame(data)
        df.to_csv(filename, index=False, encoding='utf-8')
        logger.info(f"Saved {len(data)} records to {filename}")

    def save_to_json(self, data: List[Dict], filename: str):
        """Save data to JSON"""
        if not data:
            logger.warning(f"No data to save to {filename}")
            return

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(data)} records to {filename}")


def main():
    """Example usage"""
    scraper = ATGAPIScraper(delay=1.0)

    # Scrape today
    today = datetime.now().strftime('%Y-%m-%d')

    # Or scrape historical data (last 7 days)
    # start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    # data = scraper.scrape_date_range(start_date, days=7)

    # Scrape just today
    data = scraper.scrape_date(today)

    # Save results
    if data:
        scraper.save_to_csv(data, 'atg_api_data.csv')
        scraper.save_to_json(data, 'atg_api_data.json')

        # Show summary
        df = pd.DataFrame(data)
        print(f"\n{'='*60}")
        print("DATA SUMMARY")
        print(f"{'='*60}")
        print(f"Total starts: {len(df)}")
        print(f"Unique tracks: {df['track_name'].nunique()}")
        print(f"Unique races: {df['race_id'].nunique()}")
        print(f"\nTracks: {df['track_name'].unique()}")
        print(f"\nColumns ({len(df.columns)}): {list(df.columns)}")
    else:
        print("No data collected")


if __name__ == "__main__":
    main()
