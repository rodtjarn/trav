#!/usr/bin/env python3
"""
Data processor for ATG API data
Prepares scraped API data for Random Forest ML models
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class APIDataProcessor:
    """Process ATG API data for machine learning"""

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load API data from CSV"""
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} records from {filepath}")
        return df

    def process_race_data(self, race_data_list: list) -> pd.DataFrame:
        """
        Process race data from API into ML-ready DataFrame

        Args:
            race_data_list: List of race data dictionaries from get_race_details()

        Returns:
            Processed DataFrame ready for prediction
        """
        # Extract horse data from each race
        all_horse_data = []

        for race_data in race_data_list:
            if not race_data:
                continue

            # Extract data for each start/horse
            for start in race_data.get('starts', []):
                horse_data = self._extract_horse_data(start, race_data)
                all_horse_data.append(horse_data)

        if not all_horse_data:
            logger.warning("No horse data extracted from race data")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(all_horse_data)

        # Apply all processing steps
        df = self.add_temporal_features(df)
        df = self.add_post_position_features(df)
        df = self.add_driver_trainer_rates(df)
        df = self.add_speed_features(df)
        df = self.add_track_importance(df)
        df = self.encode_categorical(df)

        # Don't create target variables for prediction data

        # Prepare for ML (handle missing values, etc)
        df, feature_cols, metadata_cols = self.prepare_for_ml(df)

        logger.info(f"Processed {len(df)} horses from {len(race_data_list)} race(s)")
        return df

    def _extract_horse_data(self, start: dict, race_info: dict) -> dict:
        """
        Extract all relevant data for one horse/start
        (Same logic as ATGAPIScraper.extract_horse_data)

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
            'start_method': race_info.get('startMethod'),
            'start_time': race_info.get('startTime'),

            # Horse info
            'horse_id': horse.get('id'),
            'horse_name': horse.get('name'),
            'horse_age': horse.get('age'),
            'horse_sex': horse.get('sex'),
            'horse_money': horse.get('money'),
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

            # Driver stats
            'driver_starts': driver_year_stats.get('starts', 0),
            'driver_wins': driver_year_stats.get('wins', 0),
            'driver_second': driver_year_stats.get('second', 0),
            'driver_third': driver_year_stats.get('third', 0),
            'driver_earnings': driver_year_stats.get('money', 0),

            # Trainer info
            'trainer_id': trainer.get('id'),
            'trainer_first_name': trainer.get('firstName'),
            'trainer_last_name': trainer.get('lastName'),
            'trainer_short_name': trainer.get('shortName'),
            'trainer_license': trainer.get('license'),

            # Trainer stats
            'trainer_starts': trainer_year_stats.get('starts', 0),
            'trainer_wins': trainer_year_stats.get('wins', 0),
            'trainer_second': trainer_year_stats.get('second', 0),
            'trainer_third': trainer_year_stats.get('third', 0),
            'trainer_earnings': trainer_year_stats.get('money', 0),

            # Equipment
            'shoes_front': start.get('horse', {}).get('shoes', {}).get('front'),
            'shoes_back': start.get('horse', {}).get('shoes', {}).get('back'),
            'sulky_type': start.get('horse', {}).get('sulky'),

            # Results (for completed races)
            'finish_place': result.get('place', 0) if result else 0,
            'finish_order': result.get('order', 0) if result else 0,
            'finish_time': finish_seconds if finish_seconds > 0 else None,
            'galloped': result.get('galloped', False) if result else False,
            'prize_money': result.get('prizeMoney', 0) if result else 0,
            'final_odds': result.get('finalOdds', None) if result else None,
        }

        return data

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features from race date/time"""
        df = df.copy()

        # Parse datetime
        df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
        df['race_date'] = pd.to_datetime(df['race_date'], errors='coerce')

        # Extract features
        df['hour'] = df['start_time'].dt.hour
        df['day_of_week'] = df['start_time'].dt.dayofweek
        df['month'] = df['start_time'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        logger.info("Added temporal features")
        return df

    def add_post_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from post position"""
        df = df.copy()

        # Count horses per race
        df['horses_in_race'] = df.groupby('race_id')['start_number'].transform('count')

        # Post position features
        df['is_inside_post'] = (df['post_position'] <= 5).astype(int)
        df['is_outside_post'] = (df['post_position'] >= 8).astype(int)
        df['position_ratio'] = df['post_position'] / df['horses_in_race']

        logger.info("Added post position features")
        return df

    def add_driver_trainer_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate win rates and top-3 rates"""
        df = df.copy()

        # Driver rates
        df['driver_win_rate'] = np.where(
            df['driver_starts'] > 0,
            df['driver_wins'] / df['driver_starts'],
            0
        )
        df['driver_top3_rate'] = np.where(
            df['driver_starts'] > 0,
            (df['driver_wins'] + df['driver_second'] + df['driver_third']) / df['driver_starts'],
            0
        )

        # Trainer rates
        df['trainer_win_rate'] = np.where(
            df['trainer_starts'] > 0,
            df['trainer_wins'] / df['trainer_starts'],
            0
        )
        df['trainer_top3_rate'] = np.where(
            df['trainer_starts'] > 0,
            (df['trainer_wins'] + df['trainer_second'] + df['trainer_third']) / df['trainer_starts'],
            0
        )

        logger.info("Calculated driver/trainer rates")
        return df

    def add_speed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate speed-related features"""
        df = df.copy()

        # Speed in km/min from record time
        df['record_speed_kmmin'] = np.where(
            df['record_time'] > 0,
            (df['distance'] / 1000) / (df['record_time'] / 60),
            np.nan
        )

        # Speed from finish time (for completed races)
        df['finish_speed_kmmin'] = np.where(
            df['finish_time'] > 0,
            (df['distance'] / 1000) / (df['finish_time'] / 60),
            np.nan
        )

        logger.info("Added speed features")
        return df

    def add_track_importance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add track importance indicator"""
        df = df.copy()

        # Major Swedish tracks
        major_tracks = ['Solvalla', 'Jägersro', 'Åby', 'Gävle', 'Eskilstuna']
        df['is_major_track'] = df['track_name'].isin(major_tracks).astype(int)

        logger.info("Added track importance features")
        return df

    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode categorical variables"""
        df = df.copy()

        # Track encoding (only for tracks with enough samples)
        track_counts = df['track_name'].value_counts()
        frequent_tracks = track_counts[track_counts >= 20].index.tolist()

        df['track_encoded'] = df['track_name'].where(
            df['track_name'].isin(frequent_tracks),
            'Other'
        )
        track_dummies = pd.get_dummies(df['track_encoded'], prefix='track')
        df = pd.concat([df, track_dummies], axis=1)

        # Start method encoding (monte vs auto)
        if 'start_method' in df.columns:
            start_dummies = pd.get_dummies(df['start_method'], prefix='start')
            df = pd.concat([df, start_dummies], axis=1)

        # Track condition encoding
        if 'track_condition' in df.columns:
            condition_dummies = pd.get_dummies(df['track_condition'], prefix='condition', dummy_na=True)
            df = pd.concat([df, condition_dummies], axis=1)

        logger.info("Encoded categorical variables")
        return df

    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ML target variables"""
        df = df.copy()

        # Handle finish_place = 0 (unplaced/galloped)
        # If galloped is True, they're disqualified (oplacerad)
        # If galloped is False and finish_place is 0, might be scratched/cancelled

        # Winner (1st place, and didn't gallop)
        df['is_winner'] = ((df['finish_place'] == 1) & (~df['galloped'])).astype(int)

        # Top 3 (1-3 place, and didn't gallop)
        df['is_top3'] = ((df['finish_place'].isin([1, 2, 3])) & (~df['galloped'])).astype(int)

        # Placed (any valid position, didn't gallop)
        df['is_placed'] = ((df['finish_place'] > 0) & (~df['galloped'])).astype(int)

        # For regression: use finish_place directly (0 = unplaced)
        df['finish_position'] = df['finish_place']

        logger.info("Created target variables")
        return df

    def filter_for_training(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter data to only completed races with results"""
        df = df.copy()

        # Keep only races with results (finish_place is not null)
        df_filtered = df[df['finish_place'].notna()].copy()

        logger.info(f"Filtered to {len(df_filtered)} starts with results (from {len(df)} total)")
        return df_filtered

    def prepare_for_ml(self, df: pd.DataFrame) -> tuple:
        """
        Final preparation for ML

        Returns:
            (df, feature_cols, metadata_cols)
        """
        df = df.copy()

        # Define metadata columns (not used as features)
        metadata_cols = [
            'race_id', 'race_date', 'race_number', 'start_time',
            'horse_id', 'horse_name', 'horse_color',
            'driver_id', 'driver_first_name', 'driver_last_name', 'driver_short_name',
            'trainer_id', 'trainer_first_name', 'trainer_last_name', 'trainer_short_name',
            'track_name', 'track_encoded', 'start_method', 'track_condition',
            # Targets
            'is_winner', 'is_top3', 'is_placed', 'finish_position',
            'finish_place', 'finish_order', 'finish_time', 'finish_speed_kmmin',
            # Result data (not features)
            'prize_money', 'final_odds', 'record_code', 'driver_license', 'trainer_license',
            'shoes_front', 'shoes_back', 'sulky_type',
        ]

        # Feature columns = all numeric columns except metadata
        all_cols = set(df.columns)
        excluded_cols = set(metadata_cols)
        feature_cols = sorted(list(all_cols - excluded_cols))

        # Keep only numeric features
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = numeric_cols

        # Handle missing values
        for col in feature_cols:
            if df[col].isna().any():
                # For ratio features, fill with 0
                if 'rate' in col or 'ratio' in col or 'percentage' in col:
                    df[col] = df[col].fillna(0)
                else:
                    # For other features, fill with median
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)

        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        for col in feature_cols:
            df[col] = df[col].fillna(0)

        logger.info(f"Prepared {len(feature_cols)} features for ML")
        return df, feature_cols, metadata_cols

    def process_pipeline(self, input_file: str, output_file: str = None,
                        training_only: bool = True) -> pd.DataFrame:
        """
        Complete processing pipeline

        Args:
            input_file: Input CSV file from ATG API scraper
            output_file: Output CSV file (optional)
            training_only: If True, filter to only completed races with results

        Returns:
            Processed DataFrame ready for ML
        """
        logger.info(f"Starting processing pipeline for {input_file}")

        # Load data
        df = self.load_data(input_file)

        # Apply transformations
        df = self.add_temporal_features(df)
        df = self.add_post_position_features(df)
        df = self.add_driver_trainer_rates(df)
        df = self.add_speed_features(df)
        df = self.add_track_importance(df)
        df = self.encode_categorical(df)
        df = self.create_target_variables(df)

        # Filter to training data
        if training_only:
            df = self.filter_for_training(df)

        # Prepare for ML
        df, feature_cols, metadata_cols = self.prepare_for_ml(df)

        # Save if requested
        if output_file:
            df.to_csv(output_file, index=False)
            logger.info(f"Saved processed data to {output_file}")

        # Print summary
        logger.info(f"\n{'='*70}")
        logger.info("PROCESSING COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Total records: {len(df)}")
        logger.info(f"Features: {len(feature_cols)}")
        logger.info(f"Targets available:")
        for target in ['is_winner', 'is_top3', 'is_placed']:
            if target in df.columns:
                pos = df[target].sum()
                logger.info(f"  {target}: {pos} / {len(df)} ({pos/len(df)*100:.1f}%)")

        logger.info(f"\nFeature list ({len(feature_cols)} features):")
        for col in feature_cols[:30]:  # Show first 30
            logger.info(f"  - {col}")
        if len(feature_cols) > 30:
            logger.info(f"  ... and {len(feature_cols) - 30} more")

        return df, feature_cols


def main():
    """Example usage"""
    processor = APIDataProcessor()

    # Process the historical data
    df, feature_cols = processor.process_pipeline(
        input_file='atg_historical_data.csv',
        output_file='processed_ml_data.csv',
        training_only=True
    )

    print(f"\n{'='*70}")
    print("SAMPLE DATA")
    print(f"{'='*70}")
    print(df[['horse_name', 'track_name', 'is_winner', 'is_top3', 'is_placed',
              'post_position', 'driver_win_rate', 'record_speed_kmmin']].head(10))


if __name__ == "__main__":
    main()
