#!/usr/bin/env python3
"""
Temporal Data Processor - NO DATA LEAKAGE
Calculates features using only historical data available at prediction time
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TemporalDataProcessor:
    """
    Process trotting data with strict temporal awareness
    Features are calculated using ONLY past data (no future leakage)
    """

    def __init__(self):
        self.data = None

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load raw race data"""
        logger.info(f"Loading data from {filepath}")
        df = pd.read_csv(filepath)

        # Ensure date column is datetime
        # Handle both 'date' and 'race_date' column names
        if 'race_date' in df.columns:
            df['date'] = pd.to_datetime(df['race_date'])
        else:
            df['date'] = pd.to_datetime(df['date'])

        # Sort by date to ensure temporal ordering
        df = df.sort_values('date').reset_index(drop=True)

        logger.info(f"Loaded {len(df)} records")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"Unique dates: {df['date'].nunique()}")

        self.data = df
        return df

    def process_race_data(self, race_data_list: list, feature_cols: list = None) -> pd.DataFrame:
        """
        Process race data from API for live prediction

        Args:
            race_data_list: List of race data dictionaries from get_race_details()
            feature_cols: List of feature columns expected by the model (optional)

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
                horse_data = self._extract_horse_data_from_api(start, race_data)
                all_horse_data.append(horse_data)

        if not all_horse_data:
            logger.warning("No horse data extracted from race data")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(all_horse_data)

        # Filter out scratched horses (struken)
        if 'scratched' in df.columns:
            scratched_count = df['scratched'].sum()
            if scratched_count > 0:
                scratched_horses = df[df['scratched'] == True]['horse_name'].tolist()
                logger.info(f"Filtering out {scratched_count} scratched horse(s): {', '.join(scratched_horses)}")
                df = df[df['scratched'] != True].reset_index(drop=True)

        # Ensure date column exists
        if 'race_date' in df.columns:
            df['date'] = pd.to_datetime(df['race_date'])
        elif 'start_time' in df.columns:
            # Try to extract date from start_time if it's a full datetime
            df['date'] = pd.to_datetime(df['start_time'], errors='coerce')
        else:
            df['date'] = pd.Timestamp.now()

        # Apply feature engineering
        df = self._add_temporal_features(df)
        df = self._add_post_position_features(df)
        df = self._add_speed_features(df)
        df = self._add_track_features(df)

        # For live prediction, use stats from API (not temporal calculation)
        df = self._add_driver_trainer_features_from_api(df)

        # For horse form, use defaults (we don't have historical data)
        df = self._add_default_horse_form(df)

        # If feature_cols provided, ensure all features exist (fill missing with 0)
        if feature_cols:
            missing_features = [feat for feat in feature_cols if feat not in df.columns]
            if missing_features:
                # Create all missing columns at once for better performance
                missing_df = pd.DataFrame(0, index=df.index, columns=missing_features)
                df = pd.concat([df, missing_df], axis=1)

        # Replace infinity and NaN values
        df = df.replace([np.inf, -np.inf], 0)
        df = df.fillna(0)

        logger.info(f"Processed {len(df)} horses from {len(race_data_list)} race(s)")
        return df

    def _extract_horse_data_from_api(self, start: dict, race_info: dict) -> dict:
        """Extract horse data from API response"""
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

        # Get driver statistics (use most recent year)
        driver_stats = driver.get('statistics', {}).get('years', {})
        driver_year = max(driver_stats.keys()) if driver_stats else None
        driver_year_stats = driver_stats.get(driver_year, {}) if driver_year else {}

        # Get trainer statistics
        trainer_stats = trainer.get('statistics', {}).get('years', {})
        trainer_year = max(trainer_stats.keys()) if trainer_stats else None
        trainer_year_stats = trainer_stats.get(trainer_year, {}) if trainer_year else {}

        data = {
            # Race info
            'race_id': race_info.get('id'),
            'race_date': race_info.get('date'),
            'race_number': race_info.get('number'),
            'track_name': track.get('name'),
            'track_id': track.get('id'),
            'track_condition': track.get('condition'),
            'distance': race_info.get('distance'),
            'start_method': race_info.get('startMethod'),
            'start_time': race_info.get('startTime'),

            # Horse info
            'horse_id': horse.get('id'),
            'horse_name': horse.get('name'),
            'horse_age': horse.get('age'),
            'horse_sex': horse.get('sex'),
            'horse_money': horse.get('money'),

            # Start info
            'start_number': start.get('number'),
            'post_position': start.get('postPosition'),
            'distance_handicap': start.get('distance', 0) - race_info.get('distance', 0),

            # Record/performance
            'record_time': record_seconds,

            # Driver info
            'driver_id': driver.get('id'),
            'driver_first_name': driver.get('firstName'),
            'driver_last_name': driver.get('lastName'),
            'driver_starts_api': driver_year_stats.get('starts', 0),
            'driver_wins_api': driver_year_stats.get('wins', 0),
            'driver_second_api': driver_year_stats.get('second', 0),
            'driver_third_api': driver_year_stats.get('third', 0),
            'driver_earnings_api': driver_year_stats.get('money', 0),

            # Trainer info
            'trainer_id': trainer.get('id'),
            'trainer_first_name': trainer.get('firstName'),
            'trainer_last_name': trainer.get('lastName'),
            'trainer_starts_api': trainer_year_stats.get('starts', 0),
            'trainer_wins_api': trainer_year_stats.get('wins', 0),
            'trainer_second_api': trainer_year_stats.get('second', 0),
            'trainer_third_api': trainer_year_stats.get('third', 0),
            'trainer_earnings_api': trainer_year_stats.get('money', 0),

            # Results (for completed races)
            'finish_place': result.get('place', 0) if result else 0,
            'galloped': result.get('galloped', False) if result else False,

            # Scratched status (horse withdrawn from race)
            'scratched': start.get('scratched', False),
        }

        return data

    def _add_driver_trainer_features_from_api(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate driver/trainer features from API-provided stats"""
        df = df.copy()

        # Driver features
        df['driver_starts'] = df['driver_starts_api']
        df['driver_wins'] = df['driver_wins_api']
        df['driver_second'] = df['driver_second_api']
        df['driver_third'] = df['driver_third_api']
        df['driver_earnings'] = df['driver_earnings_api']

        df['driver_win_rate'] = np.where(
            df['driver_starts'] > 0,
            df['driver_wins'] / df['driver_starts'],
            0
        )
        df['driver_win_percentage'] = np.where(
            df['driver_starts'] > 0,
            (df['driver_wins'] / df['driver_starts']) * 100,
            0
        )
        df['driver_top3_rate'] = np.where(
            df['driver_starts'] > 0,
            (df['driver_wins'] + df['driver_second'] + df['driver_third']) / df['driver_starts'],
            0
        )

        # Trainer features
        df['trainer_starts'] = df['trainer_starts_api']
        df['trainer_wins'] = df['trainer_wins_api']
        df['trainer_second'] = df['trainer_second_api']
        df['trainer_third'] = df['trainer_third_api']
        df['trainer_earnings'] = df['trainer_earnings_api']

        df['trainer_win_rate'] = np.where(
            df['trainer_starts'] > 0,
            df['trainer_wins'] / df['trainer_starts'],
            0
        )
        df['trainer_win_percentage'] = np.where(
            df['trainer_starts'] > 0,
            (df['trainer_wins'] / df['trainer_starts']) * 100,
            0
        )
        df['trainer_top3_rate'] = np.where(
            df['trainer_starts'] > 0,
            (df['trainer_wins'] + df['trainer_second'] + df['trainer_third']) / df['trainer_starts'],
            0
        )

        return df

    def _add_default_horse_form(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add horse form features with default values (no historical data available)"""
        df = df.copy()

        # Use defaults for horse form (would need historical data to calculate properly)
        df['horse_recent_starts'] = 0
        df['horse_recent_wins'] = 0
        df['horse_recent_avg_position'] = 5.0  # Neutral default
        df['horse_days_since_last_race'] = 30  # Assume 30 days default

        return df

    def calculate_temporal_features(self, cutoff_date: str = None) -> pd.DataFrame:
        """
        Calculate features for all races, using only data BEFORE cutoff_date

        Args:
            cutoff_date: Only use data before this date for feature calculation
                        If None, calculates for all data (for initial training)

        Returns:
            DataFrame with features calculated temporally
        """
        if cutoff_date:
            cutoff_dt = pd.to_datetime(cutoff_date)
            logger.info(f"Calculating features using data up to {cutoff_date}")
        else:
            cutoff_dt = None
            logger.info("Calculating features using all available data")

        df = self.data.copy()

        # Add basic temporal features
        df = self._add_temporal_features(df)

        # Add post position features
        df = self._add_post_position_features(df)

        # Add speed features
        df = self._add_speed_features(df)

        # Add track features
        df = self._add_track_features(df)

        # Calculate rolling statistics (TEMPORAL)
        df = self._calculate_temporal_driver_stats(df, cutoff_date=cutoff_dt)
        df = self._calculate_temporal_trainer_stats(df, cutoff_date=cutoff_dt)
        df = self._calculate_temporal_horse_form(df, cutoff_date=cutoff_dt)

        # Create targets
        df = self._create_targets(df)

        logger.info(f"Processed {len(df)} records with {len(df.columns)} columns")

        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        df = df.copy()

        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        # Hour from start_time if available
        if 'start_time' in df.columns:
            df['hour'] = pd.to_datetime(df['start_time'], format='%H:%M', errors='coerce').dt.hour
            df['hour'] = df['hour'].fillna(14)  # Default to afternoon
        else:
            df['hour'] = 14

        return df

    def _add_post_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add post position features"""
        df = df.copy()

        # Inside/outside post
        df['is_inside_post'] = (df['post_position'] <= 5).astype(int)
        df['is_outside_post'] = (df['post_position'] > 7).astype(int)

        # Position ratio
        track_col = 'track_name' if 'track_name' in df.columns else 'track'
        df['horses_in_race'] = df.groupby(['date', track_col, 'race_number'])['start_number'].transform('count')
        df['position_ratio'] = df['post_position'] / df['horses_in_race']

        return df

    def _add_speed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add speed-based features"""
        df = df.copy()

        # Convert record time to speed (km/min)
        if 'record_time' in df.columns and 'distance' in df.columns:
            df['record_speed_kmmin'] = (df['distance'] / 1000) / (df['record_time'] / 60)
            df['record_speed_kmmin'] = df['record_speed_kmmin'].fillna(df['record_speed_kmmin'].median())

        return df

    def _add_track_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add track-based features"""
        df = df.copy()

        # Handle track column name
        track_col = 'track_name' if 'track_name' in df.columns else 'track'

        # Major tracks (high prize money)
        major_tracks = ['Solvalla', 'Åby', 'Jägersro', 'Axevalla', 'Bergsåker']
        df['is_major_track'] = df[track_col].isin(major_tracks).astype(int)

        # One-hot encode tracks
        track_dummies = pd.get_dummies(df[track_col], prefix='track')
        df = pd.concat([df, track_dummies], axis=1)

        # Start method encoding
        if 'start_method' in df.columns:
            df['start_auto'] = (df['start_method'] == 'auto').astype(int)
            df['start_volte'] = (df['start_method'] == 'volte').astype(int)

        # Track condition encoding
        if 'track_condition' in df.columns:
            df['condition_light'] = (df['track_condition'] == 'light').astype(int)
            df['condition_nan'] = df['track_condition'].isna().astype(int)

        return df

    def _calculate_temporal_driver_stats(self, df: pd.DataFrame, cutoff_date=None, window_days=90) -> pd.DataFrame:
        """
        Calculate driver statistics using ONLY past data

        For each race, we look back window_days and calculate stats
        from races that happened BEFORE this race date

        Args:
            df: DataFrame with race data
            cutoff_date: Only use data before this date
            window_days: Days to look back for statistics
        """
        logger.info(f"Calculating temporal driver stats (lookback: {window_days} days)")

        df = df.copy()
        df = df.sort_values('date').reset_index(drop=True)

        # Initialize columns
        df['driver_starts'] = 0
        df['driver_wins'] = 0
        df['driver_second'] = 0
        df['driver_third'] = 0
        df['driver_earnings'] = 0
        df['driver_win_rate'] = 0.0
        df['driver_top3_rate'] = 0.0
        df['driver_win_percentage'] = 0.0

        # Group by driver and process chronologically
        for driver_id in df['driver_id'].unique():
            if pd.isna(driver_id):
                continue

            driver_mask = df['driver_id'] == driver_id
            driver_races = df[driver_mask].copy()

            for idx in driver_races.index:
                race_date = df.loc[idx, 'date']

                # If cutoff_date is set and race is after cutoff, skip calculation
                if cutoff_date and race_date > cutoff_date:
                    continue

                # Look back window_days from this race
                lookback_start = race_date - timedelta(days=window_days)

                # Get historical races for this driver BEFORE current race
                historical = driver_races[
                    (driver_races['date'] >= lookback_start) &
                    (driver_races['date'] < race_date)  # STRICTLY BEFORE
                ]

                if len(historical) > 0:
                    # Calculate stats from historical data only
                    starts = len(historical)
                    wins = (historical['finish_place'] == 1).sum()
                    second = (historical['finish_place'] == 2).sum()
                    third = (historical['finish_place'] == 3).sum()
                    earnings = historical['prize_money'].fillna(0).sum()

                    df.loc[idx, 'driver_starts'] = starts
                    df.loc[idx, 'driver_wins'] = wins
                    df.loc[idx, 'driver_second'] = second
                    df.loc[idx, 'driver_third'] = third
                    df.loc[idx, 'driver_earnings'] = earnings
                    df.loc[idx, 'driver_win_rate'] = wins / starts if starts > 0 else 0
                    df.loc[idx, 'driver_top3_rate'] = (wins + second + third) / starts if starts > 0 else 0
                    df.loc[idx, 'driver_win_percentage'] = (wins / starts * 100) if starts > 0 else 0

        return df

    def _calculate_temporal_trainer_stats(self, df: pd.DataFrame, cutoff_date=None, window_days=90) -> pd.DataFrame:
        """Calculate trainer statistics using ONLY past data"""
        logger.info(f"Calculating temporal trainer stats (lookback: {window_days} days)")

        df = df.copy()
        df = df.sort_values('date').reset_index(drop=True)

        # Initialize columns
        df['trainer_starts'] = 0
        df['trainer_wins'] = 0
        df['trainer_second'] = 0
        df['trainer_third'] = 0
        df['trainer_earnings'] = 0
        df['trainer_win_rate'] = 0.0
        df['trainer_top3_rate'] = 0.0
        df['trainer_win_percentage'] = 0.0

        # Group by trainer and process chronologically
        for trainer_id in df['trainer_id'].unique():
            if pd.isna(trainer_id):
                continue

            trainer_mask = df['trainer_id'] == trainer_id
            trainer_races = df[trainer_mask].copy()

            for idx in trainer_races.index:
                race_date = df.loc[idx, 'date']

                if cutoff_date and race_date > cutoff_date:
                    continue

                lookback_start = race_date - timedelta(days=window_days)

                historical = trainer_races[
                    (trainer_races['date'] >= lookback_start) &
                    (trainer_races['date'] < race_date)
                ]

                if len(historical) > 0:
                    starts = len(historical)
                    wins = (historical['finish_place'] == 1).sum()
                    second = (historical['finish_place'] == 2).sum()
                    third = (historical['finish_place'] == 3).sum()
                    earnings = historical['prize_money'].fillna(0).sum()

                    df.loc[idx, 'trainer_starts'] = starts
                    df.loc[idx, 'trainer_wins'] = wins
                    df.loc[idx, 'trainer_second'] = second
                    df.loc[idx, 'trainer_third'] = third
                    df.loc[idx, 'trainer_earnings'] = earnings
                    df.loc[idx, 'trainer_win_rate'] = wins / starts if starts > 0 else 0
                    df.loc[idx, 'trainer_top3_rate'] = (wins + second + third) / starts if starts > 0 else 0
                    df.loc[idx, 'trainer_win_percentage'] = (wins / starts * 100) if starts > 0 else 0

        return df

    def _calculate_temporal_horse_form(self, df: pd.DataFrame, cutoff_date=None, last_n_races=5) -> pd.DataFrame:
        """Calculate horse form using ONLY past races"""
        logger.info(f"Calculating temporal horse form (last {last_n_races} races)")

        df = df.copy()
        df = df.sort_values('date').reset_index(drop=True)

        # Initialize columns
        df['horse_recent_starts'] = 0
        df['horse_recent_wins'] = 0
        df['horse_recent_avg_position'] = 0.0
        df['horse_days_since_last_race'] = 999

        # Group by horse and process chronologically
        for horse_id in df['horse_id'].unique():
            if pd.isna(horse_id):
                continue

            horse_mask = df['horse_id'] == horse_id
            horse_races = df[horse_mask].copy()

            for idx in horse_races.index:
                race_date = df.loc[idx, 'date']

                if cutoff_date and race_date > cutoff_date:
                    continue

                # Get historical races BEFORE current race
                historical = horse_races[horse_races['date'] < race_date]

                if len(historical) > 0:
                    # Get last N races
                    recent = historical.tail(last_n_races)

                    df.loc[idx, 'horse_recent_starts'] = len(recent)
                    df.loc[idx, 'horse_recent_wins'] = (recent['finish_place'] == 1).sum()

                    # Average position (only for finished races)
                    finished = recent[recent['finish_place'].notna()]
                    if len(finished) > 0:
                        df.loc[idx, 'horse_recent_avg_position'] = finished['finish_place'].mean()

                    # Days since last race
                    last_race_date = historical['date'].max()
                    days_diff = (race_date - last_race_date).days
                    df.loc[idx, 'horse_days_since_last_race'] = days_diff

        return df

    def _create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables"""
        df = df.copy()

        # Winner
        df['is_winner'] = (df['finish_place'] == 1).astype(int)

        # Top 3
        df['is_top3'] = df['finish_place'].isin([1, 2, 3]).astype(int)

        # Top 5 (placed)
        df['is_placed'] = df['finish_place'].isin([1, 2, 3, 4, 5]).astype(int)

        return df

    def get_train_test_split(self, df: pd.DataFrame, train_end_date: str, test_start_date: str = None):
        """
        Split data temporally

        Args:
            df: Processed DataFrame
            train_end_date: Last date to include in training (inclusive)
            test_start_date: First date in test set (if None, use day after train_end_date)

        Returns:
            train_df, test_df
        """
        train_end = pd.to_datetime(train_end_date)

        if test_start_date:
            test_start = pd.to_datetime(test_start_date)
        else:
            test_start = train_end + timedelta(days=1)

        train_df = df[df['date'] <= train_end].copy()
        test_df = df[df['date'] >= test_start].copy()

        logger.info(f"\nTemporal split:")
        logger.info(f"  Training: {train_df['date'].min()} to {train_df['date'].max()} ({len(train_df)} samples)")
        logger.info(f"  Testing:  {test_df['date'].min()} to {test_df['date'].max()} ({len(test_df)} samples)")

        return train_df, test_df


def main():
    """Example usage"""
    processor = TemporalDataProcessor()

    # Load data
    df = processor.load_data('trotting_data_2025_final.csv')

    # Calculate features temporally (no cutoff for initial processing)
    df_processed = processor.calculate_temporal_features()

    # Save processed data
    df_processed.to_csv('temporal_processed_data.csv', index=False)
    logger.info("\nSaved to temporal_processed_data.csv")

    # Example split: Train on Jan-Oct, Test on Nov-Dec
    train_df, test_df = processor.get_train_test_split(
        df_processed,
        train_end_date='2025-10-31',
        test_start_date='2025-11-01'
    )

    logger.info(f"\nTraining samples: {len(train_df)}")
    logger.info(f"Testing samples: {len(test_df)}")


if __name__ == "__main__":
    main()
