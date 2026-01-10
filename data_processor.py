#!/usr/bin/env python3
"""
Data Processing Pipeline for Swedish Trotting ML Model
Cleans and prepares scraped data for Random Forest modeling
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from typing import Dict, List, Tuple


class TrottingDataProcessor:
    """Process and prepare trotting data for machine learning"""
    
    def __init__(self):
        self.driver_stats = {}
        self.trainer_stats = {}
        self.horse_stats = {}
    
    def load_raw_data(self, filepath: str) -> pd.DataFrame:
        """Load raw scraped data"""
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif filepath.endswith('.json'):
            df = pd.read_json(filepath)
        else:
            raise ValueError("Unsupported file format")
        
        return df
    
    def extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features from race data"""
        df = df.copy()
        
        if 'start_time' in df.columns:
            df['start_time'] = pd.to_datetime(df['start_time'])
            df['hour'] = df['start_time'].dt.hour
            df['day_of_week'] = df['start_time'].dt.dayofweek
            df['month'] = df['start_time'].dt.month
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        return df
    
    def parse_record_time(self, time_str: str) -> float:
        """
        Parse Swedish trotting time format to seconds
        
        Examples: '1.15,0' -> 75.0, '1.13,5a' -> 73.5
        """
        if pd.isna(time_str) or time_str == '':
            return np.nan
        
        try:
            # Remove letters (like 'a' for auto start)
            time_str = re.sub(r'[a-zA-Z]', '', str(time_str))
            
            # Handle different formats
            if '.' in time_str and ',' in time_str:
                # Format: 1.15,0 (1 min 15.0 sec)
                parts = time_str.replace(',', '.').split('.')
                minutes = int(parts[0])
                seconds = int(parts[1])
                tenths = int(parts[2]) if len(parts) > 2 else 0
                return minutes * 60 + seconds + tenths / 10
            elif ',' in time_str:
                # Format: 75,0 (75.0 sec)
                return float(time_str.replace(',', '.'))
            else:
                return float(time_str)
        except:
            return np.nan
    
    def parse_distance(self, distance_str: str) -> int:
        """
        Parse distance from various formats
        
        Examples: '2140m', '2140', '2140 meter'
        """
        if pd.isna(distance_str):
            return np.nan
        
        try:
            # Extract numbers
            numbers = re.findall(r'\d+', str(distance_str))
            if numbers:
                return int(numbers[0])
        except:
            pass
        
        return np.nan
    
    def calculate_speed(self, time_seconds: float, distance_meters: int) -> float:
        """
        Calculate speed in km/min (standard for trotting)
        
        Args:
            time_seconds: Race time in seconds
            distance_meters: Race distance in meters
            
        Returns:
            Speed in km/min
        """
        if pd.isna(time_seconds) or pd.isna(distance_meters) or time_seconds == 0:
            return np.nan
        
        # Speed = (distance in km) / (time in minutes)
        speed = (distance_meters / 1000) / (time_seconds / 60)
        return round(speed, 2)
    
    def encode_post_position(self, position: int, total_horses: int) -> Dict:
        """
        Create features from post position
        
        Args:
            position: Starting position (1-12 typically)
            total_horses: Total number of horses in race
            
        Returns:
            Dictionary of position features
        """
        features = {
            'post_position': position,
            'is_inside_post': 1 if position <= 5 else 0,
            'is_outside_post': 1 if position >= 8 else 0,
            'position_ratio': position / total_horses if total_horses > 0 else 0
        }
        
        return features
    
    def calculate_driver_stats(self, df: pd.DataFrame, 
                               window_days: int = 90) -> pd.DataFrame:
        """
        Calculate rolling driver statistics
        
        Args:
            df: DataFrame with driver and result columns
            window_days: Days to look back for statistics
        """
        df = df.copy()
        df['date'] = pd.to_datetime(df['date']) if 'date' in df.columns else datetime.now()
        
        driver_features = []
        
        for idx, row in df.iterrows():
            driver = row.get('driver', 'Unknown')
            race_date = row['date']
            
            # Get historical races for this driver
            cutoff_date = race_date - timedelta(days=window_days)
            driver_history = df[
                (df['driver'] == driver) & 
                (df['date'] < race_date) & 
                (df['date'] >= cutoff_date)
            ]
            
            if len(driver_history) > 0:
                stats = {
                    'driver_starts_90d': len(driver_history),
                    'driver_wins_90d': (driver_history['position'] == 1).sum() if 'position' in driver_history else 0,
                    'driver_win_rate_90d': (driver_history['position'] == 1).mean() if 'position' in driver_history else 0,
                    'driver_avg_position_90d': driver_history['position'].mean() if 'position' in driver_history else np.nan,
                    'driver_top3_rate_90d': (driver_history['position'] <= 3).mean() if 'position' in driver_history else 0,
                }
            else:
                stats = {
                    'driver_starts_90d': 0,
                    'driver_wins_90d': 0,
                    'driver_win_rate_90d': 0,
                    'driver_avg_position_90d': np.nan,
                    'driver_top3_rate_90d': 0,
                }
            
            driver_features.append(stats)
        
        # Add to dataframe
        driver_df = pd.DataFrame(driver_features)
        df = pd.concat([df, driver_df], axis=1)
        
        return df
    
    def calculate_horse_form(self, df: pd.DataFrame, 
                            last_n_races: int = 5) -> pd.DataFrame:
        """
        Calculate horse form from recent races
        
        Args:
            df: DataFrame with horse performance data
            last_n_races: Number of recent races to consider
        """
        df = df.copy()
        
        horse_form = []
        
        for idx, row in df.iterrows():
            horse = row.get('horse_name', 'Unknown')
            race_date = row.get('date', datetime.now())
            
            # Get last N races for this horse
            horse_history = df[
                (df['horse_name'] == horse) & 
                (df['date'] < race_date)
            ].sort_values('date', ascending=False).head(last_n_races)
            
            if len(horse_history) > 0:
                form = {
                    'horse_races_count': len(horse_history),
                    'horse_avg_position': horse_history['position'].mean() if 'position' in horse_history else np.nan,
                    'horse_best_position': horse_history['position'].min() if 'position' in horse_history else np.nan,
                    'horse_worst_position': horse_history['position'].max() if 'position' in horse_history else np.nan,
                    'horse_wins_last_5': (horse_history['position'] == 1).sum() if 'position' in horse_history else 0,
                    'horse_days_since_last_race': (race_date - horse_history.iloc[0]['date']).days if len(horse_history) > 0 else np.nan,
                    'horse_avg_speed_last_5': horse_history['speed'].mean() if 'speed' in horse_history else np.nan,
                }
            else:
                form = {
                    'horse_races_count': 0,
                    'horse_avg_position': np.nan,
                    'horse_best_position': np.nan,
                    'horse_worst_position': np.nan,
                    'horse_wins_last_5': 0,
                    'horse_days_since_last_race': np.nan,
                    'horse_avg_speed_last_5': np.nan,
                }
            
            horse_form.append(form)
        
        form_df = pd.DataFrame(horse_form)
        df = pd.concat([df, form_df], axis=1)
        
        return df
    
    def encode_track(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode track/venue"""
        if 'track' in df.columns:
            track_dummies = pd.get_dummies(df['track'], prefix='track')
            df = pd.concat([df, track_dummies], axis=1)
        return df
    
    def create_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variables for prediction
        
        Creates:
        - is_winner: Binary (1 if horse won, 0 otherwise)
        - is_top3: Binary (1 if horse placed in top 3)
        - finish_position: Actual finishing position
        """
        df = df.copy()
        
        if 'position' in df.columns or 'finish_position' in df.columns:
            pos_col = 'position' if 'position' in df.columns else 'finish_position'
            
            df['is_winner'] = (df[pos_col] == 1).astype(int)
            df['is_top3'] = (df[pos_col] <= 3).astype(int)
            df['finish_position'] = df[pos_col]
        
        return df
    
    def prepare_for_ml(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Final preparation for machine learning
        
        Returns:
            Processed dataframe and list of feature column names
        """
        df = df.copy()
        
        # Define feature columns (exclude targets and metadata)
        exclude_cols = [
            'horse_name', 'driver', 'trainer', 'date', 'race_id', 
            'is_winner', 'is_top3', 'finish_position', 'position',
            'start_time', 'track'  # Categorical already encoded
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Handle missing values
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
        
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df[feature_cols] = df[feature_cols].fillna(0)
        
        return df, feature_cols
    
    def process_pipeline(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete processing pipeline
        
        Args:
            raw_df: Raw scraped data
            
        Returns:
            Processed DataFrame ready for ML
        """
        print("Starting data processing pipeline...")
        
        # 1. Extract time features
        print("Extracting time features...")
        df = self.extract_time_features(raw_df)
        
        # 2. Parse and calculate speed
        print("Calculating speed metrics...")
        if 'record' in df.columns:
            df['record_seconds'] = df['record'].apply(self.parse_record_time)
        
        if 'distance' in df.columns:
            df['distance_meters'] = df['distance'].apply(self.parse_distance)
            
        if 'record_seconds' in df.columns and 'distance_meters' in df.columns:
            df['speed'] = df.apply(
                lambda row: self.calculate_speed(row['record_seconds'], row['distance_meters']), 
                axis=1
            )
        
        # 3. Encode post position
        print("Encoding post positions...")
        if 'post_position' in df.columns:
            # Calculate number of horses per race
            df['horses_in_race'] = df.groupby('race_id')['horse_name'].transform('count')
        
        # 4. Calculate driver statistics
        print("Calculating driver statistics...")
        if 'driver' in df.columns and 'date' in df.columns:
            df = self.calculate_driver_stats(df)
        
        # 5. Calculate horse form
        print("Calculating horse form...")
        if 'horse_name' in df.columns:
            df = self.calculate_horse_form(df)
        
        # 6. Encode categorical variables
        print("Encoding categorical variables...")
        df = self.encode_track(df)
        
        # 7. Create target variables
        print("Creating target variables...")
        df = self.create_target_variable(df)
        
        print(f"Processing complete! Dataset shape: {df.shape}")
        
        return df


def main():
    """Example usage"""
    
    processor = TrottingDataProcessor()
    
    # Load raw data
    print("Loading raw data...")
    # raw_df = processor.load_raw_data('atg_races.csv')
    
    # For demo, create sample data
    sample_data = {
        'race_id': ['R1', 'R1', 'R1', 'R2', 'R2'],
        'horse_name': ['Häst A', 'Häst B', 'Häst C', 'Häst A', 'Häst D'],
        'driver': ['Kusk 1', 'Kusk 2', 'Kusk 1', 'Kusk 1', 'Kusk 3'],
        'trainer': ['Tränare 1', 'Tränare 2', 'Tränare 1', 'Tränare 1', 'Tränare 3'],
        'post_position': [1, 5, 8, 3, 6],
        'distance': ['2140m', '2140m', '2140m', '1640m', '1640m'],
        'record': ['1.15,0', '1.15,5', '1.16,2', '1.13,0', '1.14,0'],
        'position': [1, 2, 3, 2, 1],
        'date': ['2025-01-01', '2025-01-01', '2025-01-01', '2025-01-08', '2025-01-08'],
        'track': ['Solvalla', 'Solvalla', 'Solvalla', 'Åby', 'Åby'],
        'odds': ['3.5', '5.2', '8.1', '4.0', '2.8'],
    }
    
    raw_df = pd.DataFrame(sample_data)
    
    # Process data
    processed_df = processor.process_pipeline(raw_df)
    
    # Prepare for ML
    ml_ready_df, feature_cols = processor.prepare_for_ml(processed_df)
    
    print(f"\nFeature columns ({len(feature_cols)}):")
    for col in feature_cols[:20]:
        print(f"  - {col}")
    
    print(f"\nTarget variables:")
    for col in ['is_winner', 'is_top3', 'finish_position']:
        if col in ml_ready_df.columns:
            print(f"  - {col}: {ml_ready_df[col].value_counts().to_dict()}")
    
    # Save processed data
    ml_ready_df.to_csv('processed_trotting_data.csv', index=False)
    print("\nProcessed data saved to 'processed_trotting_data.csv'")


if __name__ == "__main__":
    main()
