#!/usr/bin/env python3
"""
V85 Race Predictor
Uses trained Random Forest model to predict winners in V85 races
"""

import pandas as pd
import pickle
import json
from datetime import datetime, timedelta
from atg_api_scraper import ATGAPIScraper
from api_data_processor import APIDataProcessor
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class V85Predictor:
    """Predict V85 race winners using trained RF model"""

    def __init__(self, model_path='trotting_rf_model_extended.pkl',
                 metadata_path='trotting_rf_metadata_extended.json'):
        """
        Initialize predictor

        Args:
            model_path: Path to trained model
            metadata_path: Path to model metadata
        """
        # Load model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        logger.info(f"Loaded model from {model_path}")

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.feature_cols = self.metadata['feature_columns']
        logger.info(f"Model uses {len(self.feature_cols)} features")

        self.scraper = ATGAPIScraper(delay=0.5)
        self.processor = APIDataProcessor()

    def find_v85_races(self, days_ahead=7):
        """
        Find upcoming V85 races

        Args:
            days_ahead: Number of days to search ahead

        Returns:
            List of dates with V85 races
        """
        logger.info("Searching for V85 races...")
        v85_dates = []

        for day_offset in range(days_ahead):
            date = (datetime.now() + timedelta(days=day_offset)).strftime('%Y-%m-%d')

            # Get calendar
            calendar = self.scraper.get_calendar_for_date(date)

            if calendar:
                # Look for V85 races
                for track in calendar.get('tracks', []):
                    for race in track.get('races', []):
                        # V85 races have specific betting pools
                        pools = race.get('mergedPools', [])
                        for pool in pools:
                            bet_types = pool.get('betTypes', [])
                            # V85 is indicated by specific bet type
                            if 'v85' in [bt.lower() for bt in bet_types]:
                                v85_dates.append({
                                    'date': date,
                                    'track': track.get('name'),
                                    'race_id': race.get('id')
                                })
                                logger.info(f"Found V85 on {date} at {track.get('name')}")

        return v85_dates

    def get_v85_races_for_date(self, date):
        """
        Get all races for a specific V85 date

        V85 consists of 8 races (can be any races at the track, not necessarily 1-8)

        Args:
            date: Date string in YYYY-MM-DD format

        Returns:
            DataFrame with all race data including v85_race_number column
        """
        logger.info(f"Fetching V85 races for {date}")

        # Get V85 game info to find which races are included
        v85_info = self.scraper.get_v85_info(date)

        if not v85_info:
            logger.error(f"No V85 game found for {date}")
            return None

        # Scrape the date
        race_data = self.scraper.scrape_date(date)

        if not race_data:
            logger.error(f"No data found for {date}")
            return None

        df = pd.DataFrame(race_data)

        # Filter to only V85 races using the race_id mapping
        df = df[df['race_id'].isin(v85_info['race_ids'])]

        # Add V85 race number (1-8) based on the mapping
        df['v85_race_number'] = df['race_id'].map(v85_info['v85_race_mapping'])

        logger.info(f"V85 Game: {v85_info['game_id']}")
        logger.info(f"Loaded {len(df)} horses from {df['v85_race_number'].nunique()} V85 races")
        logger.info(f"Track race numbers: {sorted(df['race_number'].unique())}")
        logger.info(f"V85 race numbers: {sorted(df['v85_race_number'].unique())}")

        return df

    def prepare_race_data(self, df_raw):
        """
        Prepare race data for prediction (same pipeline as training)

        Args:
            df_raw: Raw race data from scraper

        Returns:
            Processed DataFrame ready for prediction
        """
        # Apply same transformations as training
        df = df_raw.copy()

        df = self.processor.add_temporal_features(df)
        df = self.processor.add_post_position_features(df)
        df = self.processor.add_driver_trainer_rates(df)
        df = self.processor.add_speed_features(df)
        df = self.processor.add_track_importance(df)
        df = self.processor.encode_categorical(df)

        # Don't create targets for prediction data

        # Get feature columns
        all_cols = set(df.columns)
        metadata_cols = {
            'race_id', 'race_date', 'race_number', 'start_time',
            'horse_id', 'horse_name', 'horse_color',
            'driver_id', 'driver_first_name', 'driver_last_name', 'driver_short_name',
            'trainer_id', 'trainer_first_name', 'trainer_last_name', 'trainer_short_name',
            'track_name', 'track_encoded', 'start_method', 'track_condition',
            'finish_place', 'finish_order', 'finish_time', 'finish_speed_kmmin',
            'prize_money', 'final_odds', 'record_code', 'driver_license', 'trainer_license',
            'shoes_front', 'shoes_back', 'sulky_type', 'galloped',
        }

        potential_features = all_cols - metadata_cols
        numeric_features = []
        for col in sorted(potential_features):
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_features.append(col)

        # Ensure we have all required features (fill missing with 0)
        for feat in self.feature_cols:
            if feat not in df.columns:
                df[feat] = 0

        # Fill NaN values
        for col in self.feature_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        return df

    def predict_race(self, df_race):
        """
        Predict winners for a specific race

        Args:
            df_race: DataFrame with horses in the race

        Returns:
            DataFrame with predictions
        """
        # Ensure we have all features
        X = df_race[self.feature_cols]

        # Predict probabilities
        probabilities = self.model.predict_proba(X)[:, 1]

        # Create results
        results = pd.DataFrame({
            'v85_race_number': df_race['v85_race_number'],
            'track_race_number': df_race['race_number'],
            'horse_name': df_race['horse_name'],
            'start_number': df_race['start_number'],
            'post_position': df_race['post_position'],
            'driver': df_race['driver_first_name'] + ' ' + df_race['driver_last_name'],
            'trainer': df_race['trainer_first_name'] + ' ' + df_race['trainer_last_name'],
            'win_probability': probabilities,
            'final_odds': df_race.get('final_odds', None),
        })

        results = results.sort_values('win_probability', ascending=False)

        return results

    def predict_v85(self, date):
        """
        Predict all V85 races for a specific date

        Args:
            date: Date string in YYYY-MM-DD format

        Returns:
            Dictionary with predictions for each V85 race number (1-8)
        """
        # Get race data
        df_raw = self.get_v85_races_for_date(date)

        if df_raw is None or len(df_raw) == 0:
            logger.error("No V85 races found")
            return None

        # Process data
        df_processed = self.prepare_race_data(df_raw)

        # Predict each race (using V85 race numbers 1-8)
        predictions = {}

        for v85_race_num in sorted(df_processed['v85_race_number'].unique()):
            df_race = df_processed[df_processed['v85_race_number'] == v85_race_num]
            race_pred = self.predict_race(df_race)
            predictions[v85_race_num] = race_pred

        return predictions

    def display_v85_predictions(self, predictions, show_top_n=5):
        """Display V85 predictions in a nice format"""

        print("\n" + "="*100)
        print("V85 RACE PREDICTIONS - RANDOM FOREST MODEL")
        print("="*100)

        for v85_race_num in sorted(predictions.keys()):
            race_pred = predictions[v85_race_num]

            # Get track race number for reference
            track_race_num = race_pred.iloc[0]['track_race_number'] if 'track_race_number' in race_pred.columns else '?'

            print(f"\n{'─'*100}")
            print(f"V85 RACE {v85_race_num} (Track Race {track_race_num}) - Top {show_top_n} Predictions")
            print(f"{'─'*100}")
            print(f"{'Rank':<6} {'Horse':<25} {'#':<4} {'Post':<6} {'Driver':<20} {'Win%':<8} {'Odds':<8}")
            print(f"{'─'*100}")

            for idx, row in race_pred.head(show_top_n).iterrows():
                rank = list(race_pred.index).index(idx) + 1
                win_pct = row['win_probability'] * 100
                odds = f"{row['final_odds']:.1f}" if pd.notna(row['final_odds']) else "N/A"

                print(f"{rank:<6} {row['horse_name']:<25} {int(row['start_number']):<4} "
                      f"{int(row['post_position']) if pd.notna(row['post_position']) else 'N/A':<6} "
                      f"{row['driver'][:20]:<20} {win_pct:>6.1f}% {odds:>7}")

        print("\n" + "="*100)
        print("SUGGESTED V85 COMBINATION (Top probability from each race):")
        print("="*100)

        combination = []
        for v85_race_num in sorted(predictions.keys()):
            top_pick = predictions[v85_race_num].iloc[0]
            combination.append(f"V85 R{v85_race_num}: #{int(top_pick['start_number'])} {top_pick['horse_name']}")

        for pick in combination:
            print(f"  {pick}")

        print("\n" + "="*100)
        print("⚠️  DISCLAIMER: For entertainment/research purposes only.")
        print("    Past performance doesn't guarantee future results.")
        print("    Gamble responsibly within your means.")
        print("="*100)


def main():
    """Main execution"""

    predictor = V85Predictor()

    # Find V85 races this week
    v85_races = predictor.find_v85_races(days_ahead=7)

    if not v85_races:
        logger.warning("No V85 races found in the next 7 days")
        logger.info("Trying today's date anyway...")
        date = datetime.now().strftime('%Y-%m-%d')
    else:
        # Use the first V85 date found
        date = v85_races[0]['date']
        logger.info(f"Using V85 date: {date}")

    # Predict V85
    predictions = predictor.predict_v85(date)

    if predictions:
        # Display predictions
        predictor.display_v85_predictions(predictions, show_top_n=5)
    else:
        logger.error("Failed to generate predictions")


if __name__ == "__main__":
    main()
