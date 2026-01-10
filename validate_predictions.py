#!/usr/bin/env python3
"""
Validate v85 predictions against actual results
"""

import pandas as pd
from atg_api_scraper import ATGAPIScraper
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_v85_predictions(date='2026-01-10', predictions=None):
    """
    Validate predictions against actual results

    Args:
        date: Date to check (YYYY-MM-DD)
        predictions: Optional dict of predictions to validate
    """

    scraper = ATGAPIScraper(delay=0.5)

    logger.info(f"Fetching results for {date}")

    # Get V85 game info to find which races are included
    v85_info = scraper.get_v85_info(date)

    if not v85_info:
        logger.error(f"No V85 game found for {date}")
        return

    # Get race data
    race_data = scraper.scrape_date(date)

    if not race_data:
        logger.error("No data found")
        return

    df = pd.DataFrame(race_data)

    # Filter to only V85 races using the race_id mapping
    df = df[df['race_id'].isin(v85_info['race_ids'])]

    # Add V85 race number (1-8) based on the mapping
    df['v85_race_number'] = df['race_id'].map(v85_info['v85_race_mapping'])

    # Filter to only completed races with results
    df = df[df['finish_place'].notna()]

    logger.info(f"Found {len(df)} horses from {df['v85_race_number'].nunique()} V85 races")
    logger.info(f"Track race numbers: {sorted(df['race_number'].unique())}")
    logger.info(f"V85 race numbers: {sorted(df['v85_race_number'].unique())}")

    # Use provided predictions or default to previous output
    if predictions is None:
        predictions = {
            1: {'horse': 'Grisle Kongen G.L.', 'number': 7, 'probability': 48.8},
            2: {'horse': 'Pack Control', 'number': 6, 'probability': 50.1},
            3: {'horse': 'Åsrud Jerven', 'number': 1, 'probability': 32.2},
            4: {'horse': 'Nephtys Boko', 'number': 1, 'probability': 41.5},
            5: {'horse': 'Krut', 'number': 5, 'probability': 46.2},
            6: {'horse': 'Freako', 'number': 1, 'probability': 51.1},
            7: {'horse': 'Hankypanky Slander', 'number': 4, 'probability': 53.6},
        }

    track_name = df['track_name'].iloc[0] if len(df) > 0 else 'Unknown'

    print("\n" + "="*100)
    print(f"V85 PREDICTION VALIDATION - {track_name} {date}")
    print(f"V85 Game: {v85_info['game_id']}")
    print("="*100)

    correct_predictions = 0
    total_races = 0
    top3_predictions = 0

    for v85_race_num in sorted(df['v85_race_number'].unique()):
        race_df = df[df['v85_race_number'] == v85_race_num].copy()
        race_df = race_df.sort_values('finish_place')

        # Get actual winner
        winners = race_df[race_df['finish_place'] == 1]

        if len(winners) == 0:
            continue

        actual_winner = winners.iloc[0]
        total_races += 1

        # Get our prediction
        prediction = predictions.get(v85_race_num)

        # Get track race number for display
        track_race_num = race_df.iloc[0]['race_number']

        print(f"\n{'─'*100}")
        print(f"V85 RACE {v85_race_num} (Track Race {track_race_num})")
        print(f"{'─'*100}")

        # Show actual top 3
        print(f"\n{'Actual Results:':<20}")
        for idx, row in race_df.head(3).iterrows():
            place = int(row['finish_place']) if row['finish_place'] > 0 else 'DNF'
            gallop_mark = ' (GALLOPED)' if row['galloped'] else ''
            print(f"  {place}. #{int(row['start_number']):<3} {row['horse_name']:<25} "
                  f"Time: {row['finish_time']:.1f}s{gallop_mark}")

        # Show our prediction
        if prediction:
            print(f"\n{'Our Prediction:':<20}")
            print(f"  #{prediction['number']} {prediction['horse']} "
                  f"(Win probability: {prediction['probability']:.1f}%)")

            # Check if correct
            if actual_winner['start_number'] == prediction['number']:
                print(f"  ✅ CORRECT! Our top pick won!")
                correct_predictions += 1
            else:
                # Check if our pick placed in top 3
                our_pick = race_df[race_df['start_number'] == prediction['number']]
                if len(our_pick) > 0:
                    our_place = our_pick.iloc[0]['finish_place']
                    if our_place <= 3 and our_place > 0:
                        print(f"  📊 Our pick finished {int(our_place)}. (Top 3)")
                        top3_predictions += 1
                    else:
                        if our_pick.iloc[0]['galloped']:
                            print(f"  ❌ Our pick galloped (disqualified)")
                        else:
                            print(f"  ❌ Our pick finished {int(our_place)}.")
                else:
                    print(f"  ❌ Incorrect prediction")

        # Show actual winner details
        print(f"\n{'Winner Details:':<20}")
        print(f"  Horse: {actual_winner['horse_name']}")
        print(f"  Number: #{int(actual_winner['start_number'])}")
        print(f"  Post: {int(actual_winner['post_position']) if pd.notna(actual_winner['post_position']) else 'N/A'}")
        print(f"  Driver: {actual_winner['driver_first_name']} {actual_winner['driver_last_name']}")
        print(f"  Final Odds: {actual_winner['final_odds']:.1f}" if pd.notna(actual_winner['final_odds']) else "  Final Odds: N/A")
        print(f"  Time: {actual_winner['finish_time']:.1f}s")

    # Summary
    print(f"\n" + "="*100)
    print("VALIDATION SUMMARY")
    print("="*100)
    print(f"Total v85 Races: {total_races}")
    print(f"Correct Winner Predictions: {correct_predictions}/{total_races} ({correct_predictions/total_races*100:.1f}%)")
    print(f"Top-3 Predictions: {top3_predictions}/{total_races} ({top3_predictions/total_races*100:.1f}%)")
    print(f"Total in Top-3 or Winner: {correct_predictions + top3_predictions}/{total_races} "
          f"({(correct_predictions + top3_predictions)/total_races*100:.1f}%)")

    # Expected performance analysis
    print(f"\n" + "="*100)
    print("PERFORMANCE ANALYSIS")
    print("="*100)

    avg_predicted_prob = sum(p['probability'] for p in predictions.values()) / len(predictions)
    print(f"Average predicted win probability: {avg_predicted_prob:.1f}%")
    print(f"Expected wins (based on probabilities): {avg_predicted_prob/100 * total_races:.1f}")
    print(f"Actual wins: {correct_predictions}")

    if correct_predictions > 0:
        print(f"\n✅ Model performed {'better than' if correct_predictions > avg_predicted_prob/100 * total_races else 'as'} expected!")
    else:
        print(f"\n⚠️  No winners predicted correctly this time.")
        print(f"   Note: With average {avg_predicted_prob:.1f}% probability, we'd expect ~{avg_predicted_prob/100 * total_races:.1f} winners.")
        print(f"   Small sample size - model needs more validation on multiple v85 rounds.")

    print(f"\n" + "="*100)


if __name__ == "__main__":
    validate_v85_predictions()
